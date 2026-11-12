/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/native_executor.h>

#include <executorch/backends/native/runtime/OffGraphCache.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/extension/llm/batching/util/session_table.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/options.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace executorch::extension::llm {

namespace {
constexpr char kMethod[] = "forward";
} // namespace

struct MuseGlimmerNativeExecutor::Native {
  std::shared_ptr<const ptn::Package> package;
  std::shared_ptr<const ptn::Program> program;
  std::unique_ptr<ptn::GgmlEngineHost> host;
  std::shared_ptr<cache::Cache> cache;
  std::unique_ptr<cache::InstallGuard> guard;
  cache::BatchControl* control = nullptr;
  std::unique_ptr<batching::util::SessionTable> sessions;
  // Declared after the host and cache, so they outlive it.
  std::unique_ptr<ptn::EngineContext> context;
  std::unique_ptr<ptn::EngineExecutable> executable;
  size_t logits_output = 0;
  int max_step_tokens = 0;
  int32_t vocab_size = 0;
};

MuseGlimmerNativeExecutor::MuseGlimmerNativeExecutor(
    std::unique_ptr<Native> native)
    : native_(std::move(native)) {}

MuseGlimmerNativeExecutor::~MuseGlimmerNativeExecutor() = default;

runtime::Result<MuseGlimmerBackend> MuseGlimmerNativeExecutor::create(
    const MuseGlimmerBackendConfig& config) {
  if (config.use_dflash || config.max_sessions <= 0) {
    ET_LOG(Error, "Muse Glimmer native: needs sessions and no DFlash");
    return runtime::Error::NotSupported;
  }
  try {
    auto native = std::make_unique<Native>();
    native->package = std::make_shared<const ptn::Package>(
        ptn::Package::load(config.model_path));
    const ptn::ByteSpan bytes = native->package->program_bytes();
    native->program = std::make_shared<const ptn::Program>(
        ptn::Program::load(bytes.data(), bytes.size()));
    const ptn::Method& method = native->program->get_method(kMethod);
    const auto geometry =
        ptn::offgraph_cache_geometry(method, *native->package);
    const std::optional<int64_t> logits_mode =
        native->package->metadata_int(kLogitsToKeepMode);
    if (!geometry || method.graph.input_ids.size() != 3 ||
        logits_mode != static_cast<int64_t>(LogitsToKeepMode::Selected)) {
      ET_LOG(
          Error,
          "Muse Glimmer native: export %s with --offgraph_cache "
          "--logits_to_keep selected",
          config.model_path.c_str());
      return runtime::Error::InvalidProgram;
    }
    const std::optional<int64_t> context =
        native->package->metadata_int(kMaxContextLen);
    if (!context || *context <= 0 ||
        *context > std::numeric_limits<int32_t>::max() ||
        config.max_session_tokens < 0 || config.max_session_tokens > *context) {
      ET_LOG(
          Error,
          "Muse Glimmer native: the session token limit must fit the "
          "package's context length");
      return runtime::Error::InvalidArgument;
    }
    const int session_tokens = config.max_session_tokens > 0
        ? config.max_session_tokens
        : static_cast<int>(*context);
    if (config.max_sessions >
        std::numeric_limits<int>::max() / session_tokens) {
      ET_LOG(Error, "Muse Glimmer native: the cell pool exceeds int range");
      return runtime::Error::InvalidArgument;
    }
    native->max_step_tokens = ptn::max_step_tokens(method);

    for (size_t i = 0; i < method.output_specs.size(); ++i) {
      if (method.output_specs[i].kind == ptn::OutputKind::UserOutput) {
        native->logits_output = i;
        const auto& sizes =
            method.graph.value(method.graph.output_ids[i]).tensor_meta().sizes;
        native->vocab_size =
            sizes.empty() ? 0 : static_cast<int32_t>(sizes.back());
        break;
      }
    }
    if (native->vocab_size <= 200092 ||
        config.bos_id >= static_cast<uint64_t>(native->vocab_size)) {
      ET_LOG(
          Error,
          "Muse Glimmer native: vocabulary %d cannot hold its special tokens",
          native->vocab_size);
      return runtime::Error::InvalidProgram;
    }

    // Every resident session may fill its context, so the pool holds them
    // all; storage grows with the cells in use.
    ptn::register_ggml_cache_builders();
    cache::CacheConfig cache_config{
        .capacity = config.max_sessions * session_tokens,
        .kv_dtype = static_cast<int>(
            config.native_kv_dtype == "fp16" ? ptn::ScalarType::Half
                                             : ptn::ScalarType::Float)};
    auto built = cache::CacheFactory::global().build(
        ptn::kGgmlCacheBackendId,
        cache::kind::kBatched,
        *geometry,
        cache_config);
    if (!built.ok()) {
      return built.error();
    }
    native->cache = built.get();
    native->control = native->cache->as<cache::BatchControl>();
    if (native->control == nullptr) {
      return runtime::Error::InvalidType;
    }
    const std::optional<int> seq_limit = native->control->max_seqs();
    if (seq_limit && config.max_sessions > *seq_limit) {
      ET_LOG(
          Error,
          "Muse Glimmer native: %d resident sessions requested, but the cache "
          "holds %d",
          config.max_sessions,
          *seq_limit);
      return runtime::Error::InvalidArgument;
    }
    native->sessions = std::make_unique<batching::util::SessionTable>(
        *native->control,
        config.max_sessions,
        session_tokens,
        native->vocab_size);

    native->guard = std::make_unique<cache::InstallGuard>(native->cache);
    runtime::BackendOptions<1> options;
    const char* key = nullptr;
    if (native->guard->set_option(options) != runtime::Error::Ok ||
        options.get_option(cache::kCacheKeyOption, key) != runtime::Error::Ok) {
      return runtime::Error::Internal;
    }
    native->host = ptn::GgmlEngineHost::create(
        {.device = config.native_device == "cpu" ? ptn::GgmlDevice::Cpu
                                                 : ptn::GgmlDevice::Gpu});
    native->context = native->host->create_context_with_options(
        native->program, native->package, {{cache::kCacheKeyOption, key}});

    auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
        MuseGlimmerPreparationSpec{
            aten::ScalarType::Float,
            /*hidden_dim=*/0,
            session_tokens,
            native->max_step_tokens,
            native->vocab_size,
            /*has_vision=*/false,
            /*max_soft_tokens=*/0,
            config.image_limits,
            config.bos_id});
    return MuseGlimmerBackend{
        std::unique_ptr<batching::Executor>(
            new MuseGlimmerNativeExecutor(std::move(native))),
        std::move(spec)};
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer native creation failed: %s", error.what());
    return runtime::Error::InvalidProgram;
  }
}

bool MuseGlimmerNativeExecutor::initialize() {
  try {
    native_->executable = native_->context->compile(kMethod);
    return true;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer native compilation failed: %s", error.what());
    return false;
  }
}

size_t MuseGlimmerNativeExecutor::preferred_batch_tokens() const {
  return static_cast<size_t>(native_->max_step_tokens);
}

std::optional<batching::SessionId> MuseGlimmerNativeExecutor::open_session() {
  return native_->sessions->open();
}

void MuseGlimmerNativeExecutor::close_session(batching::SessionId session) {
  native_->sessions->close(session);
}

std::optional<batching::SessionId> MuseGlimmerNativeExecutor::clone(
    batching::SessionId source,
    batching::Position upto) {
  return native_->sessions->clone(source, upto);
}

void MuseGlimmerNativeExecutor::set_sampling(
    batching::SessionId session,
    const batching::SamplingParams& params,
    std::optional<uint64_t> seed) {
  native_->sessions->set_sampling(session, params, seed);
}

bool MuseGlimmerNativeExecutor::execute(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());
  try {
    const auto step = native_->sessions->pack(batch);
    if (!step.ok()) {
      return false;
    }
    ptn::EngineExecutable& executable = *native_->executable;
    // A batch wider than the method runs as several forwards, in order, so a
    // slice attends the cells its predecessors wrote.
    const int total = static_cast<int>(step->tokens.size());
    std::vector<float> logits;
    for (int off = 0; off < total; off += native_->max_step_tokens) {
      const int n = std::min(native_->max_step_tokens, total - off);
      if (!native_->control->declare_step(std::vector<int32_t>(
              step->seq_ids.begin() + off, step->seq_ids.begin() + off + n))) {
        ET_LOG(Error, "Muse Glimmer native: the cache refused %d tokens", n);
        return false;
      }
      const batching::util::SliceRows rows =
          batching::util::select_rows(*step, off, n);
      const auto k = static_cast<int64_t>(rows.selector.size());
      executable.resize_input(0, {1, n});
      executable.set_input(
          0, step->tokens.data() + off, n, ptn::ScalarType::Long);
      executable.resize_input(1, {n});
      executable.set_input(
          1, step->positions.data() + off, n, ptn::ScalarType::Long);
      executable.resize_input(2, {k});
      executable.set_input(
          2, rows.selector.data(), rows.selector.size(), ptn::ScalarType::Long);
      executable.execute();
      if (rows.inputs.empty()) {
        continue;
      }
      logits.resize(static_cast<size_t>(k * native_->vocab_size));
      executable.get_output(
          native_->logits_output,
          logits.data(),
          logits.size(),
          ptn::ScalarType::Float);
      // Non-const: the sampler reduces each row in place.
      auto selected = make_tensor_ptr(
          {static_cast<int32_t>(k), native_->vocab_size}, logits);
      for (size_t row = 0; row < rows.inputs.size(); ++row) {
        const size_t input = rows.inputs[row];
        const batching::SessionId session = batch.inputs[input].sid;
        const auto token = native_->sessions->sample(
            session, *selected, static_cast<int>(row));
        if (!token) {
          return false;
        }
        out.outputs[input] = batching::Output{session, {*token}};
      }
    }
    return true;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer native execution failed: %s", error.what());
    return false;
  }
}

} // namespace executorch::extension::llm
