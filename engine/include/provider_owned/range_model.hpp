#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace dan::provider_owned {

struct RangeModelRequest {
    std::string url;
    std::string revision;
    std::string full_sha256;
    std::filesystem::path path;
    int stage_start = 0;
    int stage_end = 0;
    std::function<void(std::uint64_t, std::uint64_t, std::uint64_t)> progress;
};

struct RangeModelStats {
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    std::uint64_t downloaded_bytes = 0;
    std::uint64_t shared_bytes = 0;
    std::uint64_t header_bytes = 0;
    std::uint64_t tensors_present = 0;
    bool cache_reused = false;
};

struct ModelTensor {
    std::string name;
    std::uint64_t bytes = 0;
    // GGUF dimensions in stored order, and the GGML type id. Kept so a stage can check an
    // expert bank's shape against the metadata before anything is downloaded or loaded.
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type = 0;
};

struct ModelIndex {
    std::string architecture;
    std::uint32_t layers = 0;
    std::uint32_t hidden = 0;
    std::uint32_t heads = 0;
    std::uint32_t kv_heads = 0;
    // Mixture-of-experts geometry; zero for dense models. `experts_used` is how many experts
    // each token routes to -- never how many are resident. Every expert of an owned layer
    // stays on the worker that owns that layer.
    std::uint32_t experts = 0;
    std::uint32_t experts_used = 0;
    std::uint32_t ffn_length = 0;
    // `<architecture>.context_length`: positions the model was trained for (0 = not stated).
    std::uint32_t trained_context = 0;
    // Per-expert intermediate width (`<arch>.expert_feed_forward_length`). Qwen3-MoE sizes its
    // expert banks with this, not with `feed_forward_length`.
    std::uint32_t expert_ffn_length = 0;
    // Explicit attention head widths (`<arch>.attention.key_length` / `.value_length`), 0 when
    // the GGUF leaves them out and llama.cpp derives hidden/heads. They are NOT always that
    // quotient: Qwen3-235B-A22B declares 128 while hidden/heads is 64, so a derived width would
    // halve every KV estimate. Use `head_width_k`/`head_width_v` (planner.hpp) to read them.
    std::uint32_t head_dim_k = 0;
    std::uint32_t head_dim_v = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t header_bytes = 0;
    std::vector<ModelTensor> tensors;
};

bool inspect_range_model(const RangeModelRequest& request, ModelIndex& index,
    std::string& error);

// The same GGUF header parsing `inspect_range_model` performs, on bytes already in memory.
// False when the header is incomplete (more bytes needed) or invalid.
bool parse_model_header(const std::vector<std::uint8_t>& header, std::uint64_t logical_bytes,
    ModelIndex& index, std::string& error);
std::uint64_t stage_model_bytes(const ModelIndex& index, int begin, int end);

bool prepare_range_model(const RangeModelRequest& request, RangeModelStats& stats,
    std::string& error);

// A parsed model header kept on disk. A manifest pins its GGUF by full-file SHA-256, so the
// header of that file can never change: the index is saved once and reused by every later
// run instead of reading the header over HTTP again (several seconds per model).
bool save_model_index(const std::filesystem::path& path, const ModelIndex& index);
// False (and `index` untouched) when the file is missing or not a complete Qwen2 index.
bool load_model_index(const std::filesystem::path& path, ModelIndex& index);

} // namespace dan::provider_owned
