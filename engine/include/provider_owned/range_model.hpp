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
};

struct ModelIndex {
    std::string architecture;
    std::uint32_t layers = 0;
    std::uint32_t hidden = 0;
    std::uint32_t heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t header_bytes = 0;
    std::vector<ModelTensor> tensors;
};

bool inspect_range_model(const RangeModelRequest& request, ModelIndex& index,
    std::string& error);
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
