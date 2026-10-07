#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace dan::provider_owned {

// A pinned Qwen2 GGUF selection (config/provider-owned-*.json).
struct Manifest {
    std::string model_id;
    std::string architecture;
    std::uint32_t layers = 0;
    std::uint32_t hidden = 0;
    std::uint32_t context = 0;
    // Curated catalog rank (docs/BETA_SELECTION_PLAN.md): higher is preferred. File size
    // only breaks ties, so a larger quantization cannot outrank a better-ranked model.
    std::uint32_t quality_tier = 0;
    std::string url;
    std::string revision;
    std::string sha256;
};

Manifest load_manifest(const std::string& path);

// Reads an unsigned integer field from a flat JSON object (also used for worker metrics).
std::uint64_t json_uint64(const std::string& json, std::string_view key);

// Escapes text for a JSON string literal (reports).
std::string json_escape(std::string_view value);

} // namespace dan::provider_owned
