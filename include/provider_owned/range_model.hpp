#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace dan::provider_owned {

struct RangeModelRequest {
    std::string url;
    std::string revision;
    std::string full_sha256;
    std::filesystem::path path;
    int stage_start = 0;
    int stage_end = 0;
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

bool prepare_range_model(const RangeModelRequest& request, RangeModelStats& stats,
    std::string& error);

} // namespace dan::provider_owned
