#pragma once

// Stage placement shared by the coordinator (automatic formation), the decentralized
// client (dynamic placement) and workers (checking a proposed reservation fits).

#include "provider_owned/range_model.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dan::provider_owned {

struct StageAssignment {
    std::size_t provider = 0;   // index into the planner's candidate list
    int begin = 0;
    int end = 0;
    std::uint64_t model_bytes = 0;
    std::uint64_t kv_bytes = 0;
};

bool compatible_dense_qwen2(const ModelIndex& model, std::string* reason = nullptr);

std::uint64_t kv_bytes(const ModelIndex& model, int begin, int end,
    std::uint32_t context, std::uint32_t sessions);

// Whether layers [begin, end) plus KV for `sessions` x `context` fit in offered memory,
// keeping back max(1 GiB, 15%). Fills `assignment` (provider left at 0) either way.
bool stage_fits(const ModelIndex& model, std::uint64_t offered_mib, int begin, int end,
    std::uint32_t context, std::uint32_t sessions, StageAssignment& assignment);

// Fewest stages (at least minimum_stages) that fit, trying candidate orders; layers split
// in proportion to offered memory. At most 8 candidates.
std::optional<std::vector<StageAssignment>> plan_stages(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages = 1);

} // namespace dan::provider_owned
