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

// A split built only from ranges the candidates already hold: cached[i] lists candidate i's
// cached [begin, end) ranges for this model. Returns the plan whose stages tile 0..layers
// with each stage fitting its candidate, at most `stage_limit` stages, longest range first
// (fewer stages, fewer hops). Downloads cost minutes, so this is preferred whenever it is no
// longer than the ordinary plan.
std::optional<std::vector<StageAssignment>> plan_from_cache(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib,
    const std::vector<std::vector<std::pair<int, int>>>& cached, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages, std::size_t stage_limit);

// Fewest stages (at least minimum_stages) that fit, trying candidate orders; layers split
// in proportion to offered memory. At most 8 candidates.
std::optional<std::vector<StageAssignment>> plan_stages(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages = 1);

} // namespace dan::provider_owned
