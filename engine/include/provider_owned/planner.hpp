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

// Whether DAN can run this model as stages. Exactly the architectures whose stage loader and
// graph are audited in patches/llama-provider-owned.patch: dense Qwen2 and OLMoE. For OLMoE it
// also requires every layer's router and complete expert bank, with shapes that agree with the
// model's expert geometry. `reason` receives the first problem found.
bool compatible_stage_model(const ModelIndex& model, std::string* reason = nullptr);

// Bytes of weights a stage reads per decoded token: its tensors, except the token embedding,
// of which a token reads one row. Decoding is memory-bound, so time per token follows this.
std::uint64_t decode_bytes(const ModelIndex& model, int begin, int end);

// KV positions llama.cpp really allocates per sequence for this context and session count
// (it pads to 256), or 0 if the numbers are unusable.
std::uint32_t allocated_positions(std::uint32_t context, std::uint32_t sessions);

// The head widths llama.cpp will use for K and V: the GGUF's explicit
// `attention.key_length`/`.value_length` when present, else hidden/heads (llama-model.cpp
// defaults them that way). Qwen3-MoE declares 128 where hidden/heads is 64, so deriving the
// quotient would halve its KV estimate.
std::uint32_t head_width_k(const ModelIndex& model);
std::uint32_t head_width_v(const ModelIndex& model);

// F16 K and V for layers [begin, end), over the positions llama.cpp actually allocates.
std::uint64_t kv_bytes(const ModelIndex& model, int begin, int end,
    std::uint32_t context, std::uint32_t sessions);

// Whether layers [begin, end) plus KV for `sessions` x `context` fit in offered memory,
// keeping back max(1 GiB, 15%). Fills `assignment` (provider left at 0) either way.
bool stage_fits(const ModelIndex& model, std::uint64_t offered_mib, int begin, int end,
    std::uint32_t context, std::uint32_t sessions, StageAssignment& assignment);

// Whether the first stage [0, end) of `model` and a whole draft model, each with KV for
// `sessions` x `context`, fit together in offered memory (same reserve as stage_fits).
bool stage_with_draft_fits(const ModelIndex& model, const ModelIndex& draft,
    std::uint64_t offered_mib, int end, std::uint32_t context, std::uint32_t sessions);

// A split built only from ranges the candidates already hold: cached[i] lists candidate i's
// cached [begin, end) ranges for this model. Returns the plan whose stages tile 0..layers
// with each stage fitting its candidate, at most `stage_limit` stages, longest range first
// (fewer stages, fewer hops). Downloads cost minutes, so this is preferred whenever it is no
// longer than the ordinary plan.
std::optional<std::vector<StageAssignment>> plan_from_cache(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib,
    const std::vector<std::vector<std::pair<int, int>>>& cached, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages, std::size_t stage_limit,
    std::optional<std::size_t> head = std::nullopt);

// Fewest stages (at least minimum_stages) that fit, trying candidate orders; layers split
// in proportion to offered memory. At most 8 candidates.
//
// `head` (both planners): this candidate must run the first stage. A replica owner uses it
// to put itself at the head, where prompts arrive and the draft model runs.
std::optional<std::vector<StageAssignment>> plan_stages(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages = 1,
    std::optional<std::size_t> head = std::nullopt);

} // namespace dan::provider_owned
