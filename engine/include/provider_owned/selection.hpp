#pragma once

// The beta selection contract (docs/BETA_SELECTION_PLAN.md, M0): one policy shared by chat
// clients and replica owners for choosing among already-feasible plans. It never relaxes a
// requirement to find a plan: a plan that is too slow, unmeasured or needs more providers is
// offered only as an explicit fallback, and a plan without the requested context is never
// offered at all. Estimating a plan's numbers is the caller's job (placement, owners).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dan::provider_owned {

enum class ColdStart {
    // Answer on an eligible plan that is serving now; report a better plan that would need
    // loading separately. With nothing ready, wait for the best plan that must load.
    ready_first,
    // Always take the best eligible plan, ready or not, and wait for it to load.
    wait,
};

struct SelectionPolicy {
    // Accepted output tokens per second for one chat, as delivered to its client.
    // Speculative guesses, stream events and aggregate replica throughput do not count.
    double target_tokens_per_s = 20;
    // Warm first response for an ordinary turn: aim for this...
    double first_token_target_ms = 2000;
    // ...and never exceed this.
    double first_token_max_ms = 5000;
    // Context every plan must serve (owner decision, 2026-10-07: 32K default, up to 256K per
    // request). A plan with less is ineligible, never a fallback.
    std::uint32_t context = 32768;
    // Authenticated provider nodes per replica (an upper bound; one node may bring several
    // local GPUs). A PeerID does not prove a distinct machine or person.
    std::size_t max_providers = 5;
    // An "ordinary turn" has at most this many uncached input tokens. Longer inputs (document
    // uploads) are still served but carry no first-response promise.
    std::uint32_t ordinary_input_tokens = 2048;
    // Acceptable streaming stall: 95th percentile gap between accepted tokens.
    double stall_p95_ms = 1000;
    // Predictions must beat each limit by this fraction (0.2: a 50 ms target needs a
    // predicted 41.7 ms), absorbing estimate error on uncontrolled public hardware.
    double margin = 0.20;
    ColdStart cold_start = ColdStart::ready_first;
};

// One feasible plan, as its planner estimated it: memory, catalog, ABI and context checks
// have already passed for every stage (the worker checks them again at reservation).
struct PlanEstimate {
    std::string label;             // stable identity: breaks exact ties, names it in reasons
    int quality_tier = 0;          // curated catalog rank of the model; higher is better
    std::uint32_t context = 0;     // context every session of this plan gets
    std::size_t providers = 0;     // distinct provider nodes in the route
    bool ready = false;            // serving now (false: must reserve and load first)
    // The estimates come from observations of this model/configuration, not defaults or
    // peer claims alone. Unmeasured plans may run only as an explicitly accepted trial.
    bool measured = false;
    double token_ms = 0;           // per accepted token for one chat under expected load; 0 = unknown
    double first_token_ms = 0;     // warm, ordinary turn; 0 = unknown
    double load_ms = 0;            // reservation + loading; reported, never part of first_token_ms
    double stall_p95_ms = 0;       // observed; 0 = unknown (not held against a plan yet)
    bool cache_warm = false;       // prompt prefix or layers already where this plan needs them
    std::uint64_t resource_bytes = 0;  // memory the plan occupies across providers
};

enum class Verdict {
    meets_target,  // eligible: may be chosen
    below_target,  // too slow: an explicit fallback only
    unverified,    // unknown or unmeasured speed: an explicit trial only
    ineligible,    // wrong context, too many providers or invalid numbers: never offered
};

struct Assessment {
    Verdict verdict = Verdict::ineligible;
    std::string reason;
};

Assessment assess_plan(const PlanEstimate& plan, const SelectionPolicy& policy);

struct Selection {
    // The plan to use. Empty when no plan meets the target: the caller must not pick one
    // silently -- it reports `reason` and may offer `fallback`.
    std::optional<std::size_t> chosen;
    // ready_first only: a higher-tier eligible plan that has to load first.
    std::optional<std::size_t> better_cold;
    // The best below-target or unverified plan, for the user to accept explicitly.
    std::optional<std::size_t> fallback;
    std::string reason;
};

// Highest curated quality tier first; within a tier, a first response within the 2 s aim,
// then the faster per-chat decode, then a warm cache, then fewer resources, then the label.
// Deterministic for the same input.
Selection select_plan(const std::vector<PlanEstimate>& plans, const SelectionPolicy& policy);

// Catalog order: curated tier first; the larger file only breaks a tie between equal tiers,
// so a bigger quantization can never outrank a better-ranked model.
bool preferred_model(int left_tier, std::uint64_t left_bytes, int right_tier,
    std::uint64_t right_bytes);

} // namespace dan::provider_owned
