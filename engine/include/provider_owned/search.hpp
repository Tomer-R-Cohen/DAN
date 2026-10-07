#pragma once

// Bounded plan search (docs/BETA_SELECTION_PLAN.md, M4). Chooses which candidates form a
// route, in which order, and where the layer boundaries fall, to minimize the estimated time
// per token. For a fixed order the split is exact (dynamic programming over layer
// boundaries, using the same memory rule as stage_fits); groups and orders are enumerated
// within a fixed work budget, so a large discovery set costs bounded time. The result is the
// best plan found, not a proven global optimum unless `complete` is set.

#include "provider_owned/planner.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dan::provider_owned {

struct SearchCandidate {
    std::uint64_t offered_mib = 0;
    double us_per_gib = 0;        // decode speed for this model; 0 = default_search_speed
    double rtt_ms = 0;            // from the planning node (unknown links cost their default)
    bool relayed = false;
    std::vector<std::pair<int, int>> cached;  // [begin, end) ranges already on disk
    std::string key;              // stable tie-break (PeerID or worker ID)
    // Round trips this candidate's node measured to others, by their key (PeerID). A ring
    // link between two candidates uses one when either end measured it.
    struct Link {
        std::string peer;
        double rtt_ms = 0;
        bool relayed = false;
    };
    std::vector<Link> links;
};

struct SearchRequest {
    std::uint32_t context = 0;
    std::uint32_t sessions = 1;
    std::size_t minimum_stages = 1;
    std::size_t maximum_stages = 5;   // participant cap (SelectionPolicy::max_providers)
    std::optional<std::size_t> head;  // this candidate must run the first stage
    // Split evaluations allowed (one dynamic program over one ordered group each).
    std::size_t work_budget = 4000;
    // Candidates kept for the search, chosen for diversity from the whole pool.
    std::size_t pool_limit = 12;
    // A plan that downloads nothing is preferred when it is at most this much slower.
    double cached_tolerance = 0.10;
    double relay_penalty_ms = 25;     // added to a relayed link's round trip
};

struct SearchResult {
    std::vector<StageAssignment> plan;  // provider = index into the candidates given
    double token_ms = 0;
    bool from_cache = false;            // every stage already holds its range
    std::size_t evaluated = 0;          // split evaluations used
    bool complete = false;              // every group and order was evaluated: exact
};

inline constexpr double default_search_speed_us_per_gib = 4000;

// Time per token of `plan` (same model as estimate_token_ms): every stage reads its weights,
// and a token crosses every ring link. Link round trips are only known from the planning
// node; a link touching the head uses that measurement, any other is taken to pass through
// the planning node (a pessimistic guess later checked by real probes).
double search_token_ms(const ModelIndex& model, const std::vector<SearchCandidate>& candidates,
    const std::vector<StageAssignment>& plan, const SearchRequest& request);

// The candidates the search will consider: the head, then round-robin from the closest
// links, the most memory, the fastest speed and those already holding layers, until
// pool_limit. Indexes into `candidates`, in a deterministic order.
std::vector<std::size_t> search_pool(const std::vector<SearchCandidate>& candidates,
    const SearchRequest& request);

// Best plan found, or nullopt (with `reason`) when nothing fits.
std::optional<SearchResult> search_plan(const ModelIndex& model,
    const std::vector<SearchCandidate>& candidates, const SearchRequest& request,
    std::string* reason = nullptr);

} // namespace dan::provider_owned
