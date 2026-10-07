#include "provider_owned/search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace dan::provider_owned {
namespace {

constexpr double gib = static_cast<double>(1ull << 30);
constexpr std::uint64_t mib = 1024ull * 1024;
constexpr double infinite = std::numeric_limits<double>::infinity();

double speed_of(const SearchCandidate& candidate) {
    return candidate.us_per_gib > 0 ? candidate.us_per_gib : default_search_speed_us_per_gib;
}

double rtt_of(const SearchCandidate& candidate, const SearchRequest& request) {
    return candidate.rtt_ms + (candidate.relayed ? request.relay_penalty_ms : 0);
}

// Byte sizes by layer range without rescanning tensors: a stage's weights are the header plus
// its layers, the first stage also holds the embedding and the last the output head. Folding
// those into layer 0 and the last layer (every first stage holds layer 0, every last stage
// the last layer) makes the sizes additive.
struct Sizes {
    std::uint64_t header = 0;
    std::uint64_t embedding = 0;               // token_embd, which decoding reads one row of
    std::vector<std::uint64_t> prefix;          // prefix[l] = bytes of layers [0, l), folded
    // A tied output head reuses the embedding: a single stage holding every layer holds it
    // once, though it is folded into both the first and the last layer.
    std::uint64_t whole_overlap = 0;
    std::uint64_t kv_per_layer = 0;
    int layers = 0;

    std::uint64_t model(int begin, int end) const {
        return header + prefix[end] - prefix[begin]
            - (begin == 0 && end == layers ? whole_overlap : 0);
    }
    std::uint64_t decode(int begin, int end) const {
        const std::uint64_t bytes = model(begin, end);
        return begin == 0 && embedding < bytes ? bytes - embedding : bytes;
    }
    std::uint64_t kv(int begin, int end) const { return kv_per_layer * static_cast<std::uint64_t>(end - begin); }
};

std::optional<Sizes> sizes_of(const ModelIndex& model, std::uint32_t context, std::uint32_t sessions) {
    Sizes sizes;
    sizes.layers = static_cast<int>(model.layers);
    sizes.header = model.header_bytes;
    for (const ModelTensor& tensor : model.tensors) {
        if (tensor.name == "token_embd.weight") sizes.embedding = tensor.bytes;
    }
    sizes.prefix.assign(static_cast<std::size_t>(sizes.layers) + 1, 0);
    for (int layer = 0; layer < sizes.layers; ++layer) {
        const std::uint64_t bytes = stage_model_bytes(model, layer, layer + 1);
        if (bytes < sizes.header || bytes == 0) return std::nullopt;
        sizes.prefix[layer + 1] = sizes.prefix[layer] + bytes - sizes.header;
    }
    const std::uint64_t whole = stage_model_bytes(model, 0, sizes.layers);
    if (sizes.layers > 1) {
        if (whole == 0 || sizes.header + sizes.prefix[sizes.layers] < whole) return std::nullopt;
        sizes.whole_overlap = sizes.header + sizes.prefix[sizes.layers] - whole;
    }
    sizes.kv_per_layer = kv_bytes(model, 0, 1, context, sessions);
    // The folding is exact for every architecture DAN stages today; check it on the whole
    // model and two split points, and refuse to search a model where it is not.
    const int middle = sizes.layers / 2;
    if (sizes.kv_per_layer == 0 || sizes.model(0, sizes.layers) != stage_model_bytes(model, 0, sizes.layers)
        || (middle > 0 && (sizes.model(0, middle) != stage_model_bytes(model, 0, middle)
            || sizes.model(middle, sizes.layers) != stage_model_bytes(model, middle, sizes.layers)))
        || kv_bytes(model, 0, sizes.layers, context, sessions) != sizes.kv(0, sizes.layers)) {
        return std::nullopt;
    }
    return sizes;
}

// The stage_fits rule: weights and KV within the offer, keeping back max(1 GiB, 15%).
bool fits(const Sizes& sizes, std::uint64_t offered_mib, int begin, int end) {
    if (offered_mib > std::numeric_limits<std::uint64_t>::max() / mib) return false;
    const std::uint64_t offered = offered_mib * mib;
    const std::uint64_t reserve = std::max<std::uint64_t>(1024 * mib, offered * 15 / 100);
    if (offered <= reserve) return false;
    const std::uint64_t usable = offered - reserve;
    const std::uint64_t weights = sizes.model(begin, end);
    return weights <= usable && sizes.kv(begin, end) <= usable - weights;
}

double compute_ms(const Sizes& sizes, const SearchCandidate& candidate, int begin, int end) {
    return static_cast<double>(sizes.decode(begin, end)) / gib * speed_of(candidate) / 1000.0;
}

// One-way time a token spends on the ring's links, for members in ring order.
double links_ms(const std::vector<SearchCandidate>& candidates, const std::vector<std::size_t>& order,
    const SearchRequest& request) {
    if (order.size() < 2) return 0;
    double total = 0;
    for (std::size_t index = 0; index < order.size(); ++index) {
        const std::size_t from = order[index], to = order[(index + 1) % order.size()];
        const double from_rtt = rtt_of(candidates[from], request);
        const double to_rtt = rtt_of(candidates[to], request);
        total += request.head && from == *request.head ? to_rtt
            : request.head && to == *request.head ? from_rtt : from_rtt + to_rtt;
    }
    return total / 2;
}

// Among splits with (almost) the same total time, prefer even stage times: a pipelined prompt
// moves at the pace of its slowest stage. Small enough never to outweigh a real difference.
constexpr double balance_weight = 1e-9;  // per ms squared

// Exact best split for one ordered group: f[s][e] is the least compute time for stages
// 0..s-1 covering layers [0, e) (plus the tiny balance term). Returns the plan's compute
// time, or infinity.
double best_split(const Sizes& sizes, const std::vector<SearchCandidate>& candidates,
    const std::vector<std::size_t>& order, std::vector<StageAssignment>& plan) {
    const int layers = sizes.layers;
    const std::size_t count = order.size();
    std::vector<std::vector<double>> cost(count + 1, std::vector<double>(layers + 1, infinite));
    std::vector<std::vector<int>> from(count + 1, std::vector<int>(layers + 1, -1));
    cost[0][0] = 0;
    for (std::size_t stage = 0; stage < count; ++stage) {
        const SearchCandidate& candidate = candidates[order[stage]];
        // Every later stage needs at least one layer.
        const int last_end = layers - static_cast<int>(count - stage - 1);
        for (int begin = static_cast<int>(stage); begin < last_end; ++begin) {
            if (cost[stage][begin] == infinite) continue;
            for (int end = begin + 1; end <= last_end; ++end) {
                if (stage + 1 == count && end != layers) continue;
                // Memory only grows with the range: once it does not fit, longer ones won't.
                if (!fits(sizes, candidate.offered_mib, begin, end)) break;
                const double stage_ms = compute_ms(sizes, candidate, begin, end);
                const double value = cost[stage][begin] + stage_ms + balance_weight * stage_ms * stage_ms;
                if (value < cost[stage + 1][end]) {
                    cost[stage + 1][end] = value;
                    from[stage + 1][end] = begin;
                }
            }
        }
    }
    if (cost[count][layers] == infinite) return infinite;
    plan.assign(count, {});
    for (int stage = static_cast<int>(count), end = layers; stage > 0; --stage) {
        const int begin = from[stage][end];
        StageAssignment& assignment = plan[stage - 1];
        assignment.provider = order[stage - 1];
        assignment.begin = begin;
        assignment.end = end;
        assignment.model_bytes = sizes.model(begin, end);
        assignment.kv_bytes = sizes.kv(begin, end);
        end = begin;
    }
    double total = 0;
    for (const StageAssignment& stage : plan) {
        total += compute_ms(sizes, candidates[stage.provider], stage.begin, stage.end);
    }
    return total;
}

} // namespace

double search_token_ms(const ModelIndex& model, const std::vector<SearchCandidate>& candidates,
    const std::vector<StageAssignment>& plan, const SearchRequest& request) {
    double total = 0;
    std::vector<std::size_t> order;
    for (const StageAssignment& stage : plan) {
        total += static_cast<double>(decode_bytes(model, stage.begin, stage.end)) / gib
            * speed_of(candidates[stage.provider]) / 1000.0;
        order.push_back(stage.provider);
    }
    return total + links_ms(candidates, order, request);
}

std::vector<std::size_t> search_pool(const std::vector<SearchCandidate>& candidates,
    const SearchRequest& request) {
    std::vector<std::size_t> all(candidates.size());
    std::iota(all.begin(), all.end(), std::size_t{0});
    const auto ranked = [&](auto better) {
        std::vector<std::size_t> list = all;
        std::stable_sort(list.begin(), list.end(), [&](std::size_t left, std::size_t right) {
            const auto a = better(candidates[left]), b = better(candidates[right]);
            if (a != b) return a < b;
            return candidates[left].key < candidates[right].key;
        });
        return list;
    };
    const std::vector<std::vector<std::size_t>> lists{
        ranked([&](const SearchCandidate& c) { return rtt_of(c, request); }),
        ranked([](const SearchCandidate& c) { return -static_cast<double>(c.offered_mib); }),
        ranked([](const SearchCandidate& c) { return speed_of(c); }),
        ranked([](const SearchCandidate& c) { return c.cached.empty() ? 1.0 : 0.0; }),
    };
    std::vector<std::size_t> pool;
    std::vector<bool> taken(candidates.size(), false);
    if (request.head && *request.head < candidates.size()) {
        pool.push_back(*request.head);
        taken[*request.head] = true;
    }
    const std::size_t limit = std::min(request.pool_limit, candidates.size());
    std::vector<std::size_t> next(lists.size(), 0);
    while (pool.size() < limit) {
        for (std::size_t list = 0; list < lists.size() && pool.size() < limit; ++list) {
            while (next[list] < lists[list].size() && taken[lists[list][next[list]]]) ++next[list];
            if (next[list] == lists[list].size()) continue;
            taken[lists[list][next[list]]] = true;
            pool.push_back(lists[list][next[list]]);
        }
    }
    return pool;
}

std::optional<SearchResult> search_plan(const ModelIndex& model,
    const std::vector<SearchCandidate>& candidates, const SearchRequest& request,
    std::string* reason) {
    const auto fail = [&](std::string why) -> std::optional<SearchResult> {
        if (reason) *reason = std::move(why);
        return std::nullopt;
    };
    if (!compatible_stage_model(model)) return fail("model cannot be staged");
    if (candidates.empty() || request.context == 0 || request.sessions == 0
        || request.minimum_stages == 0 || request.maximum_stages < request.minimum_stages
        || (request.head && *request.head >= candidates.size())) return fail("invalid search request");
    const std::optional<Sizes> sizes = sizes_of(model, request.context, request.sessions);
    if (!sizes) return fail("model sizes are not additive by layer");

    const std::vector<std::size_t> pool = search_pool(candidates, request);
    const std::size_t largest = std::min({request.maximum_stages, pool.size(),
        static_cast<std::size_t>(sizes->layers)});
    SearchResult best;
    best.token_ms = infinite;
    bool exhaustive = pool.size() == candidates.size();
    const std::uint64_t total_decode = sizes->decode(0, sizes->layers);

    // Groups of each size, as combinations of pool positions in increasing order.
    for (std::size_t count = request.minimum_stages; count <= largest; ++count) {
        std::vector<std::size_t> pick(count);
        std::iota(pick.begin(), pick.end(), std::size_t{0});
        for (bool more = true; more;) {
            std::vector<std::size_t> group;
            for (const std::size_t position : pick) group.push_back(pool[position]);
            // Advance to the next combination before any `continue`.
            more = false;
            for (std::size_t slot = count; slot-- > 0;) {
                if (pick[slot] < pool.size() - count + slot) {
                    ++pick[slot];
                    for (std::size_t after = slot + 1; after < count; ++after) pick[after] = pick[after - 1] + 1;
                    more = true;
                    break;
                }
            }
            if (request.head && std::find(group.begin(), group.end(), *request.head) == group.end()) continue;

            // Cheap bounds before any split: enough memory in total, and a lower bound on
            // time (all weights at the group's fastest speed, plus its links) below the best.
            double fastest = infinite;
            long double usable = 0;
            for (const std::size_t member : group) {
                fastest = std::min(fastest, speed_of(candidates[member]));
                const long double offered = static_cast<long double>(candidates[member].offered_mib) * mib;
                usable += std::max<long double>(0, offered - std::max<long double>(1024.0L * mib, offered * 0.15L));
            }
            if (usable < static_cast<long double>(sizes->model(0, sizes->layers) - sizes->header)
                    + static_cast<long double>(sizes->header) * count + sizes->kv(0, sizes->layers)) continue;
            double link_bound = 0;
            for (const std::size_t member : group) {
                if (!request.head || member != *request.head) link_bound += rtt_of(candidates[member], request);
            }
            link_bound = count < 2 ? 0 : link_bound / 2;
            if (static_cast<double>(total_decode) / gib * fastest / 1000.0 + link_bound >= best.token_ms) continue;

            // Orders: every one for small groups; otherwise a few that matter (fastest, most
            // memory, closest first). The head always leads.
            std::vector<std::size_t> rest;
            for (const std::size_t member : group) {
                if (!request.head || member != *request.head) rest.push_back(member);
            }
            std::vector<std::vector<std::size_t>> orders;
            if (rest.size() <= 4) {
                std::sort(rest.begin(), rest.end());
                do orders.push_back(rest); while (std::next_permutation(rest.begin(), rest.end()));
            } else {
                exhaustive = false;
                for (int rule = 0; rule < 3; ++rule) {
                    std::vector<std::size_t> order = rest;
                    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
                        const SearchCandidate& a = candidates[left];
                        const SearchCandidate& b = candidates[right];
                        if (rule == 0) return speed_of(a) < speed_of(b);
                        if (rule == 1) return a.offered_mib > b.offered_mib;
                        return rtt_of(a, request) < rtt_of(b, request);
                    });
                    if (std::find(orders.begin(), orders.end(), order) == orders.end()) orders.push_back(order);
                }
            }
            for (std::vector<std::size_t>& order : orders) {
                if (request.head) order.insert(order.begin(), *request.head);
                if (best.evaluated >= request.work_budget) {
                    exhaustive = false;
                    more = false;
                    break;
                }
                ++best.evaluated;
                const double links = links_ms(candidates, order, request);
                if (links >= best.token_ms) continue;
                std::vector<StageAssignment> plan;
                const double compute = best_split(*sizes, candidates, order, plan);
                if (compute + links < best.token_ms - 1e-9) {
                    best.token_ms = compute + links;
                    best.plan = std::move(plan);
                }
            }
        }
        if (best.evaluated >= request.work_budget) break;
    }

    // A plan from layers already on disk starts in seconds instead of minutes: take it when
    // it is nearly as fast.
    std::vector<std::uint64_t> offered;
    std::vector<std::vector<std::pair<int, int>>> cached;
    std::optional<std::size_t> head;
    for (std::size_t position = 0; position < pool.size(); ++position) {
        offered.push_back(candidates[pool[position]].offered_mib);
        cached.push_back(candidates[pool[position]].cached);
        if (request.head && pool[position] == *request.head) head = position;
    }
    if (auto reuse = plan_from_cache(model, offered, cached, request.context, request.sessions,
            request.minimum_stages, largest, head)) {
        for (StageAssignment& stage : *reuse) stage.provider = pool[stage.provider];
        const double reuse_ms = search_token_ms(model, candidates, *reuse, request);
        if (best.plan.empty() || reuse_ms <= best.token_ms * (1 + request.cached_tolerance)) {
            best.plan = std::move(*reuse);
            best.token_ms = reuse_ms;
            best.from_cache = true;
        }
    }
    if (best.plan.empty()) {
        return fail(best.evaluated >= request.work_budget
            ? "no plan found within the search budget"
            : "no group of up to " + std::to_string(largest) + " candidates holds the model"
                " with this context and session count");
    }
    best.complete = exhaustive && best.evaluated < request.work_budget;
    return best;
}

} // namespace dan::provider_owned
