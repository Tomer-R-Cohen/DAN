// Bounded plan search (docs/BETA_SELECTION_PLAN.md, M4): exact on small cases (checked
// against brute force), sensible on the plan's synthetic fixtures, bounded on large pools.
// Executable in Release builds too.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "provider_owned/search.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace po = dan::provider_owned;

namespace {

constexpr std::uint64_t mib = 1024ull * 1024;

// A dense Qwen2-shaped model whose layers each weigh `layer_mib`.
po::ModelIndex model(int layers, std::uint64_t layer_mib) {
    po::ModelIndex index;
    index.architecture = "qwen2";
    index.layers = static_cast<std::uint32_t>(layers);
    index.hidden = 1024; index.heads = 16; index.kv_heads = 4;
    index.header_bytes = 4096;
    index.tensors.push_back({"token_embd.weight", 256 * mib});
    index.tensors.push_back({"output_norm.weight", 4096});
    for (int layer = 0; layer < layers; ++layer) {
        index.tensors.push_back({"blk." + std::to_string(layer) + ".weight", layer_mib * mib});
    }
    index.tensors.push_back({"output.weight", 256 * mib});
    return index;
}

po::SearchCandidate candidate(std::string key, std::uint64_t offered, double speed, double rtt = 0,
    bool relayed = false) {
    po::SearchCandidate result;
    result.key = std::move(key);
    result.offered_mib = offered;
    result.us_per_gib = speed;
    result.rtt_ms = rtt;
    result.relayed = relayed;
    return result;
}

// Every ordered group (head first when required) and every split, by brute force.
double brute_force(const po::ModelIndex& index, const std::vector<po::SearchCandidate>& candidates,
    const po::SearchRequest& request) {
    double best = std::numeric_limits<double>::infinity();
    const int layers = static_cast<int>(index.layers);
    std::vector<std::size_t> order;
    std::vector<bool> used(candidates.size(), false);
    std::function<void()> grow = [&] {
        if (order.size() >= request.minimum_stages) {
            // Every composition of the layers into order.size() positive parts.
            std::vector<po::StageAssignment> plan(order.size());
            std::function<void(std::size_t, int)> split = [&](std::size_t stage, int begin) {
                if (stage + 1 == order.size()) {
                    po::StageAssignment fit;
                    if (!po::stage_fits(index, candidates[order[stage]].offered_mib, begin, layers,
                            request.context, request.sessions, fit)) return;
                    plan[stage] = {order[stage], begin, layers};
                    best = std::min(best, po::search_token_ms(index, candidates, plan, request));
                    return;
                }
                for (int end = begin + 1; end <= layers - static_cast<int>(order.size() - stage - 1); ++end) {
                    po::StageAssignment fit;
                    if (!po::stage_fits(index, candidates[order[stage]].offered_mib, begin, end,
                            request.context, request.sessions, fit)) continue;
                    plan[stage] = {order[stage], begin, end};
                    split(stage + 1, end);
                }
            };
            split(0, 0);
        }
        if (order.size() == request.maximum_stages || order.size() == static_cast<std::size_t>(layers)) return;
        for (std::size_t next = 0; next < candidates.size(); ++next) {
            if (used[next] || (request.head && order.empty() && next != *request.head)) continue;
            used[next] = true; order.push_back(next);
            grow();
            order.pop_back(); used[next] = false;
        }
    };
    grow();
    return best;
}

bool valid(const po::ModelIndex& index, const std::vector<po::SearchCandidate>& candidates,
    const po::SearchRequest& request, const std::vector<po::StageAssignment>& plan) {
    if (plan.empty() || plan.front().begin != 0 || plan.back().end != static_cast<int>(index.layers)) return false;
    std::vector<bool> used(candidates.size(), false);
    for (std::size_t stage = 0; stage < plan.size(); ++stage) {
        if (stage > 0 && plan[stage].begin != plan[stage - 1].end) return false;
        if (used[plan[stage].provider]) return false;
        used[plan[stage].provider] = true;
        po::StageAssignment fit;
        if (!po::stage_fits(index, candidates[plan[stage].provider].offered_mib, plan[stage].begin,
                plan[stage].end, request.context, request.sessions, fit)) return false;
    }
    return !request.head || plan.front().provider == *request.head;
}

} // namespace

int main() {
    po::SearchRequest request;
    request.context = 4096;
    request.sessions = 1;

    // 1. Exact on small random cases: the search equals brute force and says it is complete.
    {
        std::mt19937 random(7);
        po::ModelIndex tied = model(6, 1024);
        tied.tensors.pop_back();  // no output.weight: the last stage reuses the embedding
        int compared = 0;
        for (int round = 0; round < 600; ++round) {
            const po::ModelIndex index = round % 2 ? tied : model(6, 1024);
            std::vector<po::SearchCandidate> candidates;
            const int count = 1 + static_cast<int>(random() % 5);
            for (int item = 0; item < count; ++item) {
                candidates.push_back(candidate("w" + std::to_string(item),
                    2048 + random() % 8192, 1000.0 + random() % 9000, random() % 120, random() % 4 == 0));
            }
            if (round % 4 >= 2) {
                // Some measured worker-to-worker links (M3): the pruning bound must still hold.
                for (auto& from : candidates) {
                    for (const auto& to : candidates) {
                        if (&from != &to && random() % 2 == 0) {
                            from.links.push_back({to.key, static_cast<double>(random() % 60), random() % 5 == 0});
                        }
                    }
                }
            }
            po::SearchRequest small = request;
            small.minimum_stages = 1 + random() % 2;
            small.maximum_stages = 2 + random() % 3;
            if (random() % 3 == 0) small.head = random() % candidates.size();
            const double expected = brute_force(index, candidates, small);
            const auto found = po::search_plan(index, candidates, small);
            assert(std::isinf(expected) == !found.has_value());
            if (!found) continue;
            ++compared;
            assert(found->complete && !found->from_cache);
            assert(valid(index, candidates, small, found->plan));
            // Equal up to the tiny balance preference among same-time splits.
            assert(std::abs(found->token_ms - expected) < 1e-3);
            assert(std::abs(po::search_token_ms(index, candidates, found->plan, small) - found->token_ms) < 1e-6);
        }
        assert(compared > 100);
    }

    const po::ModelIndex big = model(40, 1024);  // ~40.5 GiB of weights
    // 2. A slow large GPU loses to two fast ones that together hold the model.
    {
        const auto found = po::search_plan(big, {candidate("slow-large", 60000, 9000),
            candidate("fast-a", 30000, 2000, 10), candidate("fast-b", 30000, 2000, 10)}, request);
        assert(found && found->plan.size() == 2 && found->plan[0].provider != 0 && found->plan[1].provider != 0);
    }
    // ...but when the links are slow, one hop-free GPU wins despite being slower.
    {
        const auto found = po::search_plan(big, {candidate("slow-large", 60000, 3000),
            candidate("fast-a", 30000, 2000, 300), candidate("fast-b", 30000, 2000, 300)}, request);
        assert(found && found->plan.size() == 1 && found->plan[0].provider == 0);
    }
    // 3. Asymmetric links: of two equal GPUs, the far or relayed one is left out.
    {
        const auto found = po::search_plan(big, {candidate("near", 30000, 3000, 5),
            candidate("far", 30000, 3000, 250), candidate("relay", 30000, 3000, 5, true),
            candidate("mid", 30000, 3000, 20)}, request);
        assert(found && found->plan.size() == 2);
        for (const auto& stage : found->plan) assert(stage.provider == 0 || stage.provider == 3);
    }
    // 4. A useful ninth provider (the only fast one) is found although eight come first.
    {
        std::vector<po::SearchCandidate> candidates;
        for (int item = 0; item < 8; ++item) candidates.push_back(candidate("p" + std::to_string(item), 30000, 8000, 5));
        candidates.push_back(candidate("ninth", 60000, 1500, 40));
        const auto found = po::search_plan(big, candidates, request);
        assert(found && found->plan.size() == 1 && found->plan[0].provider == 8);
    }
    // 5. Participant cap: six small GPUs (7 layers each) are needed, the cap is five: no plan, with a reason.
    {
        std::vector<po::SearchCandidate> candidates;
        for (int item = 0; item < 6; ++item) candidates.push_back(candidate("s" + std::to_string(item), 9000, 3000));
        std::string reason;
        assert(!po::search_plan(big, candidates, request, &reason));
        assert(reason.find("up to 5") != std::string::npos);
        po::SearchRequest wider = request;
        wider.maximum_stages = 6;
        const auto found = po::search_plan(big, candidates, wider);
        assert(found && found->plan.size() == 6);
    }
    // 6. Heterogeneous memory: the large GPU takes the larger share.
    {
        const auto found = po::search_plan(big, {candidate("small", 16000, 3000), candidate("large", 40000, 3000)}, request);
        assert(found && found->plan.size() == 2);
        for (const auto& stage : found->plan) {
            const int layers = stage.end - stage.begin;
            assert(stage.provider == 1 ? layers > 20 : layers < 20);
        }
    }
    // 7. Cache versus speed: layers already on disk win when nearly as fast, not otherwise.
    {
        std::vector<po::SearchCandidate> candidates{candidate("a", 30000, 3000), candidate("b", 30000, 3000)};
        candidates[0].cached = {{0, 20}};
        candidates[1].cached = {{20, 40}};
        auto found = po::search_plan(big, candidates, request);
        assert(found && found->from_cache && found->plan[0].end == 20);
        candidates[1].us_per_gib = 12000;  // the cached split gives the slow GPU 20 layers, not 16
        found = po::search_plan(big, candidates, request);
        assert(found && !found->from_cache);
    }
    // 7b. Equal GPUs and equal time for any split: the split is even, not 1 layer + the rest.
    {
        const auto found = po::search_plan(big, {candidate("a", 30000, 3000), candidate("b", 30000, 3000)}, request);
        assert(found && found->plan.size() == 2);
        const int first = found->plan[0].end - found->plan[0].begin;
        assert(first >= 19 && first <= 21);
    }
    // 7c. Measured links: two workers far from the planner but close to each other win over
    // a pair near the planner once their link is known; unknown, the guess goes through the
    // planner and the near pair wins.
    {
        std::vector<po::SearchCandidate> candidates{candidate("far-a", 30000, 3000, 100),
            candidate("far-b", 30000, 3000, 100), candidate("near-c", 30000, 3000, 40),
            candidate("near-d", 30000, 3000, 40)};
        auto found = po::search_plan(big, candidates, request);
        assert(found && found->plan.size() == 2 && found->plan[0].provider >= 2 && found->plan[1].provider >= 2);
        candidates[0].links.push_back({"far-b", 5, false});
        found = po::search_plan(big, candidates, request);
        assert(found && found->plan.size() == 2 && found->plan[0].provider <= 1 && found->plan[1].provider <= 1);
        // A relayed link carries the relay penalty.
        candidates[0].links.back().relayed = true;
        candidates[0].links.back().rtt_ms = 70;
        found = po::search_plan(big, candidates, request);
        assert(found && found->plan[0].provider >= 2);
    }
    // 8. A required head always leads.
    {
        po::SearchRequest headed = request;
        headed.head = 2;
        const auto found = po::search_plan(big, {candidate("a", 30000, 2000), candidate("b", 30000, 2000),
            candidate("owner", 30000, 6000)}, headed);
        assert(found && found->plan.front().provider == 2);
    }
    // 8b. First token (M4): prompt chunks overlap across stages, each hop carries them at its
    // measured bandwidth; unknown prefill speed or bandwidth means no estimate. A plan known to
    // miss the first-token limit loses to one within it, however fast it decodes.
    {
        std::vector<po::SearchCandidate> candidates{candidate("solo", 60000, 4000, 10),
            candidate("a", 30000, 2000, 10), candidate("b", 30000, 2000, 10)};
        for (auto& item : candidates) item.prefill_us_per_gib = 30;
        po::SearchRequest timed = request;
        timed.prompt_rows = 2048;
        timed.chunk_rows = 512;
        timed.row_bytes = 1024 * 4;
        auto found = po::search_plan(big, candidates, timed);
        // The pair decodes faster; its hop's bandwidth is unknown, so no first-token estimate.
        assert(found && found->plan.size() == 2 && found->first_token_ms == 0);
        candidates[1].links.push_back({"b", 10, false, 2.0});  // 2 MiB/s
        found = po::search_plan(big, candidates, timed);
        assert(found && found->plan.size() == 2 && found->first_token_ms > 0);
        // By hand: 4 chunks of 512 rows (2 MiB each); per chunk each stage computes its GiB at
        // 30 us per GiB per row, the hop takes 1 s; after the first chunk the slowest step
        // (the hop) sets the pace. Plus the hop's and the planner's one-way latencies.
        double expected = 0, slowest = 0;
        for (const auto& stage : found->plan) {
            const double ms = static_cast<double>(po::decode_bytes(big, stage.begin, stage.end))
                / static_cast<double>(1ull << 30) * 30 / 1000.0 * 512;
            expected += ms;
            slowest = std::max(slowest, ms);
        }
        expected += 1000 + 3 * std::max(slowest, 1000.0) + 10 / 2.0 + 10 / 2.0 + 10 / 2.0;
        assert(std::abs(found->first_token_ms - expected) < 1e-6);
        assert(std::abs(po::search_first_token_ms(big, candidates, found->plan, timed) - expected) < 1e-6);
        // Over the limit (5 s with the 20% margin): the single GPU, slower to decode, wins.
        timed.first_token_limit_ms = 5000 / 1.2;
        assert(found->first_token_ms > timed.first_token_limit_ms);
        found = po::search_plan(big, candidates, timed);
        assert(found && found->plan.size() == 1 && found->plan[0].provider == 0);
        assert(found->first_token_ms > 0 && found->first_token_ms <= timed.first_token_limit_ms);
        // Nothing known to be within the limit: a plan whose first token is unknown (a hop to
        // "solo" was never measured) ranks above one known to miss it.
        candidates[0].prefill_us_per_gib = 3000;
        found = po::search_plan(big, candidates, timed);
        assert(found && found->first_token_ms == 0 && found->plan.size() == 3);
        // Only known misses left: the fastest decode is still returned (selection reports it).
        timed.maximum_stages = 2;
        candidates[0].links = {{"a", 10, false, 2.0}, {"b", 10, false, 2.0}};
        found = po::search_plan(big, candidates, timed);
        assert(found && found->plan.size() == 2 && found->plan[0].provider != 0 && found->plan[1].provider != 0);
        assert(found->first_token_ms > timed.first_token_limit_ms);
        // A one-stage route needs no bandwidth: 2048 rows of the whole model.
        const std::vector<po::StageAssignment> whole{{0, 0, 40}};
        assert(std::abs(po::search_first_token_ms(big, candidates, whole, timed)
            - (static_cast<double>(po::decode_bytes(big, 0, 40)) / static_cast<double>(1ull << 30)
                * 3000 / 1000.0 * 2048 + 10)) < 1e-6);
    }
    // 9. A large discovery set terminates within the budget, deterministically.
    {
        std::mt19937 random(11);
        std::vector<po::SearchCandidate> candidates;
        for (int item = 0; item < 200; ++item) {
            candidates.push_back(candidate("n" + std::to_string(item), 8000 + random() % 30000,
                1500.0 + random() % 8000, random() % 200, random() % 5 == 0));
        }
        po::SearchRequest bounded = request;
        bounded.work_budget = 2000;
        const auto started = std::chrono::steady_clock::now();
        const auto first = po::search_plan(big, candidates, bounded);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto second = po::search_plan(big, candidates, bounded);
        assert(first && second && !first->complete && first->evaluated <= bounded.work_budget);
        assert(valid(big, candidates, bounded, first->plan) && first->plan.size() <= 5);
        assert(first->token_ms == second->token_ms && first->plan.size() == second->plan.size());
        assert(seconds < 5);  // generous: a correctness bound, not a benchmark
    }
    return 0;
}
