// The M0 selection contract's worked examples (docs/BETA_SELECTION_PLAN.md): which plan wins
// and why, including no qualifying plan. Executable in Release builds too.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "provider_owned/selection.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace po = dan::provider_owned;

namespace {

constexpr std::uint32_t context = 32768;

// A measured, ready plan well inside every limit (20 tok/s = 50 ms; with the 20% margin a
// plan must predict at most 41.7 ms per token and 4167 ms to the first token).
po::PlanEstimate plan(std::string label, int tier, double token_ms, double first_token_ms = 800) {
    po::PlanEstimate estimate;
    estimate.label = std::move(label);
    estimate.quality_tier = tier;
    estimate.context = context;
    estimate.providers = 2;
    estimate.ready = true;
    estimate.measured = true;
    estimate.token_ms = token_ms;
    estimate.first_token_ms = first_token_ms;
    estimate.resource_bytes = 1;
    return estimate;
}

} // namespace

int main() {
    const po::SelectionPolicy policy;

    // 1. A better-ranked model that meets the target beats a faster, lower-ranked one.
    {
        const auto selection = po::select_plan({plan("7b", 30, 15), plan("14b", 40, 35)}, policy);
        assert(selection.chosen == 1u && !selection.fallback);
    }
    // 2. An under-target large replica cannot win on size or tier: 45 ms * 1.2 > 50 ms.
    {
        const auto selection = po::select_plan({plan("32b", 50, 45), plan("14b", 40, 30)}, policy);
        assert(selection.chosen == 1u);
        assert(po::assess_plan(plan("32b", 50, 45), policy).verdict == po::Verdict::below_target);
    }
    // 3. Too little context is never offered, not even as a fallback.
    {
        auto short_context = plan("7b-512", 30, 10);
        short_context.context = 512;
        const auto selection = po::select_plan({short_context}, policy);
        assert(!selection.chosen && !selection.fallback);
        assert(selection.reason.find("context") != std::string::npos);
    }
    // 4. More providers than the cap is ineligible; the cap is configurable.
    {
        auto six = plan("70b-six", 60, 30);
        six.providers = 6;
        assert(po::assess_plan(six, policy).verdict == po::Verdict::ineligible);
        po::SelectionPolicy wider = policy;
        wider.max_providers = 6;
        assert(po::assess_plan(six, wider).verdict == po::Verdict::meets_target);
    }
    // 5. Unknown or unmeasured speed is never zero: it is an explicit trial at best.
    {
        auto unknown = plan("new-gpu", 50, 0);
        auto unmeasured = plan("claimed-fast", 50, 5);
        unmeasured.measured = false;
        assert(po::assess_plan(unknown, policy).verdict == po::Verdict::unverified);
        assert(po::assess_plan(unmeasured, policy).verdict == po::Verdict::unverified);
        const auto selection = po::select_plan({unknown, plan("7b", 30, 20)}, policy);
        assert(selection.chosen == 1u && !selection.fallback);
    }
    // 6. No qualifying plan: nothing is chosen; the best slower option is offered explicitly.
    {
        const auto selection = po::select_plan({plan("14b-slow", 40, 60),
            plan("7b-slow", 30, 45), plan("32b-trial", 50, 0)}, policy);
        assert(!selection.chosen && selection.fallback == 2u);
        assert(selection.reason.starts_with("no plan meets the target"));
    }
    // 7. ready_first: answer on the ready plan, offer the better one that has to load.
    {
        auto cold = plan("14b-cold", 40, 30);
        cold.ready = false;
        cold.load_ms = 200000;
        const std::vector<po::PlanEstimate> plans{plan("7b-ready", 30, 20), cold};
        const auto ready_first = po::select_plan(plans, policy);
        assert(ready_first.chosen == 0u && ready_first.better_cold == 1u);
        po::SelectionPolicy waiting = policy;
        waiting.cold_start = po::ColdStart::wait;
        const auto wait = po::select_plan(plans, waiting);
        assert(wait.chosen == 1u && !wait.better_cold);
        // With nothing ready, ready_first waits for the best plan that must load.
        const auto only_cold = po::select_plan({cold}, policy);
        assert(only_cold.chosen == 0u);
    }
    // 8. First response: inside the 2 s aim beats 2-5 s; over 5 s (with margin) is too slow.
    {
        const auto selection = po::select_plan({plan("a", 30, 20, 3000), plan("b", 30, 30, 1200)},
            policy);
        assert(selection.chosen == 1u);
        assert(po::assess_plan(plan("c", 30, 20, 4500), policy).verdict == po::Verdict::below_target);
    }
    // 9. Equal tier and speed: warm cache, then fewer resources, then the label.
    {
        auto warm = plan("b-warm", 30, 20);
        warm.cache_warm = true;
        assert(po::select_plan({plan("a", 30, 20), warm}, policy).chosen == 1u);
        auto small = plan("z-small", 30, 20);
        small.resource_bytes = 0;
        assert(po::select_plan({plan("a", 30, 20), small}, policy).chosen == 1u);
        assert(po::select_plan({plan("b", 30, 20), plan("a", 30, 20)}, policy).chosen == 1u);
    }
    // 10. Invalid numbers from a peer are refused, not trusted.
    {
        assert(po::assess_plan(plan("nan", 30, std::nan("")), policy).verdict == po::Verdict::ineligible);
        assert(po::assess_plan(plan("negative", 30, -1), policy).verdict == po::Verdict::ineligible);
    }
    // 11. Catalog order: tier first; file size only breaks ties.
    assert(po::preferred_model(40, 8, 30, 20));
    assert(!po::preferred_model(30, 20, 40, 8));
    assert(po::preferred_model(30, 20, 30, 8));
    // 12. Streaming stalls beyond the p95 limit make a plan too slow.
    {
        auto stalls = plan("stalls", 30, 20);
        stalls.stall_p95_ms = 900;
        assert(po::assess_plan(stalls, policy).verdict == po::Verdict::below_target);
    }
    // 13. Deterministic: the same input gives the same answer in any order.
    {
        const auto forward = po::select_plan({plan("x", 30, 20), plan("y", 30, 20)}, policy);
        const auto backward = po::select_plan({plan("y", 30, 20), plan("x", 30, 20)}, policy);
        assert(forward.chosen == 0u && backward.chosen == 1u);
    }
    return 0;
}
