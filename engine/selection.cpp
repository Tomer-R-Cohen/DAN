#include "provider_owned/selection.hpp"

#include <cmath>
#include <tuple>

namespace dan::provider_owned {
namespace {

bool usable_number(double value) {
    return std::isfinite(value) && value >= 0;
}

std::string milliseconds(double value) {
    return std::to_string(static_cast<long long>(std::llround(value))) + " ms";
}

// Whether `left` should be preferred over `right`, both already eligible or both fallbacks.
bool better(const PlanEstimate& left, const PlanEstimate& right, const SelectionPolicy& policy) {
    if (left.quality_tier != right.quality_tier) return left.quality_tier > right.quality_tier;
    // Unknown (0) sorts after any known value, so an unmeasured trial never looks fastest.
    const auto known = [](double value) { return value > 0 ? value : INFINITY; };
    const bool left_quick = left.first_token_ms > 0
        && left.first_token_ms * (1 + policy.margin) <= policy.first_token_target_ms;
    const bool right_quick = right.first_token_ms > 0
        && right.first_token_ms * (1 + policy.margin) <= policy.first_token_target_ms;
    if (left_quick != right_quick) return left_quick;
    if (known(left.token_ms) != known(right.token_ms)) return known(left.token_ms) < known(right.token_ms);
    if (left.ready != right.ready) return left.ready;
    if (left.cache_warm != right.cache_warm) return left.cache_warm;
    return std::tie(left.resource_bytes, left.providers, left.label)
        < std::tie(right.resource_bytes, right.providers, right.label);
}

} // namespace

Assessment assess_plan(const PlanEstimate& plan, const SelectionPolicy& policy) {
    if (!usable_number(plan.token_ms) || !usable_number(plan.first_token_ms)
        || !usable_number(plan.load_ms) || !usable_number(plan.stall_p95_ms)) {
        return {Verdict::ineligible, "invalid estimate"};
    }
    if (plan.context < policy.context) {
        return {Verdict::ineligible, "context " + std::to_string(plan.context) + " < required "
            + std::to_string(policy.context)};
    }
    if (plan.providers == 0 || plan.providers > policy.max_providers) {
        return {Verdict::ineligible, std::to_string(plan.providers) + " providers (limit "
            + std::to_string(policy.max_providers) + ")"};
    }
    if (!plan.measured || plan.token_ms == 0 || plan.first_token_ms == 0) {
        return {Verdict::unverified, "speed not measured yet"};
    }
    const double token_limit = 1000.0 / policy.target_tokens_per_s;
    if (plan.token_ms * (1 + policy.margin) > token_limit) {
        return {Verdict::below_target, milliseconds(plan.token_ms) + " per token (limit "
            + milliseconds(token_limit) + " with margin)"};
    }
    if (plan.first_token_ms * (1 + policy.margin) > policy.first_token_max_ms) {
        return {Verdict::below_target, "first token " + milliseconds(plan.first_token_ms)
            + " (limit " + milliseconds(policy.first_token_max_ms) + " with margin)"};
    }
    if (plan.stall_p95_ms * (1 + policy.margin) > policy.stall_p95_ms) {
        return {Verdict::below_target, "p95 stall " + milliseconds(plan.stall_p95_ms)};
    }
    return {Verdict::meets_target, {}};
}

Selection select_plan(const std::vector<PlanEstimate>& plans, const SelectionPolicy& policy) {
    Selection selection;
    std::optional<std::size_t> best_ready, best_cold, best_any;
    const auto keep = [&](std::optional<std::size_t>& best, std::size_t index) {
        if (!best || better(plans[index], plans[*best], policy)) best = index;
    };
    for (std::size_t index = 0; index < plans.size(); ++index) {
        const Verdict verdict = assess_plan(plans[index], policy).verdict;
        if (verdict == Verdict::meets_target) {
            keep(plans[index].ready ? best_ready : best_cold, index);
            keep(best_any, index);
        } else if (verdict != Verdict::ineligible) {
            keep(selection.fallback, index);
        }
    }
    if (policy.cold_start == ColdStart::wait || !best_ready) {
        selection.chosen = best_any;
    } else {
        selection.chosen = best_ready;
        if (best_cold && plans[*best_cold].quality_tier > plans[*best_ready].quality_tier) {
            selection.better_cold = best_cold;
        }
    }
    if (selection.chosen) {
        const PlanEstimate& plan = plans[*selection.chosen];
        selection.reason = plan.label + (plan.ready ? " (ready)" : " (needs loading)")
            + ": tier " + std::to_string(plan.quality_tier) + ", "
            + milliseconds(plan.token_ms) + " per token, first token "
            + milliseconds(plan.first_token_ms);
        // A fallback is offered only when nothing qualifies.
        selection.fallback.reset();
    } else if (selection.fallback) {
        selection.reason = "no plan meets the target; slower option: " + plans[*selection.fallback].label
            + " (" + assess_plan(plans[*selection.fallback], policy).reason + ")";
    } else {
        selection.reason = plans.empty() ? "no feasible plan" : "no eligible plan: "
            + assess_plan(plans.front(), policy).reason;
    }
    return selection;
}

bool preferred_model(int left_tier, std::uint64_t left_bytes, int right_tier,
    std::uint64_t right_bytes) {
    if (left_tier != right_tier) return left_tier > right_tier;
    return left_bytes > right_bytes;
}

} // namespace dan::provider_owned
