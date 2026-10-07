// Idle replica upgrades (docs/BETA_SELECTION_PLAN.md, M6). Executable in Release builds too.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "provider_owned/upgrade.hpp"

#include <cassert>

namespace po = dan::provider_owned;

int main() {
    const po::UpgradeRules rules;
    const std::string small(64, 'a'), big(64, 'b');
    const po::CurrentReplica current{10, small, 100, {"self", "m1"}};
    po::UpgradeState idle;
    idle.idle_since_ms = 0;
    const std::int64_t now = rules.idle_ms + 1;

    // A higher-tier model that needs a new GPU: give way.
    const po::UpgradeOption bigger{20, big, 300, false, {"self", "m1", "peer"}};
    auto choice = po::upgrade_choice(current, {bigger}, idle, now, rules);
    assert(choice && choice->model_sha256 == big);

    // Never with an open or queued conversation, before the quiet period, or during backoff.
    po::UpgradeState busy = idle;
    busy.open_sessions = 1;
    assert(!po::upgrade_choice(current, {bigger}, busy, now, rules));
    busy = idle;
    busy.queued = 1;
    assert(!po::upgrade_choice(current, {bigger}, busy, now, rules));
    assert(!po::upgrade_choice(current, {bigger}, idle, rules.idle_ms - 1, rules));
    po::UpgradeState waiting = idle;
    waiting.not_before_ms = now + 1;
    assert(!po::upgrade_choice(current, {bigger}, waiting, now, rules));

    // A plan using only the current members is no reason (nothing new to gain).
    const po::UpgradeOption same_members{20, big, 50, true, {"self", "m1"}};
    assert(!po::upgrade_choice(current, {same_members}, idle, now, rules));

    // Same model: only clearly faster, measured, counts; a lower tier never does.
    const po::UpgradeOption slightly_faster{10, small, 90, true, {"self", "peer"}};
    assert(!po::upgrade_choice(current, {slightly_faster}, idle, now, rules));
    const po::UpgradeOption much_faster{10, small, 70, true, {"self", "peer"}};
    assert(po::upgrade_choice(current, {much_faster}, idle, now, rules));
    const po::UpgradeOption guessed_faster{10, small, 70, false, {"self", "peer"}};
    assert(!po::upgrade_choice(current, {guessed_faster}, idle, now, rules));
    const po::UpgradeOption lower{5, std::string(64, 'c'), 10, true, {"self", "peer"}};
    assert(!po::upgrade_choice(current, {lower}, idle, now, rules));
    po::CurrentReplica unmeasured = current;
    unmeasured.measured_ms = 0;
    assert(!po::upgrade_choice(unmeasured, {much_faster}, idle, now, rules));

    // Among several, the highest tier, then the fastest.
    const po::UpgradeOption biggest_slow{30, std::string(64, 'd'), 900, false, {"peer"}};
    const po::UpgradeOption biggest_fast{30, std::string(64, 'd'), 400, false, {"peer", "other"}};
    choice = po::upgrade_choice(current, {bigger, biggest_slow, much_faster, biggest_fast}, idle, now, rules);
    assert(choice && choice->quality_tier == 30 && choice->token_ms == 400);

    // Backoff doubles after a failed upgrade, is capped, and resets on success.
    std::int64_t backoff = po::next_backoff_ms(0, false, rules);
    assert(backoff == rules.first_backoff_ms);
    backoff = po::next_backoff_ms(backoff, false, rules);
    assert(backoff == 2 * rules.first_backoff_ms);
    for (int round = 0; round < 20; ++round) backoff = po::next_backoff_ms(backoff, false, rules);
    assert(backoff == rules.max_backoff_ms);
    assert(po::next_backoff_ms(backoff, true, rules) == 0);
    return 0;
}
