#pragma once

// When an idle replica should give way to a better one (docs/BETA_SELECTION_PLAN.md, M6).
// Each replica owner decides only for its own replica, from what it can see: its members,
// free peers and peers whose GPUs sit in other idle replicas. Giving way dissolves the
// replica (members keep their layers loaded) so the next formation can use the freed GPUs.
// The rules keep it from oscillating: a real gain is required, only after a quiet period,
// never with open or queued conversations, and failed upgrades back off exponentially.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dan::provider_owned {

struct UpgradeRules {
    std::int64_t idle_ms = 120000;           // no request and no open session for this long
    std::int64_t check_interval_ms = 60000;  // between looks (jittered by the caller)
    double speed_margin = 0.25;              // same tier: at least 25% less time per token
    std::int64_t first_backoff_ms = 600000;  // after an upgrade that did not happen
    std::int64_t max_backoff_ms = 4 * 3600 * 1000;
};

struct CurrentReplica {
    int quality_tier = 0;
    std::string model_sha256;
    double measured_ms = 0;           // owner-measured time per token; 0 = unknown
    std::vector<std::string> members; // PeerIDs
};

struct UpgradeOption {
    int quality_tier = 0;
    std::string model_sha256;
    double token_ms = 0;              // estimated time per token
    bool measured = false;            // estimate backed by measurements of every stage
    std::vector<std::string> peers;   // PeerIDs the plan would use
};

struct UpgradeState {
    std::int64_t idle_since_ms = 0;   // last request finished / replica became ready
    std::uint32_t open_sessions = 0;  // client sessions held (retained conversations count)
    std::size_t queued = 0;           // requests waiting or running
    std::int64_t not_before_ms = 0;   // backoff after a failed upgrade
};

// The option worth giving way for, or nothing. Higher tier wins; at equal tier the same
// model only when clearly faster by a measured estimate. A plan that uses no GPU outside
// the current replica is never a reason: the replica could not gain anything new.
inline std::optional<UpgradeOption> upgrade_choice(const CurrentReplica& current,
    const std::vector<UpgradeOption>& options, const UpgradeState& state, std::int64_t now_ms,
    const UpgradeRules& rules) {
    if (state.open_sessions != 0 || state.queued != 0 || now_ms < state.not_before_ms
        || now_ms - state.idle_since_ms < rules.idle_ms) return std::nullopt;
    std::optional<UpgradeOption> best;
    for (const UpgradeOption& option : options) {
        const bool adds_gpu = std::any_of(option.peers.begin(), option.peers.end(),
            [&](const std::string& peer) {
                return std::find(current.members.begin(), current.members.end(), peer)
                    == current.members.end();
            });
        if (!adds_gpu) continue;
        const bool higher_tier = option.quality_tier > current.quality_tier;
        const bool faster = option.quality_tier == current.quality_tier
            && option.model_sha256 == current.model_sha256 && option.measured
            && current.measured_ms > 0 && option.token_ms > 0
            && option.token_ms * (1 + rules.speed_margin) < current.measured_ms;
        if (!higher_tier && !faster) continue;
        if (!best || option.quality_tier > best->quality_tier
            || (option.quality_tier == best->quality_tier && option.token_ms < best->token_ms)) {
            best = option;
        }
    }
    return best;
}

// Backoff after giving way: if the replica formed next is not at least as good as the one
// hoped for, wait twice as long as last time before trying again; success resets it.
inline std::int64_t next_backoff_ms(std::int64_t previous_ms, bool upgraded, const UpgradeRules& rules) {
    if (upgraded) return 0;
    return previous_ms == 0 ? rules.first_backoff_ms
        : std::min(previous_ms * 2, rules.max_backoff_ms);
}

} // namespace dan::provider_owned
