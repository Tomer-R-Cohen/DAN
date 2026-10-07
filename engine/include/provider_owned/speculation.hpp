#pragma once

// Speculative decoding, the acceptance rule.
//
// A draft model proposes the next few tokens; the real stage verifies them in one batch.
// The batch is [current, proposal 0, proposal 1, ...] and the stage samples one token per
// position, so sample i is "what really follows the first i+1 inputs". Sample 0 is always
// correct (it only depends on already-committed text). Every later sample is correct only
// while the proposal in front of it matched what the real model would have said, so the
// commit stops at the first wrong guess -- the output is exactly what plain decoding would
// have produced, just obtained in fewer passes.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dan::provider_owned {

inline constexpr std::uint32_t max_speculative_width = 32;

// The tokens a verified batch commits, in order. `samples` is one sampled token per batch
// position; `proposals` is what the draft guessed for positions 1..n.
inline std::vector<std::uint32_t> accept_speculation(const std::vector<std::uint32_t>& proposals,
    const std::vector<std::uint32_t>& samples) {
    std::vector<std::uint32_t> accepted;
    if (samples.empty()) return accepted;
    accepted.push_back(samples.front());
    for (std::size_t index = 0; index < proposals.size() && index + 1 < samples.size(); ++index) {
        if (proposals[index] != accepted.back()) break;
        accepted.push_back(samples[index + 1]);
    }
    return accepted;
}

// How many guessed tokens still fit: the verify batch is the current token plus the guesses,
// and it is decoded at `position`, so it must end inside the context. Near the end a plain
// step still fits where a full batch would not.
inline std::uint32_t speculation_room(std::uint32_t context, std::uint32_t position) {
    return position < context && context - position > 1 ? context - position - 1 : 0;
}

// Session-local search over verify widths, including 1 (ordinary decoding). Compare four
// complete rounds by elapsed time / committed tokens, rather than acceptance alone. Probe
// 2, 4, ... up to the worker's cap; keep a winner only with a 5% improvement. Recheck after
// sixteen held rounds, or sooner if the winner becomes slower than the plain baseline.
// ponytail: bounded doubling search; intermediate widths need a finer search if WAN tests
// show a useful optimum between these probes.
class AdaptiveDraftWidth {
public:
    explicit AdaptiveDraftWidth(std::uint32_t limit = 4)
        : limit_(std::clamp(limit, 1u, max_speculative_width)) {}

    std::uint32_t width() const { return width_; }

    void observe(std::uint32_t width, std::uint32_t committed, std::uint64_t elapsed_ns) {
        // Clipped context/EOG batches, unfinished rounds and malformed observations cannot
        // teach the cost of the selected width. Callers exclude terminal rounds as well.
        if (limit_ == 1 || width != width_ || committed == 0 || committed > width
            || elapsed_ns == 0) return;
        elapsed_ += static_cast<double>(elapsed_ns);
        committed_ += committed;
        if (++rounds_ != 4) return;
        const double cost = elapsed_ / committed_;
        rounds_ = 0;
        elapsed_ = 0;
        committed_ = 0;
        if (searching_) {
            if (width_ == 1) {
                baseline_ = best_cost_ = cost;
                best_ = 1;
                width_ = 2;
            } else if (cost < best_cost_ * 0.95) {
                best_ = width_;
                best_cost_ = cost;
                if (width_ < limit_) width_ = std::min(limit_, width_ * 2);
                else searching_ = false;
            } else {
                width_ = best_;
                searching_ = false;
            }
        } else if ((width_ > 1 && cost > baseline_ * 1.05) || ++held_blocks_ == 4) {
            width_ = 1;
            searching_ = true;
            held_blocks_ = 0;
        }
    }

private:
    std::uint32_t limit_ = 4, width_ = 1, best_ = 1, rounds_ = 0, held_blocks_ = 0;
    std::uint32_t committed_ = 0;
    double elapsed_ = 0, baseline_ = 0, best_cost_ = 0;
    bool searching_ = true;
};

} // namespace dan::provider_owned
