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

#include <cstdint>
#include <vector>

namespace dan::provider_owned {

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

} // namespace dan::provider_owned
