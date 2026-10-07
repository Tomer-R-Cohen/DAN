#pragma once

// Observed stage speeds (docs/BETA_SELECTION_PLAN.md, M3). A worker times its own ordinary
// traffic -- no benchmark runs -- and keeps a small table of speeds, one per configuration
// that behaves differently: model, prefill or decode, context size and how many sessions
// were active. Each entry carries its sample count and age. Planners read these as hints
// (a peer could lie); an unknown configuration is "not measured", never "free".

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace dan::provider_owned {

enum class Phase : std::uint8_t { decode, prefill };

inline constexpr std::size_t max_speed_records = 64;            // per worker, oldest dropped
inline constexpr std::int64_t speed_max_age_s = 7 * 24 * 3600;  // older entries are ignored
inline constexpr std::uint32_t confident_samples = 32;           // enough to trust an entry
inline constexpr double max_us_per_gib = 1e9;                    // larger is nonsense

// Contexts in powers of two from 512: KV size and attention cost grow with context.
inline std::uint32_t context_bucket(std::uint32_t context) {
    std::uint32_t bucket = 512;
    while (bucket < context && bucket < (1u << 20)) bucket *= 2;
    return bucket;
}

// Active sessions as 1, 2, 4, 8, ... 64: sessions on one GPU share its time.
inline std::uint32_t sessions_bucket(std::uint32_t sessions) {
    std::uint32_t bucket = 1;
    while (bucket < sessions && bucket < 64) bucket *= 2;
    return bucket;
}

struct SpeedKey {
    std::string model_sha256;  // lowercase; dense vs. MoE follows from the model
    Phase phase = Phase::decode;
    std::uint32_t context = 0;   // context_bucket
    std::uint32_t sessions = 0;  // sessions_bucket
    auto tie() const { return std::tie(model_sha256, phase, context, sessions); }
    bool operator==(const SpeedKey& other) const { return tie() == other.tie(); }
};

struct SpeedRecord {
    SpeedKey key;
    // Microseconds per GiB of the stage's weights: per token for decode, per prompt row for
    // prefill. Dividing by weight bytes lets one measurement estimate other layer ranges.
    double us_per_gib = 0;
    std::uint32_t samples = 0;
    std::int64_t updated_unix_s = 0;
};

class SpeedTable {
public:
    // Folds `samples` new observations averaging `us_per_gib` into the entry for `key`.
    // Older evidence counts for at most confident_samples, so speed changes show up.
    void observe(const SpeedKey& key, double us_per_gib, std::uint32_t samples, std::int64_t now) {
        if (!(us_per_gib > 0) || us_per_gib > max_us_per_gib || samples == 0) return;
        auto found = std::find_if(records_.begin(), records_.end(),
            [&](const SpeedRecord& record) { return record.key == key; });
        if (found == records_.end()) {
            if (records_.size() >= max_speed_records) {
                records_.erase(std::min_element(records_.begin(), records_.end(),
                    [](const SpeedRecord& left, const SpeedRecord& right) {
                        return left.updated_unix_s < right.updated_unix_s;
                    }));
            }
            records_.push_back({key, us_per_gib, samples, now});
            return;
        }
        const double old_weight = std::min(found->samples, confident_samples);
        found->us_per_gib = (found->us_per_gib * old_weight + us_per_gib * samples)
            / (old_weight + samples);
        found->samples = found->samples > UINT32_MAX - samples ? UINT32_MAX : found->samples + samples;
        found->updated_unix_s = now;
    }

    void add(const SpeedRecord& record) {
        observe(record.key, record.us_per_gib, record.samples, record.updated_unix_s);
    }

    const std::vector<SpeedRecord>& records() const { return records_; }

private:
    std::vector<SpeedRecord> records_;
};

// Wire and file form: <sha256>:<d|p>:<context>:<sessions>:<us per GiB>:<samples>:<age s>.
// Age rather than a timestamp, so clocks on different machines need not agree.
inline std::string speed_text(const SpeedRecord& record, std::int64_t now) {
    return record.key.model_sha256 + ":" + (record.key.phase == Phase::decode ? "d" : "p") + ":"
        + std::to_string(record.key.context) + ":" + std::to_string(record.key.sessions) + ":"
        + std::to_string(static_cast<std::uint64_t>(std::llround(record.us_per_gib))) + ":"
        + std::to_string(record.samples) + ":"
        + std::to_string(std::max<std::int64_t>(0, now - record.updated_unix_s));
}

// Strict: any malformed or implausible field rejects the entry.
inline bool parse_speed_text(std::string_view text, SpeedRecord& record, std::int64_t now) {
    std::string_view fields[7];
    for (std::size_t index = 0; index < 7; ++index) {
        const std::size_t colon = text.find(':');
        if ((colon == std::string_view::npos) != (index == 6)) return false;
        fields[index] = text.substr(0, colon);
        if (colon != std::string_view::npos) text.remove_prefix(colon + 1);
    }
    const auto number = [](std::string_view value, auto& output) {
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), output);
        return error == std::errc{} && end == value.data() + value.size() && !value.empty();
    };
    if (fields[0].size() != 64 || !std::all_of(fields[0].begin(), fields[0].end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }) || (fields[1] != "d" && fields[1] != "p")) return false;
    std::uint64_t speed = 0, age = 0;
    SpeedRecord parsed;
    parsed.key.model_sha256 = fields[0];
    parsed.key.phase = fields[1] == "d" ? Phase::decode : Phase::prefill;
    if (!number(fields[2], parsed.key.context) || !number(fields[3], parsed.key.sessions)
        || !number(fields[4], speed) || !number(fields[5], parsed.samples) || !number(fields[6], age)
        || parsed.key.context != context_bucket(parsed.key.context)
        || parsed.key.sessions != sessions_bucket(parsed.key.sessions)
        || speed == 0 || speed > max_us_per_gib || parsed.samples == 0
        || age > static_cast<std::uint64_t>(speed_max_age_s) * 4) return false;
    parsed.us_per_gib = static_cast<double>(speed);
    parsed.updated_unix_s = now - static_cast<std::int64_t>(age);
    record = std::move(parsed);
    return true;
}

struct SpeedEstimate {
    double us_per_gib = 0;
    bool measured = false;  // backed by enough recent samples of a comparable configuration
};

// The speed to plan with for `wanted`: the closest recent entry with enough samples, exact or
// measured under heavier conditions (larger context, more sessions), which can only overstate
// the time. A few samples are noise -- on small stages fixed per-step costs dominate -- so
// thin entries are ignored. With nothing comparable, `fallback` (a generic or default speed)
// is used and the estimate is not measured.
inline SpeedEstimate estimate_speed(const std::vector<SpeedRecord>& records, const SpeedKey& wanted,
    std::int64_t now, double fallback) {
    const SpeedRecord* best = nullptr;
    for (const SpeedRecord& record : records) {
        if (record.key.model_sha256 != wanted.model_sha256 || record.key.phase != wanted.phase
            || record.key.context < wanted.context || record.key.sessions < wanted.sessions
            || record.samples < confident_samples
            || now - record.updated_unix_s > speed_max_age_s) continue;
        if (!best || std::tie(record.key.context, record.key.sessions)
                < std::tie(best->key.context, best->key.sessions)) best = &record;
    }
    if (!best) return {fallback, false};
    return {best->us_per_gib, true};
}

} // namespace dan::provider_owned
