#pragma once

// Stage-worker hello (provider_available) and assignment (assign_stage) messages.
// Placement itself lives in planner.hpp.

#include "provider_owned/measurement.hpp"
#include "provider_owned/planner.hpp"
#include "provider_owned/protocol.hpp"
#include "provider_owned/range_model.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dan::provider_owned {

// A layer range this worker already has on disk for one model.
struct CachedRange {
    std::string model_sha256;
    int begin = 0;
    int end = 0;

    bool operator==(const CachedRange&) const = default;
};

struct ProviderCapability {
    std::string id;
    std::string gpu;
    std::uint64_t offered_vram_mib = 0;
    std::string ring_endpoint;
    // Serve mode (decentralized) only; automatic mode leaves these empty and never sends them.
    std::string state;                // available | reserved | loading | serving
    std::string runtime_abi;
    std::uint32_t max_context = 0;
    std::uint32_t max_sessions = 0;
    std::vector<std::string> models;  // GGUF SHA-256 values in the worker's catalog
    std::vector<CachedRange> cached;  // ranges already downloaded (a planning hint)
    // Measured decode speed: microseconds to run one GiB of this GPU's weights for one token
    // (0 = not measured yet). Replica formation estimates a route's time per token with it.
    std::uint64_t speed_us_per_gib = 0;
    // Speeds this worker observed per configuration (measurement.hpp), at most
    // max_speed_records. Hints: the worker could misreport them.
    std::vector<SpeedRecord> speeds;
    // A worker holding a stage: seconds since that stage last computed, and the sessions
    // open on it. An idle replica's GPUs (no session, quiet for a while) may be reclaimed by
    // a better replica once their owner gives way (upgrade.hpp). Absent when available.
    std::optional<std::uint32_t> idle_s;
    std::uint32_t open_sessions = 0;
    // Round trips this worker's node measured to other peers (its sidecar's pings, kept
    // fresh for peers it has DAN streams with). Planners use them for worker-to-worker ring
    // links instead of guessing through the planning node. Hints, like every claim here.
    // Bandwidth: KiB/s at which this worker received large frames from that peer (ring
    // traffic it carried); 0 = not measured. Prefill chunks are megabytes, so it sets how
    // fast a long prompt crosses the hop.
    struct PeerLink {
        std::string peer;
        std::uint32_t rtt_ms = 0;
        bool relayed = false;
        std::uint32_t kib_per_s = 0;
    };
    std::vector<PeerLink> links;
    bool replica_owner = false;       // this node runs a replica owner (replica=auto)
    // Activation formats this worker accepts and sends when a route asks (f32 always).
    bool f16_activations = false;
    bool fp8_activations = false;
};

struct ModelAssignment {
    std::string model_id;
    std::string url;
    std::string revision;
    std::string sha256;
    int begin = 0;
    int end = 0;
    std::uint32_t context = 0;
    std::uint32_t sessions = 0;
    std::string next_endpoint;
    std::string previous_peer_id;

    bool operator==(const ModelAssignment&) const = default;
    bool same_stage(const ModelAssignment& other) const {
        return model_id == other.model_id && url == other.url && revision == other.revision
            && sha256 == other.sha256 && begin == other.begin && end == other.end
            && context == other.context && sessions == other.sessions;
    }
};

inline bool hex_string(std::string_view value, std::size_t length) {
    return value.size() == length && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    });
}

inline std::string assignment_message(const ModelAssignment& assignment) {
    return "model_id=" + assignment.model_id + "\nurl=" + assignment.url
        + "\nrevision=" + assignment.revision + "\nsha256=" + assignment.sha256
        + "\nbegin=" + std::to_string(assignment.begin)
        + "\nend=" + std::to_string(assignment.end)
        + "\ncontext=" + std::to_string(assignment.context)
        + "\nsessions=" + std::to_string(assignment.sessions)
        + (assignment.next_endpoint.empty() ? "" : "\nnext=" + assignment.next_endpoint)
        + (assignment.previous_peer_id.empty() ? ""
            : "\nprevious_peer=" + assignment.previous_peer_id);
}

inline bool parse_assignment(std::string_view text, ModelAssignment& assignment) {
    assignment = {};
    auto number = [](std::string_view value, auto& output) {
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), output);
        return ec == std::errc{} && end == value.data() + value.size();
    };
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        const std::string_view line = text.substr(0, newline);
        const std::size_t equal = line.find('=');
        if (equal == std::string_view::npos) return false;
        const auto key = line.substr(0, equal), value = line.substr(equal + 1);
        if (key == "model_id") assignment.model_id = value;
        else if (key == "url") assignment.url = value;
        else if (key == "revision") assignment.revision = value;
        else if (key == "sha256") assignment.sha256 = value;
        else if (key == "begin") { if (!number(value, assignment.begin)) return false; }
        else if (key == "end") { if (!number(value, assignment.end)) return false; }
        else if (key == "context") { if (!number(value, assignment.context)) return false; }
        else if (key == "sessions") { if (!number(value, assignment.sessions)) return false; }
        else if (key == "next") assignment.next_endpoint = value;
        else if (key == "previous_peer") assignment.previous_peer_id = value;
        else return false;
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return !assignment.model_id.empty()
        && assignment.model_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") == std::string::npos
        && assignment.url.starts_with("https://") && assignment.url.find_first_of("\r\n") == std::string::npos
        && hex_string(assignment.revision, 40) && hex_string(assignment.sha256, 64)
        && assignment.begin >= 0 && assignment.end > assignment.begin
        && assignment.context != 0 && assignment.sessions != 0
        && valid_ring_target(assignment.next_endpoint)
        && (assignment.previous_peer_id.empty()
            || (!assignment.next_endpoint.empty() && valid_peer_id(assignment.previous_peer_id)));
}

// Automatic formation: plan over registered providers' offered memory.
inline std::optional<std::vector<StageAssignment>> plan_replica(const ModelIndex& model,
    const std::vector<ProviderCapability>& providers, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages = 1) {
    std::vector<std::uint64_t> offered;
    offered.reserve(providers.size());
    for (const ProviderCapability& provider : providers) offered.push_back(provider.offered_vram_mib);
    return plan_stages(model, offered, context, sessions, minimum_stages);
}

inline std::string available_message(const ProviderCapability& provider) {
    std::string text = "id=" + provider.id + "\ngpu=" + provider.gpu
        + "\nvram_mib=" + std::to_string(provider.offered_vram_mib)
        + (provider.ring_endpoint.empty() ? "" : "\nring=" + provider.ring_endpoint);
    if (!provider.state.empty()) text += "\nstate=" + provider.state;
    if (!provider.runtime_abi.empty()) text += "\nabi=" + provider.runtime_abi;
    if (provider.max_context != 0) text += "\nmax_context=" + std::to_string(provider.max_context);
    if (provider.max_sessions != 0) text += "\nmax_sessions=" + std::to_string(provider.max_sessions);
    for (const std::string& model : provider.models) text += "\nmodel=" + model;
    for (const CachedRange& range : provider.cached) {
        text += "\ncached=" + range.model_sha256 + ":" + std::to_string(range.begin)
            + "-" + std::to_string(range.end);
    }
    if (provider.speed_us_per_gib != 0) text += "\nspeed=" + std::to_string(provider.speed_us_per_gib);
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    for (const SpeedRecord& record : provider.speeds) text += "\nmeasured=" + speed_text(record, now);
    if (provider.idle_s) {
        text += "\nidle_s=" + std::to_string(*provider.idle_s)
            + "\nopen_sessions=" + std::to_string(provider.open_sessions);
    }
    for (const ProviderCapability::PeerLink& link : provider.links) {
        text += "\nlink=" + link.peer + ":" + std::to_string(link.rtt_ms) + ":"
            + (link.relayed ? "relay" : "direct");
        if (link.kib_per_s != 0) {
            text += "\nlink_bw=" + link.peer + ":" + std::to_string(link.kib_per_s);
        }
    }
    if (provider.replica_owner) text += "\nowner=1";
    if (provider.f16_activations || provider.fp8_activations) {
        text += std::string("\nactivations=") + (provider.f16_activations ? "f16" : "")
            + (provider.f16_activations && provider.fp8_activations ? "," : "")
            + (provider.fp8_activations ? "fp8" : "");
    }
    return text;
}

// Bounds on a greeting (BETA_SELECTION_PLAN.md M7): a public peer may send anything, and a
// greeting outside them is refused whole rather than clamped.
inline constexpr std::size_t max_greeting_bytes = 64 * 1024;
inline constexpr std::size_t max_greeting_name = 128;
inline constexpr std::uint64_t max_greeting_vram_mib = 16ull << 20;  // 16 TiB
inline constexpr std::uint32_t max_greeting_context = 1u << 20;
inline constexpr std::uint32_t max_greeting_sessions = 256;
inline constexpr std::size_t max_greeting_models = 64;
inline constexpr std::size_t max_greeting_cached = 256;
inline constexpr int max_greeting_layer = 4096;
inline constexpr std::size_t max_greeting_links = 32;
inline constexpr std::uint32_t max_greeting_kib_per_s = 100u << 20;  // 100 GiB/s

inline bool parse_available(std::string_view text, ProviderCapability& provider) {
    provider = {};
    if (text.size() > max_greeting_bytes) return false;
    const auto number = [](std::string_view value, auto& output) {
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), output);
        return ec == std::errc{} && end == value.data() + value.size();
    };
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        const std::string_view line = text.substr(0, newline);
        const std::size_t equal = line.find('=');
        if (equal == std::string_view::npos) return false;
        const auto key = line.substr(0, equal), value = line.substr(equal + 1);
        if (key == "id") provider.id = value;
        else if (key == "gpu") provider.gpu = value;
        else if (key == "vram_mib") { if (!number(value, provider.offered_vram_mib)) return false; }
        else if (key == "ring") provider.ring_endpoint = value;
        else if (key == "state") provider.state = value;
        else if (key == "abi") provider.runtime_abi = value;
        else if (key == "max_context") { if (!number(value, provider.max_context)) return false; }
        else if (key == "max_sessions") { if (!number(value, provider.max_sessions)) return false; }
        else if (key == "model") {
            if (!hex_string(value, 64) || provider.models.size() >= max_greeting_models) return false;
            provider.models.emplace_back(value);
        } else if (key == "cached") {
            // <sha256>:<begin>-<end>
            const std::size_t colon = value.find(':'), dash = value.find('-', colon + 1);
            if (colon == std::string_view::npos || dash == std::string_view::npos
                || !hex_string(value.substr(0, colon), 64)) return false;
            CachedRange range;
            range.model_sha256 = value.substr(0, colon);
            if (!number(value.substr(colon + 1, dash - colon - 1), range.begin)
                || !number(value.substr(dash + 1), range.end)
                || range.begin < 0 || range.end <= range.begin || range.end > max_greeting_layer
                || provider.cached.size() >= max_greeting_cached) return false;
            provider.cached.push_back(range);
        } else if (key == "speed") {
            if (!number(value, provider.speed_us_per_gib)
                || provider.speed_us_per_gib > static_cast<std::uint64_t>(max_us_per_gib)) return false;
        } else if (key == "measured") {
            const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            SpeedRecord record;
            if (provider.speeds.size() >= max_speed_records
                || !parse_speed_text(value, record, now)) return false;
            provider.speeds.push_back(std::move(record));
        } else if (key == "idle_s") {
            std::uint32_t seconds = 0;
            if (!number(value, seconds)) return false;
            provider.idle_s = seconds;
        } else if (key == "open_sessions") {
            if (!number(value, provider.open_sessions)) return false;
        } else if (key == "link") {
            // <PeerID>:<rtt ms>:<direct|relay>
            const std::size_t first = value.find(':'), second = value.find(':', first + 1);
            ProviderCapability::PeerLink link;
            if (first == std::string_view::npos || second == std::string_view::npos
                || provider.links.size() >= max_greeting_links
                || !valid_peer_id(value.substr(0, first))
                || !number(value.substr(first + 1, second - first - 1), link.rtt_ms)
                || link.rtt_ms > 60000) return false;
            const std::string_view path = value.substr(second + 1);
            if (path != "direct" && path != "relay") return false;
            link.peer = value.substr(0, first);
            link.relayed = path == "relay";
            provider.links.push_back(std::move(link));
        } else if (key == "link_bw") {
            // <PeerID>:<KiB/s>, for a link= line already given (its own key, so older
            // parsers skip it).
            const std::size_t colon = value.find(':');
            std::uint32_t kib_per_s = 0;
            if (colon == std::string_view::npos || !number(value.substr(colon + 1), kib_per_s)
                || kib_per_s == 0 || kib_per_s > max_greeting_kib_per_s) return false;
            const std::string_view peer = value.substr(0, colon);
            const auto found = std::find_if(provider.links.begin(), provider.links.end(),
                [&](const ProviderCapability::PeerLink& link) { return link.peer == peer; });
            if (found == provider.links.end() || found->kib_per_s != 0) return false;
            found->kib_per_s = kib_per_s;
        } else if (key == "owner") {
            provider.replica_owner = value == "1";
        } else if (key == "activations") {
            // A comma-separated list, e.g. "f16,fp8".
            for (std::string_view rest = value; !rest.empty();) {
                const std::size_t comma = rest.find(',');
                const std::string_view format = rest.substr(0, comma);
                if (format == "f16") provider.f16_activations = true;
                if (format == "fp8") provider.fp8_activations = true;
                if (comma == std::string_view::npos) break;
                rest.remove_prefix(comma + 1);
            }
        }
        // Unknown keys are skipped, so newer workers can add fields without breaking older
        // clients.
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return !provider.id.empty() && provider.id.find_first_of("\r\n") == std::string::npos
        && provider.id.size() <= max_greeting_name && provider.gpu.size() <= max_greeting_name
        && provider.runtime_abi.size() <= max_greeting_name
        && provider.offered_vram_mib <= max_greeting_vram_mib
        && provider.max_context <= max_greeting_context
        && provider.max_sessions <= max_greeting_sessions
        && provider.open_sessions <= max_greeting_sessions
        && !provider.gpu.empty() && provider.offered_vram_mib != 0
        && valid_ring_target(provider.ring_endpoint)
        && (provider.state.empty() || provider.state == "available" || provider.state == "reserved"
            || provider.state == "loading" || provider.state == "serving")
        && provider.runtime_abi.find_first_of("\r\n") == std::string::npos;
}

} // namespace dan::provider_owned
