#pragma once

// Stage-worker hello (provider_available) and assignment (assign_stage) messages.
// Placement itself lives in planner.hpp.

#include "provider_owned/planner.hpp"
#include "provider_owned/protocol.hpp"
#include "provider_owned/range_model.hpp"

#include <algorithm>
#include <charconv>
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
    return text;
}

inline bool parse_available(std::string_view text, ProviderCapability& provider) {
    provider = {};
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
            if (!hex_string(value, 64)) return false;
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
                || range.begin < 0 || range.end <= range.begin) return false;
            provider.cached.push_back(range);
        } else return false;
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return !provider.id.empty() && provider.id.find_first_of("\r\n") == std::string::npos
        && !provider.gpu.empty() && provider.offered_vram_mib != 0
        && valid_ring_target(provider.ring_endpoint)
        && (provider.state.empty() || provider.state == "available" || provider.state == "reserved"
            || provider.state == "loading" || provider.state == "serving")
        && provider.runtime_abi.find_first_of("\r\n") == std::string::npos;
}

} // namespace dan::provider_owned
