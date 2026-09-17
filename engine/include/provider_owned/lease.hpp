#pragma once

// Decentralized placement: a client claims a worker for one route with a short lease.
//
//   client                          worker (one lease at a time; first reserve wins)
//   reserve {stage request}   ->    checks model, limits and memory fit; AVAILABLE -> RESERVED
//   assign_stage {same}       ->    RESERVED -> LOADING (no expiry while loading) -> SERVING
//                             <-    stage_ready
//   release_route {route_id}  ->    back to AVAILABLE (before assignment)
//   connection closes               back to AVAILABLE (weights stay cached)
//
// A reservation that is not followed by assign_stage within lease_ms expires.

#include "provider_owned/formation.hpp"

#include <chrono>
#include <charconv>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace dan::provider_owned {

inline constexpr std::uint32_t max_lease_ms = 60000;

// The stage a client asks a worker to serve. The model is named by SHA-256 only; the
// worker resolves it in its own catalog and never downloads from a client-chosen URL.
struct StageRequest {
    std::string route_id;      // 32 hex digits, random per route
    std::string model_sha256;  // 64 hex digits
    int begin = 0;
    int end = 0;
    std::uint32_t context = 0;
    std::uint32_t sessions = 0;
    std::uint32_t lease_ms = 0;  // reserve only
    // Speculative decoding: a small model from this worker's own catalog that the first
    // stage runs to propose the next few tokens. Empty = plain decoding.
    std::string draft_sha256;

    bool same_stage(const StageRequest& other) const {
        return route_id == other.route_id && model_sha256 == other.model_sha256
            && begin == other.begin && end == other.end && context == other.context
            && sessions == other.sessions && draft_sha256 == other.draft_sha256;
    }
};

inline std::string stage_request_message(const StageRequest& request) {
    std::string text = "route_id=" + request.route_id + "\nmodel_sha256=" + request.model_sha256
        + "\nbegin=" + std::to_string(request.begin) + "\nend=" + std::to_string(request.end)
        + "\ncontext=" + std::to_string(request.context)
        + "\nsessions=" + std::to_string(request.sessions);
    if (request.lease_ms != 0) text += "\nlease_ms=" + std::to_string(request.lease_ms);
    if (!request.draft_sha256.empty()) text += "\ndraft_sha256=" + request.draft_sha256;
    return text;
}

inline bool parse_stage_request(std::string_view text, StageRequest& request) {
    request = {};
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
        if (key == "route_id") request.route_id = value;
        else if (key == "model_sha256") request.model_sha256 = value;
        else if (key == "begin") { if (!number(value, request.begin)) return false; }
        else if (key == "end") { if (!number(value, request.end)) return false; }
        else if (key == "context") { if (!number(value, request.context)) return false; }
        else if (key == "sessions") { if (!number(value, request.sessions)) return false; }
        else if (key == "lease_ms") { if (!number(value, request.lease_ms)) return false; }
        else if (key == "draft_sha256") request.draft_sha256 = value;
        else return false;
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return hex_string(request.route_id, 32) && hex_string(request.model_sha256, 64)
        && request.begin >= 0 && request.end > request.begin
        && request.context != 0 && request.sessions != 0 && request.lease_ms <= max_lease_ms
        && (request.draft_sha256.empty() || hex_string(request.draft_sha256, 64));
}

// The worker's single lease. Thread-safe; every transition names the route it applies to.
class WorkerLease {
public:
    using Clock = std::chrono::steady_clock;
    enum class State { available, reserved, loading, serving };

    // First reservation wins; false if any route already holds the lease.
    bool reserve(const StageRequest& request, Clock::time_point now) {
        std::lock_guard lock(mutex_);
        expire_locked(now);
        if (state_ != State::available || request.lease_ms == 0) return false;
        state_ = State::reserved;
        request_ = request;
        deadline_ = now + std::chrono::milliseconds(request.lease_ms);
        return true;
    }

    // RESERVED -> LOADING for the same stage, if the reservation has not expired.
    bool begin_loading(const StageRequest& assignment, Clock::time_point now) {
        std::lock_guard lock(mutex_);
        expire_locked(now);
        if (state_ != State::reserved || !request_.same_stage(assignment)) return false;
        state_ = State::loading;
        return true;
    }

    bool serving(const std::string& route_id) {
        std::lock_guard lock(mutex_);
        if (state_ != State::loading || request_.route_id != route_id) return false;
        state_ = State::serving;
        return true;
    }

    // Gives the lease back if `route_id` holds it; returns whether it did.
    bool release(const std::string& route_id) {
        std::lock_guard lock(mutex_);
        if (state_ == State::available || request_.route_id != route_id) return false;
        state_ = State::available;
        request_ = {};
        return true;
    }

    State state(Clock::time_point now) {
        std::lock_guard lock(mutex_);
        expire_locked(now);
        return state_;
    }

    std::optional<StageRequest> current(Clock::time_point now) {
        std::lock_guard lock(mutex_);
        expire_locked(now);
        if (state_ == State::available) return std::nullopt;
        return request_;
    }

    static const char* name(State state) {
        switch (state) {
        case State::available: return "available";
        case State::reserved: return "reserved";
        case State::loading: return "loading";
        case State::serving: return "serving";
        }
        return "available";
    }

private:
    void expire_locked(Clock::time_point now) {
        if (state_ == State::reserved && now >= deadline_) {
            state_ = State::available;
            request_ = {};
        }
    }

    std::mutex mutex_;
    State state_ = State::available;
    StageRequest request_;
    Clock::time_point deadline_{};
};

} // namespace dan::provider_owned
