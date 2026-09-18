#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace dan {

enum class ProviderUiStatus {
    starting, connecting, available, preparing, downloading, loading, contributing,
    reconnecting, waiting_gpu, recovering, action_required, error
};

struct ProviderUiState {
    std::string version = "1.0.1";
    std::string gpu_name;
    std::size_t offered_vram_mib = 0;
    std::size_t used_vram_mib = 0;
    ProviderUiStatus status = ProviderUiStatus::starting;
    bool network_connected = false;
    std::string model_name;
    std::string quantization;
    std::size_t tokens_participated = 0;
    std::size_t requests_participated = 0;
    std::string stage;
    std::string cache_status;
    int download_percent = -1;
    std::size_t downloaded_bytes = 0;
    std::size_t download_total_bytes = 0;
    std::size_t download_bytes_per_second = 0;
    std::string message = "Checking system...";
    std::string diagnostics;

    // Decentralized node (network=dht). When set, the node dashboard is shown instead.
    bool dht_mode = false;
    std::string peer_id;
    std::size_t relay_addresses = 0;
    bool public_ipv6 = false;
    std::size_t peers = 0;
    std::string route_id;
    std::string layers;          // e.g. "10-17 of 24"
    std::string previous_peer;   // empty: this node is the first stage
    std::string next_peer;       // PeerID, or "client"
    std::string link_in;         // e.g. "relay", "direct quic-v1"
    std::string link_out;
    std::size_t routes_served = 0;
    std::string replica;         // this node's replica owner, one line (empty: none)
    std::vector<double> throughput;       // tokens/s samples, newest last
    std::uint64_t uptime_seconds = 0;
    std::vector<std::string> activity;    // "HH:MM:SS  text", newest last
};

std::string_view provider_status_label(ProviderUiStatus status);
std::string format_token_count(std::size_t value);
std::string render_provider_dashboard(const ProviderUiState& state, std::size_t width,
    bool colors, std::size_t animation_frame = 0);
// The decentralized node dashboard; `unicode` selects box drawing and block characters.
std::string render_node_dashboard(const ProviderUiState& state, std::size_t width,
    bool colors, bool unicode, std::size_t animation_frame = 0);
// Appends a time-stamped line to state.activity, keeping the newest few.
void add_activity(ProviderUiState& state, std::string text);
// "12D3KooWAbc...xyz9" style short form of a PeerID.
std::string short_peer(std::string_view peer);

class ProviderTerminalUi {
public:
    ProviderTerminalUi(ProviderUiState initial, bool enabled);
    ~ProviderTerminalUi();
    void update(const std::function<void(ProviderUiState&)>& change);
private:
    void run(std::stop_token stop);
    bool enabled_;
    bool interactive_;
    std::mutex mutex_;
    std::condition_variable_any changed_;
    ProviderUiState state_;
    std::size_t revision_ = 1;
    std::jthread renderer_;
};

} // namespace dan
