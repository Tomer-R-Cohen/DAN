#include "provider_ui.hpp"
#include "platform.hpp"

#include <cassert>
#include <cstdio>
#include <string_view>

int main(int argc, char** argv)
{
    using dan::ProviderUiStatus;
    assert(dan::provider_status_label(ProviderUiStatus::available) == "Available");
    assert(dan::provider_status_label(ProviderUiStatus::reconnecting) == "Reconnecting");
    assert(dan::provider_status_label(ProviderUiStatus::waiting_gpu) == "Waiting for GPU resources");
    assert(dan::provider_status_label(ProviderUiStatus::recovering) == "Recovering");
    assert(dan::provider_status_label(ProviderUiStatus::action_required) == "Action required");
    assert(dan::format_token_count(221) == "221");
    assert(dan::format_token_count(1204882) == "1,204,882");
    dan::ProviderUiState state;
    state.gpu_name="NVIDIA RTX"; state.offered_vram_mib=6656; state.used_vram_mib=768;
    state.status=ProviderUiStatus::available; state.network_connected=true;
    auto compact = dan::render_provider_dashboard(state, 45, false);
    assert(compact.find("Available") != std::string::npos && compact.find("\x1b") == std::string::npos);
    assert(compact.find("0.8 GB used now") != std::string::npos);
    assert(compact.find("6.5 GB offered") != std::string::npos);
    state.status=ProviderUiStatus::downloading; state.download_percent=73; state.model_name="Model";
    state.stage="Assigned layers 0-6"; state.requests_participated=12;
    state.downloaded_bytes=2400000000; state.download_total_bytes=3300000000;
    state.download_bytes_per_second=18600000;
    auto preparing=dan::render_provider_dashboard(state,70,false,2);
    assert(preparing.find("Downloading required files") != std::string::npos);
    assert(preparing.find("73%") != std::string::npos);
    assert(preparing.find("2.4 / 3.3 GB at 18.6 MB/s") != std::string::npos);
    assert(preparing.find("ETA") != std::string::npos);
    assert(preparing.find("Assigned layers 0-6") != std::string::npos);
    state.status=ProviderUiStatus::contributing; state.tokens_participated=12481;
    auto ready=dan::render_provider_dashboard(state,70,false);
    assert(ready.find("Ready for work") != std::string::npos);
    assert(ready.find("12,481 tokens participated in") != std::string::npos);
    assert(ready.find("12 requests") != std::string::npos);
    assert(ready.find("12,481 tokens participated in",
        ready.find("12,481 tokens participated in") + 1) == std::string::npos);
    state.status=ProviderUiStatus::action_required; state.diagnostics="provider.log";
    auto failed=dan::render_provider_dashboard(state,70,false);
    assert(failed.find("Action required") != std::string::npos);
    assert(failed.find("Diagnostics: provider.log") != std::string::npos);
    assert(dan::render_provider_dashboard(state,45,false).find("Diagnostics: provider.log")
        != std::string::npos);

    // Decentralized node dashboard.
    assert(dan::short_peer("12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C")
        == "12D3KooW..mbKW8C");
    assert(dan::short_peer("client") == "client");
    dan::ProviderUiState node;
    node.dht_mode = true; node.gpu_name = "NVIDIA GeForce RTX 4070"; node.offered_vram_mib = 10752;
    node.peer_id = "12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C";
    node.network_connected = true; node.relay_addresses = 1; node.public_ipv6 = true; node.peers = 7;
    node.status = ProviderUiStatus::contributing; node.model_name = "qwen2.5-0.5b-instruct-q4-k-m";
    node.route_id = "3f9a0c2d11aa22bb33cc44dd55ee66ff"; node.layers = "10-17 of 24";
    node.previous_peer = "12D3KooWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    node.next_peer = "client"; node.link_in = "relay"; node.link_out = "direct quic-v1";
    node.routes_served = 12; node.requests_participated = 1204; node.tokens_participated = 38912;
    node.throughput = {10, 20, 30, 32.4}; node.uptime_seconds = 8040; node.message.clear();
    for (int index = 0; index < 7; ++index) dan::add_activity(node, "event " + std::to_string(index));
    assert(node.activity.size() == 5 && node.activity.back().ends_with("event 6"));

    const auto plain_node = dan::render_node_dashboard(node, 80, false, false);
    assert(plain_node.find("DAN NODE") != std::string::npos);
    assert(plain_node.find("SERVING INFERENCE") != std::string::npos);
    assert(plain_node.find("layers 10-17 of 24") != std::string::npos);
    assert(plain_node.find("client") != std::string::npos);
    assert(plain_node.find("relay ready") != std::string::npos);
    assert(plain_node.find("38,912 tokens") != std::string::npos);
    assert(plain_node.find("32.4 tok/s") != std::string::npos);
    assert(plain_node.find("2h 14m") != std::string::npos);
    assert(plain_node.find("\x1b") == std::string::npos);
    assert(plain_node.find("\xe2") == std::string::npos);  // ASCII only
    // Every framed line is exactly 80 columns wide.
    for (std::size_t start = 0, end; (end = plain_node.find('\n', start)) != std::string::npos; start = end + 1) {
        const std::string line = plain_node.substr(start, end - start);
        if (line.starts_with('|') || line.starts_with('+')) assert(line.size() == 80);
    }
    const auto fancy = dan::render_node_dashboard(node, 120, true, true, 3);
    assert(fancy.find("\xe2\x95\xad") != std::string::npos);   // rounded corner
    assert(fancy.find("\x1b[35m") != std::string::npos);        // serving tint
    node.status = ProviderUiStatus::downloading; node.download_percent = 64;
    node.downloaded_bytes = 1200000000; node.download_total_bytes = 1900000000;
    node.download_bytes_per_second = 18600000;
    node.message = "A very long message that must be cut to fit inside the dashboard frame without breaking it";
    const auto downloading = dan::render_node_dashboard(node, 64, false, false);
    assert(downloading.find("DOWNLOADING MODEL LAYERS") != std::string::npos);
    assert(downloading.find("64%") != std::string::npos);
    assert(downloading.find("...") != std::string::npos);
    const auto small = dan::render_node_dashboard(node, 40, false, false);
    assert(small.find("Downloading") != std::string::npos && small.find("Download 64%") != std::string::npos);
    assert(dan::render_provider_dashboard(node, 80, false).find("DAN NODE") != std::string::npos);

    // `provider_ui_test --preview` shows the node dashboard in this terminal
    // (--preview-unicode forces box drawing without colors, e.g. for a file).
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (mode == "--preview" || mode == "--preview-unicode") {
        dan::platform::configure_output();
        node.status = ProviderUiStatus::contributing; node.download_percent = -1; node.message.clear();
        node.throughput = {12, 18, 22, 25, 31, 30, 33, 29, 32, 34, 33, 32.4};
        const bool vt = dan::platform::color_stdout();
        if (mode == "--preview-unicode") {
            const std::string screen = dan::render_node_dashboard(node, 80, false, true, 2);
            std::fwrite(screen.data(), 1, screen.size(), stdout);
            return 0;
        }
        const std::string screen = dan::render_node_dashboard(node, 80, vt, vt, 2);
        std::fwrite(screen.data(), 1, screen.size(), stdout);
    }
}
