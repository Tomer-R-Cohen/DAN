#pragma once

#include "platform.hpp"

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace dan {

struct AdminProvider {
    std::string name, id, gpu, role, state;
    std::size_t offered_vram_mib = 0, used_vram_mib = 0, tokens_participated = 0;
    bool online = false;
    long long last_seen_seconds = 0;
    std::size_t downloaded_bytes = 0, download_total_bytes = 0;
    std::size_t download_bytes_per_second = 0;
};

struct AdminTotals {
    std::size_t online = 0, assigned = 0, spare = 0, offline = 0;
    std::size_t offered_vram_mib = 0, used_vram_mib = 0;
};

struct AdminEvent { std::string time, text, level; };

struct AdminSnapshot {
    std::string network_status, provider_status, replica_status, runtime_status, setup_address;
    std::string model_name, model_id, model_version, quantization, chat_url;
    std::size_t model_bytes = 0, replicas_total = 0, replicas_ready = 0;
    std::size_t generated_tokens = 0, completed_requests = 0;
    double latest_tokens_per_second = 0;
    std::vector<AdminProvider> providers;
    std::vector<AdminEvent> events;
};

AdminTotals calculate_admin_totals(const std::vector<AdminProvider>& providers);
std::string admin_snapshot_json(const AdminSnapshot& snapshot);
void render_admin_terminal(const AdminSnapshot& snapshot);
std::size_t parse_generated_tokens(std::string_view prometheus_metrics);
std::size_t parse_completed_requests(std::string_view prometheus_metrics);
void diagnostic_log(std::string_view component, std::string_view text);

class EventTimeline {
public:
    explicit EventTimeline(std::size_t limit = 200) : limit_(limit) { }
    void add(std::string text, std::string level = "info");
    std::vector<AdminEvent> entries() const;
private:
    std::size_t limit_;
    std::deque<AdminEvent> events_;
};

class AdminDashboard {
public:
    ~AdminDashboard();
    bool start(std::string_view port, std::string& error);
    void update(std::string json);
private:
    void serve(std::stop_token stop);
    platform::Socket listener_ = platform::invalid_socket;
    std::jthread thread_;
    std::mutex mutex_;
    std::string json_ = "{}";
};

std::string http_get(std::string_view host, std::string_view port, std::string_view path);

} // namespace dan
