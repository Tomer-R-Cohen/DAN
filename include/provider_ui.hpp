#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

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
};

std::string_view provider_status_label(ProviderUiStatus status);
std::string format_token_count(std::size_t value);
std::string render_provider_dashboard(const ProviderUiState& state, std::size_t width,
    bool colors, std::size_t animation_frame = 0);

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
