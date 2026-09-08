#include "provider_ui.hpp"
#include "platform.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>

namespace dan {
namespace {
bool animated(ProviderUiStatus status)
{
    return status == ProviderUiStatus::starting || status == ProviderUiStatus::connecting
        || status == ProviderUiStatus::preparing || status == ProviderUiStatus::downloading
        || status == ProviderUiStatus::loading || status == ProviderUiStatus::reconnecting
        || status == ProviderUiStatus::waiting_gpu || status == ProviderUiStatus::recovering;
}
std::string fit(std::string text, std::size_t width)
{
    if (text.size() > width) {
        text.resize(width > 3 ? width - 3 : 0);
        text += "...";
    }
    text.append(width > text.size() ? width - text.size() : 0, ' ');
    return text;
}
std::string progress(int percent, std::size_t width)
{
    percent = std::clamp(percent, 0, 100);
    const std::size_t filled = static_cast<std::size_t>(percent) * width / 100;
    return "[" + std::string(filled, '#') + std::string(width - filled, '-') + "] "
        + std::to_string(percent) + '%';
}
std::string download_detail(const ProviderUiState& state)
{
    std::ostringstream out;
    out.setf(std::ios::fixed); out.precision(1);
    out << state.downloaded_bytes / 1000000000.0 << " / "
        << state.download_total_bytes / 1000000000.0 << " GB at "
        << state.download_bytes_per_second / 1000000.0 << " MB/s";
    if (state.download_bytes_per_second && state.download_total_bytes > state.downloaded_bytes) {
        const auto seconds = (state.download_total_bytes - state.downloaded_bytes)
            / state.download_bytes_per_second;
        out << "  /  ETA " << seconds / 60 << 'm' << seconds % 60 << 's';
    }
    return out.str();
}
const char* color(ProviderUiStatus status)
{
    if (status == ProviderUiStatus::contributing) return "\x1b[32m";
    if (status == ProviderUiStatus::action_required || status == ProviderUiStatus::error) return "\x1b[31m";
    if (status == ProviderUiStatus::available) return "\x1b[0m";
    return "\x1b[33m";
}
}

std::string_view provider_status_label(ProviderUiStatus status)
{
    switch (status) {
    case ProviderUiStatus::starting: return "Starting";
    case ProviderUiStatus::connecting: return "Connecting";
    case ProviderUiStatus::available: return "Available";
    case ProviderUiStatus::preparing: return "Preparing";
    case ProviderUiStatus::downloading: return "Downloading";
    case ProviderUiStatus::loading: return "Starting GPU worker";
    case ProviderUiStatus::contributing: return "Ready for work";
    case ProviderUiStatus::reconnecting: return "Reconnecting";
    case ProviderUiStatus::waiting_gpu: return "Waiting for GPU resources";
    case ProviderUiStatus::recovering: return "Recovering";
    case ProviderUiStatus::action_required: return "Action required";
    case ProviderUiStatus::error: return "Error";
    }
    return "Error";
}

std::string format_token_count(std::size_t value)
{
    std::string result = std::to_string(value);
    for (std::ptrdiff_t position = static_cast<std::ptrdiff_t>(result.size()) - 3;
        position > 0; position -= 3) result.insert(static_cast<std::size_t>(position), 1, ',');
    return result;
}

std::string render_provider_dashboard(const ProviderUiState& s, std::size_t width,
    bool colors, std::size_t frame)
{
    const char spinner[] = "|/-\\";
    const std::string marker = animated(s.status) ? std::string(1, spinner[frame % 4]) : "*";
    const std::string status = marker + " " + std::string(provider_status_label(s.status));
    const std::string model = s.model_name.empty() ? "Waiting for useful work"
        : s.model_name + (s.quantization.empty() ? "" : "  /  " + s.quantization);
    const std::string memory = [&] { std::ostringstream out; out.setf(std::ios::fixed); out.precision(1);
        out << s.used_vram_mib/1024.0 << " GB used now  /  " << s.offered_vram_mib/1024.0 << " GB offered to DAN"; return out.str(); }();
    if (width < 60) {
        std::ostringstream out;
        out << "DAN Provider v" << s.version << "\n" << s.gpu_name << "\n" << memory << "\n\n"
            << status << "\n" << model << "\n";
        if (!s.stage.empty()) out << s.stage << "\n";
        if (s.download_percent >= 0) {
            out << "Downloading required files " << progress(s.download_percent, 12) << "\n";
            if (s.download_total_bytes) out << download_detail(s) << "\n";
        }
        if (!s.message.empty()) out << s.message << "\n";
        if ((s.status == ProviderUiStatus::action_required || s.status == ProviderUiStatus::error)
            && !s.diagnostics.empty()) out << "Diagnostics: " << s.diagnostics << "\n";
        if (!s.cache_status.empty()) out << "Cache: " << s.cache_status << "\n";
        out << "\n" << format_token_count(s.requests_participated) << " requests  /  "
            << format_token_count(s.tokens_participated) << " tokens\n"
            << (s.network_connected ? "* DAN network connected" : "- DAN network disconnected")
            << "\n\nCtrl+C to stop contributing\n";
        return out.str();
    }
    width = std::min<std::size_t>(width, 76); const std::size_t inner = width - 4;
    std::ostringstream out; const std::string border = "+" + std::string(width-2, '-') + "+\n";
    const auto row = [&](std::string text, const char* tint = "") {
        out << "| " << (colors ? tint : "") << fit(std::move(text), inner)
            << (colors && *tint ? "\x1b[0m" : "") << " |\n";
    };
    out << border; row("DAN Provider v" + s.version
        + std::string(width > 66 ? "                               * LIVE" : "  * LIVE"), "\x1b[1m");
    out << border; row(""); row(s.gpu_name); row(memory); row(""); row(status, color(s.status)); row(model);
    if (!s.stage.empty()) row(s.stage);
    if (s.download_percent >= 0) {
        row("Downloading required files  " + progress(s.download_percent, 18));
        if (s.download_total_bytes) row(download_detail(s));
    }
    if (!s.message.empty()) row(s.message);
    if (!s.cache_status.empty()) row("Cache: " + s.cache_status);
    row("");
    if ((s.status == ProviderUiStatus::action_required || s.status == ProviderUiStatus::error)
        && !s.diagnostics.empty()) row("Diagnostics: " + s.diagnostics);
    row(format_token_count(s.requests_participated) + " requests  /  "
        + format_token_count(s.tokens_participated) + " tokens participated in"); row("");
    row(s.network_connected ? "* DAN network connected" : "- DAN network disconnected",
        s.network_connected ? "\x1b[32m" : "\x1b[33m"); row(""); out << border;
    row("Ctrl+C to stop contributing", "\x1b[90m"); out << border;
    return out.str();
}

ProviderTerminalUi::ProviderTerminalUi(ProviderUiState initial, bool enabled)
    : enabled_(enabled), interactive_(enabled && platform::interactive_stdout()), state_(std::move(initial))
{
    if (enabled_) renderer_ = std::jthread([this](std::stop_token stop){ run(stop); });
}
ProviderTerminalUi::~ProviderTerminalUi()
{
    renderer_.request_stop(); changed_.notify_all();
    if (renderer_.joinable()) renderer_.join();
}
void ProviderTerminalUi::update(const std::function<void(ProviderUiState&)>& change)
{
    if (!enabled_) return;
    { std::lock_guard lock(mutex_); change(state_); ++revision_; }
    changed_.notify_all();
}
void ProviderTerminalUi::run(std::stop_token stop)
{
    using namespace std::chrono_literals;
    std::size_t shown_revision = 0, frame = 0;
    auto last_plain = std::chrono::steady_clock::now() - 5s;
    while (!stop.stop_requested()) {
        ProviderUiState snapshot; std::size_t revision;
        {
            std::unique_lock lock(mutex_);
            changed_.wait_for(lock, 250ms);
            if (stop.stop_requested()) break;
            snapshot = state_; revision = revision_;
        }
        const bool animate = interactive_ && animated(snapshot.status);
        if (interactive_ && (revision != shown_revision || animate)) {
            platform::clear_console();
            const std::string screen = render_provider_dashboard(snapshot, platform::terminal_width(),
                platform::color_stdout(), frame++);
            std::fwrite(screen.data(), 1, screen.size(), stdout); std::fflush(stdout);
        } else if (!interactive_ && revision != shown_revision
            && std::chrono::steady_clock::now() - last_plain >= 1s) {
            std::printf("DAN Provider | %s | %s | %s tokens participated in\n",
                provider_status_label(snapshot.status).data(),
                snapshot.network_connected ? "network connected" : "network disconnected",
                format_token_count(snapshot.tokens_participated).c_str());
            last_plain = std::chrono::steady_clock::now();
            shown_revision = revision;
        }
        if (interactive_) shown_revision = revision;
    }
}
} // namespace dan
