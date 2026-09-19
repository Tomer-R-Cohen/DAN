// Dashboard for a decentralized DAN node (dan-provider network=dht).
#include "provider_ui.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

namespace dan {
namespace {

// UTF-8 byte escapes keep the source ASCII for every compiler.
constexpr const char* unicode_spinner[] = {"\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9",
    "\xe2\xa0\xb8", "\xe2\xa0\xbc", "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7",
    "\xe2\xa0\x87", "\xe2\xa0\x8f"};
constexpr const char* ascii_spinner[] = {"|", "/", "-", "\\"};
constexpr const char* unicode_spark[] = {"\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83",
    "\xe2\x96\x84", "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"};
constexpr const char* ascii_spark[] = {"_", "_", ".", ".", "-", "-", "=", "#"};

struct Glyphs {
    const char* h, *v, *top_left, *top_right, *bottom_left, *bottom_right, *left_tee, *right_tee;
    const char* on, *off, *mark, *arrow, *play, *full, *empty, *ellipsis, *check, *dot;
    const char* const* spinner;
    std::size_t spinner_count;
    const char* const* spark;
};

const Glyphs unicode_glyphs{"\xe2\x94\x80", "\xe2\x94\x82", "\xe2\x95\xad", "\xe2\x95\xae",
    "\xe2\x95\xb0", "\xe2\x95\xaf", "\xe2\x94\x9c", "\xe2\x94\xa4",
    "\xe2\x97\x8f", "\xe2\x97\x8b", "\xe2\x97\x86", "\xe2\x94\x80\xe2\x94\x80\xe2\x96\xb6",
    "\xe2\x96\xb6", "\xe2\x96\x88", "\xe2\x96\x91", "\xe2\x80\xa6", "\xe2\x9c\x93", "\xc2\xb7",
    unicode_spinner, std::size(unicode_spinner), unicode_spark};
const Glyphs ascii_glyphs{"-", "|", "+", "+", "+", "+", "+", "+",
    "*", "o", "#", "-->", ">", "#", "-", "...", "yes", "|",
    ascii_spinner, std::size(ascii_spinner), ascii_spark};

struct Tints {
    const char* dim, *bold, *green, *yellow, *red, *cyan, *magenta, *reset;
};
const Tints colored{"\x1b[90m", "\x1b[1m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[36m",
    "\x1b[35m", "\x1b[0m"};
const Tints plain{"", "", "", "", "", "", "", ""};

struct Segment {
    std::string text;
    const char* tint = "";
};

// Display width: code points, not bytes (every glyph used here is one column wide).
std::size_t display_width(std::string_view text)
{
    return static_cast<std::size_t>(std::count_if(text.begin(), text.end(),
        [](char byte) { return (static_cast<unsigned char>(byte) & 0xC0) != 0x80; }));
}

std::string first_columns(std::string_view text, std::size_t columns)
{
    std::size_t index = 0, seen = 0;
    while (index < text.size()) {
        if ((static_cast<unsigned char>(text[index]) & 0xC0) != 0x80) {
            if (seen == columns) break;
            ++seen;
        }
        ++index;
    }
    return std::string(text.substr(0, index));
}

class Screen {
public:
    Screen(std::size_t width, const Glyphs& glyphs, const Tints& tints)
        : width_(width), glyphs_(glyphs), tints_(tints) {}

    void rule(const char* left, const char* right)
    {
        out_ << tints_.dim << left;
        for (std::size_t column = 0; column + 2 < width_; ++column) out_ << glyphs_.h;
        out_ << right << tints_.reset << '\n';
    }
    void top() { rule(glyphs_.top_left, glyphs_.top_right); }
    void divider() { rule(glyphs_.left_tee, glyphs_.right_tee); }
    void bottom() { rule(glyphs_.bottom_left, glyphs_.bottom_right); }
    void blank() { row({}); }

    void row(const std::vector<Segment>& segments)
    {
        std::size_t remaining = width_ - 4;
        out_ << tints_.dim << glyphs_.v << tints_.reset << ' ';
        for (const Segment& segment : segments) {
            if (remaining == 0) break;
            std::string text = segment.text;
            if (display_width(text) > remaining) {
                const std::size_t keep = remaining - std::min(remaining, display_width(glyphs_.ellipsis));
                text = first_columns(text, keep) + glyphs_.ellipsis;
            }
            remaining -= std::min(remaining, display_width(text));
            out_ << segment.tint << text << (*segment.tint ? tints_.reset : "");
        }
        out_ << std::string(remaining, ' ') << ' ' << tints_.dim << glyphs_.v << tints_.reset << '\n';
    }

    void line(std::string_view text) { out_ << text << '\n'; }
    std::string str() const { return out_.str(); }

private:
    std::size_t width_;
    const Glyphs& glyphs_;
    const Tints& tints_;
    std::ostringstream out_;
};

std::string gigabytes(std::size_t mib)
{
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << mib / 1024.0 << " GB";
    return out.str();
}

std::string uptime_text(std::uint64_t seconds)
{
    std::ostringstream out;
    if (seconds >= 3600) out << seconds / 3600 << "h " << seconds / 60 % 60 << "m";
    else out << seconds / 60 << "m " << (seconds % 60 < 10 ? "0" : "") << seconds % 60 << "s";
    return out.str();
}

std::string bar(int percent, std::size_t width, const Glyphs& glyphs)
{
    percent = std::clamp(percent, 0, 100);
    const std::size_t filled = static_cast<std::size_t>(percent) * width / 100;
    std::string result;
    for (std::size_t index = 0; index < width; ++index) result += index < filled ? glyphs.full : glyphs.empty;
    return result + " " + std::to_string(percent) + "%";
}

std::string download_text(const ProviderUiState& s, const Glyphs& glyphs)
{
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << s.downloaded_bytes / 1e9 << " / " << s.download_total_bytes / 1e9 << " GB "
        << glyphs.dot << ' ' << s.download_bytes_per_second / 1e6 << " MB/s";
    if (s.download_bytes_per_second && s.download_total_bytes > s.downloaded_bytes) {
        const auto seconds = (s.download_total_bytes - s.downloaded_bytes) / s.download_bytes_per_second;
        out << ' ' << glyphs.dot << " ETA " << seconds / 60 << 'm' << seconds % 60 << 's';
    }
    return out.str();
}

std::string sparkline(const std::vector<double>& samples, std::size_t width, const Glyphs& glyphs)
{
    const std::size_t count = std::min(samples.size(), width);
    const auto first = samples.end() - static_cast<std::ptrdiff_t>(count);
    const double peak = count ? *std::max_element(first, samples.end()) : 0.0;
    std::string result;
    for (std::size_t index = count; index < width; ++index) result += glyphs.spark[0];
    for (auto sample = first; sample != samples.end(); ++sample) {
        const int level = peak > 0 ? static_cast<int>(*sample / peak * 7.0 + 0.5) : 0;
        result += glyphs.spark[std::clamp(level, 0, 7)];
    }
    return result;
}

struct Headline {
    std::string text;
    const char* tint;
    bool animated;
};

Headline headline(const ProviderUiState& s, const Tints& t)
{
    switch (s.status) {
    case ProviderUiStatus::starting:
    case ProviderUiStatus::connecting: return {"JOINING THE DAN NETWORK", t.yellow, true};
    case ProviderUiStatus::reconnecting: return {"RECONNECTING", t.yellow, true};
    case ProviderUiStatus::available: return {"READY " "- waiting for work", t.green, false};
    case ProviderUiStatus::preparing: return {"RESERVED FOR A ROUTE", t.cyan, true};
    case ProviderUiStatus::downloading: return {"DOWNLOADING MODEL LAYERS", t.cyan, true};
    case ProviderUiStatus::loading: return {"LOADING LAYERS ONTO THE GPU", t.cyan, true};
    case ProviderUiStatus::contributing: return {"SERVING INFERENCE", t.magenta, false};
    case ProviderUiStatus::waiting_gpu: return {"WAITING FOR GPU MEMORY", t.yellow, true};
    case ProviderUiStatus::recovering: return {"RECOVERING", t.yellow, true};
    case ProviderUiStatus::action_required: return {"ACTION REQUIRED", t.red, false};
    case ProviderUiStatus::error: return {"ERROR", t.red, false};
    }
    return {"ERROR", t.red, false};
}

// Falling green "digital rain" (The Matrix) beside the dashboard when the terminal is wide
// enough, with Morpheus's line in the middle.
constexpr std::size_t rain_width = 36;
constexpr std::string_view rain_quote = "Welcome to the real world.";
constexpr std::string_view rain_glyphs = "0123456789ZXCVBNMASDFGHJKLQWERTYUIOP:.=*+-<>|";

std::uint32_t mix(std::uint32_t a, std::uint32_t b)
{
    std::uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u + (a << 6) + (a >> 2));
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

// One row of the rain panel. Each column has its own speed, trail length and start, and its
// characters change every few frames; the quote row keeps a clear gap around the words.
std::string rain_row(std::size_t row, std::size_t rows, std::size_t frame, const Tints& t,
    bool colors)
{
    constexpr const char* white = "\x1b[97m";
    constexpr const char* faint = "\x1b[2;32m";
    const std::size_t quote_row = rows / 2;
    const std::size_t quote_start = (rain_width - rain_quote.size()) / 2;
    std::string out;
    const char* current = "";
    const auto paint = [&](const char* tint, char glyph) {
        if (colors && tint != current) {
            out += t.reset;
            out += tint;
            current = tint;
        }
        out += glyph;
    };
    for (std::size_t column = 0; column < rain_width; ++column) {
        if (row == quote_row && column + 1 >= quote_start
            && column <= quote_start + rain_quote.size()) {
            const bool word = column >= quote_start && column < quote_start + rain_quote.size();
            paint(white, word ? rain_quote[column - quote_start] : ' ');
            continue;
        }
        if (!colors) { out += ' '; continue; }
        const auto c = static_cast<std::uint32_t>(column);
        const std::uint32_t speed = 1 + mix(c, 1) % 2;
        const std::uint32_t trail = 5 + mix(c, 2) % 9;
        const std::uint32_t period = static_cast<std::uint32_t>(rows) + trail + mix(c, 3) % 12;
        const std::uint32_t head = (static_cast<std::uint32_t>(frame) * speed + mix(c, 4)) % period;
        const std::int64_t behind = static_cast<std::int64_t>(head) - static_cast<std::int64_t>(row);
        if (behind < 0 || behind >= trail) { paint("", ' '); continue; }
        const std::uint32_t pick = mix(c * 131 + static_cast<std::uint32_t>(row),
            (static_cast<std::uint32_t>(frame) + mix(c, 5) % 7) / 4);
        paint(behind == 0 ? white : behind < 3 ? t.green : faint, rain_glyphs[pick % rain_glyphs.size()]);
    }
    if (colors) out += t.reset;
    return out;
}
std::string label(std::string text)
{
    text.resize(std::max<std::size_t>(text.size(), 10), ' ');
    return text;
}

std::string peer_or_client(const std::string& peer)
{
    return peer.empty() || peer == "client" ? "client" : short_peer(peer);
}

std::string compact(const ProviderUiState& s)
{
    std::ostringstream out;
    out << "DAN node v" << s.version << "\n" << s.gpu_name << " (" << gigabytes(s.offered_vram_mib)
        << " offered)\n";
    if (!s.peer_id.empty()) out << "Node " << short_peer(s.peer_id) << "\n";
    out << (s.network_connected ? "Online" : "Offline") << ", relay "
        << (s.relay_addresses ? "ready" : "not yet") << ", IPv6 " << (s.public_ipv6 ? "yes" : "no")
        << ", " << s.peers << " peers\n\n" << provider_status_label(s.status) << "\n";
    if (!s.route_id.empty()) out << s.model_name << " layers " << s.layers << "\n";
    if (!s.replica.empty()) out << "Replica: " << s.replica << "\n";
    if (s.download_percent >= 0) out << "Download " << s.download_percent << "%\n";
    if (!s.message.empty()) out << s.message << "\n";
    out << "\n" << s.routes_served << " routes, " << format_token_count(s.requests_participated)
        << " requests, " << format_token_count(s.tokens_participated) << " tokens\n";
    for (const std::string& entry : s.activity) out << entry << "\n";
    out << "\nCtrl+C to stop\n";
    return out.str();
}

} // namespace

std::string short_peer(std::string_view peer)
{
    if (peer.size() <= 18) return std::string(peer);
    return std::string(peer.substr(0, 8)) + ".." + std::string(peer.substr(peer.size() - 6));
}

void add_activity(ProviderUiState& state, std::string text)
{
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[16];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &local);
    state.activity.push_back(std::string(stamp) + "  " + std::move(text));
    constexpr std::size_t kept = 5;
    if (state.activity.size() > kept) {
        state.activity.erase(state.activity.begin(),
            state.activity.end() - static_cast<std::ptrdiff_t>(kept));
    }
}

std::string render_node_dashboard(const ProviderUiState& s, std::size_t width, bool colors,
    bool unicode, std::size_t frame)
{
    if (width < 60) return compact(s);
    // A line exactly as wide as the window wraps on Windows consoles: keep a margin.
    const bool rain = width >= 80 + 3 + rain_width + 2;
    width = std::min<std::size_t>(width, 80);
    const Glyphs& g = unicode ? unicode_glyphs : ascii_glyphs;
    const Tints& t = colors ? colored : plain;
    Screen screen(width, g, t);
    const auto key = [&](const char* name) { return Segment{label(name), t.dim}; };
    const std::string sep = std::string(" ") + g.dot + " ";

    // Header.
    screen.top();
    const std::string online = std::string(s.network_connected ? g.on : g.off)
        + (s.network_connected ? " ONLINE" : " OFFLINE");
    const std::string title = std::string(g.mark) + " DAN NODE";
    const std::string version = "  v" + s.version + "  decentralized";
    const std::size_t gap = width - 4 - std::min(width - 4,
        display_width(title) + display_width(version) + display_width(online));
    screen.row({{title, t.bold}, {version, t.dim}, {std::string(gap, ' ')},
        {online, s.network_connected ? t.green : t.yellow}});
    screen.divider();

    // Machine and network.
    screen.row({key("GPU"), {s.gpu_name}, {sep + gigabytes(s.offered_vram_mib) + " offered", t.dim},
        {s.used_vram_mib ? sep + gigabytes(s.used_vram_mib) + " in use" : "", t.dim}});
    screen.row({key("NODE"), {s.peer_id.empty() ? "starting..." : short_peer(s.peer_id), t.cyan}});
    screen.row({key("NETWORK"),
        {std::string(s.relay_addresses ? g.on : g.off) + (s.relay_addresses ? " relay ready" : " no relay yet"),
            s.relay_addresses ? t.green : t.yellow},
        {sep + "IPv6 "}, {s.public_ipv6 ? g.check : "-", s.public_ipv6 ? t.green : t.dim},
        {sep + std::to_string(s.peers) + (s.peers == 1 ? " peer" : " peers")}});
    screen.divider();

    // Current work.
    const Headline head = headline(s, t);
    const std::string icon = head.animated ? g.spinner[frame % g.spinner_count]
        : (s.status == ProviderUiStatus::contributing ? g.play : g.on);
    screen.row({key("STATUS"), {icon + std::string(" ") + head.text, head.tint}});
    if (!s.route_id.empty()) {
        screen.row({key("MODEL"), {s.model_name}, {sep + "layers " + s.layers, t.dim}});
        screen.row({key("ROUTE"), {peer_or_client(s.previous_peer), t.dim},
            {std::string(" ") + g.arrow + " "}, {"this node", t.bold},
            {std::string(" ") + g.arrow + " "}, {peer_or_client(s.next_peer), t.dim}});
        if (!s.link_in.empty() || !s.link_out.empty()) {
            screen.row({key("LINKS"), {"in " + (s.link_in.empty() ? std::string("-") : s.link_in)},
                {sep + "out " + (s.link_out.empty() ? std::string("-") : s.link_out)}});
        }
        screen.row({key(""), {"route " + s.route_id.substr(0, 12), t.dim}});
    }
    if (!s.replica.empty()) screen.row({key("REPLICA"), {s.replica, t.cyan}});
    if (s.download_percent >= 0) {
        screen.row({key("DOWNLOAD"), {bar(s.download_percent, 20, g), t.cyan}});
        if (s.download_total_bytes) screen.row({key(""), {download_text(s, g), t.dim}});
    }
    if (!s.message.empty()) screen.row({key(""), {s.message, t.dim}});
    if ((s.status == ProviderUiStatus::action_required || s.status == ProviderUiStatus::error)
        && !s.diagnostics.empty()) {
        screen.row({key("LOGS"), {s.diagnostics, t.red}});
    }
    screen.divider();

    // Totals.
    std::ostringstream rate;
    rate.setf(std::ios::fixed);
    rate.precision(1);
    rate << (s.throughput.empty() ? 0.0 : s.throughput.back()) << " tok/s";
    screen.row({key("SPEED"), {sparkline(s.throughput, 24, g), t.magenta}, {"  " + rate.str()}});
    screen.row({key("SERVED"), {std::to_string(s.routes_served) + (s.routes_served == 1 ? " route" : " routes")},
        {sep + format_token_count(s.requests_participated) + " requests"},
        {sep + format_token_count(s.tokens_participated) + " tokens"}});
    screen.row({key("UPTIME"), {uptime_text(s.uptime_seconds)}});
    screen.divider();

    // Recent events.
    screen.row({{"RECENT", t.dim}});
    if (s.activity.empty()) screen.row({{"  waiting for the first route", t.dim}});
    for (const std::string& entry : s.activity) screen.row({{"  " + entry}});
    screen.bottom();
    screen.line(std::string(t.dim) + "  Ctrl+C to stop" + (s.diagnostics.empty() ? ""
        : sep + "logs " + s.diagnostics) + t.reset);
    if (!rain) return screen.str();
    // Beside the box, from its second line down; the box itself is unchanged.
    std::vector<std::string> lines;
    std::istringstream text(screen.str());
    for (std::string line; std::getline(text, line);) lines.push_back(std::move(line));
    const std::size_t rows = lines.size() - 1;
    std::ostringstream out;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        out << lines[index];
        if (index >= 1) out << "   " << rain_row(index - 1, rows, frame, t, colors);
        out << '\n';
    }
    return out.str();
}

} // namespace dan
