#include "persistent_runtime.hpp"
#include "protocol.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <charconv>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <string_view>

namespace dan {
namespace {

bool parse_port(std::string_view text)
{
    unsigned int port = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), port);
    return !text.empty() && error == std::errc{} && end == text.data() + text.size()
        && port > 0 && port <= 65535;
}

bool private_endpoint(std::string_view endpoint)
{
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || !parse_port(endpoint.substr(colon + 1))) {
        return false;
    }
    const std::string host(endpoint.substr(0, colon));
    in_addr ipv4{};
    if (inet_pton(AF_INET, host.c_str(), &ipv4) == 1) {
        const std::uint32_t address = ntohl(ipv4.s_addr);
        return (address >> 24) == 10 || (address >> 24) == 127
            || (address >> 20) == 0xac1 || (address >> 16) == 0xc0a8;
    }
    return false;
}

bool valid_endpoints(std::string_view endpoints)
{
    if (endpoints.empty()) return false;
    while (true) {
        const std::size_t comma = endpoints.find(',');
        if (!private_endpoint(endpoints.substr(0, comma))) return false;
        if (comma == std::string_view::npos) return true;
        endpoints.remove_prefix(comma + 1);
        if (endpoints.empty()) return false;
    }
}

int connect_tcp(const std::string& host, const std::string& port,
    std::chrono::seconds timeout)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) return -1;
    int connected = -1;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        connected = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (connected == -1) continue;
        timeval value{static_cast<time_t>(timeout.count()), 0};
        setsockopt(connected, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value));
        setsockopt(connected, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value));
        if (connect(connected, address->ai_addr, address->ai_addrlen) == 0) break;
        close(connected);
        connected = -1;
    }
    freeaddrinfo(addresses);
    return connected;
}

bool http_request(const PersistentRuntimeConfig& config, std::string_view method,
    std::string_view path, std::string_view body, std::string& response)
{
    const int socket_fd = connect_tcp(config.host, config.port,
        method == "GET" ? std::chrono::seconds(1) : config.request_timeout);
    if (socket_fd == -1) return false;
    std::string request = std::string(method) + ' ' + std::string(path)
        + " HTTP/1.1\r\nHost: " + config.host + ':' + config.port
        + "\r\nConnection: close\r\n";
    if (!body.empty()) {
        request += "Content-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\n";
    }
    request += "\r\n";
    request += body;
    if (!send_all(socket_fd, request.data(), request.size())) { close(socket_fd); return false; }
    char buffer[4096];
    while (response.size() <= max_message_size) {
        const ssize_t count = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (count == -1 && errno == EINTR) continue;
        if (count <= 0) break;
        response.append(buffer, static_cast<std::size_t>(count));
    }
    close(socket_fd);
    const std::size_t header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos || response.size() > max_message_size
        || (!response.starts_with("HTTP/1.1 200 ") && !response.starts_with("HTTP/1.0 200 "))) {
        return false;
    }
    response.erase(0, header_end + 4);
    return true;
}

std::string json_escape(std::string_view value)
{
    std::string escaped;
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (byte < 0x20) {
                char encoded[7];
                std::snprintf(encoded, sizeof(encoded), "\\u%04x", byte);
                escaped += encoded;
            } else escaped += static_cast<char>(byte);
        }
    }
    return escaped;
}

bool json_content(std::string_view json, std::string& content)
{
    const std::size_t key = json.find("\"content\"");
    if (key == std::string_view::npos) return false;
    std::size_t position = json.find(':', key + 9);
    if (position == std::string_view::npos) return false;
    do { ++position; } while (position < json.size()
        && std::isspace(static_cast<unsigned char>(json[position])));
    if (position >= json.size() || json[position++] != '"') return false;
    while (position < json.size()) {
        const char byte = json[position++];
        if (byte == '"') return true;
        if (byte != '\\') { content += byte; continue; }
        if (position >= json.size()) return false;
        switch (json[position++]) {
        case '"': content += '"'; break;
        case '\\': content += '\\'; break;
        case '/': content += '/'; break;
        case 'b': content += '\b'; break;
        case 'f': content += '\f'; break;
        case 'n': content += '\n'; break;
        case 'r': content += '\r'; break;
        case 't': content += '\t'; break;
        default: return false;
        }
    }
    return false;
}

bool healthy(const PersistentRuntimeConfig& config)
{
    std::string response;
    return http_request(config, "GET", "/health", {}, response);
}

bool port_open(const PersistentRuntimeConfig& config)
{
    const int socket_fd = connect_tcp(config.host, config.port, std::chrono::seconds(1));
    if (socket_fd == -1) return false;
    close(socket_fd);
    return true;
}

bool complete(const PersistentRuntimeConfig& config, std::string_view prompt,
    std::string& response)
{
    const std::string body = "{\"prompt\":\"" + json_escape(prompt)
        + "\",\"n_predict\":256,\"stream\":false,\"cache_prompt\":false}";
    std::string json;
    return http_request(config, "POST", "/completion", body, json)
        && json_content(json, response) && !response.empty();
}

void close_other_fds(int preserved)
{
    const long limit = sysconf(_SC_OPEN_MAX);
    for (int fd = 3; fd < limit; ++fd) if (fd != preserved) close(fd);
}

} // namespace

std::string_view runtime_state_name(RuntimeState state)
{
    switch (state) {
    case RuntimeState::stopped: return "STOPPED";
    case RuntimeState::starting: return "STARTING";
    case RuntimeState::ready: return "READY";
    case RuntimeState::error: return "ERROR";
    }
    return "ERROR";
}

PersistentRuntime::~PersistentRuntime() { cleanup(); }

void PersistentRuntime::terminate(pid_t& child)
{
    if (child == -1) return;
    kill(child, SIGTERM);
    for (int attempt = 0; attempt < 20; ++attempt) {
        if (waitpid(child, nullptr, WNOHANG) == child) { child = -1; return; }
        usleep(50000);
    }
    kill(child, SIGKILL);
    while (waitpid(child, nullptr, 0) == -1 && errno == EINTR) { }
    child = -1;
}

void PersistentRuntime::cleanup()
{
    if (result_fd_ != -1) { close(result_fd_); result_fd_ = -1; }
    terminate(request_pid_);
    terminate(pid_);
}

bool PersistentRuntime::start(const PersistentRuntimeConfig& config,
    std::string endpoints, std::string tensor_split, std::string& error)
{
    if (state_ == RuntimeState::starting || state_ == RuntimeState::ready) return true;
    if ((config.host != "127.0.0.1" && config.host != "::1")
        || !parse_port(config.port) || !valid_endpoints(endpoints)) {
        error = last_error_ = "invalid managed runtime or private RPC endpoint";
        state_ = RuntimeState::error;
        return false;
    }
    if (port_open(config)) {
        error = last_error_ = "managed runtime endpoint is already in use";
        state_ = RuntimeState::error;
        return false;
    }
    cleanup();
    config_ = config;
    endpoints_ = std::move(endpoints);
    tensor_split_ = std::move(tensor_split);
    last_error_.clear();
    pid_ = fork();
    if (pid_ == -1) {
        error = last_error_ = "could not fork persistent runtime";
        state_ = RuntimeState::error;
        return false;
    }
    if (pid_ == 0) {
        const pid_t parent = getppid();
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1 || getppid() != parent) _exit(127);
        std::string devices;
        std::size_t count = 1;
        for (const char byte : endpoints_) if (byte == ',') ++count;
        for (std::size_t index = 0; index < count; ++index) {
            if (!devices.empty()) devices += ',';
            devices += "RPC" + std::to_string(index);
        }
        std::vector<std::string> arguments{config_.executable, "--model", config_.model,
            "--rpc", endpoints_, "--device", devices, "--n-gpu-layers", "99",
            "--ctx-size", std::to_string(config_.context_size), "--split-mode", "layer",
            "--tensor-split", tensor_split_, "--host", config_.host, "--port", config_.port};
        arguments.insert(arguments.end(), config_.arguments.begin(), config_.arguments.end());
        std::vector<char*> argv;
        for (auto& argument : arguments) argv.push_back(argument.data());
        argv.push_back(nullptr);
        close_other_fds(-1);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    state_ = RuntimeState::starting;
    startup_deadline_ = std::chrono::steady_clock::now() + config_.startup_timeout;
    std::printf("Managed runtime STARTING pid=%d endpoints=%s\n", pid_, endpoints_.c_str());
    return true;
}

void PersistentRuntime::update()
{
    if (pid_ == -1) return;
    int status = 0;
    if (waitpid(pid_, &status, WNOHANG) == pid_) {
        pid_ = -1;
        mark_error("persistent runtime exited unexpectedly");
        return;
    }
    if (state_ != RuntimeState::starting) return;
    if (healthy(config_)) {
        state_ = RuntimeState::ready;
        std::printf("Managed runtime READY pid=%d\n", pid_);
    } else if (std::chrono::steady_clock::now() >= startup_deadline_) {
        mark_error("persistent runtime startup timed out");
    }
}

void PersistentRuntime::stop()
{
    cleanup();
    state_ = RuntimeState::stopped;
}

void PersistentRuntime::mark_error(std::string error)
{
    cleanup();
    last_error_ = std::move(error);
    state_ = RuntimeState::error;
    std::fprintf(stderr, "Managed runtime ERROR: %s\n", last_error_.c_str());
}

bool PersistentRuntime::submit(std::uint64_t request_id, const std::string& prompt,
    std::string& error)
{
    if (state_ != RuntimeState::ready || busy()) {
        error = "persistent runtime is not available";
        return false;
    }
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == -1) {
        error = "could not create managed request channel";
        return false;
    }
    request_pid_ = fork();
    if (request_pid_ == -1) {
        close(sockets[0]); close(sockets[1]);
        error = "could not fork managed HTTP request";
        return false;
    }
    if (request_pid_ == 0) {
        close(sockets[0]);
        const pid_t parent = getppid();
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1 || getppid() != parent) _exit(127);
        close_other_fds(sockets[1]);
        std::string response;
        const bool ok = complete(config_, prompt, response);
        send_message(sockets[1], (ok ? "RESPONSE\n" : "ERROR\n")
            + std::to_string(request_id) + '\n'
            + (ok ? response : "invalid or failed llama-server response"));
        close(sockets[1]);
        _exit(ok ? 0 : 1);
    }
    close(sockets[1]);
    result_fd_ = sockets[0];
    request_id_ = request_id;
    return true;
}

bool PersistentRuntime::collect(std::uint64_t& request_id, std::string& response,
    std::string& error)
{
    if (!busy() || result_fd_ == -1) { error = "no managed request is active"; return false; }
    std::string message;
    const bool received = receive_message(result_fd_, message);
    close(result_fd_);
    result_fd_ = -1;
    int status = 0;
    while (waitpid(request_pid_, &status, 0) == -1 && errno == EINTR) { }
    request_pid_ = -1;
    const std::size_t first = message.find('\n');
    const std::size_t second = first == std::string::npos ? first : message.find('\n', first + 1);
    std::uint64_t reported = 0;
    if (first != std::string::npos && second != std::string::npos) {
        const std::string_view id(message.data() + first + 1, second - first - 1);
        const auto parsed = std::from_chars(id.data(), id.data() + id.size(), reported);
        if (parsed.ec != std::errc{} || parsed.ptr != id.data() + id.size()) reported = 0;
    }
    if (!received || !WIFEXITED(status) || WEXITSTATUS(status) != 0
        || !message.starts_with("RESPONSE\n") || reported != request_id_) {
        error = second == std::string::npos ? "malformed persistent runtime response"
            : message.substr(second + 1);
        mark_error(error);
        return false;
    }
    request_id = reported;
    response = message.substr(second + 1);
    ++requests_served_;
    return true;
}

} // namespace dan
