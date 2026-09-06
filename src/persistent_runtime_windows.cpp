#include "persistent_runtime.hpp"
#include "distributed_runtime.hpp"

#include <charconv>
#include <cstdio>
#include <string>

namespace dan {
namespace {
bool valid_port(std::string_view text)
{
    unsigned int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size()
        && value > 0 && value <= 65535;
}

bool http_healthy(std::string_view host, std::string_view port)
{
    const platform::Socket socket = platform::connect_tcp(host, port);
    if (socket == platform::invalid_socket) return false;
    const std::string request = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    const int sent = ::send(socket, request.data(), static_cast<int>(request.size()), 0);
    bool failed = false;
    if (sent != static_cast<int>(request.size()) || !platform::wait_readable(socket, 1000, failed)) {
        platform::close_socket(socket);
        return false;
    }
    char response[1024]{};
    const int received = ::recv(socket, response, sizeof(response) - 1, 0);
    platform::close_socket(socket);
    return !failed && received > 0 && std::string_view(response, received).starts_with("HTTP/1.1 200");
}
}

std::string_view runtime_state_name(RuntimeState state)
{
    switch (state) {
    case RuntimeState::starting: return "STARTING";
    case RuntimeState::ready: return "READY";
    case RuntimeState::error: return "ERROR";
    default: return "STOPPED";
    }
}

PersistentRuntime::~PersistentRuntime() { cleanup(); }

bool PersistentRuntime::start(const PersistentRuntimeConfig& config, std::string endpoints,
    std::string tensor_split, std::string& error)
{
    if (state_ == RuntimeState::starting || state_ == RuntimeState::ready) return true;
    if ((config.host != "127.0.0.1" && config.host != "::1")
        || !valid_port(config.port) || split_rpc_endpoints(endpoints).empty()) {
        error = "invalid managed runtime or private RPC endpoint";
        return false;
    }
    cleanup();
    config_ = config;
    endpoints_ = std::move(endpoints);
    tensor_split_ = std::move(tensor_split);
    std::string devices;
    const auto endpoint_list = split_rpc_endpoints(endpoints_);
    for (std::size_t index = 0; index < endpoint_list.size(); ++index) {
        if (!devices.empty()) devices += ',';
        devices += "RPC" + std::to_string(index);
    }
    std::vector<std::string> arguments{config_.executable, "--model", config_.model,
        "--rpc", endpoints_, "--device", devices, "--n-gpu-layers", "99",
        "--ctx-size", std::to_string(config_.context_size), "--split-mode", "layer",
        "--tensor-split", tensor_split_, "--host", config_.host, "--port", config_.port};
    arguments.insert(arguments.end(), config_.arguments.begin(), config_.arguments.end());
    process_ = std::make_unique<platform::Process>();
    if (!process_->start(arguments, error, false, true)) { process_.reset(); return false; }
    pid_ = static_cast<pid_t>(process_->pid());
    state_ = RuntimeState::starting;
    startup_deadline_ = std::chrono::steady_clock::now() + config_.startup_timeout;
    std::printf("\nStarting the DAN chat interface...\n"
        "The first start can take a minute while the model loads.\n");
    return true;
}

void PersistentRuntime::update()
{
    if (!process_) return;
    if (!process_->running()) { mark_error("persistent runtime exited unexpectedly"); return; }
    if (state_ != RuntimeState::starting) return;
    if (http_healthy(config_.host, config_.port)) {
        state_ = RuntimeState::ready;
        std::printf("\nDAN is ready.\nChat: http://127.0.0.1:%s\n"
            "Opening your browser now. Keep this window and the provider open.\n",
            config_.port.c_str());
        std::string error;
        platform::open_url("http://127.0.0.1:" + config_.port, error);
    } else if (std::chrono::steady_clock::now() >= startup_deadline_) {
        mark_error("persistent runtime startup timed out");
    }
}

void PersistentRuntime::terminate(pid_t& child)
{
    if (child == -1) return;
    if (process_) process_->stop();
    child = -1;
}
void PersistentRuntime::cleanup()
{
    if (process_) { process_->stop(); process_.reset(); }
    pid_ = -1;
    request_pid_ = -1;
    result_fd_ = -1;
}
void PersistentRuntime::stop() { cleanup(); state_ = RuntimeState::stopped; }
void PersistentRuntime::mark_error(std::string error)
{
    cleanup();
    last_error_ = std::move(error);
    state_ = RuntimeState::error;
    std::fprintf(stderr, "Managed runtime ERROR: %s\n", last_error_.c_str());
}
bool PersistentRuntime::submit(std::uint64_t, const std::string&, std::string& error)
{
    error = "use the Windows web chat interface";
    return false;
}
bool PersistentRuntime::collect(std::uint64_t&, std::string&, std::string& error)
{
    error = "use the Windows web chat interface";
    return false;
}

} // namespace dan
