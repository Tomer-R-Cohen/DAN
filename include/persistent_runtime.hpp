#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#ifdef _WIN32
#include <cstdint>
#include <memory>
#include "platform.hpp"
using pid_t = std::intptr_t;
#else
#include <sys/types.h>
#endif

namespace dan {

enum class RuntimeState { stopped, starting, ready, error };

std::string_view runtime_state_name(RuntimeState state);

struct PersistentRuntimeConfig {
    std::string executable;
    std::string model;
    std::string host = "127.0.0.1";
    std::string port;
    std::size_t context_size = 32768;
    std::chrono::seconds startup_timeout{1800};
    std::chrono::seconds request_timeout{1800};
    std::vector<std::string> arguments;
};

class PersistentRuntime {
public:
    PersistentRuntime() = default;
    ~PersistentRuntime();
    PersistentRuntime(const PersistentRuntime&) = delete;
    PersistentRuntime& operator=(const PersistentRuntime&) = delete;

    bool start(const PersistentRuntimeConfig& config, std::string endpoints,
        std::string tensor_split, std::string& error);
    void update();
    void stop();
    void mark_error(std::string error);
    bool submit(std::uint64_t request_id, const std::string& prompt, std::string& error);
    bool collect(std::uint64_t& request_id, std::string& response, std::string& error);

    RuntimeState state() const { return state_; }
    pid_t pid() const { return pid_; }
    int result_fd() const { return result_fd_; }
    bool busy() const { return request_pid_ != -1; }
    std::size_t requests_served() const { return requests_served_; }
    const std::string& last_error() const { return last_error_; }
    const std::string& endpoints() const { return endpoints_; }

private:
    void terminate(pid_t& child);
    void cleanup();

    PersistentRuntimeConfig config_;
    RuntimeState state_ = RuntimeState::stopped;
    pid_t pid_ = -1;
    pid_t request_pid_ = -1;
    int result_fd_ = -1;
    std::uint64_t request_id_ = 0;
    std::size_t requests_served_ = 0;
    std::string endpoints_;
    std::string tensor_split_;
    std::string last_error_;
    std::chrono::steady_clock::time_point startup_deadline_;
#ifdef _WIN32
    std::unique_ptr<platform::Process> process_;
#endif
};

} // namespace dan
