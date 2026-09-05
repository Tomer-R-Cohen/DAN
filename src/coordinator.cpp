#include "protocol.hpp"
#include "distributed_runtime.hpp"
#include "model_registry.hpp"

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr const char* default_port = "9000";
constexpr int backlog = 16;

struct Request {
    std::uint64_t id;
    std::string prompt;
    std::string model;
    std::string model_path;
    std::string target_group;
    bool allow_single = true;
    bool allow_distributed = true;

    Request(std::uint64_t request_id, std::string request_prompt,
        std::string request_model = {}, std::string group = {})
        : id(request_id), prompt(std::move(request_prompt)),
          model(std::move(request_model)), target_group(std::move(group)) { }
};

enum class ExecutionTargetType { single_provider, distributed_group };

struct Provider {
    ExecutionTargetType type = ExecutionTargetType::single_provider;
    int socket = -1;
    std::size_t id = 0;
    std::string reported_id;
    std::string device_type;
    std::string gpu_name;
    std::string vram;
    std::string model_name;
    std::string backend;
    bool busy = false;
    std::optional<Request> request;
    std::chrono::steady_clock::time_point started;
    std::size_t completed_requests = 0;
    double last_elapsed_ms = 0.0;
    double total_elapsed_ms = 0.0;
};

struct DistributedGroup {
    ExecutionTargetType type = ExecutionTargetType::distributed_group;
    dan::DistributedConfig config;
    bool busy = false;
    int result_socket = -1;
    pid_t child = -1;
    std::optional<Request> request;
    std::chrono::steady_clock::time_point started;
    std::size_t completed_requests = 0;
    double total_elapsed_ms = 0.0;
};

void print_group(const DistributedGroup& group)
{
    std::cout << "Execution target: distributed group\n  ID: " << group.config.id
              << "\n  Status: available"
              << "\n  Model: " << group.config.model
              << "\n  RPC participants: " << group.config.endpoints
              << "\n  Tensor split: " << group.config.tensor_split << '\n';
}

bool start_group_request(DistributedGroup& group, Request request)
{
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == -1) return false;
    const pid_t child = fork();
    if (child == -1) { close(sockets[0]); close(sockets[1]); return false; }
    if (child == 0) {
        close(sockets[0]);
        // A group job must not keep other provider connections alive after disconnect.
        const long limit = sysconf(_SC_OPEN_MAX);
        for (int fd = 3; fd < limit; ++fd) if (fd != sockets[1]) close(fd);
        std::string response;
        const bool ok = dan::run_distributed_inference(
            group.config, request.prompt, response);
        const std::string result = (ok ? "RESPONSE\n" : "ERROR\n")
            + std::to_string(request.id) + '\n'
            + (ok ? response : "distributed inference failed");
        dan::send_message(sockets[1], result);
        close(sockets[1]);
        _exit(ok ? 0 : 1);
    }
    close(sockets[1]);
    group.busy = true;
    group.result_socket = sockets[0];
    group.child = child;
    group.request = std::move(request);
    group.started = std::chrono::steady_clock::now();
    return true;
}

void dispatch_group_requests(std::vector<DistributedGroup>& groups,
    std::deque<Request>& requests)
{
    for (DistributedGroup& group : groups) {
        if (group.busy) continue;
        for (auto request = requests.begin(); request != requests.end(); ++request) {
            if (request->target_group == group.config.id) {
                Request work = std::move(*request);
                requests.erase(request);
                if (start_group_request(group, std::move(work))) {
                    std::printf("Dispatched request %llu to distributed group %s\n",
                        static_cast<unsigned long long>(group.request->id),
                        group.config.id.c_str());
                } else {
                    std::fprintf(stderr, "Could not start distributed group %s\n",
                        group.config.id.c_str());
                }
                break;
            }
        }
    }
}

bool parse_capabilities(std::string_view message, Provider& provider)
{
    constexpr std::string_view prefix = "CAPABILITIES\n";
    if (!message.starts_with(prefix)) return false;
    message.remove_prefix(prefix.size());
    while (!message.empty()) {
        const std::size_t newline = message.find('\n');
        const std::string_view line = message.substr(0, newline);
        const std::size_t equals = line.find('=');
        if (equals == std::string_view::npos) return false;
        const std::string_view key = line.substr(0, equals);
        const std::string value(line.substr(equals + 1));
        if (key == "provider_id") provider.reported_id = value;
        else if (key == "device_type") provider.device_type = value;
        else if (key == "gpu_name") provider.gpu_name = value;
        else if (key == "vram") provider.vram = value;
        else if (key == "model_name") provider.model_name = value;
        else if (key == "backend") provider.backend = value;
        if (newline == std::string_view::npos) break;
        message.remove_prefix(newline + 1);
    }
    return !provider.reported_id.empty() && !provider.device_type.empty()
        && !provider.model_name.empty() && !provider.backend.empty();
}

void print_provider(const Provider& provider, std::size_t provider_count)
{
    std::printf("\nProvider %zu registered (%zu total)\n", provider.id, provider_count);
    std::cout << "  Target type: single provider\n  Status: available\n  ID: " << provider.reported_id
              << "\n  Device: " << provider.device_type
              << "\n  GPU: " << provider.gpu_name
              << "\n  VRAM: " << provider.vram
              << "\n  Model: " << provider.model_name
              << "\n  Backend: " << provider.backend << '\n';
}

void remove_provider(std::vector<Provider>& providers, std::size_t index,
    std::size_t& next_provider, std::deque<Request>& requests,
    std::deque<Request>& model_requests)
{
    Provider& provider = providers[index];
    if (provider.request) {
        std::printf("Provider %zu disconnected; re-queued request %llu\n",
            provider.id, static_cast<unsigned long long>(provider.request->id));
        if (provider.request->model.empty()) requests.push_front(std::move(*provider.request));
        else model_requests.push_front(std::move(*provider.request));
    } else {
        std::printf("Provider %zu disconnected\n", provider.id);
    }
    close(provider.socket);
    providers.erase(providers.begin() + static_cast<std::ptrdiff_t>(index));
    if (providers.empty()) next_provider = 0;
    else if (index < next_provider) --next_provider;
    else if (next_provider >= providers.size()) next_provider = 0;
}

void dispatch_waiting(std::vector<Provider>& providers,
    std::size_t& next_provider, std::deque<Request>& requests,
    std::deque<Request>& model_requests)
{
    while (!requests.empty() && !providers.empty()) {
        std::optional<std::size_t> selected;
        for (std::size_t offset = 0; offset < providers.size(); ++offset) {
            const std::size_t index = (next_provider + offset) % providers.size();
            if (!providers[index].busy) {
                selected = index;
                break;
            }
        }
        if (!selected) return;

        const std::size_t index = *selected;
        const Request& request = requests.front();
        const std::string message = "PROMPT\n" + std::to_string(request.id)
            + '\n' + request.prompt;
        if (!dan::send_message(providers[index].socket, message)) {
            remove_provider(providers, index, next_provider, requests, model_requests);
            continue;
        }

        Provider& provider = providers[index];
        provider.busy = true;
        provider.request = std::move(requests.front());
        requests.pop_front();
        provider.started = std::chrono::steady_clock::now();
        std::printf("Dispatched request %llu to provider %zu (%zu queued)\n",
            static_cast<unsigned long long>(provider.request->id), provider.id,
            requests.size());
        next_provider = (index + 1) % providers.size();
    }
}

std::string_view model_file_name(std::string_view path)
{
    const std::size_t slash = path.find_last_of('/');
    return path.substr(slash == std::string_view::npos ? 0 : slash + 1);
}

bool model_matches(std::string_view requested, const Provider& provider)
{
    return requested == provider.model_name;
}

bool model_matches(std::string_view requested, const DistributedGroup& group)
{
    return requested == group.config.model
        || requested == model_file_name(group.config.model);
}

void dispatch_model_requests(std::vector<Provider>& providers,
    std::vector<DistributedGroup>& groups, std::deque<Request>& requests,
    std::deque<Request>& legacy_requests, std::size_t& next_target,
    std::size_t& next_provider)
{
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto request = requests.begin(); request != requests.end();) {
            const std::size_t target_count = providers.size() + groups.size();
            std::optional<std::size_t> selected;
            bool compatible = false;
            for (std::size_t offset = 0; offset < target_count; ++offset) {
                const std::size_t target = (next_target + offset) % target_count;
                if (target < providers.size()) {
                    const Provider& provider = providers[target];
                    if (request->allow_single && model_matches(request->model, provider)) {
                        compatible = true;
                        if (!provider.busy) { selected = target; break; }
                    }
                } else {
                    const DistributedGroup& group = groups[target - providers.size()];
                    if (request->allow_distributed
                        && (model_matches(request->model, group)
                            || (!request->model_path.empty()
                                && request->model_path == group.config.model))) {
                        compatible = true;
                        if (!group.busy) { selected = target; break; }
                    }
                }
            }
            if (!compatible) {
                std::fprintf(stderr,
                    "Request %llu rejected: no execution target serves model %s\n",
                    static_cast<unsigned long long>(request->id), request->model.c_str());
                request = requests.erase(request);
                changed = true;
                continue;
            }
            if (!selected) { ++request; continue; }

            const std::size_t target = *selected;
            if (target < providers.size()) {
                Provider& provider = providers[target];
                const std::string message = "PROMPT\n" + std::to_string(request->id)
                    + '\n' + request->prompt;
                if (!dan::send_message(provider.socket, message)) {
                    remove_provider(providers, target, next_provider,
                        legacy_requests, requests);
                    next_target = 0;
                    changed = true;
                    break;
                }
                provider.busy = true;
                provider.request = std::move(*request);
                provider.started = std::chrono::steady_clock::now();
                std::printf("Automatically dispatched request %llu for model %s to provider %zu\n",
                    static_cast<unsigned long long>(provider.request->id),
                    provider.request->model.c_str(), provider.id);
            } else {
                DistributedGroup& group = groups[target - providers.size()];
                Request work = std::move(*request);
                if (!start_group_request(group, std::move(work))) {
                    std::fprintf(stderr, "Could not start distributed group %s\n",
                        group.config.id.c_str());
                    request = requests.erase(request);
                    changed = true;
                    continue;
                }
                std::printf("Automatically dispatched request %llu for model %s to distributed group %s\n",
                    static_cast<unsigned long long>(group.request->id),
                    group.request->model.c_str(), group.config.id.c_str());
            }
            request = requests.erase(request);
            next_target = target_count == 0 ? 0 : (target + 1) % target_count;
            changed = true;
        }
    }
}

bool split_result(std::string_view message, std::string_view prefix,
    std::uint64_t& request_id, std::string& payload)
{
    if (!message.starts_with(prefix)) return false;
    message.remove_prefix(prefix.size());
    const std::size_t newline = message.find('\n');
    if (newline == std::string_view::npos) return false;
    try {
        std::size_t consumed = 0;
        request_id = std::stoull(std::string(message.substr(0, newline)), &consumed);
        if (consumed != newline) return false;
    } catch (...) {
        return false;
    }
    payload = message.substr(newline + 1);
    return true;
}

bool any_busy(const std::vector<Provider>& providers)
{
    for (const Provider& provider : providers) {
        if (provider.busy) return true;
    }
    return false;
}

bool any_group_busy(const std::vector<DistributedGroup>& groups)
{
    for (const auto& group : groups) if (group.busy) return true;
    return false;
}
}

int main(int argc, char* argv[])
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* port = default_port;
    int argument = 1;
    if (argument < argc && !std::string_view(argv[argument]).starts_with("--")) port = argv[argument++];
    dan::ModelRegistry model_registry;
    bool has_model_registry = false;
    std::vector<DistributedGroup> groups;
    while (argument < argc) {
        if (std::string_view(argv[argument]) == "--models" && argument + 1 < argc) {
            std::string error;
            if (!model_registry.load(argv[argument + 1], error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
            has_model_registry = true;
            argument += 2;
            continue;
        }
        if (std::string_view(argv[argument]) != "--group" || argument + 5 >= argc) {
            std::fprintf(stderr, "Usage: %s [port] [--models registry] [--group id rpc-llama model-or-@id endpoints split]...\n", argv[0]);
            return 1;
        }
        dan::DistributedConfig config{argv[argument + 1], argv[argument + 2],
            argv[argument + 3], argv[argument + 4], argv[argument + 5]};
        if (dan::split_rpc_endpoints(config.endpoints).size() < 2
            || (!config.model.starts_with('@')
                && access(config.model.c_str(), R_OK) == -1)) {
            std::fprintf(stderr, "Invalid distributed group %s\n", config.id.c_str());
            return 1;
        }
        DistributedGroup group;
        group.config = std::move(config);
        groups.push_back(std::move(group));
        argument += 6;
    }
    for (auto& group : groups) {
        if (!group.config.model.starts_with('@')) continue;
        const auto* model = model_registry.find(group.config.model.substr(1));
        if (!model || !model->distributed) {
            std::fprintf(stderr, "Unknown or non-distributed group model %s\n",
                group.config.model.c_str());
            return 1;
        }
        group.config.model = model->path;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* addresses = nullptr;
    const int address_status = getaddrinfo(nullptr, port, &hints, &addresses);
    if (address_status != 0) {
        std::fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(address_status));
        return 1;
    }
    const int listening_socket = socket(
        addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
    if (listening_socket == -1) {
        perror("socket");
        freeaddrinfo(addresses);
        return 1;
    }
    const int reuse_address = 1;
    if (setsockopt(listening_socket, SOL_SOCKET, SO_REUSEADDR,
            &reuse_address, sizeof(reuse_address)) == -1
        || bind(listening_socket, addresses->ai_addr, addresses->ai_addrlen) == -1) {
        perror("bind/listen setup");
        close(listening_socket);
        freeaddrinfo(addresses);
        return 1;
    }
    freeaddrinfo(addresses);
    if (listen(listening_socket, backlog) == -1) {
        perror("listen");
        close(listening_socket);
        return 1;
    }

    std::vector<Provider> providers;
    std::deque<Request> requests;
    std::deque<Request> model_requests;
    std::deque<Request> group_requests;
    std::size_t next_provider = 0;
    std::size_t next_target = 0;
    std::size_t next_provider_id = 1;
    std::uint64_t next_request_id = 1;
    bool accepting_input = true;
    std::string input_line;
    std::printf("Listening for providers on port %s\n", port);
    if (has_model_registry) {
        std::printf("Model registry (%zu models):\n", model_registry.models().size());
        for (const auto& model : model_registry.models()) {
            std::printf("  %s [%s] family=%s memory=%zu MiB context=%zu single=%s distributed=%s\n",
                model.id.c_str(), model.role.c_str(), model.family.c_str(),
                model.memory_mib, model.context_length,
                model.single_provider ? "yes" : "no",
                model.distributed ? "yes" : "no");
        }
    }
    for (const auto& group : groups) print_group(group);
    std::cout << "Prompt (or exit): " << std::flush;

    while (accepting_input || !requests.empty() || !model_requests.empty()
        || !group_requests.empty()
        || any_busy(providers) || any_group_busy(groups)) {
        const std::size_t polled_provider_count = providers.size();
        std::vector<pollfd> poll_fds;
        poll_fds.push_back({accepting_input ? STDIN_FILENO : -1, POLLIN, 0});
        poll_fds.push_back({listening_socket, POLLIN, 0});
        for (const Provider& provider : providers) {
            poll_fds.push_back({provider.socket, POLLIN, 0});
        }
        for (const DistributedGroup& group : groups) {
            poll_fds.push_back({group.busy ? group.result_socket : -1, POLLIN, 0});
        }
        if (poll(poll_fds.data(), poll_fds.size(), -1) == -1) {
            if (errno == EINTR) continue;
            perror("poll");
            break;
        }

        for (std::size_t i = polled_provider_count; i > 0; --i) {
            const std::size_t index = i - 1;
            const short events = poll_fds[i + 1].revents;
            bool remove = false;
            if ((events & POLLIN) && providers[index].busy) {
                std::string result;
                if (!dan::receive_message(providers[index].socket, result)) {
                    remove = true;
                } else {
                    Provider& provider = providers[index];
                    std::uint64_t result_id = 0;
                    std::string payload;
                    const bool response = split_result(result, "RESPONSE\n", result_id, payload);
                    const bool error = !response
                        && split_result(result, "ERROR\n", result_id, payload);
                    if ((!response && !error) || !provider.request
                        || result_id != provider.request->id) {
                        std::fprintf(stderr, "Provider %zu returned an invalid request ID\n",
                            provider.id);
                        remove = true;
                    } else if (error) {
                        std::cerr << "Request " << result_id << " failed on provider "
                                  << provider.id << ": " << payload << '\n';
                        provider.request.reset();
                        provider.busy = false;
                        remove = true;
                    } else {
                        const auto finished = std::chrono::steady_clock::now();
                        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                            finished - provider.started).count();
                        provider.last_elapsed_ms = elapsed_ms;
                        provider.total_elapsed_ms += elapsed_ms;
                        ++provider.completed_requests;
                        std::cout << "\nResponse for request " << result_id
                                  << " from provider " << provider.id << ":\n"
                                  << payload << '\n';
                        std::printf("Performance: provider %zu | %.1f ms | %zu requests | %.1f ms average\n",
                            provider.id, provider.last_elapsed_ms,
                            provider.completed_requests,
                            provider.total_elapsed_ms / provider.completed_requests);
                        provider.request.reset();
                        provider.busy = false;
                    }
                }
            } else if ((events & POLLIN) && !providers[index].busy) {
                char byte = '\0';
                if (recv(providers[index].socket, &byte, 1,
                        MSG_PEEK | MSG_DONTWAIT) == 0) remove = true;
            }
            if (events & (POLLHUP | POLLERR | POLLNVAL)) remove = true;
            if (remove) {
                remove_provider(providers, index, next_provider, requests, model_requests);
            }
        }

        const std::size_t group_poll_start = 2 + polled_provider_count;
        for (std::size_t index = 0; index < groups.size(); ++index) {
            DistributedGroup& group = groups[index];
            if (!group.busy || !(poll_fds[group_poll_start + index].revents
                    & (POLLIN | POLLHUP | POLLERR))) continue;
            std::string result;
            std::uint64_t result_id = 0;
            std::string payload;
            const bool received = dan::receive_message(group.result_socket, result);
            const bool response = received
                && split_result(result, "RESPONSE\n", result_id, payload);
            const bool error = received && !response
                && split_result(result, "ERROR\n", result_id, payload);
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - group.started).count();
            if (response && group.request && result_id == group.request->id) {
                ++group.completed_requests;
                group.total_elapsed_ms += elapsed_ms;
                std::cout << "\nResponse for request " << result_id
                          << " from distributed group " << group.config.id
                          << ":\n" << payload << '\n';
                std::printf("Performance: group %s | %.1f ms | %zu requests | %.1f ms average\n",
                    group.config.id.c_str(), elapsed_ms, group.completed_requests,
                    group.total_elapsed_ms / group.completed_requests);
            } else {
                std::cerr << "Distributed request "
                          << (group.request ? std::to_string(group.request->id) : "unknown")
                          << " failed" << (error ? ": " + payload : "") << '\n';
            }
            close(group.result_socket);
            int status = 0;
            while (waitpid(group.child, &status, 0) == -1 && errno == EINTR) { }
            group.result_socket = -1; group.child = -1; group.busy = false;
            group.request.reset();
        }

        if (poll_fds[1].revents & POLLIN) {
            const int provider_socket = accept(listening_socket, nullptr, nullptr);
            if (provider_socket == -1) {
                perror("accept");
            } else {
                std::string greeting;
                std::string capabilities;
                Provider provider;
                provider.socket = provider_socket;
                provider.id = next_provider_id;
                if (dan::receive_message(provider_socket, greeting) && greeting == "HELLO"
                    && dan::receive_message(provider_socket, capabilities)
                    && parse_capabilities(capabilities, provider)) {
                    providers.push_back(std::move(provider));
                    print_provider(providers.back(), providers.size());
                    ++next_provider_id;
                } else {
                    std::fprintf(stderr, "Rejected invalid provider registration\n");
                    close(provider_socket);
                }
            }
        }

        if (accepting_input && (poll_fds[0].revents & (POLLIN | POLLHUP))) {
            char input_byte = 0;
            const ssize_t count = read(STDIN_FILENO, &input_byte, 1);
            if (count == -1 && errno == EINTR) continue;
            if (count > 0 && input_byte != '\n') {
                input_line += input_byte;
                continue;
            }
            std::string prompt = std::move(input_line);
            input_line.clear();
            if (!prompt.empty() && prompt.back() == '\r') prompt.pop_back();
            if ((count <= 0 && prompt.empty()) || prompt == "exit") {
                accepting_input = false;
                std::printf("Finishing queued and active requests before shutdown...\n");
            } else if (prompt.empty()) {
                std::fprintf(stderr, "Prompt must not be empty\n");
            } else {
                constexpr std::string_view group_prefix = "/group ";
                constexpr std::string_view model_prefix = "/model ";
                if (prompt.starts_with(model_prefix)) {
                    const std::size_t separator = prompt.find(' ', model_prefix.size());
                    if (separator == std::string::npos || separator + 1 == prompt.size()) {
                        std::fprintf(stderr, "Usage: /model <model-name> <prompt>\n");
                        continue;
                    }
                    const std::string requested = prompt.substr(
                        model_prefix.size(), separator - model_prefix.size());
                    model_requests.emplace_back(next_request_id,
                        prompt.substr(separator + 1), requested);
                    if (const auto* model = model_registry.find(requested)) {
                        Request& request = model_requests.back();
                        request.model = model->runtime_name;
                        request.model_path = model->path;
                        request.allow_single = model->single_provider;
                        request.allow_distributed = model->distributed;
                    } else if (has_model_registry) {
                        std::fprintf(stderr, "Request %llu rejected: model ID %s is not in the registry\n",
                            static_cast<unsigned long long>(next_request_id), requested.c_str());
                        model_requests.pop_back();
                        ++next_request_id;
                        continue;
                    }
                } else if (prompt.starts_with(group_prefix)) {
                    const std::size_t separator = prompt.find(' ', group_prefix.size());
                    if (separator == std::string::npos) {
                        std::fprintf(stderr, "Usage: /group <group-id> <prompt>\n");
                        continue;
                    }
                    const std::string group_id = prompt.substr(
                        group_prefix.size(), separator - group_prefix.size());
                    bool known = false;
                    for (const auto& group : groups) if (group.config.id == group_id) known = true;
                    if (!known) { std::fprintf(stderr, "Unknown distributed group %s\n", group_id.c_str()); continue; }
                    group_requests.emplace_back(next_request_id,
                        prompt.substr(separator + 1), std::string{}, group_id);
                } else {
                    requests.emplace_back(next_request_id, std::move(prompt));
                }
                std::printf("Queued request %llu (%zu queued)\n",
                    static_cast<unsigned long long>(next_request_id),
                    requests.size() + model_requests.size() + group_requests.size());
                ++next_request_id;
            }
        }

        dispatch_model_requests(providers, groups, model_requests, requests,
            next_target, next_provider);
        dispatch_waiting(providers, next_provider, requests, model_requests);
        dispatch_group_requests(groups, group_requests);
        if (!accepting_input && providers.empty() && !requests.empty()) {
            std::fprintf(stderr, "No providers remain; abandoning %zu queued requests\n",
                requests.size());
            requests.clear();
        }
        if (!accepting_input && providers.empty() && groups.empty()
            && !model_requests.empty()) {
            model_requests.clear();
        }
        if (accepting_input) std::cout << "Prompt (or exit): " << std::flush;
    }

    for (const Provider& provider : providers) {
        dan::send_message(provider.socket, "BYE");
        close(provider.socket);
    }
    close(listening_socket);
    return 0;
}
