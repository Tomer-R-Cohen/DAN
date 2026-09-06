#include "protocol.hpp"
#include "control_plane.hpp"
#include "distributed_runtime.hpp"
#include "model_registry.hpp"
#include "persistent_runtime.hpp"

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
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
    std::string provider_name;
    std::string device_type;
    std::string gpu_name;
    std::string vram;
    std::size_t vram_mib = 0;
    std::string model_name;
    std::string backend;
    std::string worker_endpoint;
    bool control_plane = false;
    bool online = true;
    std::vector<dan::CachedShard> cached_shards;
    std::optional<std::size_t> assigned_shard;
    std::vector<std::size_t> failed_shards;
    dan::ShardState shard_state = dan::ShardState::unassigned;
    bool auto_load = true;
    bool load_requested = false;
    std::chrono::steady_clock::time_point last_seen = std::chrono::steady_clock::now();
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
        else if (key == "provider_name") provider.provider_name = value;
        else if (key == "device_type") provider.device_type = value;
        else if (key == "gpu_name") provider.gpu_name = value;
        else if (key == "vram") provider.vram = value;
        else if (key == "vram_mib") {
            if (!dan::parse_size(value, provider.vram_mib)) return false;
        }
        else if (key == "model_name") provider.model_name = value;
        else if (key == "backend") provider.backend = value;
        else if (key == "worker_endpoint") provider.worker_endpoint = value;
        else if (key == "control_plane") {
            if (value != "0" && value != "1") return false;
            provider.control_plane = value == "1";
        } else if (key == "cached_shard") {
            dan::CachedShard shard;
            if (!dan::parse_cached_shard(value, shard)) return false;
            provider.cached_shards.push_back(std::move(shard));
        }
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
              << "\n  Name: " << (provider.provider_name.empty() ? "-" : provider.provider_name)
              << "\n  Device: " << provider.device_type
              << "\n  GPU: " << provider.gpu_name
              << "\n  VRAM: " << provider.vram
              << "\n  VRAM MiB: " << provider.vram_mib
              << "\n  Model: " << provider.model_name
              << "\n  Backend: " << provider.backend
              << "\n  Managed worker: "
              << (provider.worker_endpoint.empty() ? "not configured" : provider.worker_endpoint)
              << "\n  Control plane: " << (provider.control_plane ? "v1" : "disabled") << '\n';
}

void offline_provider(std::vector<Provider>& providers, std::size_t index,
    std::size_t& next_provider, std::deque<Request>& requests,
    std::deque<Request>& model_requests, std::string_view reason = "disconnected")
{
    Provider& provider = providers[index];
    if (!provider.online) return;
    if (provider.request) {
        std::printf("Provider %zu %.*s; re-queued request %llu\n", provider.id,
            static_cast<int>(reason.size()), reason.data(),
            static_cast<unsigned long long>(provider.request->id));
        if (provider.request->model.empty()) requests.push_front(std::move(*provider.request));
        else model_requests.push_front(std::move(*provider.request));
    } else {
        std::printf("Provider %zu %.*s\n", provider.id,
            static_cast<int>(reason.size()), reason.data());
    }
    close(provider.socket);
    provider.socket = -1;
    provider.online = false;
    provider.load_requested = false;
    provider.busy = false;
    provider.request.reset();
    if (provider.assigned_shard) {
        provider.assigned_shard.reset();
        provider.shard_state = dan::ShardState::unassigned;
    }
    if (!providers.empty()) next_provider = (index + 1) % providers.size();
}

std::size_t online_provider_count(const std::vector<Provider>& providers)
{
    std::size_t count = 0;
    for (const auto& provider : providers) if (provider.online) ++count;
    return count;
}

bool has_cached_shard(const Provider& provider, const dan::ManagedModel& model,
    const dan::ShardMetadata& shard)
{
    for (const auto& cached : provider.cached_shards) {
        if (cached.model_id == model.id && cached.version == model.version
            && cached.shard_id == shard.id && cached.hash == shard.hash) return true;
    }
    return false;
}

bool failed_shard(const Provider& provider, std::size_t shard)
{
    return std::find(provider.failed_shards.begin(), provider.failed_shards.end(), shard)
        != provider.failed_shards.end();
}

bool valid_transition(dan::ShardState from, dan::ShardState to)
{
    if (from == to) return true;
    if (to == dan::ShardState::error) return true;
    switch (from) {
    case dan::ShardState::assigned:
        return to == dan::ShardState::downloading || to == dan::ShardState::cached;
    case dan::ShardState::downloading:
        return to == dan::ShardState::cached;
    case dan::ShardState::cached:
        return to == dan::ShardState::loading || to == dan::ShardState::ready;
    case dan::ShardState::loading:
        return to == dan::ShardState::ready || to == dan::ShardState::cached;
    case dan::ShardState::ready:
        return to == dan::ShardState::cached;
    case dan::ShardState::error:
        return to == dan::ShardState::assigned || to == dan::ShardState::downloading
            || to == dan::ShardState::cached;
    default:
        return false;
    }
}

bool apply_shard_state(Provider& provider, const dan::ManagedModel& model,
    std::string_view message)
{
    dan::CachedShard reported;
    dan::ShardState state;
    if (!provider.assigned_shard
        || !dan::parse_shard_state_message(message, reported, state)
        || state == dan::ShardState::unassigned) return false;
    const auto& assigned = model.shards[*provider.assigned_shard];
    if (reported.model_id != model.id || reported.version != model.version
        || reported.shard_id != assigned.id || reported.hash != assigned.hash
        || !valid_transition(provider.shard_state, state)) return false;
    provider.shard_state = state;
    const std::size_t shard_index = *provider.assigned_shard;
    if (state == dan::ShardState::error) {
        if (!failed_shard(provider, shard_index)) provider.failed_shards.push_back(shard_index);
    } else {
        std::erase(provider.failed_shards, shard_index);
    }
    if (state == dan::ShardState::cached || state == dan::ShardState::ready) {
        bool known = false;
        for (const auto& cached : provider.cached_shards) {
            if (dan::cached_shard_value(cached) == dan::cached_shard_value(reported)) known = true;
        }
        if (!known) provider.cached_shards.push_back(std::move(reported));
    }
    return true;
}

bool send_assignment(Provider& provider, const dan::ManagedModel& model)
{
    if (!provider.online || !provider.assigned_shard) return true;
    return dan::send_message(provider.socket,
        dan::shard_assignment_message(model, model.shards[*provider.assigned_shard]));
}

dan::CachedShard assigned_identity(const Provider& provider,
    const dan::ManagedModel& model)
{
    const auto& shard = model.shards[*provider.assigned_shard];
    return {model.id, model.version, shard.id, shard.hash};
}

bool send_shard_command(Provider& provider, const dan::ManagedModel& model,
    std::string_view command)
{
    if (!provider.online || !provider.assigned_shard) return false;
    return dan::send_message(provider.socket,
        dan::shard_command_message(command, assigned_identity(provider, model)));
}

std::optional<std::size_t> select_provider(const std::vector<Provider>& providers,
    const dan::ManagedModel& model, std::size_t shard_index)
{
    std::optional<std::size_t> selected;
    for (std::size_t index = 0; index < providers.size(); ++index) {
        const Provider& candidate = providers[index];
        if (!candidate.online || !candidate.control_plane || candidate.assigned_shard
            || candidate.busy || failed_shard(candidate, shard_index)
            || candidate.vram_mib < model.shards[shard_index].min_vram_mib) continue;
        if (!selected) {
            selected = index;
            continue;
        }
        const bool candidate_cached = has_cached_shard(
            candidate, model, model.shards[shard_index]);
        const bool selected_cached = has_cached_shard(
            providers[*selected], model, model.shards[shard_index]);
        if ((candidate_cached && !selected_cached)
            || (candidate_cached == selected_cached
                && (candidate.vram_mib > providers[*selected].vram_mib
                    || (candidate.vram_mib == providers[*selected].vram_mib
                        && candidate.reported_id < providers[*selected].reported_id)))) {
            selected = index;
        }
    }
    return selected;
}

void assign_shards(std::vector<Provider>& providers, const dan::ManagedModel& model)
{
    for (Provider& owner : providers) {
        if (!owner.online || !owner.assigned_shard
            || owner.shard_state != dan::ShardState::error) continue;
        if (select_provider(providers, model, *owner.assigned_shard)) {
            std::printf("Released failed shard %s from provider %s for replacement\n",
                model.shards[*owner.assigned_shard].id.c_str(), owner.reported_id.c_str());
            owner.assigned_shard.reset();
            owner.load_requested = false;
        }
    }
    for (std::size_t shard_index = 0; shard_index < model.shards.size(); ++shard_index) {
        bool assigned = false;
        for (const auto& provider : providers) {
            if (provider.assigned_shard == shard_index) { assigned = true; break; }
        }
        if (assigned) continue;
        const auto selected = select_provider(providers, model, shard_index);
        if (!selected) continue;
        Provider& provider = providers[*selected];
        provider.assigned_shard = shard_index;
        provider.auto_load = true;
        provider.load_requested = false;
        provider.shard_state = has_cached_shard(provider, model, model.shards[shard_index])
            ? dan::ShardState::cached : dan::ShardState::assigned;
        std::printf("Assigned shard %s to provider %s%s\n",
            model.shards[shard_index].id.c_str(), provider.reported_id.c_str(),
            provider.shard_state == dan::ShardState::cached ? " from exact cache" : "");
        if (!send_assignment(provider, model)) {
            close(provider.socket);
            provider.socket = -1;
            provider.online = false;
            provider.assigned_shard.reset();
            provider.shard_state = dan::ShardState::unassigned;
        }
    }
}

std::size_t ready_shard_count(const std::vector<Provider>& providers,
    const dan::ManagedModel& model)
{
    std::size_t ready = 0;
    for (std::size_t shard = 0; shard < model.shards.size(); ++shard) {
        for (const auto& provider : providers) {
            if (provider.online && provider.assigned_shard == shard
                && provider.shard_state == dan::ShardState::ready) { ++ready; break; }
        }
    }
    return ready;
}

bool managed_runtime_layout(const std::vector<Provider>& providers,
    const dan::ManagedModel& model, std::string& endpoints, std::string& split)
{
    endpoints.clear();
    split.clear();
    for (std::size_t shard = 0; shard < model.shards.size(); ++shard) {
        const Provider* selected = nullptr;
        for (const auto& provider : providers) {
            if (provider.online && provider.assigned_shard == shard
                && provider.shard_state == dan::ShardState::ready) {
                selected = &provider;
                break;
            }
        }
        if (!selected || selected->worker_endpoint.empty() || selected->vram_mib == 0) return false;
        if (!endpoints.empty()) { endpoints += ','; split += ','; }
        endpoints += selected->worker_endpoint;
        split += std::to_string(selected->vram_mib);
    }
    return !endpoints.empty();
}

void print_control_plane(const std::vector<Provider>& providers,
    const std::optional<dan::ManagedModel>& model,
    const dan::PersistentRuntime* runtime = nullptr)
{
    std::printf("\n%-14s %-18s %-10s %-10s %-8s %-12s %-9s %-22s %s\n",
        "ID", "GPU", "VRAM", "ROLE", "SHARD", "STATE", "STATUS", "WORKER", "LAST_SEEN");
    const auto now = std::chrono::steady_clock::now();
    for (const auto& provider : providers) {
        const std::string shard = provider.assigned_shard && model
            ? model->shards[*provider.assigned_shard].id : "-";
        const char* role = provider.assigned_shard ? "ASSIGNED"
            : provider.online ? "SPARE" : "OFFLINE";
        const std::string_view state = provider.assigned_shard
            || provider.shard_state == dan::ShardState::error
            ? dan::shard_state_name(provider.shard_state) : "-";
        const auto age = std::chrono::duration_cast<std::chrono::seconds>(now - provider.last_seen).count();
        std::printf("%-14s %-18s %-10s %-10s %-8s %-12s %-9s %-22s %llds\n",
            provider.reported_id.c_str(), provider.gpu_name.c_str(), provider.vram.c_str(),
            role, shard.c_str(), state.data(),
            provider.online ? "ONLINE" : "OFFLINE",
            provider.worker_endpoint.empty() ? "-" : provider.worker_endpoint.c_str(),
            static_cast<long long>(age));
    }
    if (!model) {
        std::printf("Managed dan-main replica: disabled (use --managed-model <manifest>)\n");
        return;
    }
    const std::size_t ready = ready_shard_count(providers, *model);
    std::printf("\n%s %s\nrequired shards: %zu\nready shards: %zu\nstate: %s\n",
        model->id.c_str(), model->version.c_str(), model->shards.size(), ready,
        ready == model->shards.size() ? "READY" : "NOT_READY");
    std::printf("replica: %s\nruntime: %s\nruntime pid: %d\nrequests served: %zu\n",
        ready == model->shards.size() ? "READY" : "NOT_READY",
        runtime ? dan::runtime_state_name(runtime->state()).data() : "DISABLED",
        runtime ? runtime->pid() : -1, runtime ? runtime->requests_served() : 0);
    if (runtime && !runtime->endpoints().empty()) {
        std::printf("runtime endpoints: %s\n", runtime->endpoints().c_str());
    }
    if (runtime && !runtime->last_error().empty()) {
        std::printf("runtime error: %s\n", runtime->last_error().c_str());
    }
    if (ready != model->shards.size()) {
        for (std::size_t shard = 0; shard < model->shards.size(); ++shard) {
            const Provider* owner = nullptr;
            for (const auto& provider : providers) {
                if (provider.online && provider.assigned_shard == shard) {
                    owner = &provider;
                    break;
                }
            }
            if (owner && owner->shard_state == dan::ShardState::ready) continue;
            if (owner) {
                std::printf("preparing shard: %s\nreplacement: %s (%s)\n",
                    model->shards[shard].id.c_str(), owner->reported_id.c_str(),
                    dan::shard_state_name(owner->shard_state).data());
            } else {
                std::printf("missing shard: %s\nreplacement: NONE_ELIGIBLE\n",
                    model->shards[shard].id.c_str());
            }
        }
    }
}

void dispatch_waiting(std::vector<Provider>& providers,
    std::size_t& next_provider, std::deque<Request>& requests,
    std::deque<Request>& model_requests)
{
    while (!requests.empty() && !providers.empty()) {
        std::optional<std::size_t> selected;
        for (std::size_t offset = 0; offset < providers.size(); ++offset) {
            const std::size_t index = (next_provider + offset) % providers.size();
            if (providers[index].online && !providers[index].busy) {
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
            offline_provider(providers, index, next_provider, requests, model_requests);
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
                    if (provider.online && request->allow_single
                        && model_matches(request->model, provider)) {
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
                    offline_provider(providers, target, next_provider,
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
        if (provider.online && provider.busy) return true;
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
    std::optional<dan::ManagedModel> managed_model;
    std::optional<dan::PersistentRuntimeConfig> managed_runtime_config;
    std::chrono::seconds heartbeat_timeout{10};
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
        if (std::string_view(argv[argument]) == "--managed-model" && argument + 1 < argc
            && !managed_model) {
            dan::ManagedModel model;
            std::string error;
            if (!dan::load_managed_model(argv[argument + 1], model, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
            managed_model = std::move(model);
            argument += 2;
            continue;
        }
        if (std::string_view(argv[argument]) == "--managed-runtime" && argument + 3 < argc
            && !managed_runtime_config) {
            dan::PersistentRuntimeConfig config;
            config.executable = argv[argument + 1];
            config.model = argv[argument + 2];
            config.port = argv[argument + 3];
            managed_runtime_config = std::move(config);
            argument += 4;
            continue;
        }
        if (std::string_view(argv[argument]) == "--managed-runtime-arg"
            && argument + 1 < argc && managed_runtime_config) {
            managed_runtime_config->arguments.emplace_back(argv[argument + 1]);
            argument += 2;
            continue;
        }
        if ((std::string_view(argv[argument]) == "--managed-runtime-timeout"
                || std::string_view(argv[argument]) == "--managed-request-timeout"
                || std::string_view(argv[argument]) == "--managed-runtime-context")
            && argument + 1 < argc && managed_runtime_config) {
            std::size_t value = 0;
            if (!dan::parse_size(argv[argument + 1], value) || value == 0) {
                std::fprintf(stderr, "Managed runtime numeric options must be positive\n");
                return 1;
            }
            if (std::string_view(argv[argument]) == "--managed-runtime-timeout") {
                managed_runtime_config->startup_timeout = std::chrono::seconds(value);
            } else if (std::string_view(argv[argument]) == "--managed-request-timeout") {
                managed_runtime_config->request_timeout = std::chrono::seconds(value);
            } else managed_runtime_config->context_size = value;
            argument += 2;
            continue;
        }
        if (std::string_view(argv[argument]) == "--heartbeat-timeout"
            && argument + 1 < argc) {
            std::size_t seconds = 0;
            if (!dan::parse_size(argv[argument + 1], seconds) || seconds == 0) {
                std::fprintf(stderr, "Heartbeat timeout must be a positive number of seconds\n");
                return 1;
            }
            heartbeat_timeout = std::chrono::seconds(seconds);
            argument += 2;
            continue;
        }
        if (std::string_view(argv[argument]) != "--group" || argument + 5 >= argc) {
            std::fprintf(stderr, "Usage: %s [port] [--models registry] "
                "[--managed-model manifest] [--heartbeat-timeout seconds] "
                "[--managed-runtime llama-server model.gguf listen-port] "
                "[--managed-runtime-timeout seconds] [--managed-request-timeout seconds] "
                "[--managed-runtime-context tokens] [--managed-runtime-arg value]... "
                "[--group id rpc-llama model-or-@id endpoints split]...\n", argv[0]);
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
    if (managed_runtime_config && (!managed_model
            || access(managed_runtime_config->model.c_str(), R_OK) == -1)) {
        std::fprintf(stderr, "Managed runtime requires a dan-main manifest and readable model file\n");
        return 1;
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
    std::deque<Request> managed_requests;
    dan::PersistentRuntime managed_runtime;
    bool managed_runtime_auto_start = true;
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
    if (managed_model) {
        std::printf("Managed replica: %s %s (%zu required shards, heartbeat timeout %llds)\n",
            managed_model->id.c_str(), managed_model->version.c_str(),
            managed_model->shards.size(),
            static_cast<long long>(heartbeat_timeout.count()));
    }
    for (const auto& group : groups) print_group(group);
    std::cout << "Prompt (or exit): " << std::flush;

    const auto replica_ready = [&] {
        return managed_model
            && ready_shard_count(providers, *managed_model) == managed_model->shards.size();
    };
    const auto start_managed_runtime = [&] {
        if (!managed_runtime_config || !managed_model || !replica_ready()) return false;
        std::string endpoints;
        std::string split;
        std::string error;
        if (!managed_runtime_layout(providers, *managed_model, endpoints, split)) {
            managed_runtime.mark_error("managed replica has an invalid worker endpoint or VRAM");
            return false;
        }
        if (!managed_runtime.start(*managed_runtime_config,
                std::move(endpoints), std::move(split), error)) {
            if (!error.empty()) std::fprintf(stderr, "Could not start managed runtime: %s\n", error.c_str());
            return false;
        }
        return true;
    };
    const auto reconcile_managed_runtime = [&] {
        if (!managed_runtime_config) return;
        managed_runtime.update();
        if (!replica_ready()) {
            if (managed_runtime.state() != dan::RuntimeState::stopped) {
                if (managed_runtime.busy()) {
                    std::fprintf(stderr,
                        "Active dan-main request failed: replica lost a required provider\n");
                }
                managed_runtime.stop();
                std::fprintf(stderr, "Managed runtime STOPPED: replica is not ready\n");
            }
            if (!managed_requests.empty()) {
                std::fprintf(stderr, "Rejected %zu queued dan-main requests: replica is not ready\n",
                    managed_requests.size());
                managed_requests.clear();
            }
        } else if (managed_runtime_auto_start
            && managed_runtime.state() == dan::RuntimeState::stopped) {
            start_managed_runtime();
        }
    };

    while (accepting_input || !requests.empty() || !model_requests.empty()
        || !group_requests.empty() || !managed_requests.empty()
        || any_busy(providers) || any_group_busy(groups) || managed_runtime.busy()) {
        if (managed_model) assign_shards(providers, *managed_model);
        reconcile_managed_runtime();
        const std::size_t polled_provider_count = providers.size();
        std::vector<pollfd> poll_fds;
        poll_fds.push_back({accepting_input ? STDIN_FILENO : -1, POLLIN, 0});
        poll_fds.push_back({listening_socket, POLLIN, 0});
        for (const Provider& provider : providers) {
            poll_fds.push_back({provider.online ? provider.socket : -1, POLLIN, 0});
        }
        for (const DistributedGroup& group : groups) {
            poll_fds.push_back({group.busy ? group.result_socket : -1, POLLIN, 0});
        }
        const std::size_t managed_poll_index = poll_fds.size();
        poll_fds.push_back({managed_runtime.busy() ? managed_runtime.result_fd() : -1, POLLIN, 0});
        if (poll(poll_fds.data(), poll_fds.size(), 250) == -1) {
            if (errno == EINTR) continue;
            perror("poll");
            break;
        }

        for (std::size_t i = polled_provider_count; i > 0; --i) {
            const std::size_t index = i - 1;
            const short events = poll_fds[i + 1].revents;
            bool remove = false;
            if ((events & POLLIN) && providers[index].online) {
                std::string result;
                if (!dan::receive_message(providers[index].socket, result)) {
                    remove = true;
                } else {
                    Provider& provider = providers[index];
                    if (result == "HEARTBEAT") {
                        provider.last_seen = std::chrono::steady_clock::now();
                    } else if (result.starts_with("SHARD_STATE\n")) {
                        if (!managed_model || !apply_shard_state(provider, *managed_model, result)) {
                            std::fprintf(stderr, "Provider %zu sent invalid shard state\n", provider.id);
                            remove = true;
                        } else {
                            provider.last_seen = std::chrono::steady_clock::now();
                            std::printf("Provider %s shard %s is %s\n",
                                provider.reported_id.c_str(),
                                managed_model->shards[*provider.assigned_shard].id.c_str(),
                                dan::shard_state_name(provider.shard_state).data());
                            if (provider.shard_state == dan::ShardState::error) {
                                provider.load_requested = false;
                            } else if (provider.shard_state == dan::ShardState::cached
                                && provider.auto_load && !provider.load_requested) {
                                provider.load_requested = true;
                                if (!send_shard_command(provider, *managed_model, "LOAD_SHARD")) {
                                    remove = true;
                                }
                            }
                        }
                    } else if (!provider.busy) {
                        std::fprintf(stderr, "Provider %zu sent an unexpected control message\n",
                            provider.id);
                        remove = true;
                    } else {
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
                            provider.last_seen = std::chrono::steady_clock::now();
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
                }
            }
            if (events & (POLLHUP | POLLERR | POLLNVAL)) remove = true;
            if (remove) {
                offline_provider(providers, index, next_provider, requests, model_requests);
            }
        }

        const auto now = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < providers.size(); ++index) {
            const Provider& provider = providers[index];
            if (provider.online && provider.control_plane
                && now - provider.last_seen > heartbeat_timeout) {
                offline_provider(providers, index, next_provider, requests,
                    model_requests, "heartbeat timed out; marked OFFLINE");
            }
        }
        reconcile_managed_runtime();

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

        if (managed_runtime.busy()
            && (poll_fds[managed_poll_index].revents & (POLLIN | POLLHUP | POLLERR))) {
            std::uint64_t request_id = 0;
            std::string response;
            std::string error;
            if (managed_runtime.collect(request_id, response, error)) {
                std::cout << "\nResponse for request " << request_id
                          << " from managed dan-main:\n" << response << '\n';
                std::printf("Managed runtime: pid %d | %zu requests served\n",
                    managed_runtime.pid(), managed_runtime.requests_served());
            } else {
                std::fprintf(stderr, "Managed dan-main request failed: %s\n", error.c_str());
            }
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
                    auto existing = providers.end();
                    for (auto candidate = providers.begin(); candidate != providers.end(); ++candidate) {
                        if (candidate->reported_id == provider.reported_id) { existing = candidate; break; }
                    }
                    if (existing != providers.end() && existing->online) {
                        std::fprintf(stderr, "Rejected duplicate live provider ID %s\n",
                            provider.reported_id.c_str());
                        close(provider_socket);
                    } else {
                        Provider* registered = nullptr;
                        if (existing == providers.end()) {
                            provider.last_seen = std::chrono::steady_clock::now();
                            providers.push_back(std::move(provider));
                            registered = &providers.back();
                            ++next_provider_id;
                        } else {
                            existing->socket = provider.socket;
                            existing->provider_name = std::move(provider.provider_name);
                            existing->device_type = std::move(provider.device_type);
                            existing->gpu_name = std::move(provider.gpu_name);
                            existing->vram = std::move(provider.vram);
                            existing->vram_mib = provider.vram_mib;
                            existing->model_name = std::move(provider.model_name);
                            existing->backend = std::move(provider.backend);
                            existing->worker_endpoint = std::move(provider.worker_endpoint);
                            existing->control_plane = provider.control_plane;
                            existing->cached_shards = std::move(provider.cached_shards);
                            existing->failed_shards.clear();
                            existing->online = true;
                            existing->auto_load = true;
                            existing->load_requested = false;
                            existing->last_seen = std::chrono::steady_clock::now();
                            registered = &*existing;
                            if (managed_model && registered->assigned_shard) {
                                const auto& shard = managed_model->shards[*registered->assigned_shard];
                                if (registered->vram_mib < shard.min_vram_mib) {
                                    registered->assigned_shard.reset();
                                    registered->shard_state = dan::ShardState::unassigned;
                                } else {
                                    registered->shard_state = has_cached_shard(*registered,
                                        *managed_model, shard)
                                        ? dan::ShardState::cached : dan::ShardState::assigned;
                                }
                            } else registered->shard_state = dan::ShardState::unassigned;
                        }
                        print_provider(*registered, online_provider_count(providers));
                        if (managed_model && registered->control_plane) {
                            if (registered->assigned_shard
                                && !send_assignment(*registered, *managed_model)) {
                                close(registered->socket); registered->socket = -1;
                                registered->online = false;
                            }
                            assign_shards(providers, *managed_model);
                        }
                    }
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
            } else if (prompt == "/providers") {
                print_control_plane(providers, managed_model,
                    managed_runtime_config ? &managed_runtime : nullptr);
            } else if (prompt == "/runtime start" || prompt == "/runtime restart"
                || prompt == "/runtime stop") {
                if (!managed_runtime_config) {
                    std::fprintf(stderr, "Managed runtime is not configured\n");
                } else if (prompt == "/runtime stop") {
                    managed_runtime_auto_start = false;
                    managed_runtime.stop();
                    std::printf("Managed runtime STOPPED by operator\n");
                } else if (!replica_ready()) {
                    std::fprintf(stderr, "Cannot start managed runtime: dan-main replica is NOT_READY\n");
                } else if (prompt == "/runtime start"
                    && (managed_runtime.state() == dan::RuntimeState::starting
                        || managed_runtime.state() == dan::RuntimeState::ready)) {
                    std::printf("Managed runtime already %s pid=%d\n",
                        dan::runtime_state_name(managed_runtime.state()).data(),
                        managed_runtime.pid());
                } else {
                    managed_runtime_auto_start = true;
                    if (prompt == "/runtime restart") managed_runtime.stop();
                    start_managed_runtime();
                }
            } else if (prompt.starts_with("/load ") || prompt.starts_with("/unload ")) {
                const bool load = prompt.starts_with("/load ");
                const std::string id = prompt.substr(load ? 6 : 8);
                auto provider = providers.end();
                for (auto candidate = providers.begin(); candidate != providers.end(); ++candidate) {
                    if (candidate->reported_id == id) { provider = candidate; break; }
                }
                if (!managed_model || provider == providers.end() || !provider->online
                    || !provider->control_plane || !provider->assigned_shard) {
                    std::fprintf(stderr, "Managed provider %s is not available\n", id.c_str());
                } else {
                    provider->auto_load = load;
                    provider->load_requested = load;
                    if (!send_shard_command(*provider, *managed_model,
                            load ? "LOAD_SHARD" : "UNLOAD_SHARD")) {
                        const std::size_t index = static_cast<std::size_t>(provider - providers.begin());
                        offline_provider(providers, index, next_provider, requests, model_requests);
                    }
                }
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
                    const bool managed_request = managed_model && requested == managed_model->id;
                    if (managed_request) {
                        if (!replica_ready()) {
                            std::string_view missing = "unknown";
                            for (std::size_t shard = 0; shard < managed_model->shards.size(); ++shard) {
                                bool ready = false;
                                for (const auto& provider : providers) {
                                    if (provider.online && provider.assigned_shard == shard
                                        && provider.shard_state == dan::ShardState::ready) {
                                        ready = true;
                                        break;
                                    }
                                }
                                if (!ready) { missing = managed_model->shards[shard].id; break; }
                            }
                            std::fprintf(stderr,
                                "Request %llu rejected: dan-main replica is NOT_READY; recovering shard %.*s\n",
                                static_cast<unsigned long long>(next_request_id),
                                static_cast<int>(missing.size()), missing.data());
                            ++next_request_id;
                            continue;
                        }
                        if (!managed_runtime_config
                            || managed_runtime.state() != dan::RuntimeState::ready) {
                            std::fprintf(stderr,
                                "Request %llu rejected: dan-main runtime is %s\n",
                                static_cast<unsigned long long>(next_request_id),
                                managed_runtime_config
                                    ? dan::runtime_state_name(managed_runtime.state()).data()
                                    : "DISABLED");
                            ++next_request_id;
                            continue;
                        }
                        managed_requests.emplace_back(next_request_id,
                            prompt.substr(separator + 1), requested);
                    } else {
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
                    requests.size() + model_requests.size() + group_requests.size()
                        + managed_requests.size());
                ++next_request_id;
            }
        }

        if (!managed_runtime.busy() && !managed_requests.empty()
            && managed_runtime.state() == dan::RuntimeState::ready) {
            std::string error;
            const Request& request = managed_requests.front();
            if (managed_runtime.submit(request.id, request.prompt, error)) {
                std::printf("Dispatched request %llu to managed dan-main runtime pid=%d\n",
                    static_cast<unsigned long long>(request.id), managed_runtime.pid());
                managed_requests.pop_front();
            } else {
                std::fprintf(stderr, "Could not dispatch managed request: %s\n", error.c_str());
            }
        }
        if (managed_runtime.state() == dan::RuntimeState::error
            && !managed_requests.empty()) {
            std::fprintf(stderr, "Rejected %zu queued dan-main requests: runtime is ERROR\n",
                managed_requests.size());
            managed_requests.clear();
        }

        dispatch_model_requests(providers, groups, model_requests, requests,
            next_target, next_provider);
        dispatch_waiting(providers, next_provider, requests, model_requests);
        dispatch_group_requests(groups, group_requests);
        if (!accepting_input && online_provider_count(providers) == 0 && !requests.empty()) {
            std::fprintf(stderr, "No providers remain; abandoning %zu queued requests\n",
                requests.size());
            requests.clear();
        }
        if (!accepting_input && online_provider_count(providers) == 0 && groups.empty()
            && !model_requests.empty()) {
            model_requests.clear();
        }
        if (accepting_input) std::cout << "Prompt (or exit): " << std::flush;
    }

    managed_runtime.stop();
    for (const Provider& provider : providers) {
        if (!provider.online) continue;
        dan::send_message(provider.socket, "BYE");
        close(provider.socket);
    }
    close(listening_socket);
    return 0;
}
