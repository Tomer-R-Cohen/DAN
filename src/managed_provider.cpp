#include "control_plane.hpp"
#include "protocol.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace std::chrono_literals;

struct Options {
    std::string coordinator_host = "127.0.0.1";
    std::string coordinator_port = "9000";
    std::string id;
    std::string gpu = "unknown";
    std::string vram = "unknown";
    std::size_t vram_mib = 0;
    fs::path cache_dir;
    std::string worker;
    std::string worker_host = "127.0.0.1";
    std::string worker_port;
    std::string worker_device = "CUDA0";
    std::vector<std::string> worker_args;
    std::chrono::seconds worker_timeout{5};
};

bool safe_component(std::string_view value)
{
    if (value.empty() || value == "." || value == "..") return false;
    for (const unsigned char byte : value) {
        if (!std::isalnum(byte) && byte != '.' && byte != '_' && byte != '-') return false;
    }
    return true;
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
    return value;
}

bool run(const std::vector<std::string>& arguments, std::string& error)
{
    const pid_t child = fork();
    if (child == -1) { error = "fork failed"; return false; }
    if (child == 0) {
        std::vector<char*> argv;
        for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    int status = 0;
    while (waitpid(child, &status, 0) == -1 && errno == EINTR) { }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        error = arguments.front() + " exited unsuccessfully";
        return false;
    }
    return true;
}

bool sha256(const fs::path& path, std::string& digest, std::string& error)
{
    int output[2];
    if (pipe(output) == -1) { error = "sha256 pipe failed"; return false; }
    const pid_t child = fork();
    if (child == -1) {
        close(output[0]); close(output[1]); error = "sha256 fork failed"; return false;
    }
    if (child == 0) {
        close(output[0]);
        if (dup2(output[1], STDOUT_FILENO) == -1) _exit(127);
        close(output[1]);
        const std::string filename = path.string();
        execlp("sha256sum", "sha256sum", "--", filename.c_str(), nullptr);
        _exit(127);
    }
    close(output[1]);
    std::string result;
    char buffer[256];
    while (true) {
        const ssize_t count = read(output[0], buffer, sizeof(buffer));
        if (count == -1 && errno == EINTR) continue;
        if (count <= 0) break;
        result.append(buffer, static_cast<std::size_t>(count));
    }
    close(output[0]);
    int status = 0;
    while (waitpid(child, &status, 0) == -1 && errno == EINTR) { }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || result.size() < 65
        || !dan::valid_sha256(std::string_view(result).substr(0, 64))) {
        error = "could not calculate SHA-256";
        return false;
    }
    digest = lowercase(result.substr(0, 64));
    return true;
}

fs::path artifact_path(const Options& options, const dan::CachedShard& shard)
{
    return options.cache_dir / shard.model_id / shard.version / shard.shard_id
        / (lowercase(shard.hash) + ".artifact");
}

bool valid_identity(const dan::CachedShard& shard)
{
    return safe_component(shard.model_id) && safe_component(shard.version)
        && safe_component(shard.shard_id) && dan::valid_sha256(shard.hash);
}

bool verified(const fs::path& path, std::string_view expected, std::string& error)
{
    if (!fs::is_regular_file(path)) return false;
    std::string actual;
    return sha256(path, actual, error) && actual == lowercase(std::string(expected));
}

std::vector<dan::CachedShard> scan_cache(const Options& options)
{
    std::vector<dan::CachedShard> cached;
    std::error_code ignored;
    if (!fs::is_directory(options.cache_dir, ignored)) return cached;
    for (const auto& model : fs::directory_iterator(options.cache_dir, ignored)) {
        if (!model.is_directory() || !safe_component(model.path().filename().string())) continue;
        for (const auto& version : fs::directory_iterator(model.path(), ignored)) {
            if (!version.is_directory() || !safe_component(version.path().filename().string())) continue;
            for (const auto& shard : fs::directory_iterator(version.path(), ignored)) {
                if (!shard.is_directory() || !safe_component(shard.path().filename().string())) continue;
                for (const auto& artifact : fs::directory_iterator(shard.path(), ignored)) {
                    const std::string name = artifact.path().filename().string();
                    if (!artifact.is_regular_file() || !name.ends_with(".artifact")) continue;
                    const std::string hash = name.substr(0, name.size() - 9);
                    dan::CachedShard identity{model.path().filename().string(),
                        version.path().filename().string(), shard.path().filename().string(), hash};
                    std::string error;
                    if (valid_identity(identity) && verified(artifact.path(), hash, error)) {
                        cached.push_back(std::move(identity));
                    }
                }
            }
        }
    }
    return cached;
}

template <typename Report>
bool prepare_artifact(const Options& options, const dan::CachedShard& assignment,
    const dan::ShardMetadata& metadata, Report report, std::string& error)
{
    if (!valid_identity(assignment) || !dan::valid_source(metadata.source)) {
        error = "invalid artifact identity or source";
        return false;
    }
    const fs::path target = artifact_path(options, assignment);
    if (fs::exists(target)) {
        if (verified(target, assignment.hash, error)) {
            if (!report(dan::ShardState::cached)) error = "could not report cached artifact";
            else return true;
            return false;
        }
        std::error_code remove_error;
        fs::remove(target, remove_error);
        if (remove_error) { error = "could not reject corrupt cached artifact"; return false; }
    }
    if (!report(dan::ShardState::downloading)) {
        error = "could not report artifact download";
        return false;
    }
    std::error_code filesystem_error;
    fs::create_directories(target.parent_path(), filesystem_error);
    if (filesystem_error) { error = "could not create cache directory"; return false; }
    const fs::path temporary = target.string() + ".partial." + std::to_string(getpid());
    fs::remove(temporary, filesystem_error);
    bool downloaded = false;
    if (metadata.source.starts_with("http://") || metadata.source.starts_with("https://")) {
        downloaded = run({"curl", "--fail", "--location", "--silent", "--show-error",
            "--output", temporary.string(), metadata.source}, error);
    } else {
        std::string source = metadata.source.starts_with("file://")
            ? metadata.source.substr(7) : metadata.source;
        if (source.empty()) error = "empty local artifact source";
        else {
            fs::copy_file(source, temporary, fs::copy_options::overwrite_existing, filesystem_error);
            downloaded = !filesystem_error;
            if (!downloaded) error = "could not copy local artifact: " + filesystem_error.message();
        }
    }
    if (!downloaded) { fs::remove(temporary, filesystem_error); return false; }
    if (metadata.size_bytes != 0 && fs::file_size(temporary, filesystem_error) != metadata.size_bytes) {
        error = "downloaded artifact has the wrong size";
        fs::remove(temporary, filesystem_error);
        return false;
    }
    std::string actual;
    if (!sha256(temporary, actual, error) || actual != lowercase(assignment.hash)) {
        if (error.empty()) error = "downloaded artifact SHA-256 mismatch";
        fs::remove(temporary, filesystem_error);
        return false;
    }
    fs::rename(temporary, target, filesystem_error);
    if (filesystem_error) { error = "could not install cached artifact"; fs::remove(temporary); return false; }
    if (!report(dan::ShardState::cached)) {
        error = "could not report cached artifact";
        return false;
    }
    return true;
}

bool tcp_healthy(std::string_view host, std::string_view port)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(std::string(host).c_str(), std::string(port).c_str(), &hints, &addresses) != 0) {
        return false;
    }
    bool healthy = false;
    for (addrinfo* address = addresses; address && !healthy; address = address->ai_next) {
        const int socket_fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (socket_fd == -1) continue;
        healthy = connect(socket_fd, address->ai_addr, address->ai_addrlen) == 0;
        close(socket_fd);
    }
    freeaddrinfo(addresses);
    return healthy;
}

class ManagedWorker {
public:
    ~ManagedWorker() { stop(); }

    bool start(const Options& options, std::string& error)
    {
        last_error.clear();
        if (running()) return true;
        if (tcp_healthy(options.worker_host, options.worker_port)) {
            error = last_error = "worker endpoint is already in use";
            return false;
        }
        pid_ = fork();
        if (pid_ == -1) { error = last_error = "could not fork managed worker"; return false; }
        if (pid_ == 0) {
            const pid_t parent = getppid();
            if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1 || getppid() != parent) _exit(127);
            std::vector<std::string> arguments{options.worker, "--host", options.worker_host,
                "--port", options.worker_port, "--device", options.worker_device};
            arguments.insert(arguments.end(), options.worker_args.begin(), options.worker_args.end());
            std::vector<char*> argv;
            for (auto& argument : arguments) argv.push_back(argument.data());
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            _exit(127);
        }
        std::printf("Managed worker started: pid=%d endpoint=%s:%s\n", pid_,
            options.worker_host.c_str(), options.worker_port.c_str());
        const auto deadline = std::chrono::steady_clock::now() + options.worker_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            const pid_t result = waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                pid_ = -1; error = last_error = "managed worker exited before healthy"; return false;
            }
            if (tcp_healthy(options.worker_host, options.worker_port)) return true;
            std::this_thread::sleep_for(50ms);
        }
        error = last_error = "managed worker did not become healthy before timeout";
        stop();
        return false;
    }

    bool running()
    {
        if (pid_ == -1) return false;
        int status = 0;
        const pid_t result = waitpid(pid_, &status, WNOHANG);
        if (result == 0) return true;
        if (result == pid_) pid_ = -1;
        return false;
    }

    void stop()
    {
        if (pid_ == -1) return;
        kill(pid_, SIGTERM);
        for (int attempt = 0; attempt < 20; ++attempt) {
            int status = 0;
            if (waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; return; }
            std::this_thread::sleep_for(50ms);
        }
        kill(pid_, SIGKILL);
        while (waitpid(pid_, nullptr, 0) == -1 && errno == EINTR) { }
        pid_ = -1;
    }

    std::string last_error;

private:
    pid_t pid_ = -1;
};

bool parse_options(int argc, char* argv[], Options& options)
{
    const char* home = std::getenv("HOME");
    options.cache_dir = home ? fs::path(home) / ".dan/models" : fs::path(".dan/models");
    for (int argument = 1; argument < argc;) {
        const std::string option = argv[argument++];
        if (argument >= argc) return false;
        const std::string value = argv[argument++];
        if (option == "--host") options.coordinator_host = value;
        else if (option == "--port") options.coordinator_port = value;
        else if (option == "--id") options.id = value;
        else if (option == "--gpu") options.gpu = value;
        else if (option == "--vram") options.vram = value;
        else if (option == "--vram-mib") {
            if (!dan::parse_size(value, options.vram_mib)) return false;
        } else if (option == "--cache-dir") options.cache_dir = value;
        else if (option == "--worker") options.worker = value;
        else if (option == "--worker-host") options.worker_host = value;
        else if (option == "--worker-port") options.worker_port = value;
        else if (option == "--worker-device") options.worker_device = value;
        else if (option == "--worker-arg") options.worker_args.push_back(value);
        else if (option == "--worker-timeout") {
            std::size_t seconds = 0;
            if (!dan::parse_size(value, seconds) || seconds == 0) return false;
            options.worker_timeout = std::chrono::seconds(seconds);
        } else return false;
    }
    return !options.id.empty() && !options.worker.empty() && !options.worker_port.empty()
        && options.vram_mib > 0 && options.worker_host != "0.0.0.0"
        && options.worker_host != "::" && options.worker_host != "*";
}

int connect_to(std::string_view host, std::string_view port)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(std::string(host).c_str(), std::string(port).c_str(), &hints, &addresses) != 0) {
        return -1;
    }
    int connected = -1;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        connected = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (connected != -1 && connect(connected, address->ai_addr, address->ai_addrlen) == 0) break;
        if (connected != -1) close(connected);
        connected = -1;
    }
    freeaddrinfo(addresses);
    return connected;
}
}

int main(int argc, char* argv[])
{
    signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::fprintf(stderr, "Usage: %s --id ID --gpu NAME --vram-mib N --cache-dir DIR "
            "--worker PATH --worker-port PORT [--host HOST] [--port PORT] "
            "[--worker-host HOST] [--worker-device DEVICE] [--worker-arg ARG]...\n", argv[0]);
        return 1;
    }
    const int coordinator = connect_to(options.coordinator_host, options.coordinator_port);
    if (coordinator == -1) { std::fprintf(stderr, "Could not connect to coordinator\n"); return 1; }

    std::vector<dan::CachedShard> cached = scan_cache(options);
    std::string capabilities = "CAPABILITIES\nprovider_id=" + options.id
        + "\ndevice_type=GPU\ngpu_name=" + options.gpu + "\nvram="
        + (options.vram == "unknown" ? std::to_string(options.vram_mib) + " MiB" : options.vram)
        + "\nvram_mib=" + std::to_string(options.vram_mib)
        + "\nmodel_name=dan-main\nbackend=RPC_LLAMA\ncontrol_plane=1\nworker_endpoint="
        + options.worker_host + ':' + options.worker_port;
    for (const auto& shard : cached) capabilities += "\ncached_shard=" + dan::cached_shard_value(shard);
    if (!dan::send_message(coordinator, "HELLO") || !dan::send_message(coordinator, capabilities)) {
        close(coordinator); return 1;
    }

    std::mutex send_mutex;
    std::atomic<bool> connected{true};
    const auto send = [&](std::string_view message) {
        std::lock_guard lock(send_mutex);
        return connected.load() && dan::send_message(coordinator, message);
    };
    std::jthread heartbeat([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(1s);
            if (!stop.stop_requested() && !send("HEARTBEAT")) break;
        }
    });
    const auto finish = [&](int status) {
        heartbeat.request_stop();
        connected = false;
        shutdown(coordinator, SHUT_RDWR);
        close(coordinator);
        return status;
    };

    ManagedWorker worker;
    std::optional<dan::CachedShard> assignment;
    dan::ShardMetadata metadata;
    dan::ShardState state = dan::ShardState::unassigned;
    const auto report = [&](dan::ShardState next) {
        state = next;
        return assignment && send(dan::shard_state_message(*assignment, next));
    };
    while (true) {
        pollfd watched{coordinator, POLLIN, 0};
        const int poll_result = poll(&watched, 1, 250);
        if (poll_result == -1 && errno == EINTR) continue;
        if (poll_result == -1 || (watched.revents & (POLLHUP | POLLERR | POLLNVAL))) return finish(1);
        if ((state == dan::ShardState::ready || state == dan::ShardState::loading)
            && !worker.running()) {
            worker.last_error = "managed worker exited unexpectedly";
            std::fprintf(stderr, "Managed worker exited unexpectedly\n");
            if (!report(dan::ShardState::error)) return finish(1);
        }
        if (!(watched.revents & POLLIN)) continue;
        std::string message;
        if (!dan::receive_message(coordinator, message)) return finish(1);
        if (message == "BYE") break;
        if (message.starts_with("ASSIGN_SHARD\n")) {
            dan::CachedShard next_assignment;
            dan::ShardMetadata next_metadata;
            if (!dan::parse_shard_assignment(message, next_assignment, next_metadata)
                || !valid_identity(next_assignment)) {
                std::fprintf(stderr, "Invalid shard assignment\n");
                return finish(1);
            }
            worker.stop();
            assignment = std::move(next_assignment);
            metadata = std::move(next_metadata);
            const fs::path path = artifact_path(options, *assignment);
            std::string error;
            if (verified(path, assignment->hash, error)) {
                if (!report(dan::ShardState::cached)) return finish(1);
            } else {
                if (!report(dan::ShardState::assigned)
                    || !prepare_artifact(options, *assignment, metadata, report, error)) {
                    std::fprintf(stderr, "Artifact preparation failed: %s\n", error.c_str());
                    if (!report(dan::ShardState::error)) return finish(1);
                }
            }
            continue;
        }
        const bool load = message.starts_with("LOAD_SHARD\n");
        const bool unload = message.starts_with("UNLOAD_SHARD\n");
        if (load || unload) {
            dan::CachedShard command;
            if (!assignment || !dan::parse_shard_command(message,
                    load ? "LOAD_SHARD" : "UNLOAD_SHARD", command)
                || dan::cached_shard_value(command) != dan::cached_shard_value(*assignment)) {
                std::fprintf(stderr, "Invalid managed shard command\n");
                return finish(1);
            }
            if (unload) {
                worker.stop();
                std::string error;
                if (!verified(artifact_path(options, *assignment), assignment->hash, error)) {
                    worker.last_error = error.empty() ? "cached artifact is invalid" : error;
                    if (!report(dan::ShardState::error)) return finish(1);
                } else if (!report(dan::ShardState::cached)) return finish(1);
                continue;
            }
            if (worker.running()) {
                if (!report(dan::ShardState::ready)) return finish(1);
                continue;
            }
            std::string error;
            const bool already_cached = verified(
                artifact_path(options, *assignment), assignment->hash, error);
            if (!already_cached
                && !prepare_artifact(options, *assignment, metadata, report, error)) {
                std::fprintf(stderr, "Artifact recovery failed: %s\n", error.c_str());
                if (!report(dan::ShardState::error)) return finish(1);
                continue;
            }
            if (already_cached && state == dan::ShardState::error
                && !report(dan::ShardState::cached)) return finish(1);
            if (!report(dan::ShardState::loading)) return finish(1);
            if (!worker.start(options, error)) {
                worker.last_error = error;
                std::fprintf(stderr, "Managed worker failed: %s\n", error.c_str());
                if (!report(dan::ShardState::error)) return finish(1);
            } else if (!report(dan::ShardState::ready)) return finish(1);
            continue;
        }
        std::fprintf(stderr, "Unexpected or malformed control message\n");
        return finish(1);
    }
    worker.stop();
    return finish(0);
}
