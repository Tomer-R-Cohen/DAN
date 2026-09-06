#include "control_plane.hpp"
#include "platform.hpp"
#include "protocol.hpp"
#include "provider_ui.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
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
    std::string name;
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
    std::chrono::seconds reconnect_delay{0};
    bool friendly = false;
};

void log_event(std::string_view text)
{
    const fs::path directory = dan::platform::data_directory() / "logs";
    std::error_code error; fs::create_directories(directory, error);
    std::ofstream log(directory / "provider.log", std::ios::app);
    if (log) log << std::time(nullptr) << ' ' << text << '\n';
}

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

bool sha256(const fs::path& path, std::string& digest, std::string& error)
{
    return dan::platform::sha256_file(path, digest, error);
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
    const dan::ShardMetadata& metadata, Report report, dan::ProviderTerminalUi* ui,
    std::string& error)
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
        if (ui) ui->update([](auto& state) {
            state.status = dan::ProviderUiStatus::preparing;
            state.message = "Cached model data was invalid. Repairing automatically...";
        });
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
    if (metadata.size_bytes != 0) {
        const auto space = fs::space(target.parent_path(), filesystem_error);
        if (filesystem_error || !dan::artifact_fits_disk(metadata.size_bytes, space.available)) {
            error = "insufficient disk space";
            if (ui && !filesystem_error) ui->update([&](auto& state) {
                state.status = dan::ProviderUiStatus::action_required;
                state.message = "Not enough disk space. DAN needs "
                    + std::to_string((metadata.size_bytes + 1073741823) / 1073741824) + " GB free.";
            });
            return false;
        }
    }
    const fs::path temporary = target.string() + ".partial."
        + std::to_string(dan::platform::process_id());
    fs::remove(temporary, filesystem_error);
    bool downloaded = false;
    if (metadata.source.starts_with("http://") || metadata.source.starts_with("https://")) {
        std::vector<std::string> arguments{"curl", "--fail", "--location", "--show-error",
            "--retry", "3", "--retry-delay", "2",
            "--silent",
            "--output", temporary.string(), metadata.source};
        if (ui) ui->update([](auto& state) {
            state.status = dan::ProviderUiStatus::downloading;
            state.message = "Downloading model...";
            state.download_percent = -1;
        });
        dan::platform::Process download;
        if (download.start(arguments, error, false, true)) {
            while (download.running() && !dan::platform::stop_requested()) {
                if (ui && metadata.size_bytes != 0) {
                    std::error_code size_error;
                    const std::uintmax_t bytes = fs::file_size(temporary, size_error);
                    if (!size_error) ui->update([&](auto& state) {
                        state.download_percent = static_cast<int>(std::min<std::uintmax_t>(100,
                            bytes * 100 / metadata.size_bytes));
                    });
                }
                std::this_thread::sleep_for(250ms);
            }
            if (dan::platform::stop_requested()) {
                download.stop();
                error = "download stopped";
            } else {
                downloaded = fs::is_regular_file(temporary);
                if (!downloaded) error = "model download failed";
            }
        }
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
    return dan::platform::tcp_healthy(host, port);
}

bool wait_for_gpu_memory(const Options& options, const dan::ShardMetadata& metadata,
    dan::ProviderTerminalUi* ui, std::string& error)
{
    const std::size_t required = dan::required_current_vram_mib(metadata);
    const std::string prefix = "CUDA";
    const std::string device = options.worker_device.starts_with(prefix)
        ? options.worker_device.substr(prefix.size()) : "0";
    for (int attempt = 0; attempt < 15; ++attempt) {
        std::string output, command_error;
        if (!dan::platform::run({"nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits", "--id=" + device}, command_error, &output)) {
            return true; // Existing Linux/headless setups may intentionally provide GPU access differently.
        }
        std::size_t available = 0;
        const auto parsed = std::from_chars(output.data(), output.data() + output.size(), available);
        if (parsed.ec != std::errc{}) return true;
        if (available >= required) return true;
        if (ui) ui->update([](auto& state) {
            state.status = dan::ProviderUiStatus::waiting_gpu;
            state.message = "Your GPU is busy. DAN will retry automatically.";
        });
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    error = "not enough current free GPU memory";
    return false;
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
        std::vector<std::string> arguments{options.worker, "--host", options.worker_host,
            "--port", options.worker_port, "--device", options.worker_device};
        arguments.insert(arguments.end(), options.worker_args.begin(), options.worker_args.end());
        if (!process_.start(arguments, error, false, options.friendly)) {
            last_error = error = "could not start managed worker: " + error; return false;
        }
        if (!options.friendly) std::printf("Managed worker started: pid=%llu endpoint=%s:%s\n",
            static_cast<unsigned long long>(process_.pid()), options.worker_host.c_str(),
            options.worker_port.c_str());
        const auto deadline = std::chrono::steady_clock::now() + options.worker_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!process_.running()) {
                error = last_error = "managed worker exited before healthy"; return false;
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
        return process_.running();
    }

    void stop()
    {
        process_.stop();
    }

    std::string last_error;

private:
    dan::platform::Process process_;
};

bool parse_options(int argc, char* argv[], Options& options)
{
    options.cache_dir = dan::platform::data_directory() / "models";
    for (int argument = 1; argument < argc;) {
        const std::string option = argv[argument++];
        if (option == "--friendly") { options.friendly = true; continue; }
        if (argument >= argc) return false;
        const std::string value = argv[argument++];
        if (option == "--host") options.coordinator_host = value;
        else if (option == "--port") options.coordinator_port = value;
        else if (option == "--id") options.id = value;
        else if (option == "--name") options.name = value;
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
        else if (option == "--reconnect-seconds") {
            std::size_t seconds = 0;
            if (!dan::parse_size(value, seconds) || seconds == 0) return false;
            options.reconnect_delay = std::chrono::seconds(seconds);
        }
        else if (option == "--worker-timeout") {
            std::size_t seconds = 0;
            if (!dan::parse_size(value, seconds) || seconds == 0) return false;
            options.worker_timeout = std::chrono::seconds(seconds);
        } else return false;
    }
    return !options.id.empty() && !options.worker.empty() && !options.worker_port.empty()
        && options.vram_mib > 0 && options.worker_host != "0.0.0.0"
        && options.worker_host != "::" && options.worker_host != "*"
        && (options.name.empty() || safe_component(options.name));
}

}

int main(int argc, char* argv[])
{
    std::string platform_error;
    if (!dan::platform::initialize(platform_error)) {
        std::fprintf(stderr, "%s\n", platform_error.c_str()); return 1;
    }
    struct Cleanup { ~Cleanup() { dan::platform::cleanup(); } } cleanup;
    dan::platform::install_stop_handlers();
    dan::platform::configure_output();
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::fprintf(stderr, "Usage: %s --id ID [--name NAME] --gpu NAME --vram-mib N --cache-dir DIR "
            "--worker PATH --worker-port PORT [--host HOST] [--port PORT] "
            "[--worker-host HOST] [--worker-device DEVICE] [--worker-arg ARG] "
            "[--reconnect-seconds N]...\n", argv[0]);
        return 1;
    }
    std::size_t total_vram_mib = options.vram_mib;
    std::from_chars(options.vram.data(), options.vram.data() + options.vram.size(), total_vram_mib);
    dan::ProviderUiState initial_ui;
    initial_ui.gpu_name = options.gpu;
    initial_ui.total_vram_mib = total_vram_mib;
    initial_ui.offered_vram_mib = options.vram_mib;
    initial_ui.status = dan::ProviderUiStatus::connecting;
    initial_ui.message = "Connecting to DAN automatically...";
    initial_ui.diagnostics = (dan::platform::data_directory() / "logs" / "provider.log").string();
    dan::ProviderTerminalUi terminal_ui(std::move(initial_ui), options.friendly);
    dan::ProviderTerminalUi* ui = options.friendly ? &terminal_ui : nullptr;
    ManagedWorker worker;
    std::optional<dan::CachedShard> assignment;
    dan::ShardMetadata metadata;
    dan::ShardState state = dan::ShardState::unassigned;
    std::size_t tokens_participated = 0;
    auto retry_delay = std::min(options.reconnect_delay, std::chrono::seconds(30));
    const auto wait_to_reconnect = [&] {
        if (ui) ui->update([](auto& state) {
            state.network_connected = false;
            state.status = dan::ProviderUiStatus::reconnecting;
            state.message = "DAN network temporarily unavailable. Trying again automatically...";
        });
        else std::fprintf(stderr, "Coordinator: DISCONNECTED; retrying in %llds\n",
                static_cast<long long>(retry_delay.count()));
        const auto deadline = std::chrono::steady_clock::now() + retry_delay;
        while (!dan::platform::stop_requested() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(100ms);
        retry_delay = std::min(retry_delay * 2, std::chrono::seconds(30));
    };
    while (!dan::platform::stop_requested()) {
        const dan::platform::Socket coordinator = dan::platform::connect_tcp(
            options.coordinator_host, options.coordinator_port);
        if (coordinator == dan::platform::invalid_socket) {
            log_event("coordinator connection failed; retrying");
            if (options.reconnect_delay.count() == 0) {
                std::fprintf(stderr, "Could not connect to coordinator\n");
                return 1;
            }
            wait_to_reconnect();
            continue;
        }

        std::vector<dan::CachedShard> cached = scan_cache(options);
        std::string capabilities = "CAPABILITIES\nprovider_id=" + options.id
            + "\nprotocol_version=" + std::to_string(dan::protocol_version)
            + "\nbuild_version=" + std::string(dan::build_version);
        if (!options.name.empty()) capabilities += "\nprovider_name=" + options.name;
        capabilities += "\ndevice_type=GPU\ngpu_name=" + options.gpu + "\nvram="
            + (options.vram == "unknown" ? std::to_string(options.vram_mib) + " MiB" : options.vram)
            + "\nvram_mib=" + std::to_string(options.vram_mib)
            + "\nmodel_name=dan-main\nbackend=RPC_LLAMA\ncontrol_plane=1\nworker_endpoint="
            + options.worker_host + ':' + options.worker_port;
        for (const auto& shard : cached) {
            capabilities += "\ncached_shard=" + dan::cached_shard_value(shard);
        }
        if (!dan::send_message(coordinator, "HELLO", !options.friendly)
            || !dan::send_message(coordinator, capabilities, !options.friendly)) {
            dan::platform::close_socket(coordinator);
            if (options.reconnect_delay.count() == 0) return 1;
            wait_to_reconnect();
            continue;
        }
        retry_delay = std::min(options.reconnect_delay, std::chrono::seconds(30));
        log_event("provider registered with coordinator; protocol=1 build=testnet-ui-v1 gpu="
            + options.gpu + " offered_vram_mib=" + std::to_string(options.vram_mib));
        if (ui) ui->update([&](auto& state) {
            state.network_connected = true;
            state.status = assignment && worker.running()
                ? dan::ProviderUiStatus::contributing : dan::ProviderUiStatus::available;
            state.message = assignment && worker.running()
                ? "Your GPU is contributing to DAN." : "Waiting for useful work";
        });
        else std::printf("Coordinator: CONNECTED\nRole: %s\n", assignment ? "ASSIGNED" : "SPARE");

        std::mutex send_mutex;
        std::atomic<bool> connected{true};
        const auto send = [&](std::string_view message) {
            std::lock_guard lock(send_mutex);
            return connected.load() && dan::send_message(coordinator, message, !options.friendly);
        };
        const auto report = [&](dan::ShardState next) {
            state = next;
            if (ui) ui->update([&](auto& screen) {
                screen.download_percent = -1;
                screen.status = dan::ProviderUiStatus::preparing;
                screen.message = "Preparing model...";
                if (next == dan::ShardState::downloading) {
                    screen.status = dan::ProviderUiStatus::downloading;
                    screen.message = "Downloading model...";
                } else if (next == dan::ShardState::cached) {
                    screen.message = "Model downloaded and verified.";
                } else if (next == dan::ShardState::loading) {
                    screen.status = dan::ProviderUiStatus::loading;
                    screen.message = "Starting GPU worker...";
                } else if (next == dan::ShardState::ready) {
                    screen.status = dan::ProviderUiStatus::contributing;
                    screen.message = "Your GPU is contributing to DAN.";
                } else if (next == dan::ShardState::error) {
                    if (screen.status != dan::ProviderUiStatus::action_required
                        && screen.status != dan::ProviderUiStatus::waiting_gpu) {
                        screen.status = dan::ProviderUiStatus::recovering;
                        screen.message = "DAN is recovering automatically...";
                    }
                }
            }); else std::printf("State: %s\n", dan::shard_state_name(next).data());
            return assignment && send(dan::shard_state_message(*assignment, next));
        };
        std::jthread heartbeat([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                std::this_thread::sleep_for(500ms);
                if (!stop.stop_requested() && !send("HEARTBEAT")) break;
            }
        });

        bool fatal = false;
        bool bye = false;
        while (!dan::platform::stop_requested()) {
            bool socket_failed = false;
            const bool readable = dan::platform::wait_readable(coordinator, 250, socket_failed);
            if (socket_failed) break;
            if ((state == dan::ShardState::ready || state == dan::ShardState::loading)
                && !worker.running()) {
                worker.last_error = "managed worker exited unexpectedly";
                log_event("worker stopped unexpectedly; starting bounded recovery");
                if (ui) ui->update([](auto& screen) {
                    screen.status = dan::ProviderUiStatus::recovering;
                    screen.message = "GPU worker stopped unexpectedly. Recovering automatically...";
                });
                else std::fprintf(stderr, "Managed worker exited unexpectedly\n");
                if (!report(dan::ShardState::cached)) break;
                std::string recovery_error;
                bool recovered = false;
                for (int attempt = 1; attempt <= 3 && !recovered; ++attempt) {
                    recovered = worker.start(options, recovery_error);
                    if (!recovered && attempt < 3) std::this_thread::sleep_for(2s);
                }
                if (recovered) {
                    log_event("worker recovery succeeded");
                    if (!report(dan::ShardState::ready)) break;
                } else {
                    log_event("worker recovery failed: " + recovery_error);
                    if (!report(dan::ShardState::error)) break;
                }
            }
            if (!readable) continue;
            std::string message;
            if (!dan::receive_message(coordinator, message, !options.friendly)) break;
            if (message == "BYE") { bye = true; break; }
            if (message.starts_with("SESSION_METRICS\ntokens_participated=")) {
                std::size_t tokens = 0;
                const std::string_view value(message.data()
                    + std::string_view("SESSION_METRICS\ntokens_participated=").size(),
                    message.size() - std::string_view("SESSION_METRICS\ntokens_participated=").size());
                if (dan::parse_size(value, tokens) && tokens != tokens_participated) {
                    tokens_participated = tokens;
                    if (ui) ui->update([&](auto& screen) { screen.tokens_participated = tokens; });
                }
                continue;
            }
            if (message.starts_with("INCOMPATIBLE\n")) {
                const std::string reason = message.substr(std::string_view("INCOMPATIBLE\n").size());
                log_event("incompatible provider: " + reason);
                if (ui) ui->update([&](auto& screen) {
                    screen.status = dan::ProviderUiStatus::action_required;
                    screen.message = reason;
                }); else std::fprintf(stderr, "\nACTION REQUIRED\n%s\n", reason.c_str());
                fatal = true;
                break;
            }
            if (message.starts_with("ASSIGN_SHARD\n")) {
                dan::CachedShard next_assignment;
                dan::ShardMetadata next_metadata;
                if (!dan::parse_shard_assignment(message, next_assignment, next_metadata)
                    || !valid_identity(next_assignment)) {
                    log_event("invalid shard assignment received");
                    if (ui) ui->update([](auto& screen) {
                        screen.status = dan::ProviderUiStatus::action_required;
                        screen.message = "DAN received an incompatible assignment.";
                    }); else std::fprintf(stderr, "Invalid shard assignment\n");
                    fatal = true;
                    break;
                }
                const bool same = assignment
                    && dan::cached_shard_value(*assignment)
                        == dan::cached_shard_value(next_assignment);
                if (!same) worker.stop();
                assignment = std::move(next_assignment);
                metadata = std::move(next_metadata);
                log_event("assignment model=" + assignment->model_id + " version="
                    + assignment->version + " shard=" + assignment->shard_id
                    + " hash=" + assignment->hash);
                if (ui) ui->update([&](auto& screen) {
                    screen.status = dan::ProviderUiStatus::preparing;
                    screen.model_name = metadata.model_name.empty() ? assignment->version : metadata.model_name;
                    screen.quantization = metadata.quantization;
                    screen.message = "Preparing assigned model...";
                });
                else std::printf("Role: ASSIGNED\nAssignment: %s %s shard %s\n",
                    assignment->model_id.c_str(), assignment->version.c_str(),
                    assignment->shard_id.c_str());
                const fs::path path = artifact_path(options, *assignment);
                std::string error;
                if (verified(path, assignment->hash, error)) {
                    if (!report(dan::ShardState::cached)) break;
                } else if (!report(dan::ShardState::assigned)
                    || !prepare_artifact(options, *assignment, metadata, report, ui, error)) {
                    log_event("artifact preparation failed: " + error);
                    if (ui) ui->update([](auto& screen) {
                        screen.status = dan::ProviderUiStatus::recovering;
                        screen.message = "Model preparation was interrupted. DAN will retry safely.";
                    });
                    else std::fprintf(stderr, "Artifact preparation failed: %s\n", error.c_str());
                    if (!report(dan::ShardState::error)) break;
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
                    log_event("invalid managed shard command received");
                    if (ui) ui->update([](auto& screen) {
                        screen.status = dan::ProviderUiStatus::action_required;
                        screen.message = "DAN received an incompatible worker command.";
                    }); else std::fprintf(stderr, "Invalid managed shard command\n");
                    fatal = true;
                    break;
                }
                if (unload) {
                    worker.stop();
                    std::string error;
                    if (!verified(artifact_path(options, *assignment), assignment->hash, error)) {
                        worker.last_error = error.empty() ? "cached artifact is invalid" : error;
                        if (!report(dan::ShardState::error)) break;
                    } else if (!report(dan::ShardState::cached)) break;
                    continue;
                }
                if (worker.running()) {
                    if (!report(dan::ShardState::ready)) break;
                    continue;
                }
                std::string error;
                const bool already_cached = verified(
                    artifact_path(options, *assignment), assignment->hash, error);
                if (!already_cached
                    && !prepare_artifact(options, *assignment, metadata, report, ui, error)) {
                    log_event("artifact recovery failed: " + error);
                    if (ui) ui->update([](auto& screen) {
                        screen.status = dan::ProviderUiStatus::recovering;
                        screen.message = "Model preparation was interrupted. DAN will retry safely.";
                    });
                    else std::fprintf(stderr, "Artifact recovery failed: %s\n", error.c_str());
                    if (!report(dan::ShardState::error)) break;
                    continue;
                }
                if (already_cached && state == dan::ShardState::error
                    && !report(dan::ShardState::cached)) break;
                if (!wait_for_gpu_memory(options, metadata, ui, error)) {
                    if (!report(dan::ShardState::error)) break;
                    if (ui) ui->update([](auto& screen) {
                        screen.status = dan::ProviderUiStatus::waiting_gpu;
                        screen.message = "Your GPU is busy. DAN will retry when memory is available.";
                    });
                    continue;
                }
                if (!report(dan::ShardState::loading)) break;
                bool started = false;
                for (int attempt = 1; attempt <= 3 && !started; ++attempt) {
                    started = worker.start(options, error);
                    if (!started && attempt < 3) {
                        if (ui) ui->update([](auto& screen) {
                            screen.status = dan::ProviderUiStatus::recovering;
                            screen.message = "GPU worker could not start. Retrying automatically...";
                        });
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    }
                }
                if (!started) {
                    worker.last_error = error;
                    log_event("worker start failed: " + error);
                    if (ui) ui->update([](auto& screen) {
                        screen.status = dan::ProviderUiStatus::action_required;
                        screen.message = "DAN could not start the GPU worker.";
                    });
                    else std::fprintf(stderr, "Managed worker failed: %s\n", error.c_str());
                    if (!report(dan::ShardState::error)) break;
                } else if (!report(dan::ShardState::ready)) break;
                else log_event("worker ready");
                continue;
            }
            log_event("unexpected or malformed control message");
            if (ui) ui->update([](auto& screen) {
                screen.status = dan::ProviderUiStatus::action_required;
                screen.message = "DAN received an incompatible network message.";
            }); else std::fprintf(stderr, "Unexpected or malformed control message\n");
            fatal = true;
            break;
        }
        heartbeat.request_stop();
        log_event("coordinator connection lost; reconnecting");
        {
            std::lock_guard lock(send_mutex);
            connected = false;
            dan::platform::shutdown_socket(coordinator);
            dan::platform::close_socket(coordinator);
        }
        if (fatal) return 1;
        if (dan::platform::stop_requested() || (bye && options.reconnect_delay.count() == 0)) break;
        if (options.reconnect_delay.count() == 0) return 1;
        wait_to_reconnect();
    }
    worker.stop();
    return 0;
}
