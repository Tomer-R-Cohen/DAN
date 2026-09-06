#include "control_plane.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace fs = std::filesystem;

struct Gpu {
    std::size_t index = 0;
    std::string name;
    std::size_t total_vram_mib = 0;
    std::string uuid;
};

struct Options {
    std::string coordinator;
    fs::path cache_dir;
    fs::path state_dir;
    std::string rpc_worker;
    std::optional<std::size_t> device;
    std::size_t reserve_vram_mib = 1536;
    std::string provider_name;
    std::string advertise_host;
    std::string worker_port;
    std::size_t reconnect_seconds = 2;
    std::string nvidia_smi = "nvidia-smi";
    bool check_only = false;
};

std::string trim(std::string_view value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

bool safe_name(std::string_view value)
{
    if (value.empty() || value == "." || value == "..") return false;
    for (const unsigned char byte : value) {
        if (!std::isalnum(byte) && byte != '.' && byte != '_' && byte != '-') return false;
    }
    return true;
}

bool set_option(Options& options, std::string_view key, const std::string& value,
    std::string& error)
{
    std::size_t number = 0;
    if (key == "coordinator") options.coordinator = value;
    else if (key == "cache_dir") options.cache_dir = value;
    else if (key == "state_dir") options.state_dir = value;
    else if (key == "rpc_worker") options.rpc_worker = value;
    else if (key == "device") {
        if (!dan::parse_size(value, number)) { error = "device must be a GPU index"; return false; }
        options.device = number;
    } else if (key == "reserve_vram_mib") {
        if (!dan::parse_size(value, options.reserve_vram_mib)) {
            error = "reserve_vram_mib must be an integer"; return false;
        }
    } else if (key == "provider_name") options.provider_name = value;
    else if (key == "advertise_host") options.advertise_host = value;
    else if (key == "worker_port") options.worker_port = value;
    else if (key == "reconnect_seconds") {
        if (!dan::parse_size(value, options.reconnect_seconds)
            || options.reconnect_seconds == 0) {
            error = "reconnect_seconds must be positive"; return false;
        }
    } else if (key == "nvidia_smi") options.nvidia_smi = value;
    else { error = "unknown provider configuration key: " + std::string(key); return false; }
    return true;
}

bool load_config(const fs::path& path, Options& options, std::string& error)
{
    std::ifstream input(path);
    if (!input) { error = "could not open provider config: " + path.string(); return false; }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string clean = trim(line);
        if (clean.empty() || clean[0] == '#') continue;
        const std::size_t equals = clean.find('=');
        if (equals == std::string::npos || equals == 0
            || !set_option(options, trim(std::string_view(clean).substr(0, equals)),
                trim(std::string_view(clean).substr(equals + 1)), error)) {
            if (error.empty()) error = "malformed provider config";
            error += " at line " + std::to_string(line_number);
            return false;
        }
    }
    return true;
}

bool command_output(const std::string& executable, std::string& output)
{
    int pipe_fds[2];
    if (pipe(pipe_fds) == -1) return false;
    const pid_t child = fork();
    if (child == -1) { close(pipe_fds[0]); close(pipe_fds[1]); return false; }
    if (child == 0) {
        close(pipe_fds[0]);
        if (dup2(pipe_fds[1], STDOUT_FILENO) == -1) _exit(127);
        close(pipe_fds[1]);
        execlp(executable.c_str(), executable.c_str(),
            "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader,nounits",
            nullptr);
        _exit(127);
    }
    close(pipe_fds[1]);
    char buffer[4096];
    while (output.size() < 1024 * 1024) {
        const ssize_t count = read(pipe_fds[0], buffer, sizeof(buffer));
        if (count == -1 && errno == EINTR) continue;
        if (count <= 0) break;
        output.append(buffer, static_cast<std::size_t>(count));
    }
    close(pipe_fds[0]);
    int status = 0;
    while (waitpid(child, &status, 0) == -1 && errno == EINTR) { }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 && !output.empty();
}

bool parse_gpus(std::string_view output, std::vector<Gpu>& gpus, std::string& error)
{
    while (!output.empty()) {
        const std::size_t newline = output.find('\n');
        const std::string line = trim(output.substr(0, newline));
        if (!line.empty()) {
            std::vector<std::string> fields;
            std::string_view remaining = line;
            while (true) {
                const std::size_t comma = remaining.find(',');
                fields.push_back(trim(remaining.substr(0, comma)));
                if (comma == std::string_view::npos) break;
                remaining.remove_prefix(comma + 1);
            }
            Gpu gpu;
            if (fields.size() != 4 || !dan::parse_size(fields[0], gpu.index)
                || fields[1].empty() || !dan::parse_size(fields[2], gpu.total_vram_mib)
                || gpu.total_vram_mib == 0 || fields[3].empty()) {
                error = "malformed nvidia-smi GPU output";
                return false;
            }
            gpu.name = std::move(fields[1]);
            gpu.uuid = std::move(fields[3]);
            gpus.push_back(std::move(gpu));
        }
        if (newline == std::string_view::npos) break;
        output.remove_prefix(newline + 1);
    }
    if (gpus.empty()) { error = "nvidia-smi reported no NVIDIA GPUs"; return false; }
    std::sort(gpus.begin(), gpus.end(), [](const Gpu& left, const Gpu& right) {
        return left.index < right.index;
    });
    return true;
}

bool private_ipv4(std::string_view text)
{
    in_addr address{};
    if (inet_pton(AF_INET, std::string(text).c_str(), &address) != 1) return false;
    const std::uint32_t host = ntohl(address.s_addr);
    const unsigned first = host >> 24;
    const unsigned second = (host >> 16) & 0xff;
    return first == 10 || first == 127 || (first == 172 && second >= 16 && second <= 31)
        || (first == 192 && second == 168) || (first == 100 && second >= 64 && second <= 127);
}

bool split_endpoint(std::string_view endpoint, std::string& host, std::string& port)
{
    const std::size_t colon = endpoint.rfind(':');
    std::size_t number = 0;
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == endpoint.size()
        || !dan::parse_size(endpoint.substr(colon + 1), number)
        || number == 0 || number > 65535) return false;
    host = std::string(endpoint.substr(0, colon));
    port = std::string(endpoint.substr(colon + 1));
    return true;
}

std::string free_port(std::string_view host)
{
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == -1) return {};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    if (inet_pton(AF_INET, std::string(host).c_str(), &address.sin_addr) != 1
        || bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == -1) {
        close(socket_fd);
        return {};
    }
    socklen_t size = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &size) == -1) {
        close(socket_fd);
        return {};
    }
    // ponytail: brief bind/exec race; retain a reservation only if testnet collisions appear.
    close(socket_fd);
    return std::to_string(ntohs(address.sin_port));
}

std::string provider_id(const fs::path& state_dir, std::string& error)
{
    const fs::path path = state_dir / "provider-id";
    std::ifstream existing(path);
    std::string id;
    if (existing && std::getline(existing, id)) {
        id = trim(id);
        if (safe_name(id)) return id;
        error = "invalid persistent provider ID: " + path.string();
        return {};
    }
    std::error_code filesystem_error;
    fs::create_directories(state_dir, filesystem_error);
    if (filesystem_error) { error = "could not create provider state directory"; return {}; }
    std::random_device random;
    std::ostringstream generated;
    generated << "node-" << std::hex << std::setfill('0');
    for (int byte = 0; byte < 6; ++byte) generated << std::setw(2) << (random() & 0xff);
    id = generated.str();
    std::ofstream output(path, std::ios::trunc);
    if (!output || !(output << id << '\n')) {
        error = "could not persist provider ID: " + path.string();
        return {};
    }
    return id;
}

void usage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s [--config FILE] [--coordinator HOST:PORT] [--rpc-worker PATH] "
        "[--advertise-host PRIVATE_IP] [--device INDEX] [--reserve-vram-mib N] "
        "[--provider-name NAME] [--cache-dir DIR] [--check]\n", program);
}
}

int main(int argc, char* argv[])
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* home = std::getenv("HOME");
    Options options;
    options.state_dir = home ? fs::path(home) / ".dan" : fs::path(".dan");
    options.cache_dir = options.state_dir / "models";
    fs::path config = options.state_dir / "provider.conf";
    bool explicit_config = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--config" && index + 1 < argc) {
            config = argv[++index];
            explicit_config = true;
        }
    }
    std::string error;
    if ((explicit_config || fs::exists(config)) && !load_config(config, options, error)) {
        std::fprintf(stderr, "Provider configuration error: %s\n", error.c_str());
        return 1;
    }
    for (int index = 1; index < argc;) {
        const std::string option = argv[index++];
        if (option == "--help") { usage(argv[0]); return 0; }
        if (option == "--check") { options.check_only = true; continue; }
        if (index >= argc) { usage(argv[0]); return 1; }
        const std::string value = argv[index++];
        if (option == "--config") continue;
        std::string key;
        if (option == "--coordinator") key = "coordinator";
        else if (option == "--cache-dir") key = "cache_dir";
        else if (option == "--state-dir") key = "state_dir";
        else if (option == "--rpc-worker") key = "rpc_worker";
        else if (option == "--device") key = "device";
        else if (option == "--reserve-vram-mib") key = "reserve_vram_mib";
        else if (option == "--provider-name") key = "provider_name";
        else if (option == "--advertise-host") key = "advertise_host";
        else if (option == "--worker-port") key = "worker_port";
        else if (option == "--reconnect-seconds") key = "reconnect_seconds";
        else if (option == "--nvidia-smi") key = "nvidia_smi";
        else { std::fprintf(stderr, "Unknown provider option: %s\n", option.c_str()); return 1; }
        if (!set_option(options, key, value, error)) {
            std::fprintf(stderr, "Provider configuration error: %s\n", error.c_str());
            return 1;
        }
    }

    std::string coordinator_host;
    std::string coordinator_port;
    if (!split_endpoint(options.coordinator, coordinator_host, coordinator_port)
        || options.rpc_worker.empty() || access(options.rpc_worker.c_str(), X_OK) == -1
        || !private_ipv4(options.advertise_host)
        || (!options.provider_name.empty() && !safe_name(options.provider_name))) {
        std::fprintf(stderr, "Provider setup requires coordinator=HOST:PORT, an executable "
            "rpc_worker, a numeric private advertise_host, and a safe provider_name\n");
        return 1;
    }
    std::string gpu_output;
    std::vector<Gpu> gpus;
    if (!command_output(options.nvidia_smi, gpu_output)
        || !parse_gpus(gpu_output, gpus, error)) {
        std::fprintf(stderr, "NVIDIA GPU detection failed%s%s\n",
            error.empty() ? "" : ": ", error.c_str());
        return 1;
    }
    const Gpu* selected = &gpus.front();
    if (options.device) {
        selected = nullptr;
        for (const auto& gpu : gpus) if (gpu.index == *options.device) selected = &gpu;
        if (!selected) {
            std::fprintf(stderr, "Configured CUDA device %zu was not reported by nvidia-smi\n",
                *options.device);
            return 1;
        }
    }
    if (options.reserve_vram_mib >= selected->total_vram_mib) {
        std::fprintf(stderr, "VRAM reserve must be smaller than detected total VRAM\n");
        return 1;
    }
    if (options.worker_port.empty()) options.worker_port = free_port(options.advertise_host);
    std::size_t worker_port_number = 0;
    if (!dan::parse_size(options.worker_port, worker_port_number)
        || worker_port_number == 0 || worker_port_number > 65535) {
        std::fprintf(stderr, "Could not select a valid managed worker port\n");
        return 1;
    }
    const std::string id = provider_id(options.state_dir, error);
    if (id.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    const std::size_t usable_vram = selected->total_vram_mib - options.reserve_vram_mib;
    std::printf("DAN Provider\n\nProvider ID: %s\nName: %s\nGPU: %s\nGPU UUID: %s\nDevice: CUDA%zu\n"
        "VRAM total: %zu MiB\nReserved: %zu MiB\nAvailable to DAN: %zu MiB\n"
        "Coordinator: %s\nCache: %s\nWorker: STOPPED (%s:%s)\n",
        id.c_str(), options.provider_name.empty() ? "-" : options.provider_name.c_str(),
        selected->name.c_str(), selected->uuid.c_str(), selected->index, selected->total_vram_mib,
        options.reserve_vram_mib, usable_vram, options.coordinator.c_str(),
        options.cache_dir.c_str(), options.advertise_host.c_str(), options.worker_port.c_str());
    if (options.check_only) return 0;

    std::error_code filesystem_error;
    const fs::path executable = fs::read_symlink("/proc/self/exe", filesystem_error)
        .parent_path() / "managed_provider";
    if (filesystem_error || access(executable.c_str(), X_OK) == -1) {
        std::fprintf(stderr, "Could not find managed_provider beside dan-provider\n");
        return 1;
    }
    std::vector<std::string> arguments{executable.string(), "--id", id};
    if (!options.provider_name.empty()) {
        arguments.insert(arguments.end(), {"--name", options.provider_name});
    }
    arguments.insert(arguments.end(), {"--gpu", selected->name,
        "--vram", std::to_string(selected->total_vram_mib) + " MiB total",
        "--vram-mib", std::to_string(usable_vram), "--cache-dir", options.cache_dir.string(),
        "--worker", options.rpc_worker, "--worker-host", options.advertise_host,
        "--worker-port", options.worker_port, "--worker-device",
        "CUDA" + std::to_string(selected->index), "--host", coordinator_host,
        "--port", coordinator_port, "--reconnect-seconds",
        std::to_string(options.reconnect_seconds)});
    std::vector<char*> exec_arguments;
    for (auto& argument : arguments) exec_arguments.push_back(argument.data());
    exec_arguments.push_back(nullptr);
    execv(executable.c_str(), exec_arguments.data());
    std::perror("Could not start managed_provider");
    return 1;
}
