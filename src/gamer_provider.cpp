#include "control_plane.hpp"
#include "platform.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
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
    std::string stage_worker;
    std::optional<std::size_t> device;
    std::size_t reserve_vram_mib = 1536;
    std::string provider_name;
    std::string advertise_host;
    std::string nvidia_smi = "nvidia-smi";
    bool check_only = false;
    bool verbose = false;
    bool manage_network = false;
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
    else if (key == "stage_worker") options.stage_worker = value;
    else if (key == "device") {
        if (!dan::parse_size(value, number)) { error = "device must be a GPU index"; return false; }
        options.device = number;
    } else if (key == "reserve_vram_mib") {
        if (!dan::parse_size(value, options.reserve_vram_mib)) {
            error = "reserve_vram_mib must be an integer"; return false;
        }
    } else if (key == "provider_name") options.provider_name = value;
    else if (key == "advertise_host") options.advertise_host = value;
    else if (key == "nvidia_smi") options.nvidia_smi = value;
    else if (key == "network" && value == "tailscale") options.manage_network = true;
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

bool command_output(const std::string& executable, std::string& output, std::string& error)
{
    return dan::platform::run({executable, "--query-gpu=index,name,memory.total,uuid",
        "--format=csv,noheader,nounits"}, error, &output) && !output.empty();
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
    return dan::platform::private_ipv4(text);
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

bool tailscale_ip(bool configure, const fs::path& package_dir, std::string& address,
    std::string& error)
{
    fs::path client = dan::platform::network_client_executable();
    if (!dan::platform::executable_file(client)) {
        if (!configure) { error = "official Tailscale is not installed"; return false; }
        std::cout << "Installing the official private-network component...\n"
            << "Approve the Windows prompt to continue.\n";
        if (!dan::platform::install_msi(package_dir / "runtime" / "tailscale.msi", error))
            return false;
        client = dan::platform::network_client_executable();
        if (!dan::platform::executable_file(client)) {
            error = "Tailscale installation completed but its client was not found"; return false;
        }
    }
    const auto discover = [&] {
        std::string output;
        if (!dan::platform::run({client.string(), "ip", "-4"}, error, &output)) return false;
        address = trim(std::string_view(output).substr(0, output.find('\n')));
        return private_ipv4(address);
    };
    if (discover()) return true;
    if (!configure) { error = "private network authentication is incomplete"; return false; }
    if (!dan::platform::start_network_client(client, error)) return false;
    for (int attempt = 0; attempt < 10; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (discover()) return true;
    }
    std::cout << "Connecting the private DAN network...\n"
        << "A browser may open once so you can approve secure enrollment.\n";
    std::string onboarding;
    dan::platform::run({client.string(), "up", "--timeout", "2s"}, error, &onboarding);
    if (discover()) return true;
    for (int attempt = 0; attempt < 3
        && onboarding.find("https://login.tailscale.com/") == std::string::npos; ++attempt) {
        onboarding.clear();
        dan::platform::run({client.string(), "up", "--force-reauth", "--timeout", "2s"},
            error, &onboarding);
        if (onboarding.find("https://login.tailscale.com/") == std::string::npos)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    const std::size_t start = onboarding.find("https://login.tailscale.com/");
    if (start == std::string::npos) {
        error = "the private network service did not provide a sign-in page; run DAN again";
        return false;
    }
    const std::size_t end = onboarding.find_first_of(" \t\r\n\"", start);
    const std::string_view url = std::string_view(onboarding).substr(start,
        end == std::string::npos ? end : end - start);
    if (!dan::platform::open_url(url, error)) return false;
    std::cout << "Finish the secure sign-in in your browser. DAN will continue automatically...\n";
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (!dan::platform::stop_requested() && std::chrono::steady_clock::now() < deadline) {
        if (discover()) return true;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (discover()) return true;
    error = "private network setup did not finish; run DAN again to retry";
    return false;
}

bool first_run_setup(const fs::path& config, const fs::path& package_dir,
    const Gpu& gpu, Options& options, std::string& error)
{
    std::cout << "DAN Provider\n\nGPU detected:\n" << gpu.name << '\n'
        << gpu.total_vram_mib << " MiB VRAM\n\nSetting up private DAN network...\n";
    if (!tailscale_ip(true, package_dir, options.advertise_host, error)) return false;
    options.manage_network = true;
    std::cout << "Private network: Connected\n\n"
        << "Ask Tomer for the DAN coordinator address.\n"
        << "Coordinator (for example 100.80.10.1:50200): ";
    if (!std::getline(std::cin, options.coordinator)
        || options.coordinator.find_first_of("\r\n") != std::string::npos) {
        error = "invalid coordinator address";
        return false;
    }
    options.coordinator = trim(options.coordinator);
    options.provider_name = "windows-pc";
    std::error_code filesystem_error;
    fs::create_directories(options.state_dir, filesystem_error);
    if (filesystem_error) { error = "could not create the DAN data directory"; return false; }
    std::ofstream output(config, std::ios::trunc);
    if (!output || !(output << "coordinator=" << options.coordinator
        << "\nprovider_name=" << options.provider_name
        << "\nnetwork=tailscale"
        << "\nadvertise_host=" << options.advertise_host
        << "\ncache_dir=" << options.cache_dir.string()
        << "\nstage_worker=runtime\\dan-stage-worker.exe"
        << "\nreserve_vram_mib=" << options.reserve_vram_mib << '\n')) {
        error = "could not create provider configuration";
        return false;
    }
    std::cout << "\nSetup saved. Starting DAN...\n\n";
    return true;
}

void usage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s [--config FILE] [--coordinator HOST:PORT] [--stage-worker PATH] "
        "[--advertise-host PRIVATE_IP] [--device INDEX] [--reserve-vram-mib N] "
        "[--provider-name NAME] [--cache-dir DIR] [--check] [--verbose]\n", program);
}
}

int provider_main(int argc, char* argv[])
{
    std::string error;
    if (!dan::platform::initialize(error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    struct Cleanup { ~Cleanup() { dan::platform::cleanup(); } } cleanup;
    dan::platform::install_stop_handlers();
    dan::platform::configure_output();
    if (dan::platform::is_windows() && argc == 1
        && !dan::platform::acquire_single_instance("DAN.Provider.v1")) {
        std::printf("DAN Provider is already running.\n\n"
            "Use the existing DAN Provider window.\n");
        return 0;
    }
    Options options;
    options.state_dir = dan::platform::data_directory();
    options.cache_dir = options.state_dir / "models";
    fs::path package_dir;
    if (dan::platform::is_windows()) {
        package_dir = dan::platform::current_executable(error).parent_path();
        if (!error.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
        options.stage_worker = (package_dir / "runtime" / "dan-stage-worker.exe").string();
    }
    fs::path config = options.state_dir / "provider-v1.0.1.conf";
    bool explicit_config = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--config" && index + 1 < argc) {
            config = argv[++index];
            explicit_config = true;
        }
    }
    if ((explicit_config || fs::exists(config)) && !load_config(config, options, error)) {
        std::fprintf(stderr, "Provider configuration error: %s\n", error.c_str());
        return 1;
    }
    for (int index = 1; index < argc;) {
        const std::string option = argv[index++];
        if (option == "--help") { usage(argv[0]); return 0; }
        if (option == "--check") { options.check_only = true; continue; }
        if (option == "--verbose") { options.verbose = true; continue; }
        if (index >= argc) { usage(argv[0]); return 1; }
        const std::string value = argv[index++];
        if (option == "--config") continue;
        std::string key;
        if (option == "--coordinator") key = "coordinator";
        else if (option == "--cache-dir") key = "cache_dir";
        else if (option == "--state-dir") key = "state_dir";
        else if (option == "--stage-worker") key = "stage_worker";
        else if (option == "--device") key = "device";
        else if (option == "--reserve-vram-mib") key = "reserve_vram_mib";
        else if (option == "--provider-name") key = "provider_name";
        else if (option == "--advertise-host") key = "advertise_host";
        else if (option == "--nvidia-smi") key = "nvidia_smi";
        else { std::fprintf(stderr, "Unknown provider option: %s\n", option.c_str()); return 1; }
        if (!set_option(options, key, value, error)) {
            std::fprintf(stderr, "Provider configuration error: %s\n", error.c_str());
            return 1;
        }
    }
    if (dan::platform::is_windows() && fs::path(options.stage_worker).is_relative()) {
        options.stage_worker = (package_dir / options.stage_worker).lexically_normal().string();
    }
    std::string gpu_output;
    std::vector<Gpu> gpus;
    if (!command_output(options.nvidia_smi, gpu_output, error)
        || !parse_gpus(gpu_output, gpus, error)) {
        if (dan::platform::is_windows() && argc == 1) {
            std::fprintf(stderr, "No supported NVIDIA GPU was found.\n\n"
                "DAN currently supports NVIDIA GPUs only.\n");
        } else std::fprintf(stderr, "NVIDIA GPU detection failed%s%s\n",
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
    const bool first_run = dan::platform::is_windows() && argc == 1 && !fs::exists(config);
    if (first_run && !first_run_setup(config, package_dir, *selected, options, error)) {
        std::fprintf(stderr, "First setup failed: %s\n", error.c_str());
        return 1;
    }
    if (options.manage_network && !first_run
        && !tailscale_ip(!options.check_only, package_dir, options.advertise_host, error)) {
        std::fprintf(stderr, "Private network check failed: %s\n", error.c_str());
        return 1;
    }
    std::string coordinator_host;
    std::string coordinator_port;
    if (!split_endpoint(options.coordinator, coordinator_host, coordinator_port)
        || options.stage_worker.empty() || !dan::platform::executable_file(options.stage_worker)
        || !private_ipv4(options.advertise_host)
        || (!options.provider_name.empty() && !safe_name(options.provider_name))) {
        std::fprintf(stderr, "Provider setup requires coordinator=HOST:PORT, a bundled "
            "provider-owned runtime, a private network address, and a safe provider_name\n");
        return 1;
    }
    const std::string id = provider_id(options.state_dir, error);
    if (id.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    std::error_code filesystem_error;
    fs::create_directories(options.cache_dir, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "Could not create cache directory: %s\n",
            filesystem_error.message().c_str()); return 1;
    }
    const std::size_t usable_vram = selected->total_vram_mib - options.reserve_vram_mib;
    const std::string cache_display = options.cache_dir.string();
    if (dan::platform::is_windows() && !options.verbose) {
        std::printf("DAN Provider\n\nPC: %s\nGPU: %s\nAvailable to DAN: %zu MiB\n\n"
            "Local configuration: OK\nDiagnostics: %s\nStatus: Starting\n"
            "Connecting to DAN automatically...\n\n",
            options.provider_name.empty() ? "Windows PC" : options.provider_name.c_str(),
            selected->name.c_str(), usable_vram,
            (dan::platform::data_directory() / "logs" / "provider.log").string().c_str());
    } else {
        std::printf("DAN Provider\n\nProvider ID: %s\nName: %s\nGPU: %s\nGPU UUID: %s\nDevice: CUDA%zu\n"
            "VRAM total: %zu MiB\nReserved: %zu MiB\nAvailable to DAN: %zu MiB\n"
            "Coordinator: %s\nCache: %s\nRuntime: provider-owned v1.0.1\n",
            id.c_str(), options.provider_name.empty() ? "-" : options.provider_name.c_str(),
            selected->name.c_str(), selected->uuid.c_str(), selected->index,
            selected->total_vram_mib, options.reserve_vram_mib, usable_vram,
            options.coordinator.c_str(), cache_display.c_str());
    }
    if (options.check_only) {
        std::printf("Bundled provider-owned runtime: OK\n");
        return 0;
    }

    std::vector<std::string> arguments{options.stage_worker,
        "--coordinator", options.coordinator,
        "--provider-id", id,
        "--gpu", selected->name,
        "--vram-mib", std::to_string(usable_vram),
        "--cache-dir", options.cache_dir.string(),
        "--tui"};
    const int result = dan::platform::replace_with_provider(arguments, error);
    if (result != 0) std::fprintf(stderr, "%s\n", error.c_str());
    return result;
}

int main(int argc, char* argv[])
{
    const int result = provider_main(argc, argv);
    if (result != 0 && argc == 1 && dan::platform::is_windows()) {
        std::fputs("\nPress Enter to close DAN...", stderr);
        std::string ignored;
        std::getline(std::cin, ignored);
    }
    return result;
}
