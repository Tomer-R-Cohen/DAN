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
    std::string coordinator_peer;
    std::vector<std::string> relays;
    std::string host_manifest;
    std::string serve_endpoint;
    std::string provider_listen = "0.0.0.0:50201";
    fs::path metadata_cache;
    fs::path cache_dir;
    fs::path state_dir;
    std::string stage_worker;
    std::string sidecar;
    std::optional<std::size_t> device;
    std::size_t reserve_vram_mib = 1536;
    std::string provider_name;
    std::string advertise_host;
    std::string nvidia_smi = "nvidia-smi";
    std::size_t ring_port = 50202;
    bool check_only = false;
    bool verbose = false;
    bool manage_network = false;
    bool peer_network = false;
    // network=dht: no coordinator. Join the DAN DHT through `bootstrap`, keep a relay
    // reservation (the bootstrap nodes unless `relay` is given), and serve the models in
    // `catalog` to clients that reserve this worker.
    bool dht_network = false;
    std::vector<std::string> bootstrap;
    std::vector<std::string> catalog;
    std::size_t max_context = 4096;
    std::size_t max_sessions = 1;
    std::size_t listen_port = 0;
    bool simulate_nat = false;  // test only: the sidecar accepts only relayed connections
    // Persistent replicas: "auto" also runs this node's replica owner (dan-client --form),
    // which forms a replica around this worker whenever it is free and serves chats on it.
    bool replica = false;
    std::string replica_client;           // dan-client; default: next to dan-provider
    std::size_t replica_sessions = 1;     // sessions per replica (KV is planned for them)
    std::size_t replica_max_edge_rtt_ms = 150;
    bool replica_relay_edges = true;
    bool replica_speculate = false;
    std::size_t replica_min_stages = 1;   // tests: split even a model one GPU could hold
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
    else if (key == "host_manifest") options.host_manifest = value;
    else if (key == "serve") options.serve_endpoint = value;
    else if (key == "provider_listen") options.provider_listen = value;
    else if (key == "metadata_cache") options.metadata_cache = value;
    else if (key == "coordinator_peer") {
        options.coordinator_peer = value;
        options.peer_network = true;
    }
    else if (key == "relay") {
        if (value.empty()) { error = "relay must be a peer multiaddress"; return false; }
        options.relays.push_back(value);
    }
    else if (key == "cache_dir") options.cache_dir = value;
    else if (key == "state_dir") options.state_dir = value;
    else if (key == "stage_worker") options.stage_worker = value;
    else if (key == "sidecar") options.sidecar = value;
    else if (key == "device") {
        if (!dan::parse_size(value, number)) { error = "device must be a GPU index"; return false; }
        options.device = number;
    } else if (key == "reserve_vram_mib") {
        if (!dan::parse_size(value, options.reserve_vram_mib)) {
            error = "reserve_vram_mib must be an integer"; return false;
        }
    } else if (key == "ring_port") {
        if (!dan::parse_size(value, options.ring_port) || options.ring_port == 0
            || options.ring_port > 65535) {
            error = "ring_port must be a TCP port"; return false;
        }
    } else if (key == "provider_name") options.provider_name = value;
    else if (key == "advertise_host") options.advertise_host = value;
    else if (key == "nvidia_smi") options.nvidia_smi = value;
    else if (key == "network" && value == "tailscale") options.manage_network = true;
    else if (key == "network" && value == "libp2p") options.peer_network = true;
    else if (key == "network" && value == "dht") options.dht_network = true;
    else if (key == "bootstrap") {
        if (!value.starts_with('/') || value.find("/p2p/") == std::string::npos) {
            error = "bootstrap must be a peer multiaddress ending in /p2p/<PeerID>"; return false;
        }
        options.bootstrap.push_back(value);
    }
    else if (key == "simulate_nat") options.simulate_nat = value == "true";
    else if (key == "replica") {
        if (value != "auto" && value != "off") { error = "replica must be auto or off"; return false; }
        options.replica = value == "auto";
    } else if (key == "replica_client") options.replica_client = value;
    else if (key == "replica_sessions") {
        if (!dan::parse_size(value, options.replica_sessions) || options.replica_sessions == 0) {
            error = "replica_sessions must be a positive integer"; return false;
        }
    } else if (key == "replica_max_edge_rtt_ms") {
        if (!dan::parse_size(value, options.replica_max_edge_rtt_ms)
            || options.replica_max_edge_rtt_ms == 0) {
            error = "replica_max_edge_rtt_ms must be a positive integer"; return false;
        }
    } else if (key == "replica_relay_edges") options.replica_relay_edges = value != "false";
    else if (key == "replica_speculate") options.replica_speculate = value == "true";
    else if (key == "replica_min_stages") {
        if (!dan::parse_size(value, options.replica_min_stages) || options.replica_min_stages == 0) {
            error = "replica_min_stages must be a positive integer"; return false;
        }
    }
    else if (key == "catalog") {
        if (value.empty()) { error = "catalog must be a model manifest path"; return false; }
        options.catalog.push_back(value);
    } else if (key == "max_context") {
        if (!dan::parse_size(value, options.max_context) || options.max_context == 0) {
            error = "max_context must be a positive integer"; return false;
        }
    } else if (key == "max_sessions") {
        if (!dan::parse_size(value, options.max_sessions) || options.max_sessions == 0) {
            error = "max_sessions must be a positive integer"; return false;
        }
    } else if (key == "listen_port") {
        if (!dan::parse_size(value, options.listen_port) || options.listen_port > 65535) {
            error = "listen_port must be 0 or a port"; return false;
        }
    }
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

// Starts the sidecar and waits for its ready file: the first line is our PeerID, the
// following lines our peer addresses.
bool start_sidecar(dan::platform::Process& sidecar, std::vector<std::string> arguments,
    const fs::path& state_dir, const std::string& id, std::vector<std::string>& addresses,
    std::string& error)
{
    std::error_code filesystem_error;
    const fs::path ready_file = state_dir / ("sidecar-ready-"
        + std::to_string(dan::platform::process_id()) + ".txt");
    fs::remove(ready_file, filesystem_error);
    arguments.insert(arguments.end(), {"-ready-file", ready_file.string(), "-log",
        (state_dir / "logs" / "sidecar.log").string()});
    if (!sidecar.start(arguments, error, false, true)) {
        error = "could not start encrypted networking: " + error;
        return false;
    }
    bool ready = false;
    // Home nodes wait for a relay reservation before they are ready.
    for (int attempt = 0; attempt < 1400 && sidecar.running(); ++attempt) {
        if (fs::exists(ready_file)) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) { error = "encrypted networking did not start; see sidecar.log"; return false; }
    std::ifstream input(ready_file);
    std::string ready_id;
    if (!std::getline(input, ready_id) || trim(ready_id) != id) {
        error = "encrypted networking returned the wrong provider identity";
        return false;
    }
    for (std::string address; std::getline(input, address);) {
        address = trim(address);
        if (address.empty()) continue;
        if (!address.starts_with('/') || address.find("/p2p/" + id) == std::string::npos) {
            error = "encrypted networking returned an invalid peer address";
            return false;
        }
        addresses.push_back(address);
    }
    input.close();
    fs::remove(ready_file, filesystem_error);
    return true;
}

bool reserve_local_ports(std::size_t count, std::vector<std::string>& ports)
{
    while (ports.size() < count) {
        const std::string port = dan::platform::free_tcp_port("127.0.0.1");
        if (port.empty()) return false;
        if (std::find(ports.begin(), ports.end(), port) == ports.end()) ports.push_back(port);
    }
    return true;
}

void usage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s [--config FILE] [--coordinator HOST:PORT] [--stage-worker PATH] "
        "[--host-coordinator MANIFEST --serve HOST:PORT [--provider-listen HOST:PORT]] "
        "[--coordinator-peer MULTIADDR] [--relay MULTIADDR ...] [--sidecar PATH] [--ring-port PORT] "
        "[--advertise-host PRIVATE_IP] [--device INDEX] [--reserve-vram-mib N] "
        "[--provider-name NAME] [--cache-dir DIR] [--check] [--verbose]\n"
        "   or: %s --network dht --bootstrap MULTIADDR [...] --catalog MANIFEST [...] "
        "[--relay MULTIADDR ...] [--max-context N] [--max-sessions N] [--listen-port PORT] "
        "[--device INDEX] [--reserve-vram-mib N] [--cache-dir DIR] [--check]\n", program, program);
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
    const fs::path package_dir = dan::platform::current_executable(error).parent_path();
    if (!error.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    options.stage_worker = (package_dir / "runtime"
        / (dan::platform::is_windows() ? "dan-stage-worker.exe" : "dan-stage-worker")).string();
    options.sidecar = (package_dir / "runtime"
        / (dan::platform::is_windows() ? "dan-sidecar.exe" : "dan-sidecar")).string();
    fs::path config = options.state_dir / "provider-v1.1.conf";
    const fs::path bundled_config = package_dir / "config" / "provider.conf";
    bool explicit_config = false;
    bool command_line_dht = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--config" && index + 1 < argc) {
            config = argv[++index];
            explicit_config = true;
        } else if (std::string_view(argv[index]) == "--network" && index + 1 < argc) {
            command_line_dht = std::string_view(argv[++index]) == "dht";
        }
    }
    // "--network dht" never picks up a saved coordinator-mode configuration; it uses an
    // explicit --config or the package's config/provider-dht.conf (bootstrap, catalog).
    if (command_line_dht && !explicit_config) config = package_dir / "config" / "provider-dht.conf";
    const fs::path selected_config = explicit_config || fs::exists(config) ? config
        : (command_line_dht ? fs::path{} : bundled_config);
    const bool has_config = fs::exists(selected_config);
    if ((explicit_config || has_config) && !load_config(selected_config, options, error)) {
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
        else if (option == "--host-coordinator") key = "host_manifest";
        else if (option == "--serve") key = "serve";
        else if (option == "--provider-listen") key = "provider_listen";
        else if (option == "--metadata-cache") key = "metadata_cache";
        else if (option == "--coordinator-peer") key = "coordinator_peer";
        else if (option == "--relay") key = "relay";
        else if (option == "--cache-dir") key = "cache_dir";
        else if (option == "--state-dir") key = "state_dir";
        else if (option == "--stage-worker") key = "stage_worker";
        else if (option == "--sidecar") key = "sidecar";
        else if (option == "--device") key = "device";
        else if (option == "--reserve-vram-mib") key = "reserve_vram_mib";
        else if (option == "--ring-port") key = "ring_port";
        else if (option == "--provider-name") key = "provider_name";
        else if (option == "--advertise-host") key = "advertise_host";
        else if (option == "--nvidia-smi") key = "nvidia_smi";
        else if (option == "--network") key = "network";
        else if (option == "--bootstrap") key = "bootstrap";
        else if (option == "--catalog") key = "catalog";
        else if (option == "--max-context") key = "max_context";
        else if (option == "--max-sessions") key = "max_sessions";
        else if (option == "--listen-port") key = "listen_port";
        else { std::fprintf(stderr, "Unknown provider option: %s\n", option.c_str()); return 1; }
        if (!set_option(options, key, value, error)) {
            std::fprintf(stderr, "Provider configuration error: %s\n", error.c_str());
            return 1;
        }
    }
    if (fs::path(options.stage_worker).is_relative()) {
        options.stage_worker = (package_dir / options.stage_worker).lexically_normal().string();
    }
    if (fs::path(options.sidecar).is_relative()) {
        options.sidecar = (package_dir / options.sidecar).lexically_normal().string();
    }
    if (options.replica_client.empty()) {
        options.replica_client = (package_dir
            / (dan::platform::is_windows() ? "dan-client.exe" : "dan-client")).string();
    } else if (fs::path(options.replica_client).is_relative()) {
        options.replica_client = (package_dir / options.replica_client).lexically_normal().string();
    }
    // A relative catalog path means the current directory if it exists there, else the
    // package (where bundled model manifests live).
    for (std::string& manifest : options.catalog) {
        if (fs::path(manifest).is_relative()) {
            manifest = (fs::exists(manifest) ? fs::absolute(manifest)
                : package_dir / manifest).lexically_normal().string();
        }
    }
    if (options.dht_network && options.relays.empty()) options.relays = options.bootstrap;
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
    const bool first_run = dan::platform::is_windows() && argc == 1 && !has_config;
    if (first_run && !first_run_setup(config, package_dir, *selected, options, error)) {
        std::fprintf(stderr, "First setup failed: %s\n", error.c_str());
        return 1;
    }
    if (options.manage_network && !first_run
        && !tailscale_ip(!options.check_only, package_dir, options.advertise_host, error)) {
        std::fprintf(stderr, "Private network check failed: %s\n", error.c_str());
        return 1;
    }
    std::string coordinator_host, coordinator_port;
    const bool hosted = !options.host_manifest.empty();
    const bool direct_ready = split_endpoint(options.coordinator, coordinator_host, coordinator_port);
    const bool dht = options.dht_network;
    if (dht) {
        std::string problem;
        if (options.bootstrap.empty()) problem = "set at least one bootstrap address";
        else if (options.catalog.empty()) problem = "set at least one catalog model manifest";
        else if (!dan::platform::executable_file(options.sidecar)) {
            problem = "the networking program is missing: " + options.sidecar;
        } else if (hosted || options.peer_network || options.manage_network
            || !options.coordinator.empty()) {
            problem = "remove coordinator settings (coordinator, host_manifest, other network modes)";
        }
        for (const std::string& manifest : options.catalog) {
            if (problem.empty() && !fs::is_regular_file(manifest)) {
                problem = "catalog manifest not found: " + manifest;
            }
        }
        if (!problem.empty()) {
            std::fprintf(stderr, "network=dht: %s\n", problem.c_str());
            return 1;
        }
    }
    if ((!dht && !hosted && !options.peer_network && !direct_ready)
        || (options.peer_network && (options.coordinator_peer.empty()
            || !dan::platform::executable_file(options.sidecar)))
        || (hosted && (!options.coordinator.empty() || options.peer_network
            || options.serve_endpoint.empty()))
        || (options.peer_network && options.manage_network)
        || (!options.peer_network && !dht && !options.relays.empty())
        || options.stage_worker.empty() || !dan::platform::executable_file(options.stage_worker)
        || (!options.peer_network && !dht && !private_ipv4(options.advertise_host))
        || (!options.provider_name.empty() && !safe_name(options.provider_name))) {
        std::fprintf(stderr, "Provider setup requires a coordinator, the bundled runtime, "
            "and a safe provider_name\n");
        return 1;
    }
    std::error_code filesystem_error;
    fs::create_directories(options.state_dir / "logs", filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "Could not create state directory: %s\n",
            filesystem_error.message().c_str()); return 1;
    }
    std::string id;
    const fs::path identity_key = options.state_dir / "identity.key";
    if (options.peer_network || dht) {
        if (!dan::platform::run({options.sidecar, "-key", identity_key.string(), "-id"},
                error, &id)) {
            std::fprintf(stderr, "Could not create provider identity: %s\n", error.c_str());
            return 1;
        }
        id = trim(id);
        if (!safe_name(id) || id.size() < 32 || id.size() > 90) {
            std::fprintf(stderr, "The sidecar returned an invalid provider identity\n");
            return 1;
        }
    } else id = provider_id(options.state_dir, error);
    if (id.empty()) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
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
            (dht ? options.state_dir / "logs" / "stage-worker.log"
                : dan::platform::data_directory() / "logs" / "provider.log").string().c_str());
    } else {
        std::printf("DAN Provider\n\nProvider ID: %s\nName: %s\nGPU: %s\nGPU UUID: %s\nDevice: CUDA%zu\n"
            "VRAM total: %zu MiB\nReserved: %zu MiB\nAvailable to DAN: %zu MiB\n"
            "Coordinator: %s\nPeerID: %s\nCache: %s\nRuntime: provider-owned v1.0.1\n",
            id.c_str(), options.provider_name.empty() ? "-" : options.provider_name.c_str(),
            selected->name.c_str(), selected->uuid.c_str(), selected->index,
            selected->total_vram_mib, options.reserve_vram_mib, usable_vram,
            dht ? "none (DAN DHT)" : (hosted ? "hosted on this provider" : (options.peer_network
                ? "encrypted peer network" : options.coordinator.c_str())),
            options.peer_network || dht ? id.c_str() : "-", cache_display.c_str());
    }
    if (options.check_only) {
        std::printf("Bundled provider-owned runtime: OK\n");
        return 0;
    }

    if (dht) {
        if (options.replica && options.replica_sessions > options.max_sessions) {
            std::fprintf(stderr, "replica_sessions cannot exceed max_sessions\n"); return 1;
        }
        if (options.replica && !dan::platform::executable_file(options.replica_client)) {
            std::fprintf(stderr, "replica=auto needs dan-client: %s\n", options.replica_client.c_str());
            return 1;
        }
        std::vector<std::string> ports;
        if (!reserve_local_ports(options.replica ? 6 : 3, ports)) {
            std::fprintf(stderr, "Could not reserve local network ports\n"); return 1;
        }
        const std::string control = "127.0.0.1:" + ports[0];
        const std::string ring = "127.0.0.1:" + ports[1];
        const std::string proxy = "127.0.0.1:" + ports[2];
        const fs::path status = options.state_dir / "worker-status.json";
        const fs::path replica_status = options.state_dir / "replica-status.json";
        const fs::path network_status = options.state_dir / "network-status.json";
        std::vector<std::string> sidecar_arguments{options.sidecar, "-key", identity_key.string(),
            "-listen", "/ip4/0.0.0.0/tcp/" + std::to_string(options.listen_port),
            "-dht", "client", "-reachability", "private", "-status-file", status.string(),
            "-inbound", control, "-allow-any", "-ring-inbound", ring, "-ring-proxy", proxy,
            "-net-status-file", network_status.string()};
        for (const std::string& peer : options.bootstrap) {
            sidecar_arguments.insert(sidecar_arguments.end(), {"-bootstrap", peer});
        }
        for (const std::string& relay : options.relays) {
            sidecar_arguments.insert(sidecar_arguments.end(), {"-relay", relay});
        }
        if (options.simulate_nat) sidecar_arguments.push_back("-simulate-nat");
        std::string candidate_api, session_listen;
        if (options.replica) {
            // The owner shares this sidecar: discovery and link probes (candidate API), the
            // ring's returns to the owner, and clients' sessions on its front door.
            candidate_api = "127.0.0.1:" + ports[3];
            session_listen = "127.0.0.1:" + ports[5];
            sidecar_arguments.insert(sidecar_arguments.end(), {"-candidate-api", candidate_api,
                "-return-inbound", "127.0.0.1:" + ports[4], "-session-inbound", session_listen,
                "-replica-status", replica_status.string()});
        }
        dan::platform::Process sidecar;
        std::vector<std::string> addresses;
        if (!start_sidecar(sidecar, sidecar_arguments, options.state_dir, id, addresses, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        std::printf("Joined the DAN network. Relay addresses: %zu\n", static_cast<std::size_t>(
            std::count_if(addresses.begin(), addresses.end(), [](const std::string& address) {
                return address.find("/p2p-circuit") != std::string::npos;
            })));
        std::vector<std::string> arguments{options.stage_worker,
            "--control-listen", control, "--ring-listen", ring,
            "--peer-header", "--ring-proxy", proxy, "--ring-target", "/p2p/" + id,
            "--status-file", status.string(),
            "--net-status-file", network_status.string(),
            "--provider-id", id, "--gpu", selected->name,
            "--vram-mib", std::to_string(usable_vram),
            "--cache-dir", options.cache_dir.string(),
            "--ctx", std::to_string(options.max_context),
            "--max-sessions", std::to_string(options.max_sessions)};
        for (const std::string& manifest : options.catalog) {
            arguments.insert(arguments.end(), {"--catalog", manifest});
        }
        if (!options.verbose) arguments.push_back("--tui");
        if (options.replica) {
            arguments.insert(arguments.end(), {"--replica-status-file", replica_status.string()});
        }
        dan::platform::Process owner;
        if (options.replica) {
            std::vector<std::string> owner_arguments{options.replica_client, "--form",
                "--discover", candidate_api, "--self-control", control,
                "--session-listen", session_listen, "--replica-status", replica_status.string(),
                "--sessions", std::to_string(options.replica_sessions),
                "--context", std::to_string(options.max_context),
                "--max-edge-rtt-ms", std::to_string(options.replica_max_edge_rtt_ms),
                "--min-stages", std::to_string(options.replica_min_stages),
                "--rank-delay", "--log", (options.state_dir / "logs" / "replica-owner.log").string()};
            if (!options.replica_relay_edges) owner_arguments.push_back("--no-relay-edges");
            if (options.replica_speculate) owner_arguments.push_back("--speculate");
            for (const std::string& manifest : options.catalog) {
                owner_arguments.insert(owner_arguments.end(), {"--manifest", manifest});
            }
            if (!owner.start(owner_arguments, error, false, true)) {
                std::fprintf(stderr, "Could not start the replica owner: %s\n", error.c_str());
                return 1;
            }
        }
        const int result = dan::platform::replace_with_provider(arguments, error);
        if (result != 0) std::fprintf(stderr, "%s\n", error.c_str());
        return result;
    }
    std::string worker_coordinator = options.coordinator;
    std::string ring_listen;
    std::string ring_proxy;
    std::string ring_target;
    dan::platform::Process sidecar;
    if (options.peer_network) {
        const std::string port = dan::platform::free_tcp_port("127.0.0.1");
        if (port.empty()) {
            std::fprintf(stderr, "Could not reserve a local network port\n"); return 1;
        }
        worker_coordinator = "127.0.0.1:" + port;
        std::string ring_port;
        do { ring_port = dan::platform::free_tcp_port("127.0.0.1"); }
        while (!ring_port.empty() && ring_port == port);
        std::string proxy_port;
        do { proxy_port = dan::platform::free_tcp_port("127.0.0.1"); }
        while (!proxy_port.empty() && (proxy_port == port || proxy_port == ring_port));
        if (ring_port.empty() || proxy_port.empty()) {
            std::fprintf(stderr, "Could not reserve local ring ports\n"); return 1;
        }
        ring_listen = "127.0.0.1:" + ring_port;
        ring_proxy = "127.0.0.1:" + proxy_port;
        std::vector<std::string> sidecar_arguments{options.sidecar, "-key", identity_key.string(),
                "-listen", "/ip4/0.0.0.0/tcp/0", "-forward",
                worker_coordinator + "=" + options.coordinator_peer,
                "-ring-inbound", ring_listen, "-ring-proxy", ring_proxy};
        for (const std::string& relay : options.relays) {
            sidecar_arguments.insert(sidecar_arguments.end(), {"-relay", relay});
        }
        std::vector<std::string> addresses;
        if (!start_sidecar(sidecar, sidecar_arguments, options.state_dir, id, addresses, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        for (const std::string& address : addresses) {
            if (!ring_target.empty()) ring_target += ',';
            ring_target += address;
        }
        if (ring_target.empty()) {
            std::fprintf(stderr, "Encrypted networking returned no ring address\n");
            return 1;
        }
    }
    std::vector<std::string> arguments{options.stage_worker,
        "--provider-id", id,
        "--gpu", selected->name,
        "--vram-mib", std::to_string(usable_vram),
        "--cache-dir", options.cache_dir.string()};
    if (hosted) {
        if (options.metadata_cache.empty()) {
            options.metadata_cache = options.state_dir / "coordinator-model-index.gguf";
        }
        arguments.insert(arguments.end(), {"--host-coordinator",
            fs::absolute(options.host_manifest).string(), "--serve", options.serve_endpoint,
            "--provider-listen", options.provider_listen, "--metadata-cache",
            fs::absolute(options.metadata_cache).string()});
    } else {
        arguments.insert(arguments.end(), {"--coordinator", worker_coordinator});
    }
    if (options.peer_network) {
        arguments.insert(arguments.end(), {"--ring-listen", ring_listen,
            "--ring-proxy", ring_proxy, "--ring-target", ring_target});
    } else {
        arguments.insert(arguments.end(), {"--ring-listen", options.advertise_host + ':'
            + std::to_string(options.ring_port)});
    }
    if (!options.verbose) arguments.push_back("--tui");
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
