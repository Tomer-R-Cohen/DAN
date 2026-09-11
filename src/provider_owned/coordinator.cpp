#include "provider_owned/protocol.hpp"
#include "provider_owned/fair_queue.hpp"
#include "provider_owned/formation.hpp"
#include "provider_owned/range_model.hpp"
#include "platform.hpp"
#ifdef DAN_HAS_LLAMA
#include "llama.h"
#include "ggml-backend.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace po = dan::provider_owned;

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

class Connection {
public:
    explicit Connection(po::socket_t socket) : socket_(socket) {
        if (socket_ == po::invalid_socket) throw std::runtime_error("invalid provider socket");
        set_timeout();
    }

    explicit Connection(std::string_view endpoint) {
        const std::size_t colon = endpoint.rfind(':');
        if (colon == std::string_view::npos || colon == 0 || colon + 1 == endpoint.size()) {
            throw std::runtime_error("invalid provider endpoint");
        }
        const std::string host(endpoint.substr(0, colon));
        const std::string port(endpoint.substr(colon + 1));
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* addresses = nullptr;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) {
            throw std::runtime_error("could not resolve provider endpoint");
        }
        for (addrinfo* address = addresses; address; address = address->ai_next) {
            socket_ = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
            if (socket_ == po::invalid_socket) continue;
            if (connect(socket_, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) break;
            po::close_socket(socket_);
            socket_ = po::invalid_socket;
        }
        freeaddrinfo(addresses);
        if (socket_ == po::invalid_socket) throw std::runtime_error("could not connect to provider");
        set_timeout();
    }

    Connection(Connection&& other) noexcept : socket_(std::exchange(other.socket_, po::invalid_socket)) {}
    Connection& operator=(Connection&& other) noexcept {
        if (this != &other) {
            if (socket_ != po::invalid_socket) po::close_socket(socket_);
            socket_ = std::exchange(other.socket_, po::invalid_socket);
        }
        return *this;
    }

public:
    void set_timeout(std::uint32_t milliseconds = 30000) {
#ifdef _WIN32
        const DWORD timeout = milliseconds;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO,
            reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
        const timeval timeout{static_cast<long>(milliseconds / 1000),
            static_cast<long>((milliseconds % 1000) * 1000)};
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
    }

    ~Connection() { if (socket_ != po::invalid_socket) po::close_socket(socket_); }
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    std::pair<po::Frame, std::uint64_t> exchange(const po::Frame& input) {
        std::string error;
        const auto start = Clock::now();
        if (!po::send_frame(socket_, input, error)) throw std::runtime_error(error);
        po::Frame output;
        if (!po::recv_frame(socket_, output, error)) throw std::runtime_error(error);
        if (output.type == po::Type::error) {
            throw std::runtime_error(std::string(output.payload.begin(), output.payload.end()));
        }
        return {std::move(output), elapsed_ns(start)};
    }

    void send(const po::Frame& input) {
        std::string error;
        if (!po::send_frame(socket_, input, error)) throw std::runtime_error(error);
    }

    po::Frame receive() {
        po::Frame output;
        std::string error;
        if (!po::recv_frame(socket_, output, error)) throw std::runtime_error(error);
        if (output.type == po::Type::error) {
            throw std::runtime_error(std::string(output.payload.begin(), output.payload.end()));
        }
        return output;
    }

    bool receive_peer_id(std::string& peer_id) {
        return po::recv_peer_id(socket_, peer_id);
    }

private:
    po::socket_t socket_ = po::invalid_socket;
};

struct Manifest {
    std::string model_id;
    std::string architecture;
    std::uint32_t layers = 0;
    std::uint32_t hidden = 0;
    std::uint32_t context = 0;
    std::string url;
    std::string revision;
    std::string sha256;
};

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open manifest");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t value_start(const std::string& json, std::string_view key) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    const std::size_t found = json.find(quoted);
    if (found == std::string::npos) throw std::runtime_error("manifest missing " + std::string(key));
    std::size_t position = json.find(':', found + quoted.size());
    if (position == std::string::npos) throw std::runtime_error("invalid manifest");
    do { ++position; } while (position < json.size()
        && (json[position] == ' ' || json[position] == '\t'
            || json[position] == '\r' || json[position] == '\n'));
    return position;
}

std::string json_string(const std::string& json, std::string_view key) {
    std::size_t position = value_start(json, key);
    if (position >= json.size() || json[position++] != '"') {
        throw std::runtime_error("manifest field is not a string: " + std::string(key));
    }
    std::string value;
    while (position < json.size() && json[position] != '"') {
        if (json[position] == '\\') throw std::runtime_error("escaped manifest strings unsupported");
        value += json[position++];
    }
    if (position >= json.size() || value.empty()) throw std::runtime_error("invalid manifest string");
    return value;
}

std::string optional_json_string(const std::string& json, std::string_view key) {
    return json.find("\"" + std::string(key) + "\"") == std::string::npos
        ? std::string{} : json_string(json, key);
}

std::uint32_t json_uint(const std::string& json, std::string_view key) {
    std::size_t position = value_start(json, key);
    std::uint64_t value = 0;
    const std::size_t begin = position;
    while (position < json.size() && json[position] >= '0' && json[position] <= '9') {
        value = value * 10 + static_cast<unsigned>(json[position++] - '0');
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("manifest integer overflow");
        }
    }
    if (position == begin) throw std::runtime_error("manifest field is not an integer");
    return static_cast<std::uint32_t>(value);
}

std::uint64_t json_uint64(const std::string& json, std::string_view key) {
    std::size_t position = value_start(json, key);
    std::uint64_t value = 0;
    const std::size_t begin = position;
    while (position < json.size() && json[position] >= '0' && json[position] <= '9') {
        const unsigned digit = static_cast<unsigned>(json[position++] - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::runtime_error("manifest integer overflow");
        }
        value = value * 10 + digit;
    }
    if (position == begin) throw std::runtime_error("JSON field is not an integer");
    return value;
}

std::uint32_t optional_json_uint(const std::string& json, std::string_view key) {
    return json.find("\"" + std::string(key) + "\"") == std::string::npos
        ? 0 : json_uint(json, key);
}

Manifest load_manifest(const std::string& path) {
    const std::string json = read_file(path);
    Manifest manifest;
    manifest.model_id = json_string(json, "model_id");
    manifest.architecture = optional_json_string(json, "architecture");
    manifest.layers = optional_json_uint(json, "layers");
    manifest.hidden = optional_json_uint(json, "hidden_size");
    manifest.context = json_uint(json, "context_size");
    manifest.revision = json_string(json, "artifact_revision");
    manifest.sha256 = json_string(json, "artifact_sha256");
    manifest.url = optional_json_string(json, "artifact_url");
    if (manifest.url.empty()) {
        const std::string repository = json_string(json, "hf_repo");
        const std::string filename = json_string(json, "gguf_filename");
        const auto unsafe = [](std::string_view value, bool slash) {
            const std::string_view allowed = slash
                ? "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./"
                : "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.";
            return value.empty() || value.find_first_not_of(allowed) != std::string_view::npos
                || value.contains("..") || value.front() == '/' || value.back() == '/';
        };
        if (unsafe(repository, true) || unsafe(filename, false)) {
            throw std::runtime_error("invalid Hugging Face repository or GGUF filename");
        }
        manifest.url = "https://huggingface.co/" + repository + "/resolve/"
            + manifest.revision + "/" + filename;
    }
    const auto hex = [](std::string_view value, std::size_t length) {
        return value.size() == length && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')
                || (c >= 'A' && c <= 'F');
        });
    };
    if (manifest.model_id.empty() || manifest.model_id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos
        || (!manifest.architecture.empty() && manifest.architecture != "qwen2")
        || manifest.context == 0 || !manifest.url.starts_with("https://")
        || manifest.url.find_first_of("\r\n") != std::string::npos
        || !hex(manifest.revision, 40) || !hex(manifest.sha256, 64)) {
        throw std::runtime_error("manifest is not a valid pinned Qwen2 GGUF selection");
    }
    return manifest;
}

class DraftModel {
public:
    DraftModel(const std::string& path, std::uint32_t context_size, int gpu_layers) {
#ifdef DAN_HAS_LLAMA
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers;
        model_ = llama_model_load_from_file(path.c_str(), model_params);
        if (!model_) throw std::runtime_error("draft model load failed");
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = context_size;
        context_params.n_batch = context_size;
        context_params.n_ubatch = std::min<std::uint32_t>(context_size, 512);
        context_ = llama_init_from_model(model_, context_params);
        if (!context_) throw std::runtime_error("draft model context creation failed");
        sampler_ = llama_sampler_init_greedy();
        if (!sampler_) throw std::runtime_error("draft sampler creation failed");
#else
        (void) path; (void) context_size; (void) gpu_layers;
        throw std::runtime_error("this coordinator was built without llama.cpp draft support");
#endif
    }

    ~DraftModel() {
#ifdef DAN_HAS_LLAMA
        if (sampler_) llama_sampler_free(sampler_);
        if (context_) llama_free(context_);
        if (model_) llama_model_free(model_);
#endif
    }

    DraftModel(const DraftModel&) = delete;
    DraftModel& operator=(const DraftModel&) = delete;

    std::uint32_t start(std::string_view prompt) {
#ifdef DAN_HAS_LLAMA
        llama_memory_clear(llama_get_memory(context_), true);
        llama_sampler_reset(sampler_);
        position_ = 0;
        const int count = -llama_tokenize(llama_model_get_vocab(model_), prompt.data(),
            static_cast<int>(prompt.size()), nullptr, 0, true, true);
        if (count <= 0) throw std::runtime_error("draft prompt tokenization failed");
        std::vector<llama_token> tokens(static_cast<std::size_t>(count));
        if (llama_tokenize(llama_model_get_vocab(model_), prompt.data(),
                static_cast<int>(prompt.size()), tokens.data(), count, true, true) != count) {
            throw std::runtime_error("draft prompt tokenization changed size");
        }
        decode(tokens);
        return position_;
#else
        (void) prompt;
        return 0;
#endif
    }

    std::vector<std::uint32_t> propose(std::uint32_t current, std::size_t count) {
        std::vector<std::uint32_t> output;
        output.reserve(count);
#ifdef DAN_HAS_LLAMA
        for (std::size_t index = 0; index < count; ++index) {
            decode({static_cast<llama_token>(current)});
            current = static_cast<std::uint32_t>(
                llama_sampler_sample(sampler_, context_, -1));
            output.push_back(current);
        }
#else
        (void) current;
#endif
        return output;
    }

    void rollback(std::uint32_t position) {
#ifdef DAN_HAS_LLAMA
        if (position > position_ || !llama_memory_seq_rm(
                llama_get_memory(context_), 0, position, -1)) {
            throw std::runtime_error("could not roll back draft model");
        }
        position_ = position;
#else
        (void) position;
#endif
    }

    std::string piece(std::uint32_t token) const {
#ifdef DAN_HAS_LLAMA
        int size = llama_token_to_piece(llama_model_get_vocab(model_),
            static_cast<llama_token>(token), nullptr, 0, 0, false);
        if (size == 0) return {};
        if (size > 0) throw std::runtime_error("unexpected draft token piece probe result");
        std::string output(static_cast<std::size_t>(-size), '\0');
        size = llama_token_to_piece(llama_model_get_vocab(model_),
            static_cast<llama_token>(token), output.data(), static_cast<int>(output.size()), 0, false);
        if (size < 0) throw std::runtime_error("draft token conversion failed");
        output.resize(static_cast<std::size_t>(size));
        return output;
#else
        (void) token;
        return {};
#endif
    }

    bool is_eog(std::uint32_t token) const {
#ifdef DAN_HAS_LLAMA
        return llama_vocab_is_eog(llama_model_get_vocab(model_),
            static_cast<llama_token>(token));
#else
        (void) token;
        return false;
#endif
    }

private:
#ifdef DAN_HAS_LLAMA
    void decode(const std::vector<llama_token>& tokens) {
        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            batch.token[index] = tokens[index];
            batch.pos[index] = static_cast<llama_pos>(position_ + index);
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = index + 1 == tokens.size();
        }
        const int result = llama_decode(context_, batch);
        llama_batch_free(batch);
        if (result != 0) throw std::runtime_error("draft model decode failed");
        position_ += static_cast<std::uint32_t>(tokens.size());
    }

    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    std::uint32_t position_ = 0;
#endif
};

struct Options {
    std::string manifest;
    std::string provider_a;
    std::string provider_b;
    std::vector<std::string> providers;
    std::vector<std::string> prompts;
    std::string report;
    std::string expected;
    std::string listen;
    std::string provider_listen;
    std::string draft_model;
    std::string ring_return;
    bool provider_peer_auth = false;
    std::filesystem::path metadata_cache;
    int tokens = 20;
    int requests = 1;
    int resident_sessions = 1;
    int queue_capacity = 128;
    int queue_timeout_ms = 30000;
    int client_threads = 64;
    int draft_tokens = 4;
    int draft_gpu_layers = 999;
    int pipeline_depth = 0;
    bool persistent = false;
    bool reset_between = false;
    bool shutdown_workers = false;
    bool interactive = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
#ifdef _WIN32
    if (argc == 1) {
        const auto package = std::filesystem::absolute(argv[0]).parent_path();
        options.manifest = (package / "config" / "active-model.json").string();
        options.provider_listen = "127.0.0.1:50201";
        options.provider_peer_auth = true;
        options.metadata_cache = "data/model-index.tmp";
        options.tokens = 256;
        options.interactive = true;
    }
#endif
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--persistent") { options.persistent = true; continue; }
        if (option == "--reset-between") { options.reset_between = true; continue; }
        if (option == "--shutdown-workers") { options.shutdown_workers = true; continue; }
        if (option == "--interactive") { options.interactive = true; continue; }
        if (option == "--provider-peer-auth") { options.provider_peer_auth = true; continue; }
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--manifest") options.manifest = value;
        else if (option == "--provider-a") options.provider_a = value;
        else if (option == "--provider-b") options.provider_b = value;
        else if (option == "--provider") options.providers.push_back(value);
        else if (option == "--prompt") options.prompts.push_back(value);
        else if (option == "--tokens") options.tokens = std::stoi(value);
        else if (option == "--requests") options.requests = std::stoi(value);
        else if (option == "--resident-sessions") options.resident_sessions = std::stoi(value);
        else if (option == "--report") options.report = value;
        else if (option == "--expected-output") options.expected = value;
        else if (option == "--listen") options.listen = value;
        else if (option == "--provider-listen") options.provider_listen = value;
        else if (option == "--metadata-cache") options.metadata_cache = value;
        else if (option == "--draft-model") options.draft_model = value;
        else if (option == "--draft-tokens") options.draft_tokens = std::stoi(value);
        else if (option == "--draft-gpu-layers") options.draft_gpu_layers = std::stoi(value);
        else if (option == "--pipeline-depth") options.pipeline_depth = std::stoi(value);
        else if (option == "--ring-return") options.ring_return = value;
        else if (option == "--queue-capacity") options.queue_capacity = std::stoi(value);
        else if (option == "--queue-timeout-ms") options.queue_timeout_ms = std::stoi(value);
        else if (option == "--client-threads") options.client_threads = std::stoi(value);
        else throw std::runtime_error("unknown option: " + option);
    }
    if (!options.provider_a.empty()) options.providers.insert(options.providers.begin(), options.provider_a);
    if (!options.provider_b.empty()) options.providers.push_back(options.provider_b);
    const bool automatic = !options.provider_listen.empty();
    if (options.manifest.empty() || (!automatic && options.providers.empty())
        || (automatic && (!options.providers.empty() || options.metadata_cache.empty()))
        || (options.listen.empty() && options.prompts.empty() && !options.interactive)
        || options.tokens < 1 || options.requests < 1
        || options.resident_sessions < 1 || (!options.persistent
            && (options.resident_sessions != 1 || options.reset_between))
        || options.queue_capacity < 1 || options.queue_timeout_ms < 1
        || options.client_threads < 1 || options.client_threads > 256
        || options.draft_tokens < 1 || options.draft_tokens > 16
        || options.draft_gpu_layers < 0
        || options.pipeline_depth < 0
        || (options.pipeline_depth > 0 && options.draft_model.empty())
        || (!options.ring_return.empty() && (automatic || options.providers.size() < 2))
        || (!options.draft_model.empty() && (options.persistent || options.interactive))
        || (!options.draft_model.empty() && options.prompts.empty() && options.listen.empty())
        || (options.provider_peer_auth && (options.provider_listen.empty()
            || !options.provider_listen.starts_with("127.0.0.1:")))
        || (!options.listen.empty() && (!options.prompts.empty() || options.persistent
            || options.reset_between || options.interactive))
        || (options.interactive && (!options.prompts.empty() || options.persistent
            || options.reset_between))) {
        throw std::runtime_error(
            "usage: dan-provider-owned-coordinator --manifest FILE (--provider HOST:PORT --provider HOST:PORT [...] [--ring-return HOST:PORT] | --provider-listen HOST:PORT --metadata-cache FILE [--provider-peer-auth]) (--interactive | --prompt TEXT [...] [--draft-model FILE --draft-tokens 4 [--pipeline-depth N]] | --listen HOST:PORT [...] [--draft-model FILE --draft-tokens 4 [--pipeline-depth N]])");
    }
    return options;
}

#ifdef _WIN32
void start_packaged_network(dan::platform::Process& sidecar, const char* program) {
    namespace fs = std::filesystem;
    std::string error;
    const fs::path package = fs::absolute(program).parent_path();
    const fs::path executable = package / "runtime" / "dan-sidecar.exe";
    if (!dan::platform::executable_file(executable)) {
        throw std::runtime_error("the bundled encrypted networking program is missing");
    }
    const fs::path state = dan::platform::data_directory() / "coordinator";
    const fs::path ready = state / "network-ready.txt";
    std::error_code filesystem_error;
    fs::create_directories(state / "logs", filesystem_error);
    fs::remove(ready, filesystem_error);
    if (filesystem_error) throw std::runtime_error("could not prepare coordinator data folder");
    std::vector<std::string> arguments{executable.string(),
            "-key", (state / "identity.key").string(),
            "-listen", "/ip4/0.0.0.0/tcp/50200",
            "-inbound", "127.0.0.1:50201", "-allow-any",
            "-ready-file", ready.string(),
            "-log", (state / "logs" / "sidecar.log").string()};
    std::ifstream relays(package / "config" / "relays.txt");
    for (std::string relay; std::getline(relays, relay);) {
        if (!relay.empty() && relay.back() == '\r') relay.pop_back();
        if (!relay.empty() && relay.front() != '#') {
            arguments.push_back("-relay");
            arguments.push_back(relay);
        }
    }
    if (!sidecar.start(arguments, error, false, true)) {
        throw std::runtime_error("could not start encrypted networking: " + error);
    }
    for (int attempt = 0; attempt < 100 && sidecar.running() && !fs::exists(ready); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::ifstream input(ready);
    std::string peer_id, address;
    if (!std::getline(input, peer_id) || peer_id.empty()) {
        throw std::runtime_error("encrypted networking did not start; see sidecar.log");
    }
    std::cout << "Coordinator identity: " << peer_id << "\n"
        << "Use one of these addresses when building the provider download:\n";
    while (std::getline(input, address)) if (!address.empty()) std::cout << "  " << address << '\n';
    std::cout << '\n';
    fs::remove(ready, filesystem_error);
}
#endif

po::socket_t listen_endpoint(std::string_view endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == endpoint.size()) {
        throw std::runtime_error("invalid listen endpoint");
    }
    const std::string host(endpoint.substr(0, colon)), port(endpoint.substr(colon + 1));
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        throw std::runtime_error("could not resolve listen endpoint");
    }
    po::socket_t listener = po::invalid_socket;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        listener = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (listener == po::invalid_socket) continue;
        const int enabled = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
            reinterpret_cast<const char*>(&enabled), sizeof(enabled));
        if (bind(listener, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0
            && listen(listener, 16) == 0) break;
        po::close_socket(listener); listener = po::invalid_socket;
    }
    freeaddrinfo(addresses);
    if (listener == po::invalid_socket) throw std::runtime_error("bind/listen failed");
    return listener;
}

struct FormedReplica {
    std::vector<std::unique_ptr<Connection>> connections;
    std::vector<po::StageAssignment> assignments;
};

FormedReplica form_replica(Manifest& manifest, const Options& options) {
    po::ModelIndex model;
    std::string error;
    if (!po::inspect_range_model({manifest.url, manifest.revision, manifest.sha256,
            options.metadata_cache, 0, 1}, model, error)) {
        throw std::runtime_error("model metadata: " + error);
    }
    if ((!manifest.architecture.empty() && model.architecture != manifest.architecture)
        || (manifest.layers != 0 && model.layers != manifest.layers)
        || (manifest.hidden != 0 && model.hidden != manifest.hidden)) {
        throw std::runtime_error("manifest does not match remote GGUF metadata");
    }
    std::string incompatibility;
    if (!po::compatible_dense_qwen2(model, &incompatibility)) {
        throw std::runtime_error("incompatible GGUF: " + incompatibility);
    }
    manifest.architecture = model.architecture;
    manifest.layers = model.layers;
    manifest.hidden = model.hidden;
    if (std::uint64_t(manifest.context) * manifest.hidden * sizeof(float) > po::max_payload - 8) {
        throw std::runtime_error("model context activation exceeds protocol frame limit");
    }

    struct Registered {
        po::ProviderCapability capability;
        std::string peer_id;
        std::unique_ptr<Connection> connection;
    };
    std::vector<Registered> registered;
    const po::socket_t listener = listen_endpoint(options.provider_listen);
    std::fprintf(stderr, "model metadata ready: layers=%u hidden=%u; providers join at %s\n",
        model.layers, model.hidden, options.provider_listen.c_str());
    std::optional<std::vector<po::StageAssignment>> plan;
    while (!plan) {
        const po::socket_t socket = accept(listener, nullptr, nullptr);
        if (socket == po::invalid_socket) { po::close_socket(listener); throw std::runtime_error("accept failed"); }
        auto connection = std::make_unique<Connection>(socket);
        std::string peer_id;
        if (options.provider_peer_auth && !connection->receive_peer_id(peer_id)) {
            std::fprintf(stderr, "rejected connection without authenticated PeerID\n");
            continue;
        }
        po::Frame hello = connection->receive();
        po::ProviderCapability capability;
        const std::string text(hello.payload.begin(), hello.payload.end());
        if (hello.type != po::Type::provider_available || hello.session != 0
            || hello.request != 0 || !po::parse_available(text, capability)
            || (!peer_id.empty() && capability.id != peer_id)
            || std::any_of(registered.begin(), registered.end(), [&](const Registered& value) {
                return value.capability.id == capability.id
                    || (!peer_id.empty() && value.peer_id == peer_id);
            })) {
            std::fprintf(stderr, "rejected invalid or duplicate provider\n");
            continue;
        }
        std::fprintf(stderr, "provider AVAILABLE id=%s peer=%s gpu=%s offered=%llu MiB\n",
            capability.id.c_str(), peer_id.empty() ? "direct-tcp" : peer_id.c_str(),
            capability.gpu.c_str(),
            static_cast<unsigned long long>(capability.offered_vram_mib));
        registered.push_back({std::move(capability), std::move(peer_id), std::move(connection)});
        std::vector<po::ProviderCapability> capabilities;
        for (const Registered& value : registered) capabilities.push_back(value.capability);
        plan = po::plan_replica(model, capabilities, manifest.context, 8);
    }
    po::close_socket(listener);

    // Send every assignment before waiting so range downloads run concurrently.
    for (const po::StageAssignment& stage : *plan) {
        po::ModelAssignment assignment{manifest.model_id, manifest.url, manifest.revision,
            manifest.sha256, stage.begin, stage.end, manifest.context, 8};
        po::Frame frame; frame.type = po::Type::assign_stage;
        const std::string payload = po::assignment_message(assignment);
        frame.payload.assign(payload.begin(), payload.end());
        registered[stage.provider].connection->set_timeout(0);
        registered[stage.provider].connection->send(frame);
        std::fprintf(stderr, "assigned %s layers=%d..%d model=%.2f GiB kv=%.2f MiB\n",
            registered[stage.provider].capability.id.c_str(), stage.begin, stage.end - 1,
            stage.model_bytes / double(1024ull * 1024 * 1024),
            stage.kv_bytes / double(1024ull * 1024));
    }
    FormedReplica formed;
    formed.assignments = *plan;
    for (const po::StageAssignment& stage : *plan) {
        po::Frame ready = registered[stage.provider].connection->receive();
        if (ready.type != po::Type::stage_ready) throw std::runtime_error("provider failed to become ready");
        registered[stage.provider].connection->set_timeout();
        formed.connections.push_back(std::move(registered[stage.provider].connection));
    }
    std::fprintf(stderr, "N-stage replica READY stages=%zu\n", formed.connections.size());
    return formed;
}

void require_ack(const po::Frame& frame, const po::Frame& input, bool allow_timing = false) {
    if (frame.type != po::Type::ack || frame.session != input.session
        || frame.request != input.request || frame.rows != 0 || frame.cols != 0
        || frame.dtype != po::DType::none
        || (!frame.payload.empty() && (!allow_timing || frame.payload.size() != 8))) {
        throw std::runtime_error("invalid provider acknowledgement");
    }
}

using StageConnections = std::vector<Connection*>;

void control_all(const StageConnections& stages, po::Type type,
    std::uint64_t session, std::uint64_t request = 0) {
    po::Frame input;
    input.type = type;
    input.session = session;
    input.request = request;
    std::uint32_t position = 0;
    for (std::size_t index = 0; index < stages.size(); ++index) {
        auto [output, ignored] = stages[index]->exchange(input);
        (void) ignored;
        require_ack(output, input);
        if (type == po::Type::end_request) {
            if (index != 0 && output.position != position) {
                throw std::runtime_error("provider session positions diverged");
            }
            position = output.position;
        }
    }
}

struct RequestMetrics {
    double latency_ms = 0;
    double prefill_ms = 0;
    double ttft_ms = 0;
    double queue_wait_ms = 0;
    std::vector<double> a_compute_ms;
    std::vector<double> middle_compute_ms;
    std::vector<double> b_compute_ms;
    std::vector<double> network_ms;
    std::size_t activation_bytes = 0;
    std::vector<std::uint32_t> token_ids;
    std::uint64_t speculative_rounds = 0;
    std::uint64_t proposed_tokens = 0;
    std::uint64_t accepted_draft_tokens = 0;
    double draft_ms = 0;
};

struct RequestResult {
    std::string output;
    std::uint32_t position = 0;
    std::uint32_t final_token = 0;
    bool eog = false;
    RequestMetrics metrics;
};

struct Activation {
    po::Frame frame;
    std::uint64_t compute_ns = 0;
};

Activation require_activation(po::Frame frame, const po::Frame& input,
    std::uint32_t hidden, po::Type expected = po::Type::activation) {
    if (frame.type != expected || frame.session != input.session
        || frame.request != input.request || frame.position != input.position
        || frame.rows == 0 || frame.cols != hidden || frame.dtype != po::DType::f32le) {
        throw std::runtime_error("invalid stage A activation metadata");
    }
    const std::uint64_t values = std::uint64_t(frame.rows) * frame.cols;
    if (values > (po::max_payload - 8) / sizeof(float)
        || frame.payload.size() != 8 + values * sizeof(float)) {
        throw std::runtime_error("invalid stage A activation size");
    }
    const std::uint64_t compute_ns = po::get64(frame.payload.data());
    return {std::move(frame), compute_ns};
}

struct Result {
    std::uint32_t token = 0;
    std::uint64_t compute_ns = 0;
    bool eog = false;
    std::string text;
    std::uint32_t position = 0;
};

Result require_result(const po::Frame& frame, const po::Frame& input) {
    const bool position_ok = input.rows != 0
        ? frame.position == input.position + input.rows
        : (input.type == po::Type::prompt
            ? frame.position > input.position
            : frame.position == input.position + 1);
    if (frame.type != po::Type::result || frame.session != input.session
        || frame.request != input.request || !position_ok
        || frame.rows != 0 || frame.cols != 0 || frame.dtype != po::DType::none
        || frame.payload.size() < 13 || frame.payload[12] > 1) {
        throw std::runtime_error("invalid stage B result");
    }
    return {po::get32(frame.payload.data()), po::get64(frame.payload.data() + 4),
        frame.payload[12] != 0,
        std::string(frame.payload.begin() + 13, frame.payload.end()), frame.position};
}

struct SpeculativeResult {
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint64_t> compute_ns;
    std::uint64_t network_ns = 0;
    std::uint32_t position = 0;
};

SpeculativeResult route_speculative(const StageConnections& stages, po::Frame input,
    std::uint32_t hidden, RequestMetrics& metrics) {
    SpeculativeResult output;
    po::Frame current = std::move(input);
    for (std::size_t index = 0; index < stages.size(); ++index) {
        auto [response, round_ns] = stages[index]->exchange(current);
        if (index + 1 == stages.size()) {
            if (response.type != po::Type::result || response.session != current.session
                || response.request != current.request
                || response.position != current.position + current.rows
                || response.rows != current.rows || response.cols != 0
                || response.dtype != po::DType::none
                || response.payload.size() != 8 + static_cast<std::size_t>(response.rows) * 4) {
                throw std::runtime_error("invalid speculative result");
            }
            const std::uint64_t compute = po::get64(response.payload.data());
            output.compute_ns.push_back(compute);
            output.network_ns += round_ns > compute ? round_ns - compute : 0;
            output.position = response.position;
            output.tokens.reserve(response.rows);
            for (std::uint32_t row = 0; row < response.rows; ++row) {
                output.tokens.push_back(po::get32(
                    response.payload.data() + 8 + static_cast<std::size_t>(row) * 4));
            }
            return output;
        }
        Activation activation = require_activation(std::move(response), current, hidden,
            po::Type::speculative_activation);
        metrics.activation_bytes += activation.frame.rows * activation.frame.cols * sizeof(float);
        output.compute_ns.push_back(activation.compute_ns);
        output.network_ns += round_ns > activation.compute_ns ? round_ns - activation.compute_ns : 0;
        current = std::move(activation.frame);
    }
    throw std::runtime_error("speculative routing failed");
}

// Ring direct-return (docs/PIPELINED_SPECULATION_V1.md phase 4): when ring_return is given,
// stages must already be configured with --next so stage 0 forwards its activation directly to
// stage 1, and so on to the tail, whose --next is the socket ring_return wraps -- the
// coordinator never sees or touches any intermediate hop. Prefill validation is looser here than
// in hub-and-spoke mode: require_result falls back to its single-stage check (final position
// greater than the start, not the exact prompt length) because the coordinator never learns the
// tokenized prompt length from an intermediate activation the way it does in hub-and-spoke mode.
// That's a real, accepted reduction in coordinator-side sanity-checking, not a correctness gap --
// the stage's own KV/position bookkeeping is the actual source of truth either way.
Result route_step(const StageConnections& stages, const po::Frame& input,
    std::uint32_t hidden, RequestMetrics& metrics, bool decode, Connection* ring_return = nullptr) {
    if (stages.empty()) throw std::runtime_error("replica has no stages");
    if (ring_return) {
        const auto start = Clock::now();
        stages.front()->send(input);
        po::Frame response = ring_return->receive();
        const std::uint64_t round_ns = elapsed_ns(start);
        Result result = require_result(response, input);
        if (decode) {
            metrics.b_compute_ms.push_back(result.compute_ns / 1e6);
            const std::uint64_t network_ns =
                round_ns > result.compute_ns ? round_ns - result.compute_ns : 0;
            metrics.network_ms.push_back(network_ns / 1e6);
        }
        return result;
    }
    po::Frame current = input;
    std::uint64_t network_ns = 0;
    for (std::size_t index = 0; index < stages.size(); ++index) {
        auto [response, round_ns] = stages[index]->exchange(current);
        if (index + 1 == stages.size()) {
            Result result = require_result(response, current);
            if (decode) {
                metrics.b_compute_ms.push_back(result.compute_ns / 1e6);
                network_ns += round_ns > result.compute_ns ? round_ns - result.compute_ns : 0;
                metrics.network_ms.push_back(network_ns / 1e6);
            }
            return result;
        }
        Activation activation = require_activation(std::move(response), current, hidden);
        metrics.activation_bytes += activation.frame.rows * activation.frame.cols * sizeof(float);
        if (decode) {
            (index == 0 ? metrics.a_compute_ms : metrics.middle_compute_ms)
                .push_back(activation.compute_ns / 1e6);
            network_ns += round_ns > activation.compute_ns ? round_ns - activation.compute_ns : 0;
        }
        current = std::move(activation.frame);
    }
    throw std::runtime_error("replica routing failed");
}

void commit_final_token(const StageConnections& stages, std::uint64_t session,
    std::uint64_t request, std::uint32_t position, std::uint32_t token,
    std::uint32_t hidden) {
    po::Frame input;
    input.type = po::Type::commit_token;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.resize(4);
    po::put32(input.payload.data(), token);
    po::Frame current = std::move(input);
    for (std::size_t index = 0; index < stages.size(); ++index) {
        auto [response, ignored] = stages[index]->exchange(current);
        (void) ignored;
        if (index + 1 == stages.size()) {
            require_ack(response, current, true);
            if (response.position != position + 1) {
                throw std::runtime_error("commit position mismatch");
            }
        } else {
            current = require_activation(std::move(response), current, hidden,
                po::Type::commit_activation).frame;
        }
    }
}

RequestResult generate(const StageConnections& stages, const Manifest& manifest,
    std::uint64_t session, std::uint64_t request, std::uint32_t position,
    const std::string& prompt, int token_limit, bool preserve_session,
    DraftModel* draft = nullptr, int draft_tokens = 4, Connection* ring_return = nullptr) {
    const auto request_start = Clock::now();
    RequestResult output;
    po::Frame input;
    input.type = po::Type::prompt;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.assign(prompt.begin(), prompt.end());

    const auto prefill_start = Clock::now();
    Result result = route_step(stages, input, manifest.hidden, output.metrics, false, ring_return);
    output.metrics.prefill_ms = elapsed_ns(prefill_start) / 1e6;
    output.metrics.ttft_ms = elapsed_ns(request_start) / 1e6;
    output.output += result.text;
    output.metrics.token_ids.push_back(result.token);
    output.position = result.position;
    output.final_token = result.token;
    output.eog = result.eog;

    if (draft) {
        if (position != 0 || draft->start(prompt) != result.position
            || draft->piece(result.token) != result.text) {
            throw std::runtime_error("draft and target tokenizers do not match");
        }
    }

    while (static_cast<int>(output.metrics.token_ids.size()) < token_limit && !output.eog) {
        if (draft) {
            const std::size_t count = static_cast<std::size_t>(std::min(
                draft_tokens, token_limit - static_cast<int>(output.metrics.token_ids.size())));
            const auto draft_start = Clock::now();
            const std::vector<std::uint32_t> guesses = draft->propose(output.final_token, count);
            output.metrics.draft_ms += elapsed_ns(draft_start) / 1e6;
            output.metrics.speculative_rounds++;
            output.metrics.proposed_tokens += guesses.size();

            po::Frame block;
            block.type = po::Type::token;
            block.session = session;
            block.request = request;
            block.position = output.position;
            block.rows = static_cast<std::uint32_t>(count);
            block.payload.resize(count * 4);
            po::put32(block.payload.data(), output.final_token);
            for (std::size_t index = 1; index < count; ++index) {
                po::put32(block.payload.data() + index * 4, guesses[index - 1]);
            }
            SpeculativeResult checked = route_speculative(
                stages, std::move(block), manifest.hidden, output.metrics);

            std::size_t accepted = 0;
            bool stopped = false;
            for (; accepted < guesses.size() && guesses[accepted] == checked.tokens[accepted];
                    ++accepted) {
                output.output += draft->piece(guesses[accepted]);
                output.metrics.token_ids.push_back(guesses[accepted]);
                output.final_token = guesses[accepted];
                if (draft->is_eog(guesses[accepted])) {
                    output.eog = true;
                    stopped = true;
                    ++accepted;
                    break;
                }
            }
            output.metrics.accepted_draft_tokens += accepted;
            if (!stopped && accepted < guesses.size()) {
                output.final_token = checked.tokens[accepted];
                output.output += draft->piece(output.final_token);
                output.metrics.token_ids.push_back(output.final_token);
                output.eog = draft->is_eog(output.final_token);
            }
            const std::size_t committed_inputs = stopped ? accepted
                : std::min<std::size_t>(count, accepted + 1);
            const std::uint32_t keep_position = output.position
                + static_cast<std::uint32_t>(committed_inputs);
            // No network round trip here: the next frame sent to the stages (the next
            // speculative round, or commit_final_token below) carries keep_position, and each
            // stage truncates its own KV to that position on receipt (implicit rollback).
            draft->rollback(keep_position);
            output.position = keep_position;

            const std::size_t produced = stopped ? accepted
                : (accepted < guesses.size() ? accepted + 1 : accepted);
            if (produced == 0) throw std::runtime_error("speculative round produced no token");
            for (std::size_t token = 0; token < produced; ++token) {
                if (checked.compute_ns.size() > 1) {
                    output.metrics.a_compute_ms.push_back(
                        checked.compute_ns.front() / 1e6 / produced);
                }
                for (std::size_t stage = 1; stage + 1 < checked.compute_ns.size(); ++stage) {
                    output.metrics.middle_compute_ms.push_back(
                        checked.compute_ns[stage] / 1e6 / produced);
                }
                output.metrics.b_compute_ms.push_back(checked.compute_ns.back() / 1e6 / produced);
                output.metrics.network_ms.push_back(checked.network_ns / 1e6 / produced);
            }
            continue;
        }
        input = {};
        input.type = po::Type::token;
        input.session = session;
        input.request = request;
        input.position = output.position;
        input.payload.resize(4);
        po::put32(input.payload.data(), output.final_token);
        result = route_step(stages, input, manifest.hidden, output.metrics, true, ring_return);
        output.output += result.text;
        output.metrics.token_ids.push_back(result.token);
        output.position = result.position;
        output.final_token = result.token;
        output.eog = result.eog;
    }
    if (preserve_session) {
        commit_final_token(stages, session, request, output.position,
            output.final_token, manifest.hidden);
        ++output.position;
    }
    control_all(stages, po::Type::end_request, session, request);
    output.metrics.latency_ms = elapsed_ns(request_start) / 1e6;
    return output;
}

// Pipelined speculative decoding (docs/PIPELINED_SPECULATION_V1.md). Keeps several
// speculative chunks in flight instead of one, so the round trip is hidden instead of
// paid per token. Opt-in via --pipeline-depth; generate() above is unchanged and remains
// the default. Committed output must match generate()'s single-chunk speculative path
// (not plain serial decode -- see the design doc's "correctness fact" section for why).
//
// Roles, one thread per role (the receiver runs on the calling thread):
//   sender:   drafts K tokens past the frontier, sends the chunk to stage 0, never waits
//             for its reply before drafting the next one (bounded by pipeline_depth).
//   relay i:  (one per stage boundary, stages 0..N-2) forwards stage i's activation reply
//             to stage i+1 untouched. Pure pass-through; needs no pipeline state.
//   receiver: reads the tail's replies in send order, applies the existing accept rule
//             per reply instead of per round, and on a real rejection bumps the epoch so
//             replies already in flight for the discarded continuation are recognized as
//             stale and dropped when they arrive.
//
// FIFO order is the only synchronization primitive that matters here: stage 0 is the only
// stage that receives coordinator-injected frames, so every stage processes the same frame
// sequence in the same order, and replies return in send order. That is what lets the
// receiver match replies to requests with a queue and a counter instead of per-frame
// identifiers, and what makes the epoch (kept in-process, not on the wire) sufficient: a
// single writer per stage socket and no redials in v1 mean a stale reply can only be one
// that was sent before the epoch advanced, never one from a different production.
struct SpeculativeChunk {
    std::uint64_t epoch = 0;
    std::uint32_t base_position = 0;
    std::vector<std::uint32_t> guesses;
};

struct PipelineState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<SpeculativeChunk> inflight;
    std::uint64_t epoch = 0;
    bool finished = false;          // receiver decided the request is done (EOG or token limit)
    // sent[i]: frames placed on stages[i]'s socket so far. stage_finished[i]: no more frames
    // will EVER be placed on stages[i]'s socket. sent[0]/stage_finished[0] are the sender's;
    // sent[i>0]/stage_finished[i>0] are set by the relay thread forwarding INTO stage i (the
    // relay reading stage i-1's replies). A downstream thread (relay i, or the receiver for
    // i == stages.size()-1) may only call receive() on stages[i] once sent[i] has grown past
    // its own processed count -- otherwise the frame proving there is something to read may
    // not exist yet, and receive() would block for up to the socket timeout waiting for a
    // frame that was never going to be the next one sent (see design doc addendum: an earlier
    // version of this function let each stage guess when its predecessor was done, which is
    // provably too early -- the predecessor cannot know it sent the LAST frame until the
    // reply for that frame has propagated all the way to the receiver, which happens strictly
    // after the predecessor already forwarded it).
    std::vector<std::size_t> sent;
    std::vector<bool> stage_finished;
    bool correction_pending = false;
    std::uint32_t correction_position = 0;
    std::uint32_t correction_token = 0;
    std::exception_ptr error;

    explicit PipelineState(std::size_t stage_count)
        : sent(stage_count, 0), stage_finished(stage_count, false) {}
};

void pipeline_sender(Connection& stage0, DraftModel& draft, int draft_tokens,
    std::uint32_t start_position, std::uint32_t start_token, std::uint64_t session,
    std::uint64_t request, std::size_t max_inflight, PipelineState& state) {
    try {
        std::uint32_t base_position = start_position;
        std::uint32_t carry = start_token;
        for (;;) {
            std::uint64_t epoch;
            bool needs_rollback = false;
            {
                std::unique_lock<std::mutex> lock(state.mutex);
                state.cv.wait(lock, [&] {
                    return state.error || state.finished || state.inflight.size() < max_inflight;
                });
                if (state.error || state.finished) break;
                if (state.correction_pending) {
                    base_position = state.correction_position;
                    carry = state.correction_token;
                    state.correction_pending = false;
                    needs_rollback = true;
                }
                epoch = state.epoch;
            }
            // The draft model free-runs forward every round (see below), so on a real
            // rejection its own KV must be rolled back to the corrected position too, or it
            // silently keeps decoding a continuation nobody committed to until it runs off
            // the end of its context.
            if (needs_rollback) draft.rollback(base_position);

            const std::size_t count = static_cast<std::size_t>(draft_tokens);
            std::vector<std::uint32_t> guesses = draft.propose(carry, count);

            po::Frame block;
            block.type = po::Type::token;
            block.session = session;
            block.request = request;
            block.position = base_position;
            block.rows = static_cast<std::uint32_t>(count);
            block.payload.resize(count * 4);
            po::put32(block.payload.data(), carry);
            for (std::size_t index = 1; index < count; ++index) {
                po::put32(block.payload.data() + index * 4, guesses[index - 1]);
            }
            stage0.send(block);

            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.inflight.push_back({epoch, base_position, guesses});
                ++state.sent[0];
            }
            state.cv.notify_all();

            base_position += static_cast<std::uint32_t>(count);
            carry = guesses.back();
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.error) state.error = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.stage_finished[0] = true;
    }
    state.cv.notify_all();
}

void validate_relay_activation(const po::Frame& frame, std::uint64_t session,
    std::uint64_t request, std::uint32_t hidden) {
    if (frame.type != po::Type::speculative_activation || frame.session != session
        || frame.request != request || frame.rows == 0 || frame.cols != hidden
        || frame.dtype != po::DType::f32le) {
        throw std::runtime_error("invalid pipelined relay activation metadata");
    }
    const std::uint64_t values = std::uint64_t(frame.rows) * frame.cols;
    if (values > (po::max_payload - 8) / sizeof(float)
        || frame.payload.size() != 8 + values * sizeof(float)) {
        throw std::runtime_error("invalid pipelined relay activation size");
    }
}

// Forwards stage[read_index]'s replies to stage[write_index] (write_index == read_index + 1).
// Only calls receive() once state.sent[read_index] proves a corresponding frame was already
// handed to that stage's socket; see the PipelineState comment for why that ordering matters.
void pipeline_relay(Connection& upstream, Connection& downstream, std::uint64_t session,
    std::uint64_t request, std::uint32_t hidden, std::size_t read_index,
    std::size_t write_index, PipelineState& state) {
    std::size_t processed = 0;
    try {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(state.mutex);
                state.cv.wait(lock, [&] {
                    return state.error || processed < state.sent[read_index]
                        || (state.stage_finished[read_index] && processed >= state.sent[read_index]);
                });
                if (state.error) break;
                if (state.stage_finished[read_index] && processed >= state.sent[read_index]) break;
            }
            po::Frame reply = upstream.receive();
            validate_relay_activation(reply, session, request, hidden);
            downstream.send(reply);
            ++processed;
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                ++state.sent[write_index];
            }
            state.cv.notify_all();
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.error) state.error = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.stage_finished[write_index] = true;
    }
    state.cv.notify_all();
}

// Runs on the calling thread. Reads the tail's replies in send order and applies the same
// accept rule as generate()'s synchronous round, per reply instead of per round.
void pipeline_receiver(Connection& tail, std::uint64_t session,
    std::uint64_t request, int token_limit, DraftModel& draft, RequestResult& output,
    std::size_t read_index, PipelineState& state) {
    std::size_t processed = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [&] {
                return state.error || processed < state.sent[read_index]
                    || (state.stage_finished[read_index] && processed >= state.sent[read_index]);
            });
            if (state.error) return;
            if (state.stage_finished[read_index] && processed >= state.sent[read_index]) return;
        }

        po::Frame response;
        try {
            response = tail.receive();
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.error) state.error = std::current_exception();
            state.cv.notify_all();
            return;
        }
        ++processed;

        SpeculativeChunk chunk;
        bool should_apply;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (state.inflight.empty()) {
                if (!state.error) {
                    state.error = std::make_exception_ptr(
                        std::runtime_error("pipelined reply with no in-flight chunk"));
                }
                state.cv.notify_all();
                return;
            }
            chunk = std::move(state.inflight.front());
            state.inflight.pop_front();
            should_apply = !state.finished && chunk.epoch == state.epoch;
        }

        try {
            if (response.type != po::Type::result || response.session != session
                || response.request != request
                || response.position != chunk.base_position + chunk.guesses.size()
                || response.rows != chunk.guesses.size() || response.cols != 0
                || response.dtype != po::DType::none
                || response.payload.size() != 8 + static_cast<std::size_t>(response.rows) * 4) {
                throw std::runtime_error("invalid pipelined speculative result");
            }

            if (should_apply) {
                std::vector<std::uint32_t> checked;
                checked.reserve(response.rows);
                for (std::uint32_t row = 0; row < response.rows; ++row) {
                    checked.push_back(po::get32(response.payload.data() + 8
                        + static_cast<std::size_t>(row) * 4));
                }
                const std::uint64_t compute = po::get64(response.payload.data());

                std::size_t accepted = 0;
                bool stopped = false;
                for (; accepted < chunk.guesses.size() && chunk.guesses[accepted] == checked[accepted];
                        ++accepted) {
                    output.output += draft.piece(chunk.guesses[accepted]);
                    output.metrics.token_ids.push_back(chunk.guesses[accepted]);
                    output.final_token = chunk.guesses[accepted];
                    // The sender always drafts a full round (no shrink-to-fit like generate()'s
                    // count = min(draft_tokens, remaining) -- it can't know the confirmed count
                    // in advance under pipelining). Stopping here instead, once token_limit is
                    // reached, produces the identical committed token sequence: verification is
                    // causal, so checked[i] depends only on chunk.guesses[0..i], never on tokens
                    // drafted after it in the same block, so truncating the applied portion post
                    // hoc is equivalent to never having drafted the rest.
                    const bool eog_now = draft.is_eog(chunk.guesses[accepted]);
                    if (eog_now) output.eog = true;
                    if (eog_now
                        || static_cast<int>(output.metrics.token_ids.size()) >= token_limit) {
                        stopped = true;
                        ++accepted;
                        break;
                    }
                }
                output.metrics.accepted_draft_tokens += accepted;
                if (!stopped && accepted < chunk.guesses.size()) {
                    output.final_token = checked[accepted];
                    output.output += draft.piece(output.final_token);
                    output.metrics.token_ids.push_back(output.final_token);
                    output.eog = draft.is_eog(output.final_token);
                }
                const std::size_t committed_inputs = stopped ? accepted
                    : std::min<std::size_t>(chunk.guesses.size(), accepted + 1);
                const std::uint32_t keep_position = chunk.base_position
                    + static_cast<std::uint32_t>(committed_inputs);
                output.position = keep_position;

                output.metrics.speculative_rounds++;
                output.metrics.proposed_tokens += chunk.guesses.size();
                const std::size_t produced = stopped ? accepted
                    : (accepted < chunk.guesses.size() ? accepted + 1 : accepted);
                if (produced == 0) throw std::runtime_error("speculative round produced no token");
                // Reduced-fidelity metrics vs. generate(): only the tail's own compute is
                // attributed here. Per-hop breakdown (A/middle stage compute, wire time) would
                // need relay threads to thread per-chunk accumulators through the pipeline;
                // deferred, since these are diagnostic, not correctness (see design doc).
                for (std::size_t token = 0; token < produced; ++token) {
                    output.metrics.b_compute_ms.push_back(compute / 1e6 / produced);
                }

                const bool fully_accepted = committed_inputs == chunk.guesses.size();
                const bool reached_limit
                    = static_cast<int>(output.metrics.token_ids.size()) >= token_limit;
                std::lock_guard<std::mutex> lock(state.mutex);
                if (output.eog || reached_limit) {
                    state.finished = true;
                } else if (!fully_accepted) {
                    ++state.epoch;
                    state.correction_pending = true;
                    state.correction_position = keep_position;
                    state.correction_token = output.final_token;
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.error) state.error = std::current_exception();
            state.cv.notify_all();
            return;
        }

        state.cv.notify_all();
        // Loop back to the top: the wait predicate there (state.sent[read_index] /
        // state.stage_finished[read_index]) decides whether more replies are coming.
    }
}

RequestResult generate_pipelined(const StageConnections& stages, const Manifest& manifest,
    std::uint64_t session, std::uint64_t request, std::uint32_t position,
    const std::string& prompt, int token_limit, bool preserve_session, DraftModel& draft,
    int draft_tokens, std::size_t pipeline_depth, Connection* ring_return = nullptr) {
    if (stages.empty()) throw std::runtime_error("replica has no stages");
    if (pipeline_depth < 1) throw std::runtime_error("pipeline depth must be at least 1");

    const auto request_start = Clock::now();
    RequestResult output;
    po::Frame input;
    input.type = po::Type::prompt;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.assign(prompt.begin(), prompt.end());

    // Only the prefill uses ring_return here -- route_speculative and the pipelined
    // sender/relay/receiver threads below remain hub-and-spoke; ring integration for the
    // speculative/pipelined path is out of scope for this phase (see the design doc).
    const auto prefill_start = Clock::now();
    Result result = route_step(stages, input, manifest.hidden, output.metrics, false, ring_return);
    output.metrics.prefill_ms = elapsed_ns(prefill_start) / 1e6;
    output.metrics.ttft_ms = elapsed_ns(request_start) / 1e6;
    output.output += result.text;
    output.metrics.token_ids.push_back(result.token);
    output.position = result.position;
    output.final_token = result.token;
    output.eog = result.eog;

    if (position != 0 || draft.start(prompt) != result.position
        || draft.piece(result.token) != result.text) {
        throw std::runtime_error("draft and target tokenizers do not match");
    }

    if (!output.eog && static_cast<int>(output.metrics.token_ids.size()) < token_limit) {
        PipelineState state(stages.size());
        std::thread sender(pipeline_sender, std::ref(*stages.front()), std::ref(draft),
            draft_tokens, output.position, output.final_token, session, request,
            pipeline_depth, std::ref(state));

        std::vector<std::thread> relays;
        relays.reserve(stages.size() > 0 ? stages.size() - 1 : 0);
        for (std::size_t index = 0; index + 1 < stages.size(); ++index) {
            relays.emplace_back(pipeline_relay, std::ref(*stages[index]),
                std::ref(*stages[index + 1]), session, request, manifest.hidden, index,
                index + 1, std::ref(state));
        }

        pipeline_receiver(*stages.back(), session, request, token_limit,
            draft, output, stages.size() - 1, state);

        sender.join();
        for (auto& relay : relays) relay.join();

        if (state.error) std::rethrow_exception(state.error);
    }

    if (preserve_session) {
        commit_final_token(stages, session, request, output.position,
            output.final_token, manifest.hidden);
        ++output.position;
    }
    control_all(stages, po::Type::end_request, session, request);
    output.metrics.latency_ms = elapsed_ns(request_start) / 1e6;
    return output;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0;
    double sum = 0;
    for (double value : values) sum += value;
    return sum / values.size();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(fraction * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

std::string json_escape(std::string_view value) {
    std::string output;
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (byte < 0x20) {
                char escaped[7];
                std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
                output += escaped;
            } else output += static_cast<char>(byte);
        }
    }
    return output;
}

std::string worker_metrics(Connection& connection) {
    po::Frame input;
    input.type = po::Type::metrics;
    auto [output, ignored] = connection.exchange(input);
    (void) ignored;
    if (output.type != po::Type::metrics || output.session != 0 || output.request != 0
        || output.rows != 0 || output.cols != 0 || output.dtype != po::DType::none
        || output.payload.empty()) {
        throw std::runtime_error("invalid worker metrics");
    }
    return {output.payload.begin(), output.payload.end()};
}

void shutdown(Connection& connection) {
    po::Frame input;
    input.type = po::Type::shutdown;
    auto [output, ignored] = connection.exchange(input);
    (void) ignored;
    require_ack(output, input);
}

class ClientError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class JobKind { create, reset, destroy, generate };
enum class JobState { queued, active, done, cancelled };

struct ClientResponse {
    bool ok = false;
    std::string error;
    RequestResult result;
};

struct Job {
    JobKind kind = JobKind::generate;
    std::uint64_t session = 0;
    std::uint64_t request = 0;
    std::string prompt;
    int tokens = 0;
    Clock::time_point enqueued;
    Clock::time_point deadline;
    std::atomic<JobState> state{JobState::queued};
    std::promise<ClientResponse> completion;
};

struct RequestKey {
    std::uint64_t session = 0;
    std::uint64_t request = 0;
    bool operator==(const RequestKey&) const = default;
};

struct RequestKeyHash {
    std::size_t operator()(const RequestKey& key) const {
        return std::hash<std::uint64_t>{}(key.session)
            ^ (std::hash<std::uint64_t>{}(key.request) << 1);
    }
};

po::socket_t listen_on(std::string_view endpoint);

// Ring direct-return (docs/PIPELINED_SPECULATION_V1.md phase 4): blocks until the tail stage's
// --next dials in. Must be called before, or concurrently with, the tail stage starting up --
// there is no retry on this side, matching listen_on's existing single-shot bind/listen/accept
// pattern used elsewhere in this file.
std::unique_ptr<Connection> accept_ring_return(const std::string& endpoint) {
    const po::socket_t listener = listen_on(endpoint);
    std::fprintf(stderr, "ring: waiting for the tail stage to connect at %s\n", endpoint.c_str());
    const po::socket_t accepted = accept(listener, nullptr, nullptr);
    po::close_socket(listener);
    if (accepted == po::invalid_socket) throw std::runtime_error("ring return accept failed");
    std::fprintf(stderr, "ring: tail stage connected\n");
    return std::make_unique<Connection>(accepted);
}

class Replica {
public:
    Replica(const Manifest& manifest, const std::vector<std::string>& providers,
        const Options& options)
        : manifest_(manifest), draft_tokens_(options.draft_tokens),
          pipeline_depth_(static_cast<std::size_t>(options.pipeline_depth)) {
        for (const std::string& endpoint : providers) {
            connections_.push_back(std::make_unique<Connection>(endpoint));
            stages_.push_back(connections_.back().get());
            kv_bytes_per_session_ += json_uint64(worker_metrics(*stages_.back()),
                "kv_bytes_per_session");
        }
        if (!options.draft_model.empty()) {
            draft_ = std::make_unique<DraftModel>(
                options.draft_model, manifest.context, options.draft_gpu_layers);
        }
        if (!options.ring_return.empty()) {
            ring_return_connection_ = accept_ring_return(options.ring_return);
            ring_return_ = ring_return_connection_.get();
        }
    }

    Replica(const Manifest& manifest, std::vector<std::unique_ptr<Connection>> connections,
        const Options& options)
        : manifest_(manifest), connections_(std::move(connections)),
          draft_tokens_(options.draft_tokens),
          pipeline_depth_(static_cast<std::size_t>(options.pipeline_depth)) {
        for (const auto& connection : connections_) {
            stages_.push_back(connection.get());
            kv_bytes_per_session_ += json_uint64(worker_metrics(*connection),
                "kv_bytes_per_session");
        }
        if (!options.draft_model.empty()) {
            draft_ = std::make_unique<DraftModel>(
                options.draft_model, manifest.context, options.draft_gpu_layers);
        }
        if (!options.ring_return.empty()) {
            ring_return_connection_ = accept_ring_return(options.ring_return);
            ring_return_ = ring_return_connection_.get();
        }
    }

    ClientResponse run(const Job& job) {
        switch (job.kind) {
        case JobKind::create:
            if (job.session == 0 || sessions_.contains(job.session)) {
                throw ClientError("session already exists or is invalid");
            }
            control_all(stages_, po::Type::create_session, job.session);
            sessions_.emplace(job.session, SessionState{});
            update_memory();
            return {.ok = true};
        case JobKind::reset: {
            SessionState& session = require_session(job.session);
            control_all(stages_, po::Type::reset_session, job.session);
            session = {};
            return {.ok = true};
        }
        case JobKind::destroy:
            require_session(job.session);
            control_all(stages_, po::Type::destroy_session, job.session);
            sessions_.erase(job.session);
            update_memory();
            return {.ok = true};
        case JobKind::generate:
            return generate_request(job);
        }
        throw ClientError("unknown queued operation");
    }

    void shutdown_workers() {
        for (Connection* stage : stages_) shutdown(*stage);
    }

    std::size_t resident_sessions() const { return resident_sessions_.load(); }
    std::uint64_t kv_memory_bytes() const { return kv_memory_bytes_.load(); }

private:
    struct SessionState {
        std::uint32_t position = 0;
    };

    SessionState& require_session(std::uint64_t id) {
        const auto found = sessions_.find(id);
        if (found == sessions_.end()) throw ClientError("unknown session ID");
        return found->second;
    }

    ClientResponse generate_request(const Job& job) {
        const bool stateless = job.session == 0;
        std::uint64_t session_id = job.session;
        SessionState* state = nullptr;
        if (stateless) {
            while (session_id == 0 || sessions_.contains(session_id)) session_id = next_ephemeral_--;
            control_all(stages_, po::Type::create_session, session_id);
        } else {
            state = &require_session(session_id);
        }

        const std::uint32_t position = state ? state->position : 0;
        const std::uint64_t provider_request = next_provider_request_++;
        RequestResult result = draft_ && pipeline_depth_ > 0
            ? generate_pipelined(stages_, manifest_, session_id, provider_request,
                position, job.prompt, job.tokens, !stateless, *draft_, draft_tokens_,
                pipeline_depth_, ring_return_)
            : generate(stages_, manifest_, session_id, provider_request,
                position, job.prompt, job.tokens, !stateless, draft_.get(), draft_tokens_,
                ring_return_);
        if (state) {
            state->position = result.position;
        } else {
            control_all(stages_, po::Type::destroy_session, session_id);
        }
        ClientResponse response;
        response.ok = true;
        response.result = std::move(result);
        return response;
    }

    void update_memory() {
        resident_sessions_.store(sessions_.size());
        kv_memory_bytes_.store(kv_bytes_per_session_ * sessions_.size());
    }

    Manifest manifest_;
    std::vector<std::unique_ptr<Connection>> connections_;
    StageConnections stages_;
    std::unordered_map<std::uint64_t, SessionState> sessions_;
    std::uint64_t next_ephemeral_ = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t next_provider_request_ = 1;
    std::uint64_t kv_bytes_per_session_ = 0;
    std::atomic<std::size_t> resident_sessions_{0};
    std::atomic<std::uint64_t> kv_memory_bytes_{0};
    // Optional: set when --draft-model is given, so the served path (--listen) can run
    // speculative (and, with --pipeline-depth, pipelined) decoding, not just the --prompt CLI
    // benchmark path. Safe unsynchronized: the scheduler's single executor thread guarantees
    // only one generate()/generate_pipelined() call is ever active against this replica.
    std::unique_ptr<DraftModel> draft_;
    int draft_tokens_ = 4;
    std::size_t pipeline_depth_ = 0;
    // Optional: set when --ring-return is given (docs/PIPELINED_SPECULATION_V1.md phase 4).
    // Only route_step (prefill and non-speculative decode) uses it; route_speculative and the
    // pipelined sender/relay/receiver stay hub-and-spoke regardless.
    std::unique_ptr<Connection> ring_return_connection_;
    Connection* ring_return_ = nullptr;
};

class Scheduler {
public:
    Scheduler(const Manifest& manifest, const Options& options)
        : queue_(static_cast<std::size_t>(options.queue_capacity)),
          replica_(manifest, options.providers, options),
          default_timeout_(options.queue_timeout_ms),
          shutdown_workers_(options.shutdown_workers), started_(Clock::now()),
          executor_() {
        executor_ = std::thread([this] { execute(); });
    }

    Scheduler(const Manifest& manifest, const Options& options,
        std::vector<std::unique_ptr<Connection>> connections)
        : queue_(static_cast<std::size_t>(options.queue_capacity)),
          replica_(manifest, std::move(connections), options),
          default_timeout_(options.queue_timeout_ms),
          shutdown_workers_(options.shutdown_workers), started_(Clock::now()),
          executor_() {
        executor_ = std::thread([this] { execute(); });
    }

    ~Scheduler() { stop(); }

    ClientResponse submit(JobKind kind, const po::Frame& frame, int tokens = 0,
        int timeout_ms = 0) {
        if (unavailable_.load() || stopping_.load()) {
            count_rejected(kind);
            return {.error = "replica_unavailable"};
        }
        auto job = std::make_shared<Job>();
        job->kind = kind;
        job->session = frame.session;
        job->request = frame.request;
        job->tokens = tokens;
        job->prompt.assign(frame.payload.begin(), frame.payload.end());
        job->enqueued = Clock::now();
        const int effective_timeout = timeout_ms > 0 ? timeout_ms : default_timeout_;
        job->deadline = job->enqueued + std::chrono::milliseconds(effective_timeout);
        std::future<ClientResponse> future = job->completion.get_future();

        if (kind == JobKind::generate) {
            std::lock_guard lock(pending_mutex_);
            const RequestKey key{job->session, job->request};
            if (pending_.contains(key)) {
                count_rejected(kind);
                return {.error = "duplicate_request"};
            }
            pending_.emplace(key, job);
        }

        const std::uint64_t fairness_key = frame.session != 0 ? frame.session : frame.request;
        if (!queue_.push(fairness_key, job)) {
            erase_pending(*job);
            count_rejected(kind);
            return {.error = "queue_full"};
        }
        update_max_depth();

        if (kind == JobKind::generate
            && future.wait_until(job->deadline) == std::future_status::timeout) {
            cancel(frame.session, frame.request, true);
        }
        return future.get();
    }

    ClientResponse cancel(std::uint64_t session, std::uint64_t request,
        bool timed_out = false) {
        std::shared_ptr<Job> job;
        {
            std::lock_guard lock(pending_mutex_);
            const auto found = pending_.find({session, request});
            if (found == pending_.end()) return {.error = "request_not_queued"};
            job = found->second;
            JobState expected = JobState::queued;
            if (!job->state.compare_exchange_strong(expected, JobState::cancelled)) {
                return {.error = "request_not_queued"};
            }
            pending_.erase(found);
        }
        queue_.remove_if([&](const std::shared_ptr<Job>& candidate) {
            return candidate == job;
        });
        {
            std::lock_guard lock(metrics_mutex_);
            if (timed_out) {
                ++failed_;
                ++timed_out_;
            } else {
                ++cancelled_;
            }
        }
        job->completion.set_value({.error = timed_out ? "queue_timeout" : "cancelled"});
        return {.ok = true};
    }

    std::string metrics_json() const {
        const auto depths = queue_.depths();
        std::lock_guard lock(metrics_mutex_);
        const double seconds = std::max(1e-9, elapsed_ns(started_) / 1e9);
        std::ostringstream output;
        output << "{\"queue_depth\":" << queue_.size()
            << ",\"queue_depth_max\":" << max_queue_depth_
            << ",\"queue_wait_ms\":" << mean(queue_wait_ms_)
            << ",\"time_to_first_token_ms\":" << mean(ttft_ms_)
            << ",\"request_latency_p50_ms\":" << percentile(latency_ms_, .50)
            << ",\"request_latency_p95_ms\":" << percentile(latency_ms_, .95)
            << ",\"request_latency_p99_ms\":" << percentile(latency_ms_, .99)
            << ",\"provider_a_compute_ms_per_step\":" << mean(a_compute_ms_)
            << ",\"middle_compute_ms_per_step\":" << mean(middle_compute_ms_)
            << ",\"network_ms_per_step\":" << mean(network_ms_)
            << ",\"provider_b_compute_ms_per_step\":" << mean(b_compute_ms_)
            << ",\"requests_completed\":" << completed_
            << ",\"requests_failed\":" << failed_
            << ",\"requests_cancelled\":" << cancelled_
            << ",\"requests_timed_out\":" << timed_out_
            << ",\"resident_sessions\":" << replica_.resident_sessions()
            << ",\"kv_memory_bytes\":" << replica_.kv_memory_bytes()
            << ",\"generated_tokens\":" << generated_tokens_
            << ",\"activation_bytes_per_generated_token\":"
                << (generated_tokens_ == 0 ? 0 : activation_bytes_ / generated_tokens_)
            << ",\"aggregate_generated_tokens_per_second\":"
                << generated_tokens_ / seconds
            << ",\"active_request\":" << (active_.load() ? "true" : "false")
            << ",\"replica_available\":" << (!unavailable_.load() ? "true" : "false")
            << ",\"model_reload_count_per_worker\":1,\"per_session_queued\":{";
        bool first = true;
        for (const auto& [session, depth] : depths) {
            if (!first) output << ',';
            first = false;
            output << '\"' << session << "\":" << depth;
        }
        output << "}}";
        return output.str();
    }

    void stop() {
        std::lock_guard stop_lock(stop_mutex_);
        if (stopped_) return;
        stopping_.store(true);
        for (auto& job : queue_.close()) {
            JobState expected = JobState::queued;
            if (!job->state.compare_exchange_strong(expected, JobState::cancelled)) continue;
            erase_pending(*job);
            if (job->kind == JobKind::generate) {
                std::lock_guard lock(metrics_mutex_);
                ++failed_;
            }
            job->completion.set_value({.error = "coordinator_stopping"});
        }
        if (executor_.joinable()) executor_.join();
        stopped_ = true;
    }

private:
    void execute() {
        while (const auto next = queue_.pop()) {
            const std::shared_ptr<Job>& job = *next;
            if (Clock::now() >= job->deadline) {
                JobState expected = JobState::queued;
                if (job->state.compare_exchange_strong(expected, JobState::cancelled)) {
                    erase_pending(*job);
                    {
                        std::lock_guard lock(metrics_mutex_);
                        if (job->kind == JobKind::generate) {
                            ++failed_;
                            ++timed_out_;
                        }
                    }
                    job->completion.set_value({.error = "queue_timeout"});
                }
                continue;
            }
            JobState expected = JobState::queued;
            if (!job->state.compare_exchange_strong(expected, JobState::active)) continue;
            active_.store(true);
            const double queue_wait = elapsed_ns(job->enqueued) / 1e6;
            try {
                ClientResponse response = replica_.run(*job);
                complete(job, std::move(response), queue_wait);
            } catch (const ClientError& error) {
                complete(job, {.error = error.what()}, queue_wait);
            } catch (const std::exception& error) {
                unavailable_.store(true);
                complete(job, {.error = std::string("provider_failure: ") + error.what()},
                    queue_wait);
                fail_pending("provider_disconnected");
            }
            active_.store(false);
            if (unavailable_.load()) break;
        }
        if (shutdown_workers_ && !unavailable_.load()) {
            try { replica_.shutdown_workers(); }
            catch (...) { unavailable_.store(true); }
        }
    }

    void complete(const std::shared_ptr<Job>& job, ClientResponse response,
        double queue_wait) {
        job->state.store(JobState::done);
        erase_pending(*job);
        if (job->kind == JobKind::generate) {
            std::lock_guard lock(metrics_mutex_);
            if (response.ok) {
                response.result.metrics.queue_wait_ms = queue_wait;
                response.result.metrics.ttft_ms += queue_wait;
                response.result.metrics.latency_ms += queue_wait;
                queue_wait_ms_.push_back(queue_wait);
                ttft_ms_.push_back(response.result.metrics.ttft_ms);
                latency_ms_.push_back(response.result.metrics.latency_ms);
                a_compute_ms_.insert(a_compute_ms_.end(),
                    response.result.metrics.a_compute_ms.begin(),
                    response.result.metrics.a_compute_ms.end());
                middle_compute_ms_.insert(middle_compute_ms_.end(),
                    response.result.metrics.middle_compute_ms.begin(),
                    response.result.metrics.middle_compute_ms.end());
                network_ms_.insert(network_ms_.end(),
                    response.result.metrics.network_ms.begin(),
                    response.result.metrics.network_ms.end());
                b_compute_ms_.insert(b_compute_ms_.end(),
                    response.result.metrics.b_compute_ms.begin(),
                    response.result.metrics.b_compute_ms.end());
                activation_bytes_ += response.result.metrics.activation_bytes;
                generated_tokens_ += response.result.metrics.token_ids.size();
                ++completed_;
            } else {
                ++failed_;
            }
        }
        job->completion.set_value(std::move(response));
    }

    void fail_pending(const std::string& error) {
        for (auto& job : queue_.close()) {
            JobState expected = JobState::queued;
            if (!job->state.compare_exchange_strong(expected, JobState::cancelled)) continue;
            erase_pending(*job);
            if (job->kind == JobKind::generate) {
                std::lock_guard lock(metrics_mutex_);
                ++failed_;
            }
            job->completion.set_value({.error = error});
        }
    }

    void erase_pending(const Job& job) {
        if (job.kind != JobKind::generate) return;
        std::lock_guard lock(pending_mutex_);
        pending_.erase({job.session, job.request});
    }

    void count_rejected(JobKind kind) {
        if (kind != JobKind::generate) return;
        std::lock_guard lock(metrics_mutex_);
        ++failed_;
    }

    void update_max_depth() {
        const std::size_t depth = queue_.size();
        std::lock_guard lock(metrics_mutex_);
        max_queue_depth_ = std::max(max_queue_depth_, depth);
    }

    po::FairQueue<std::shared_ptr<Job>> queue_;
    Replica replica_;
    const int default_timeout_;
    const bool shutdown_workers_;
    const Clock::time_point started_;
    std::thread executor_;
    std::atomic<bool> active_{false};
    std::atomic<bool> unavailable_{false};
    std::atomic<bool> stopping_{false};
    mutable std::mutex pending_mutex_;
    std::unordered_map<RequestKey, std::shared_ptr<Job>, RequestKeyHash> pending_;
    mutable std::mutex metrics_mutex_;
    std::vector<double> queue_wait_ms_;
    std::vector<double> ttft_ms_;
    std::vector<double> latency_ms_;
    std::vector<double> a_compute_ms_;
    std::vector<double> middle_compute_ms_;
    std::vector<double> network_ms_;
    std::vector<double> b_compute_ms_;
    std::uint64_t completed_ = 0;
    std::uint64_t failed_ = 0;
    std::uint64_t cancelled_ = 0;
    std::uint64_t timed_out_ = 0;
    std::uint64_t generated_tokens_ = 0;
    std::uint64_t activation_bytes_ = 0;
    std::size_t max_queue_depth_ = 0;
    std::mutex stop_mutex_;
    bool stopped_ = false;
};

po::socket_t listen_on(std::string_view endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == endpoint.size()) {
        throw std::runtime_error("invalid listen endpoint");
    }
    const std::string host(endpoint.substr(0, colon));
    const std::string port(endpoint.substr(colon + 1));
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        throw std::runtime_error("could not resolve listen endpoint");
    }
    po::socket_t listener = po::invalid_socket;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        listener = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (listener == po::invalid_socket) continue;
        const int enabled = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
            reinterpret_cast<const char*>(&enabled), sizeof(enabled));
        if (bind(listener, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0
            && listen(listener, 128) == 0) break;
        po::close_socket(listener);
        listener = po::invalid_socket;
    }
    freeaddrinfo(addresses);
    if (listener == po::invalid_socket) throw std::runtime_error("bind/listen failed");
    return listener;
}

void set_client_timeout(po::socket_t socket) {
#ifdef _WIN32
    const DWORD timeout = 30000;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
        reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
        reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    const timeval timeout{30, 0};
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

po::Frame client_reply(const po::Frame& input, ClientResponse response) {
    if (!response.ok) return po::error_frame(input, response.error);
    po::Frame output;
    output.session = input.session;
    output.request = input.request;
    if (input.type == po::Type::prompt) {
        output.type = po::Type::client_result;
        output.position = response.result.position;
        output.rows = static_cast<std::uint32_t>(response.result.metrics.token_ids.size());
        output.payload.assign(response.result.output.begin(), response.result.output.end());
    } else {
        output.type = po::Type::ack;
    }
    return output;
}

void close_listener(po::socket_t listener) {
#ifdef _WIN32
    ::shutdown(listener, SD_BOTH);
#else
    ::shutdown(listener, SHUT_RDWR);
#endif
    po::close_socket(listener);
}

void handle_client(po::socket_t client, Scheduler& scheduler,
    std::atomic<bool>& stopping, po::socket_t listener) {
    std::string error;
    while (!stopping.load()) {
        po::Frame input;
        po::Frame output;
        if (!po::recv_frame(client, input, error)) break;
        try {
        switch (input.type) {
        case po::Type::create_session:
            if (!po::empty_control(input) || input.session == 0 || input.request != 0) {
                throw ClientError("invalid create-session request");
            }
            output = client_reply(input, scheduler.submit(JobKind::create, input));
            break;
        case po::Type::reset_session:
            if (!po::empty_control(input) || input.session == 0 || input.request != 0) {
                throw ClientError("invalid reset-session request");
            }
            output = client_reply(input, scheduler.submit(JobKind::reset, input));
            break;
        case po::Type::destroy_session:
            if (!po::empty_control(input) || input.session == 0 || input.request != 0) {
                throw ClientError("invalid destroy-session request");
            }
            output = client_reply(input, scheduler.submit(JobKind::destroy, input));
            break;
        case po::Type::prompt:
            if (input.request == 0 || input.rows == 0 || input.rows > 4096
                || input.cols != 0 || input.dtype != po::DType::none
                || input.payload.empty() || input.payload.size() > 1024 * 1024) {
                throw ClientError("invalid generate request");
            }
            output = client_reply(input, scheduler.submit(JobKind::generate, input,
                static_cast<int>(input.rows), static_cast<int>(input.position)));
            break;
        case po::Type::cancel_request: {
            if (!po::empty_control(input) || input.request == 0) {
                throw ClientError("invalid cancellation request");
            }
            output = client_reply(input, scheduler.cancel(input.session, input.request));
            break;
        }
        case po::Type::metrics:
            if (!po::empty_control(input) || input.session != 0 || input.request != 0) {
                throw ClientError("invalid metrics request");
            }
            output.type = po::Type::metrics;
            {
                const std::string metrics = scheduler.metrics_json();
                output.payload.assign(metrics.begin(), metrics.end());
            }
            break;
        case po::Type::shutdown:
            if (!po::empty_control(input) || input.session != 0 || input.request != 0) {
                throw ClientError("invalid shutdown request");
            }
            output.type = po::Type::ack;
            break;
        default:
            throw ClientError("unsupported client frame type");
        }
        } catch (const std::exception& exception) {
            output = po::error_frame(input, exception.what());
        }
        if (!po::send_frame(client, output, error)) break;
        if (input.type == po::Type::shutdown && output.type == po::Type::ack) {
            if (!stopping.exchange(true)) close_listener(listener);
            break;
        }
    }
    po::close_socket(client);
}

void run_server(const Manifest& manifest, const Options& options,
    std::vector<std::unique_ptr<Connection>> connections = {}) {
    std::unique_ptr<Scheduler> scheduler = connections.empty()
        ? std::make_unique<Scheduler>(manifest, options)
        : std::make_unique<Scheduler>(manifest, options, std::move(connections));
    const po::socket_t listener = listen_on(options.listen);
    po::FairQueue<po::socket_t> clients(
        static_cast<std::size_t>(options.queue_capacity + options.client_threads));
    std::atomic<bool> stopping{false};
    std::vector<std::jthread> handlers;
    handlers.reserve(static_cast<std::size_t>(options.client_threads));
    for (int index = 0; index < options.client_threads; ++index) {
        handlers.emplace_back([&] {
            while (const auto client = clients.pop()) {
                handle_client(*client, *scheduler, stopping, listener);
            }
        });
    }
    std::uint64_t connection_id = 1;
    std::fprintf(stderr, "provider-owned v2 READY on %s queue_capacity=%d\n",
        options.listen.c_str(), options.queue_capacity);
    while (!stopping.load()) {
        const po::socket_t client = accept(listener, nullptr, nullptr);
        if (client == po::invalid_socket) {
            if (stopping.load()) break;
            throw std::runtime_error("client accept failed");
        }
        set_client_timeout(client);
        if (!clients.push(connection_id++, client)) {
            po::Frame source;
            const po::Frame rejected = po::error_frame(source, "connection_queue_full");
            std::string ignored;
            po::send_frame(client, rejected, ignored);
            po::close_socket(client);
        }
    }
    scheduler->stop();
    for (const po::socket_t client : clients.close()) {
        po::Frame source;
        const po::Frame rejected = po::error_frame(source, "coordinator_stopping");
        std::string ignored;
        po::send_frame(client, rejected, ignored);
        po::close_socket(client);
    }
    handlers.clear();
    std::fprintf(stderr, "provider-owned v2 stopped metrics=%s\n",
        scheduler->metrics_json().c_str());
}

void run_interactive(const Manifest& manifest, const Options& options,
    std::vector<std::unique_ptr<Connection>> connections) {
    if (connections.empty()) {
        for (const std::string& endpoint : options.providers) {
            connections.push_back(std::make_unique<Connection>(endpoint));
        }
    }
    StageConnections stages;
    for (const auto& connection : connections) stages.push_back(connection.get());
    if (stages.empty()) throw std::runtime_error("interactive replica has no providers");

    std::uint64_t next_id = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count()) | 1;
    const std::uint64_t session = next_id++;
    control_all(stages, po::Type::create_session, session);
    std::uint32_t position = 0;
    bool first_prompt = true;
    std::size_t requests = 0;
    std::size_t tokens = 0;

    dan::platform::clear_console();
    std::cout << "\n+------------------------------------------------------------------+\n"
        << "| DAN Coordinator v1.0.1                         REPLICA READY     |\n"
        << "+------------------------------------------------------------------+\n"
        << "| Model: " << manifest.model_id << "\n"
        << "| Providers: " << stages.size() << "  |  Context: " << manifest.context << " tokens\n"
        << "+------------------------------------------------------------------+\n"
        << "Type a message, or use /new, /stats, /quit.\n\n";

    for (;;) {
        std::cout << "You > " << std::flush;
        std::string text;
        if (!std::getline(std::cin, text) || text == "/quit") break;
        if (text.empty()) continue;
        if (text == "/new") {
            control_all(stages, po::Type::reset_session, session);
            position = 0;
            first_prompt = true;
            std::cout << "DAN > New conversation started.\n\n";
            continue;
        }
        if (text == "/stats") {
            std::cout << "DAN > " << requests << " requests, " << tokens
                << " generated tokens, " << stages.size() << " providers online.\n\n";
            continue;
        }
        const std::string prompt = (first_prompt
            ? "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
            : "\n<|im_start|>user\n")
            + text + "<|im_end|>\n<|im_start|>assistant\n";
        const RequestResult result = generate(stages, manifest, session, next_id++,
            position, prompt, options.tokens, true);
        position = result.position;
        first_prompt = false;
        ++requests;
        tokens += result.metrics.token_ids.size();
        std::cout << "DAN > " << result.output << "\n\n"
            << "      " << result.metrics.token_ids.size() << " tokens | "
            << static_cast<unsigned>(result.metrics.latency_ms) << " ms\n\n";
    }
    control_all(stages, po::Type::destroy_session, session);
    std::cout << "\nCoordinator stopped. Providers will reconnect automatically.\n";
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    int exit_code = 0;
    dan::platform::Process sidecar;
    try {
        dan::platform::configure_output();
#ifdef _WIN32
        if (argc == 1) start_packaged_network(sidecar, argv[0]);
#endif
        const Options options = parse_options(argc, argv);
        Manifest manifest = load_manifest(options.manifest);
        if (options.interactive) {
            std::cout << "+------------------------------------------------------------------+\n"
                << "| DAN Coordinator v1.0.1                         FORMING REPLICA    |\n"
                << "+------------------------------------------------------------------+\n"
                << "| Model: " << manifest.model_id << "\n"
                << "| Providers: " << (options.provider_listen.empty()
                    ? "configured endpoints" : options.provider_listen) << "\n"
                << "+------------------------------------------------------------------+\n"
                << "Reading model metadata and waiting for providers...\n\n";
        }
        if (options.provider_listen.empty() && (manifest.layers == 0 || manifest.hidden == 0)) {
            throw std::runtime_error(
                "metadata-derived manifests require automatic --provider-listen formation");
        }
        FormedReplica formed;
        if (!options.provider_listen.empty()) formed = form_replica(manifest, options);
        if (options.interactive) {
            run_interactive(manifest, options, std::move(formed.connections));
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
        if (!options.listen.empty()) {
            run_server(manifest, options, std::move(formed.connections));
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
        const auto connected_start = Clock::now();
        std::vector<std::unique_ptr<Connection>> provider_connections;
        StageConnections providers;
        if (!formed.connections.empty()) {
            provider_connections = std::move(formed.connections);
        } else {
            for (const std::string& endpoint : options.providers) {
                provider_connections.push_back(std::make_unique<Connection>(endpoint));
            }
        }
        for (const auto& connection : provider_connections) providers.push_back(connection.get());
        const double connect_ms = elapsed_ns(connected_start) / 1e6;
        std::unique_ptr<DraftModel> draft;
        if (!options.draft_model.empty()) {
            draft = std::make_unique<DraftModel>(
                options.draft_model, manifest.context, options.draft_gpu_layers);
        }
        std::unique_ptr<Connection> ring_return_connection;
        Connection* ring_return = nullptr;
        if (!options.ring_return.empty()) {
            ring_return_connection = accept_ring_return(options.ring_return);
            ring_return = ring_return_connection.get();
        }

        std::uint64_t next_id = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now().time_since_epoch()).count()) | 1;
        std::vector<std::uint64_t> persistent_sessions;
        std::unordered_map<std::uint64_t, std::uint32_t> positions;
        if (options.persistent) {
            for (int index = 0; index < options.resident_sessions; ++index) {
                const std::uint64_t session = next_id++;
                control_all(providers, po::Type::create_session, session);
                persistent_sessions.push_back(session);
                positions[session] = 0;
            }
        }

        std::vector<double> request_latencies;
        std::vector<double> a_compute;
        std::vector<double> middle_compute;
        std::vector<double> b_compute;
        std::vector<double> network;
        std::size_t activation_bytes = 0;
        std::size_t generated_tokens = 0;
        std::uint64_t speculative_rounds = 0;
        std::uint64_t proposed_tokens = 0;
        std::uint64_t accepted_draft_tokens = 0;
        double draft_ms = 0;
        std::vector<std::string> outputs;

        for (int index = 0; index < options.requests; ++index) {
            const std::uint64_t session = options.persistent
                ? persistent_sessions[static_cast<std::size_t>(index) % persistent_sessions.size()]
                : next_id++;
            if (!options.persistent) {
                control_all(providers, po::Type::create_session, session);
                positions[session] = 0;
            } else if (options.reset_between && index >= options.resident_sessions) {
                control_all(providers, po::Type::reset_session, session);
                positions[session] = 0;
            }
            const std::uint64_t request = next_id++;
            RequestResult result = options.pipeline_depth > 0
                ? generate_pipelined(providers, manifest,
                    session, request, positions[session],
                    options.prompts[static_cast<std::size_t>(index) % options.prompts.size()],
                    options.tokens, options.persistent, *draft, options.draft_tokens,
                    static_cast<std::size_t>(options.pipeline_depth), ring_return)
                : generate(providers, manifest,
                    session, request, positions[session],
                    options.prompts[static_cast<std::size_t>(index) % options.prompts.size()],
                    options.tokens, options.persistent, draft.get(), options.draft_tokens,
                    ring_return);
            positions[session] = result.position;
            std::printf("request=%d session=%llu tokens=%zu latency_ms=%.3f output=%s\n",
                index + 1, static_cast<unsigned long long>(session),
                result.metrics.token_ids.size(), result.metrics.latency_ms,
                result.output.c_str());
            request_latencies.push_back(result.metrics.latency_ms);
            a_compute.insert(a_compute.end(), result.metrics.a_compute_ms.begin(),
                result.metrics.a_compute_ms.end());
            middle_compute.insert(middle_compute.end(), result.metrics.middle_compute_ms.begin(),
                result.metrics.middle_compute_ms.end());
            b_compute.insert(b_compute.end(), result.metrics.b_compute_ms.begin(),
                result.metrics.b_compute_ms.end());
            network.insert(network.end(), result.metrics.network_ms.begin(),
                result.metrics.network_ms.end());
            activation_bytes += result.metrics.activation_bytes;
            generated_tokens += result.metrics.token_ids.size();
            speculative_rounds += result.metrics.speculative_rounds;
            proposed_tokens += result.metrics.proposed_tokens;
            accepted_draft_tokens += result.metrics.accepted_draft_tokens;
            draft_ms += result.metrics.draft_ms;
            outputs.push_back(std::move(result.output));
            if (!options.persistent) {
                control_all(providers, po::Type::destroy_session, session);
                positions.erase(session);
            }
            if (!options.expected.empty() && outputs.back() != options.expected) {
                throw std::runtime_error("deterministic output mismatch on request "
                    + std::to_string(index + 1));
            }
        }

        std::vector<std::string> provider_metrics;
        for (Connection* provider : providers) provider_metrics.push_back(worker_metrics(*provider));
        const std::string metrics_a = provider_metrics.front();
        const std::string metrics_b = provider_metrics.back();
        const double average_token_ms = mean(a_compute) + mean(middle_compute)
            + mean(b_compute) + mean(network)
            + (generated_tokens == 0 ? 0.0 : draft_ms / generated_tokens);
        const double decode_tokens_per_second = average_token_ms > 0 ? 1000 / average_token_ms : 0;
        const double p50 = percentile(request_latencies, 0.50);
        const double p95 = percentile(request_latencies, 0.95);
        const double p99 = percentile(request_latencies, 0.99);
        std::printf(
            "READY requests=%d tokens=%zu p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f decode_tok_s=%.3f A_ms/token=%.3f network_ms/token=%.3f B_ms/token=%.3f activation_bytes/token=%u speculative_rounds=%llu draft_accept=%.3f draft_ms/token=%.3f\n",
            options.requests, generated_tokens, p50, p95, p99, decode_tokens_per_second,
            mean(a_compute), mean(network), mean(b_compute),
            static_cast<unsigned>((providers.size() - 1) * manifest.hidden * 4),
            static_cast<unsigned long long>(speculative_rounds),
            proposed_tokens == 0 ? 0.0
                : static_cast<double>(accepted_draft_tokens) / proposed_tokens,
            generated_tokens == 0 ? 0.0 : draft_ms / generated_tokens);
        std::printf("provider_a_metrics=%s\nprovider_b_metrics=%s\n",
            metrics_a.c_str(), metrics_b.c_str());

        if (!options.report.empty()) {
            std::ofstream report(options.report, std::ios::binary);
            if (!report) throw std::runtime_error("could not create report");
            report << "{\n  \"runtime\": \"provider-owned-v1\",\n"
                << "  \"model\": \"" << json_escape(manifest.model_id) << "\",\n"
                << "  \"requests_served\": " << options.requests << ",\n"
                << "  \"tokens_generated\": " << generated_tokens << ",\n"
                << "  \"sessions_resident\": " << persistent_sessions.size() << ",\n"
                << "  \"connect_ms\": " << connect_ms << ",\n"
                << "  \"request_latency_p50_ms\": " << p50 << ",\n"
                << "  \"request_latency_p95_ms\": " << p95 << ",\n"
                << "  \"request_latency_p99_ms\": " << p99 << ",\n"
                << "  \"decode_tok_s\": " << decode_tokens_per_second << ",\n"
                << "  \"speculative_rounds\": " << speculative_rounds << ",\n"
                << "  \"draft_tokens_proposed\": " << proposed_tokens << ",\n"
                << "  \"draft_tokens_accepted\": " << accepted_draft_tokens << ",\n"
                << "  \"draft_acceptance\": " << (proposed_tokens == 0 ? 0.0
                    : static_cast<double>(accepted_draft_tokens) / proposed_tokens) << ",\n"
                << "  \"draft_ms_per_generated_token\": "
                    << (generated_tokens == 0 ? 0.0 : draft_ms / generated_tokens) << ",\n"
                << "  \"provider_a_compute_ms_per_token\": " << mean(a_compute) << ",\n"
                << "  \"network_ms_per_token\": " << mean(network) << ",\n"
                << "  \"provider_b_compute_ms_per_token\": " << mean(b_compute) << ",\n"
                << "  \"activation_bytes_per_token\": "
                << (providers.size() - 1) * manifest.hidden * 4 << ",\n"
                << "  \"activation_bytes_total\": " << activation_bytes << ",\n"
                << "  \"provider_a\": " << metrics_a << ",\n"
                << "  \"provider_b\": " << metrics_b << ",\n"
                << "  \"outputs\": [";
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                if (index != 0) report << ',';
                report << "\n    \"" << json_escape(outputs[index]) << "\"";
            }
            report << "\n  ]\n}\n";
        }

        for (const std::uint64_t session : persistent_sessions) {
            control_all(providers, po::Type::destroy_session, session);
        }
        if (options.shutdown_workers) {
            for (Connection* provider : providers) shutdown(*provider);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "provider-owned runtime unavailable: %s\n", error.what());
        exit_code = 1;
        if (argc == 1) {
            std::fputs("\nPress Enter to close DAN...", stderr);
            std::string ignored;
            std::getline(std::cin, ignored);
        }
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return exit_code;
}
