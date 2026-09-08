#include "provider_owned/protocol.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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
#ifdef _WIN32
        const DWORD timeout = 30000;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO,
            reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
        const timeval timeout{30, 0};
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

private:
    po::socket_t socket_ = po::invalid_socket;
};

struct Manifest {
    std::string model_id;
    std::string architecture;
    std::string quantization;
    std::uint32_t layers = 0;
    std::uint32_t hidden = 0;
    std::uint32_t split = 0;
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

Manifest load_manifest(const std::string& path) {
    const std::string json = read_file(path);
    Manifest manifest;
    manifest.model_id = json_string(json, "model_id");
    manifest.architecture = json_string(json, "architecture");
    manifest.quantization = json_string(json, "quantization");
    manifest.layers = json_uint(json, "layers");
    manifest.hidden = json_uint(json, "hidden_size");
    manifest.split = json_uint(json, "split_layer");
    if (manifest.architecture != "qwen2" || manifest.layers != 24
        || manifest.hidden != 896 || manifest.split != 12) {
        throw std::runtime_error("v1 requires the proven Qwen2.5 0.5B 12/12 manifest");
    }
    return manifest;
}

struct Options {
    std::string manifest;
    std::string provider_a;
    std::string provider_b;
    std::vector<std::string> prompts;
    std::string report;
    std::string expected;
    int tokens = 20;
    int requests = 1;
    int resident_sessions = 1;
    bool persistent = false;
    bool reset_between = false;
    bool shutdown_workers = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--persistent") { options.persistent = true; continue; }
        if (option == "--reset-between") { options.reset_between = true; continue; }
        if (option == "--shutdown-workers") { options.shutdown_workers = true; continue; }
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--manifest") options.manifest = value;
        else if (option == "--provider-a") options.provider_a = value;
        else if (option == "--provider-b") options.provider_b = value;
        else if (option == "--prompt") options.prompts.push_back(value);
        else if (option == "--tokens") options.tokens = std::stoi(value);
        else if (option == "--requests") options.requests = std::stoi(value);
        else if (option == "--resident-sessions") options.resident_sessions = std::stoi(value);
        else if (option == "--report") options.report = value;
        else if (option == "--expected-output") options.expected = value;
        else throw std::runtime_error("unknown option: " + option);
    }
    if (options.manifest.empty() || options.provider_a.empty() || options.provider_b.empty()
        || options.prompts.empty() || options.tokens < 1 || options.requests < 1
        || options.resident_sessions < 1 || (!options.persistent
            && (options.resident_sessions != 1 || options.reset_between))) {
        throw std::runtime_error(
            "usage: dan-provider-owned-coordinator --manifest FILE --provider-a HOST:PORT --provider-b HOST:PORT --prompt TEXT [--prompt TEXT] [--tokens N] [--requests N] [--persistent --resident-sessions N] [--reset-between] [--report FILE] [--expected-output TEXT] [--shutdown-workers]");
    }
    return options;
}

void require_ack(const po::Frame& frame, const po::Frame& input, bool allow_timing = false) {
    if (frame.type != po::Type::ack || frame.session != input.session
        || frame.request != input.request || frame.rows != 0 || frame.cols != 0
        || frame.dtype != po::DType::none
        || (!frame.payload.empty() && (!allow_timing || frame.payload.size() != 8))) {
        throw std::runtime_error("invalid provider acknowledgement");
    }
}

void control_both(Connection& a, Connection& b, po::Type type,
    std::uint64_t session, std::uint64_t request = 0) {
    po::Frame input;
    input.type = type;
    input.session = session;
    input.request = request;
    auto [from_a, ignored_a] = a.exchange(input);
    auto [from_b, ignored_b] = b.exchange(input);
    (void) ignored_a;
    (void) ignored_b;
    require_ack(from_a, input);
    require_ack(from_b, input);
    if (type == po::Type::end_request && from_a.position != from_b.position) {
        throw std::runtime_error("provider session positions diverged");
    }
}

struct RequestMetrics {
    double latency_ms = 0;
    double prefill_ms = 0;
    std::vector<double> a_compute_ms;
    std::vector<double> b_compute_ms;
    std::vector<double> network_ms;
    std::size_t activation_bytes = 0;
    std::vector<std::uint32_t> token_ids;
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

Result require_result(const po::Frame& frame, const po::Frame& activation) {
    if (frame.type != po::Type::result || frame.session != activation.session
        || frame.request != activation.request
        || frame.position != activation.position + activation.rows
        || frame.rows != 0 || frame.cols != 0 || frame.dtype != po::DType::none
        || frame.payload.size() < 13 || frame.payload[12] > 1) {
        throw std::runtime_error("invalid stage B result");
    }
    return {po::get32(frame.payload.data()), po::get64(frame.payload.data() + 4),
        frame.payload[12] != 0,
        std::string(frame.payload.begin() + 13, frame.payload.end()), frame.position};
}

Result route_step(Connection& a, Connection& b, const po::Frame& input,
    std::uint32_t hidden, RequestMetrics& metrics, bool decode) {
    auto [a_frame, a_round_ns] = a.exchange(input);
    Activation activation = require_activation(std::move(a_frame), input, hidden);
    const auto route_start = Clock::now();
    auto [b_frame, b_round_ns] = b.exchange(activation.frame);
    const std::uint64_t route_ns = elapsed_ns(route_start);
    Result result = require_result(b_frame, activation.frame);
    metrics.activation_bytes += activation.frame.rows * activation.frame.cols * sizeof(float);
    if (decode) {
        metrics.a_compute_ms.push_back(activation.compute_ns / 1e6);
        metrics.b_compute_ms.push_back(result.compute_ns / 1e6);
        const std::uint64_t a_network = a_round_ns > activation.compute_ns
            ? a_round_ns - activation.compute_ns : 0;
        const std::uint64_t b_network = route_ns > result.compute_ns
            ? route_ns - result.compute_ns : 0;
        metrics.network_ms.push_back((a_network + b_network) / 1e6);
    }
    (void) b_round_ns;
    return result;
}

void commit_final_token(Connection& a, Connection& b, std::uint64_t session,
    std::uint64_t request, std::uint32_t position, std::uint32_t token,
    std::uint32_t hidden) {
    po::Frame input;
    input.type = po::Type::commit_token;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.resize(4);
    po::put32(input.payload.data(), token);
    auto [a_frame, ignored_a] = a.exchange(input);
    Activation activation = require_activation(std::move(a_frame), input, hidden,
        po::Type::commit_activation);
    auto [b_frame, ignored_b] = b.exchange(activation.frame);
    require_ack(b_frame, activation.frame, true);
    if (b_frame.position != position + 1) throw std::runtime_error("commit position mismatch");
    (void) ignored_a;
    (void) ignored_b;
}

RequestResult generate(Connection& a, Connection& b, const Manifest& manifest,
    std::uint64_t session, std::uint64_t request, std::uint32_t position,
    const std::string& prompt, int token_limit, bool preserve_session) {
    const auto request_start = Clock::now();
    RequestResult output;
    po::Frame input;
    input.type = po::Type::prompt;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.assign(prompt.begin(), prompt.end());

    const auto prefill_start = Clock::now();
    Result result = route_step(a, b, input, manifest.hidden, output.metrics, false);
    output.metrics.prefill_ms = elapsed_ns(prefill_start) / 1e6;
    output.output += result.text;
    output.metrics.token_ids.push_back(result.token);
    output.position = result.position;
    output.final_token = result.token;
    output.eog = result.eog;

    while (static_cast<int>(output.metrics.token_ids.size()) < token_limit && !output.eog) {
        input = {};
        input.type = po::Type::token;
        input.session = session;
        input.request = request;
        input.position = output.position;
        input.payload.resize(4);
        po::put32(input.payload.data(), output.final_token);
        result = route_step(a, b, input, manifest.hidden, output.metrics, true);
        output.output += result.text;
        output.metrics.token_ids.push_back(result.token);
        output.position = result.position;
        output.final_token = result.token;
        output.eog = result.eog;
    }
    if (preserve_session) {
        commit_final_token(a, b, session, request, output.position,
            output.final_token, manifest.hidden);
        ++output.position;
    }
    control_both(a, b, po::Type::end_request, session, request);
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

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    int exit_code = 0;
    try {
        const Options options = parse_options(argc, argv);
        const Manifest manifest = load_manifest(options.manifest);
        const auto connected_start = Clock::now();
        Connection provider_a(options.provider_a);
        Connection provider_b(options.provider_b);
        const double connect_ms = elapsed_ns(connected_start) / 1e6;

        std::uint64_t next_id = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now().time_since_epoch()).count()) | 1;
        std::vector<std::uint64_t> persistent_sessions;
        std::unordered_map<std::uint64_t, std::uint32_t> positions;
        if (options.persistent) {
            for (int index = 0; index < options.resident_sessions; ++index) {
                const std::uint64_t session = next_id++;
                control_both(provider_a, provider_b, po::Type::create_session, session);
                persistent_sessions.push_back(session);
                positions[session] = 0;
            }
        }

        std::vector<double> request_latencies;
        std::vector<double> a_compute;
        std::vector<double> b_compute;
        std::vector<double> network;
        std::size_t activation_bytes = 0;
        std::size_t generated_tokens = 0;
        std::vector<std::string> outputs;

        for (int index = 0; index < options.requests; ++index) {
            const std::uint64_t session = options.persistent
                ? persistent_sessions[static_cast<std::size_t>(index) % persistent_sessions.size()]
                : next_id++;
            if (!options.persistent) {
                control_both(provider_a, provider_b, po::Type::create_session, session);
                positions[session] = 0;
            } else if (options.reset_between && index >= options.resident_sessions) {
                control_both(provider_a, provider_b, po::Type::reset_session, session);
                positions[session] = 0;
            }
            const std::uint64_t request = next_id++;
            RequestResult result = generate(provider_a, provider_b, manifest,
                session, request, positions[session],
                options.prompts[static_cast<std::size_t>(index) % options.prompts.size()],
                options.tokens, options.persistent);
            positions[session] = result.position;
            std::printf("request=%d session=%llu tokens=%zu latency_ms=%.3f output=%s\n",
                index + 1, static_cast<unsigned long long>(session),
                result.metrics.token_ids.size(), result.metrics.latency_ms,
                result.output.c_str());
            request_latencies.push_back(result.metrics.latency_ms);
            a_compute.insert(a_compute.end(), result.metrics.a_compute_ms.begin(),
                result.metrics.a_compute_ms.end());
            b_compute.insert(b_compute.end(), result.metrics.b_compute_ms.begin(),
                result.metrics.b_compute_ms.end());
            network.insert(network.end(), result.metrics.network_ms.begin(),
                result.metrics.network_ms.end());
            activation_bytes += result.metrics.activation_bytes;
            generated_tokens += result.metrics.token_ids.size();
            outputs.push_back(std::move(result.output));
            if (!options.persistent) {
                control_both(provider_a, provider_b, po::Type::destroy_session, session);
                positions.erase(session);
            }
            if (!options.expected.empty() && outputs.back() != options.expected) {
                throw std::runtime_error("deterministic output mismatch on request "
                    + std::to_string(index + 1));
            }
        }

        const std::string metrics_a = worker_metrics(provider_a);
        const std::string metrics_b = worker_metrics(provider_b);
        const double average_token_ms = mean(a_compute) + mean(b_compute) + mean(network);
        const double decode_tokens_per_second = average_token_ms > 0 ? 1000 / average_token_ms : 0;
        const double p50 = percentile(request_latencies, 0.50);
        const double p95 = percentile(request_latencies, 0.95);
        std::printf(
            "READY requests=%d tokens=%zu p50_ms=%.3f p95_ms=%.3f decode_tok_s=%.3f A_ms/token=%.3f network_ms/token=%.3f B_ms/token=%.3f activation_bytes/token=%u\n",
            options.requests, generated_tokens, p50, p95, decode_tokens_per_second,
            mean(a_compute), mean(network), mean(b_compute), manifest.hidden * 4);
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
                << "  \"decode_tok_s\": " << decode_tokens_per_second << ",\n"
                << "  \"provider_a_compute_ms_per_token\": " << mean(a_compute) << ",\n"
                << "  \"network_ms_per_token\": " << mean(network) << ",\n"
                << "  \"provider_b_compute_ms_per_token\": " << mean(b_compute) << ",\n"
                << "  \"activation_bytes_per_token\": " << manifest.hidden * 4 << ",\n"
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
            control_both(provider_a, provider_b, po::Type::destroy_session, session);
        }
        if (options.shutdown_workers) {
            shutdown(provider_a);
            shutdown(provider_b);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "provider-owned runtime unavailable: %s\n", error.what());
        exit_code = 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return exit_code;
}
