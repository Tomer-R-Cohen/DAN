#include "provider_owned/protocol.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace po = dan::provider_owned;
using namespace std::chrono_literals;

namespace {

po::socket_t connect_to(std::string_view endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == endpoint.size()) {
        throw std::runtime_error("invalid endpoint");
    }
    const std::string host(endpoint.substr(0, colon));
    const std::string port(endpoint.substr(colon + 1));
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        throw std::runtime_error("resolve failed");
    }
    po::socket_t socket_value = po::invalid_socket;
    // ponytail: short retry covers transient listener/backlog pressure in the load harness.
    for (int attempt = 0; attempt < 100 && socket_value == po::invalid_socket; ++attempt) {
        for (addrinfo* address = addresses; address; address = address->ai_next) {
            socket_value = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
            if (socket_value == po::invalid_socket) continue;
            if (connect(socket_value, address->ai_addr,
                static_cast<int>(address->ai_addrlen)) == 0) break;
            po::close_socket(socket_value);
            socket_value = po::invalid_socket;
        }
        if (socket_value == po::invalid_socket) std::this_thread::sleep_for(50ms);
    }
    freeaddrinfo(addresses);
    if (socket_value == po::invalid_socket) throw std::runtime_error("connect failed");
    return socket_value;
}

po::Frame call(const std::string& endpoint, const po::Frame& input) {
    const po::socket_t socket_value = connect_to(endpoint);
    std::string error;
    if (!po::send_frame(socket_value, input, error)) {
        po::close_socket(socket_value);
        throw std::runtime_error(error);
    }
    po::Frame output;
    if (!po::recv_frame(socket_value, output, error)) {
        po::close_socket(socket_value);
        throw std::runtime_error(error);
    }
    po::close_socket(socket_value);
    return output;
}

class Client {
public:
    explicit Client(const std::string& endpoint) : socket_(connect_to(endpoint)) {}
    ~Client() { po::close_socket(socket_); }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    po::Frame call(const po::Frame& input) {
        send(input);
        return receive();
    }

    void send(const po::Frame& input) {
        std::string error;
        if (!po::send_frame(socket_, input, error)) throw std::runtime_error(error);
    }

    po::Frame receive() {
        std::string error;
        po::Frame output;
        if (!po::recv_frame(socket_, output, error)) throw std::runtime_error(error);
        return output;
    }

private:
    po::socket_t socket_;
};

std::string error_text(const po::Frame& frame) {
    return {frame.payload.begin(), frame.payload.end()};
}

void require_ack(const std::string& endpoint, po::Type type, std::uint64_t session) {
    po::Frame input;
    input.type = type;
    input.session = session;
    const po::Frame output = call(endpoint, input);
    if (output.type != po::Type::ack || output.session != session) {
        throw std::runtime_error(output.type == po::Type::error
            ? error_text(output) : "invalid acknowledgement");
    }
}

po::Frame generate(const std::string& endpoint, std::uint64_t session,
    std::uint64_t request, int tokens, int timeout_ms = 0) {
    po::Frame input;
    input.type = po::Type::prompt;
    input.session = session;
    input.request = request;
    input.position = static_cast<std::uint32_t>(timeout_ms);
    input.rows = static_cast<std::uint32_t>(tokens);
    static constexpr std::string_view prompt = "The capital of France is";
    input.payload.assign(prompt.begin(), prompt.end());
    return call(endpoint, input);
}

po::Frame generate(Client& client, std::uint64_t session,
    std::uint64_t request, int tokens, int timeout_ms = 0) {
    po::Frame input;
    input.type = po::Type::prompt;
    input.session = session;
    input.request = request;
    input.position = static_cast<std::uint32_t>(timeout_ms);
    input.rows = static_cast<std::uint32_t>(tokens);
    static constexpr std::string_view prompt = "The capital of France is";
    input.payload.assign(prompt.begin(), prompt.end());
    return client.call(input);
}

std::string generate_stream(const std::string& endpoint, std::uint64_t request, int tokens) {
    Client client(endpoint);
    po::Frame input;
    input.type = po::Type::stream_prompt;
    input.request = request;
    input.rows = static_cast<std::uint32_t>(tokens);
    static constexpr std::string_view prompt = "The capital of France is";
    input.payload.assign(prompt.begin(), prompt.end());
    client.send(input);
    std::string output;
    int chunks = 0;
    for (;;) {
        po::Frame frame;
        frame = client.receive();
        if (frame.type == po::Type::error) throw std::runtime_error(error_text(frame));
        if (frame.type == po::Type::client_chunk) {
            if (frame.request != request || frame.rows != 1) {
                throw std::runtime_error("invalid streaming chunk");
            }
            output.append(frame.payload.begin(), frame.payload.end());
            ++chunks;
            continue;
        }
        if (frame.type != po::Type::client_result || frame.request != request
            || frame.rows != static_cast<std::uint32_t>(chunks) || !frame.payload.empty()) {
            throw std::runtime_error("invalid streaming completion");
        }
        return output;
    }
}

void require_ack(Client& client, po::Type type, std::uint64_t session) {
    po::Frame input;
    input.type = type;
    input.session = session;
    po::Frame output;
    do {
        output = client.call(input);
        if (output.type == po::Type::error
            && (error_text(output) == "queue_full"
                || error_text(output) == "connection_queue_full")) {
            std::this_thread::sleep_for(25ms);
        }
    } while (output.type == po::Type::error
        && (error_text(output) == "queue_full"
            || error_text(output) == "connection_queue_full"));
    if (output.type != po::Type::ack || output.session != session) {
        throw std::runtime_error(output.type == po::Type::error
            ? error_text(output) : "invalid acknowledgement");
    }
}

bool retryable(const po::Frame& frame) {
    if (frame.type != po::Type::error) return false;
    const std::string error = error_text(frame);
    return error == "queue_full" || error == "connection_queue_full";
}

struct Options {
    std::string endpoint;
    int clients = 20;
    int sessions = 50;
    int requests = 1000;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) throw std::runtime_error("missing option value");
        const std::string name = argv[index];
        const std::string value = argv[++index];
        if (name == "--coordinator") result.endpoint = value;
        else if (name == "--clients") result.clients = std::stoi(value);
        else if (name == "--sessions") result.sessions = std::stoi(value);
        else if (name == "--requests") result.requests = std::stoi(value);
        else throw std::runtime_error("unknown option: " + name);
    }
    if (result.endpoint.empty() || result.clients < 1 || result.sessions < result.clients
        || result.requests < result.clients) {
        throw std::runtime_error(
            "usage: provider_owned_concurrency_client --coordinator HOST:PORT [--clients 20] [--sessions 50] [--requests 1000]");
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    try {
        const Options config = options(argc, argv);
        constexpr std::uint64_t session_base = 1000;
        for (int index = 0; index < config.sessions; ++index) {
            require_ack(config.endpoint, po::Type::create_session, session_base + index);
        }

        std::atomic<std::uint64_t> next_request{1};
        std::atomic<int> completed{0};
        std::atomic<int> failures{0};
        std::atomic<int> backpressure{0};
        std::mutex reference_mutex;
        std::string clean_reference;

        std::atomic<bool> start_probe{false};
        std::vector<po::Frame> probe_results(32);
        std::vector<std::jthread> probes;
        for (std::size_t index = 0; index < probe_results.size(); ++index) {
            const std::uint64_t request = next_request.fetch_add(1);
            probes.emplace_back([&, index, request] {
                while (!start_probe.load()) std::this_thread::yield();
                probe_results[index] = generate(config.endpoint, 0, request, 20);
            });
        }
        start_probe.store(true);
        probes.clear();
        int queue_full_results = 0;
        for (const po::Frame& result : probe_results) {
            if (retryable(result)) ++queue_full_results;
            else if (result.type != po::Type::client_result) ++failures;
        }
        if (queue_full_results == 0) {
            throw std::runtime_error("backpressure probe did not fill the queue");
        }

        const std::uint64_t blocker_id = next_request.fetch_add(1);
        const std::uint64_t cancelled_id = next_request.fetch_add(1);
        po::Frame blocker_result;
        po::Frame cancelled_result;
        std::jthread blocker([&] {
            blocker_result = generate(config.endpoint, 0, blocker_id, 20);
        });
        std::this_thread::sleep_for(10ms);
        std::jthread cancelled([&] {
            cancelled_result = generate(config.endpoint, 0, cancelled_id, 20);
        });
        std::this_thread::sleep_for(10ms);
        po::Frame cancel_request;
        cancel_request.type = po::Type::cancel_request;
        cancel_request.request = cancelled_id;
        const po::Frame cancel_result = call(config.endpoint, cancel_request);
        blocker.join();
        cancelled.join();
        if (blocker_result.type != po::Type::client_result
            || cancel_result.type != po::Type::ack
            || cancelled_result.type != po::Type::error
            || error_text(cancelled_result) != "cancelled") {
            throw std::runtime_error("queued cancellation probe failed");
        }

        const std::uint64_t active_id = next_request.fetch_add(1);
        po::Frame active_result;
        std::jthread active([&] {
            active_result = generate(config.endpoint, 0, active_id, 4096);
        });
        std::this_thread::sleep_for(50ms);
        cancel_request.request = active_id;
        const po::Frame active_cancel = call(config.endpoint, cancel_request);
        active.join();
        if (active_cancel.type != po::Type::ack
            || active_result.type != po::Type::error
            || error_text(active_result) != "cancelled") {
            throw std::runtime_error("active cancellation probe failed");
        }

        const std::uint64_t timeout_blocker_id = next_request.fetch_add(1);
        const std::uint64_t timed_out_id = next_request.fetch_add(1);
        std::jthread timeout_blocker([&] {
            blocker_result = generate(config.endpoint, 0, timeout_blocker_id, 20);
        });
        std::this_thread::sleep_for(10ms);
        const po::Frame timed_out = generate(config.endpoint, 0, timed_out_id, 1, 1);
        timeout_blocker.join();
        if (blocker_result.type != po::Type::client_result
            || timed_out.type != po::Type::error || error_text(timed_out) != "queue_timeout") {
            throw std::runtime_error("queued timeout probe failed");
        }

        std::vector<std::jthread> clients;
        clients.reserve(static_cast<std::size_t>(config.clients));
        for (int client = 0; client < config.clients; ++client) {
            clients.emplace_back([&, client] {
                try {
                    Client connection(config.endpoint);
                    const int target = config.requests / config.clients
                        + (client < config.requests % config.clients ? 1 : 0);
                    std::vector<int> owned_sessions;
                    for (int session = client; session < config.sessions;
                        session += config.clients) owned_sessions.push_back(session);
                    std::vector<int> session_uses(owned_sessions.size(), 0);
                    int persistent_index = 0;
                    for (int index = 0; index < target; ++index) {
                        const bool stateless = index % 5 == 0;
                        const std::size_t owned = static_cast<std::size_t>(persistent_index++)
                            % owned_sessions.size();
                        const int session_index = owned_sessions[owned];
                        const std::uint64_t session = stateless ? 0 : session_base + session_index;
                        bool fresh = stateless || session_uses[owned] == 0;
                        if (!stateless && session_uses[owned] == 4) {
                            require_ack(connection, po::Type::reset_session, session);
                            session_uses[owned] = 0;
                            fresh = true;
                        }
                        const std::uint64_t request = next_request.fetch_add(1);
                        po::Frame result;
                        do {
                            result = generate(connection, session, request, 1);
                            if (retryable(result)) {
                                ++backpressure;
                                std::this_thread::sleep_for(25ms);
                            }
                        } while (retryable(result));
                        if (result.type != po::Type::client_result || result.request != request
                            || result.rows != 1) {
                            ++failures;
                            continue;
                        }
                        const std::string text(result.payload.begin(), result.payload.end());
                        if (fresh) {
                            std::lock_guard lock(reference_mutex);
                            if (clean_reference.empty()) clean_reference = text;
                            else if (text != clean_reference) ++failures;
                        }
                        if (!stateless) ++session_uses[owned];
                        ++completed;
                    }
                } catch (...) {
                    ++failures;
                }
            });
        }
        clients.clear();

        const std::string streamed = generate_stream(
            config.endpoint, next_request.fetch_add(1), 20);
        if (streamed.empty()) throw std::runtime_error("streaming produced no output");

        po::Frame metrics_request;
        metrics_request.type = po::Type::metrics;
        const po::Frame metrics = call(config.endpoint, metrics_request);
        if (metrics.type != po::Type::metrics) throw std::runtime_error("metrics failed");
        std::printf("completed=%d failures=%d queue_full=%d backpressure_retries=%d queued_cancellation=passed active_cancellation=passed timeout=passed streaming=passed reference=%s\nmetrics=%s\n",
            completed.load(), failures.load(), queue_full_results, backpressure.load(), clean_reference.c_str(),
            std::string(metrics.payload.begin(), metrics.payload.end()).c_str());

        for (int index = 0; index < config.sessions; ++index) {
            require_ack(config.endpoint, po::Type::destroy_session, session_base + index);
        }
        po::Frame shutdown_request;
        shutdown_request.type = po::Type::shutdown;
        const po::Frame stopped = call(config.endpoint, shutdown_request);
        if (stopped.type != po::Type::ack) throw std::runtime_error("shutdown failed");
        if (completed.load() != config.requests || failures.load() != 0
            || clean_reference.empty()) return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "concurrency client: %s\n", error.what());
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
}
