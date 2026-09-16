#include "provider_owned/client.hpp"
#include "provider_owned/route.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <thread>

namespace dan::provider_owned {

std::uint64_t elapsed_ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

Connection::Connection(socket_t socket) : socket_(socket) {
    if (socket_ == invalid_socket) throw std::runtime_error("invalid provider socket");
    set_timeout();
}

Connection::Connection(std::string_view endpoint) {
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
        if (socket_ == invalid_socket) continue;
        if (connect(socket_, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) break;
        close_socket(socket_);
        socket_ = invalid_socket;
    }
    freeaddrinfo(addresses);
    if (socket_ == invalid_socket) throw std::runtime_error("could not connect to provider");
    set_timeout();
}

Connection::Connection(Connection&& other) noexcept
    : socket_(std::exchange(other.socket_, invalid_socket)) {}

Connection& Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        if (socket_ != invalid_socket) close_socket(socket_);
        socket_ = std::exchange(other.socket_, invalid_socket);
    }
    return *this;
}

Connection::~Connection() { if (socket_ != invalid_socket) close_socket(socket_); }

void Connection::set_timeout(std::uint32_t milliseconds) {
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

Frame Connection::read_reply() {
    Frame output;
    std::string error;
    if (!recv_frame(socket_, output, error)) throw std::runtime_error(error);
    if (output.type == Type::error) {
        throw std::runtime_error(std::string(output.payload.begin(), output.payload.end()));
    }
    if (output.type == Type::activation || output.type == Type::speculative_activation
        || output.type == Type::prompt_chunk) {
        ++activations_received_;
    }
    return output;
}

std::pair<Frame, std::uint64_t> Connection::exchange(const Frame& input) {
    std::string error;
    const auto start = Clock::now();
    if (!send_frame(socket_, input, error)) throw std::runtime_error(error);
    Frame output = read_reply();
    return {std::move(output), elapsed_ns(start)};
}

void Connection::send(const Frame& input) {
    std::string error;
    if (!send_frame(socket_, input, error)) throw std::runtime_error(error);
}

Frame Connection::receive() { return read_reply(); }

bool Connection::receive_peer_id(std::string& peer_id) {
    return recv_peer_id(socket_, peer_id);
}

namespace {

void close_listener(socket_t listener) {
#ifdef _WIN32
    ::shutdown(listener, SD_BOTH);
#else
    ::shutdown(listener, SHUT_RDWR);
#endif
    close_socket(listener);
}

} // namespace

socket_t listen_on(std::string_view endpoint) {
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
    socket_t listener = invalid_socket;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        listener = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (listener == invalid_socket) continue;
        const int enabled = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
            reinterpret_cast<const char*>(&enabled), sizeof(enabled));
        if (bind(listener, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0
            && listen(listener, 128) == 0) break;
        close_socket(listener);
        listener = invalid_socket;
    }
    freeaddrinfo(addresses);
    if (listener == invalid_socket) throw std::runtime_error("bind/listen failed");
    return listener;
}

std::unique_ptr<Connection> accept_ring_return(socket_t listener,
    const std::string& expected_peer) {
    for (;;) {
        const socket_t accepted = accept(listener, nullptr, nullptr);
        if (accepted == invalid_socket) throw std::runtime_error("ring return accept failed");
        auto connection = std::make_unique<Connection>(accepted);
        // A TCP tunnel (SSH -R, and possibly others) can complete a LOCAL accept on the far
        // side before its forwarded channel to this listener actually exists yet, leaving the
        // far side believing it "connected" over a socket that silently goes nowhere -- and the
        // ring stage on the far end has no retry once its own connect() call returns
        // successfully (see docs/PIPELINED_RING_PHYSICAL_TEST.md). A real send/receive round
        // trip catches that dead-socket case; a bare accept() does not. On failure, go back to
        // accepting a new connection instead of giving up -- the far side may itself be
        // retrying with a fresh connect.
        try {
            if (!expected_peer.empty()) {
                std::string peer_id;
                if (!connection->receive_peer_id(peer_id) || peer_id != expected_peer) {
                    throw std::runtime_error("unexpected tail PeerID");
                }
            }
            Frame hello = connection->receive();
            if (hello.type != Type::ack) throw std::runtime_error("unexpected ring handshake hello");
            Frame reply; reply.type = Type::ack;
            connection->send(reply);
            Frame confirmed = connection->receive();
            if (confirmed.type != Type::ack) {
                throw std::runtime_error("unexpected ring handshake confirmation");
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "ring: tail stage connection failed handshake (%s) -- waiting for a new one\n",
                error.what());
            continue;
        }
        std::fprintf(stderr, "ring: tail stage connected\n");
        return connection;
    }
}

// Ring direct-return (docs/PIPELINED_SPECULATION_V1.md phase 4): blocks until the tail stage's
// --next dials in and completes a handshake round trip. Must be called before, or concurrently
// with, the tail stage starting up.
std::unique_ptr<Connection> accept_ring_return(const std::string& endpoint,
    const std::string& expected_peer) {
    const socket_t listener = listen_on(endpoint);
    std::fprintf(stderr, "ring: waiting for the tail stage to connect at %s\n", endpoint.c_str());
    try {
        auto connection = accept_ring_return(listener, expected_peer);
        close_socket(listener);
        return connection;
    } catch (...) {
        close_socket(listener);
        throw;
    }
}

bool append_token(RequestResult& output, std::uint32_t token, const std::string& piece,
    const TokenSink& sink) {
    output.output += piece;
    output.metrics.token_ids.push_back(token);
    output.cancelled = sink && !sink(piece);
    return !output.cancelled;
}

void require_ack(const Frame& frame, const Frame& input, bool allow_timing) {
    if (frame.type != Type::ack || frame.session != input.session
        || frame.request != input.request || frame.rows != 0 || frame.cols != 0
        || frame.dtype != DType::none
        || (!frame.payload.empty() && (!allow_timing || frame.payload.size() != 8))) {
        throw std::runtime_error("invalid provider acknowledgement");
    }
}

Activation require_activation(Frame frame, const Frame& input, std::uint32_t hidden,
    Type expected) {
    if (frame.type != expected || frame.session != input.session
        || frame.request != input.request || frame.position != input.position
        || frame.rows == 0 || frame.cols != hidden || frame.dtype != DType::f32le) {
        throw std::runtime_error("invalid stage A activation metadata");
    }
    const std::uint64_t values = std::uint64_t(frame.rows) * frame.cols;
    if (values > (max_payload - 8) / sizeof(float)
        || frame.payload.size() != 8 + values * sizeof(float)) {
        throw std::runtime_error("invalid stage A activation size");
    }
    const std::uint64_t compute_ns = get64(frame.payload.data());
    return {std::move(frame), compute_ns};
}

Result require_result(const Frame& frame, const Frame& input) {
    const bool position_ok = input.rows != 0
        ? frame.position == input.position + input.rows
        : (input.type == Type::prompt
            ? frame.position > input.position
            : frame.position == input.position + 1);
    if (frame.type != Type::result || frame.session != input.session
        || frame.request != input.request || !position_ok
        || frame.rows != 0 || frame.cols != 0 || frame.dtype != DType::none
        || frame.payload.size() < 13 || frame.payload[12] > 1) {
        throw std::runtime_error("invalid stage B result");
    }
    return {get32(frame.payload.data()), get64(frame.payload.data() + 4),
        frame.payload[12] != 0,
        std::string(frame.payload.begin() + 13, frame.payload.end()), frame.position};
}

void control_all(const StageConnections& stages, Type type,
    std::uint64_t session, std::uint64_t request) {
    Frame input;
    input.type = type;
    input.session = session;
    input.request = request;
    std::uint32_t position = 0;
    for (std::size_t index = 0; index < stages.size(); ++index) {
        auto [output, ignored] = stages[index]->exchange(input);
        (void) ignored;
        require_ack(output, input);
        if (type == Type::end_request) {
            if (index != 0 && output.position != position) {
                throw std::runtime_error("provider session positions diverged");
            }
            position = output.position;
        }
    }
}

void rollback_all(const StageConnections& stages, std::uint64_t session,
    std::uint64_t request, std::uint32_t position) {
    Frame input;
    input.type = Type::rollback;
    input.session = session;
    input.request = request;
    input.position = position;
    for (Connection* stage : stages) {
        auto [output, ignored] = stage->exchange(input);
        (void) ignored;
        require_ack(output, input);
        if (output.position != position) throw std::runtime_error("rollback position mismatch");
    }
}

SpeculativeResult route_speculative(const StageConnections& stages, Frame input,
    std::uint32_t hidden, RequestMetrics& metrics) {
    SpeculativeResult output;
    Frame current = std::move(input);
    for (std::size_t index = 0; index < stages.size(); ++index) {
        auto [response, round_ns] = stages[index]->exchange(current);
        if (index + 1 == stages.size()) {
            if (response.type != Type::result || response.session != current.session
                || response.request != current.request
                || response.position != current.position + current.rows
                || response.rows != current.rows || response.cols != 0
                || response.dtype != DType::none
                || response.payload.size() != 8 + static_cast<std::size_t>(response.rows) * 4) {
                throw std::runtime_error("invalid speculative result");
            }
            const std::uint64_t compute = get64(response.payload.data());
            output.compute_ns.push_back(compute);
            output.network_ns += round_ns > compute ? round_ns - compute : 0;
            output.position = response.position;
            output.tokens.reserve(response.rows);
            for (std::uint32_t row = 0; row < response.rows; ++row) {
                output.tokens.push_back(get32(
                    response.payload.data() + 8 + static_cast<std::size_t>(row) * 4));
            }
            return output;
        }
        Activation activation = require_activation(std::move(response), current, hidden,
            Type::speculative_activation);
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
// caller never sees or touches any intermediate hop. Prefill validation is looser here than
// in hub-and-spoke mode: require_result falls back to its single-stage check (final position
// greater than the start, not the exact prompt length) because the caller never learns the
// tokenized prompt length from an intermediate activation the way it does in hub-and-spoke mode.
// That's a real, accepted reduction in caller-side sanity-checking, not a correctness gap --
// the stage's own KV/position bookkeeping is the actual source of truth either way.
Result route_step(const StageConnections& stages, const Frame& input,
    std::uint32_t hidden, RequestMetrics& metrics, bool decode, Connection* ring_return) {
    if (stages.empty()) throw std::runtime_error("replica has no stages");
    if (ring_return) {
        const auto start = Clock::now();
        stages.front()->send(input);
        Frame response = ring_return->receive();
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
    Frame current = input;
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
    Frame input;
    input.type = Type::commit_token;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.resize(4);
    put32(input.payload.data(), token);
    Frame current = std::move(input);
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
                Type::commit_activation).frame;
        }
    }
}

RequestResult generate(const StageConnections& stages, std::uint32_t hidden,
    std::uint64_t session, std::uint64_t request, std::uint32_t position,
    const std::string& prompt, int token_limit, bool preserve_session,
    Drafter* draft, int draft_tokens, Connection* ring_return, const TokenSink& sink) {
    const auto request_start = Clock::now();
    RequestResult output;
    Frame input;
    input.type = Type::prompt;
    input.session = session;
    input.request = request;
    input.position = position;
    input.payload.assign(prompt.begin(), prompt.end());

    const auto prefill_start = Clock::now();
    Result result = route_step(stages, input, hidden, output.metrics, false, ring_return);
    output.metrics.prefill_ms = elapsed_ns(prefill_start) / 1e6;
    output.metrics.ttft_ms = elapsed_ns(request_start) / 1e6;
    append_token(output, result.token, result.text, sink);
    output.position = result.position;
    output.final_token = result.token;
    output.eog = result.eog;

    if (draft) {
        if (position != 0 || draft->start(prompt) != result.position
            || draft->piece(result.token) != result.text) {
            throw std::runtime_error("draft and target tokenizers do not match");
        }
    }

    while (static_cast<int>(output.metrics.token_ids.size()) < token_limit
        && !output.eog && !output.cancelled) {
        if (draft) {
            const std::size_t count = static_cast<std::size_t>(std::min(
                draft_tokens, token_limit - static_cast<int>(output.metrics.token_ids.size())));
            const auto draft_start = Clock::now();
            const std::vector<std::uint32_t> guesses = draft->propose(output.final_token, count);
            output.metrics.draft_ms += elapsed_ns(draft_start) / 1e6;
            output.metrics.speculative_rounds++;
            output.metrics.proposed_tokens += guesses.size();

            Frame block;
            block.type = Type::token;
            block.session = session;
            block.request = request;
            block.position = output.position;
            block.rows = static_cast<std::uint32_t>(count);
            block.payload.resize(count * 4);
            put32(block.payload.data(), output.final_token);
            for (std::size_t index = 1; index < count; ++index) {
                put32(block.payload.data() + index * 4, guesses[index - 1]);
            }
            SpeculativeResult checked = route_speculative(
                stages, std::move(block), hidden, output.metrics);

            std::size_t accepted = 0;
            bool stopped = false;
            for (; accepted < guesses.size() && guesses[accepted] == checked.tokens[accepted];
                    ++accepted) {
                output.final_token = guesses[accepted];
                const bool keep_going = append_token(output, guesses[accepted],
                    draft->piece(guesses[accepted]), sink);
                output.eog = draft->is_eog(guesses[accepted]);
                if (!keep_going || output.eog) {
                    stopped = true;
                    ++accepted;
                    break;
                }
            }
            output.metrics.accepted_draft_tokens += accepted;
            if (!stopped && accepted < guesses.size()) {
                output.final_token = checked.tokens[accepted];
                append_token(output, output.final_token, draft->piece(output.final_token), sink);
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
        input.type = Type::token;
        input.session = session;
        input.request = request;
        input.position = output.position;
        input.payload.resize(4);
        put32(input.payload.data(), output.final_token);
        result = route_step(stages, input, hidden, output.metrics, true, ring_return);
        append_token(output, result.token, result.text, sink);
        output.position = result.position;
        output.final_token = result.token;
        output.eog = result.eog;
    }
    if (output.cancelled) {
        rollback_all(stages, session, request, position);
    } else if (preserve_session) {
        commit_final_token(stages, session, request, output.position,
            output.final_token, hidden);
        ++output.position;
    }
    control_all(stages, Type::end_request, session, request);
    output.metrics.latency_ms = elapsed_ns(request_start) / 1e6;
    return output;
}

std::string worker_metrics(Connection& connection) {
    Frame input;
    input.type = Type::metrics;
    auto [output, ignored] = connection.exchange(input);
    (void) ignored;
    if (output.type != Type::metrics || output.session != 0 || output.request != 0
        || output.rows != 0 || output.cols != 0 || output.dtype != DType::none
        || output.payload.empty()) {
        throw std::runtime_error("invalid worker metrics");
    }
    return {output.payload.begin(), output.payload.end()};
}

void shutdown_stage(Connection& connection) {
    Frame input;
    input.type = Type::shutdown;
    auto [output, ignored] = connection.exchange(input);
    (void) ignored;
    require_ack(output, input);
}

void InferenceClient::validate_route() const {
    const std::size_t count = route_.stage_endpoints.size();
    if (count == 0) throw std::runtime_error("route has no stages");
    if (route_.hidden == 0) throw std::runtime_error("route hidden size is unknown");
    if (ring()) {
        if (route_.return_target.empty() || route_.ring_targets.size() != count
            || std::any_of(route_.ring_targets.begin() + 1, route_.ring_targets.end(),
                [](const std::string& target) { return target.empty(); })
            || (!route_.peer_ids.empty() && route_.peer_ids.size() != count)) {
            throw std::runtime_error("incomplete ring route");
        }
    } else if (!route_.ring_targets.empty() || !route_.peer_ids.empty()
        || !route_.return_target.empty()) {
        throw std::runtime_error("ring route fields need a return listener");
    }
}

InferenceClient::InferenceClient(const InferenceRoute& route) : route_(route) {
    validate_route();
    for (const std::string& endpoint : route.stage_endpoints) {
        connections_.push_back(std::make_unique<Connection>(endpoint));
        stages_.push_back(connections_.back().get());
    }
    if (ring()) return_listener_ = listen_on(route.return_listen);
}

InferenceClient::InferenceClient(const InferenceRoute& route,
    std::vector<std::unique_ptr<Connection>> connections)
    : route_(route), connections_(std::move(connections)) {
    validate_route();
    if (connections_.size() != route.stage_endpoints.size()
        || std::any_of(connections_.begin(), connections_.end(),
            [](const std::unique_ptr<Connection>& connection) { return !connection; })) {
        throw std::runtime_error("route needs one open connection per stage");
    }
    for (const auto& connection : connections_) stages_.push_back(connection.get());
    if (ring()) return_listener_ = listen_on(route.return_listen);
}

InferenceClient::~InferenceClient() {
    if (return_listener_ != invalid_socket) close_listener(return_listener_);
}

void InferenceClient::close_route() {
    if (return_listener_ != invalid_socket) close_listener(return_listener_);
    return_listener_ = invalid_socket;
    ring_return_.reset();
    stages_.clear();
    connections_.clear();
    sessions_.clear();
}

void InferenceClient::create_ring_session(std::uint64_t session) {
    const bool p2p = !route_.peer_ids.empty();
    std::vector<std::size_t> created;
    try {
        // Last stage first, so each stage knows its expected predecessor before that
        // predecessor is told to connect.
        for (std::size_t index = stages_.size(); index-- > 0;) {
            const bool last = index + 1 == stages_.size();
            const RingRoute ring{last ? route_.return_target : route_.ring_targets[index + 1],
                p2p && index != 0 ? route_.peer_ids[index - 1] : std::string{}};
            Frame input;
            input.type = Type::create_session;
            input.session = session;
            const std::string payload = route_message(ring);
            input.payload.assign(payload.begin(), payload.end());
            if (!last || ring_return_) {
                auto [output, ignored] = stages_[index]->exchange(input);
                (void) ignored;
                require_ack(output, input);
            } else {
                // The last stage connects back to us while it handles this create_session.
                std::unique_ptr<Connection> accepted;
                std::exception_ptr failure;
                std::thread acceptor([&] {
                    try {
                        accepted = accept_ring_return(return_listener_,
                            p2p ? route_.peer_ids.back() : std::string{});
                    } catch (...) { failure = std::current_exception(); }
                });
                try {
                    auto [output, ignored] = stages_[index]->exchange(input);
                    (void) ignored;
                    require_ack(output, input);
                } catch (...) {
                    close_listener(return_listener_);
                    return_listener_ = invalid_socket;
                    acceptor.join();
                    throw;
                }
                acceptor.join();
                if (failure) std::rethrow_exception(failure);
                ring_return_ = std::move(accepted);
                close_listener(return_listener_);
                return_listener_ = invalid_socket;
            }
            created.push_back(index);
        }
    } catch (...) {
        for (const std::size_t index : created) {
            Frame destroy;
            destroy.type = Type::destroy_session;
            destroy.session = session;
            try { stages_[index]->exchange(destroy); } catch (...) {}
        }
        close_route();
        throw;
    }
}

std::uint64_t InferenceClient::new_session_id() const {
    std::random_device device;
    std::uint64_t id = 0;
    while (id == 0 || sessions_.contains(id)) {
        id = (std::uint64_t(device()) << 32) | device();
    }
    return id;
}

InferenceClient::SessionState& InferenceClient::require_session(std::uint64_t session) {
    const auto found = sessions_.find(session);
    if (found == sessions_.end()) throw std::runtime_error("unknown session ID");
    return found->second;
}

std::uint64_t InferenceClient::create_session() {
    if (stages_.empty()) throw std::runtime_error("route is closed");
    const std::uint64_t session = new_session_id();
    if (ring()) create_ring_session(session);
    else control_all(stages_, Type::create_session, session);
    sessions_.emplace(session, SessionState{});
    return session;
}

void InferenceClient::reset_session(std::uint64_t session) {
    SessionState& state = require_session(session);
    control_all(stages_, Type::reset_session, session);
    state.position = 0;
}

void InferenceClient::destroy_session(std::uint64_t session) {
    require_session(session);
    control_all(stages_, Type::destroy_session, session);
    sessions_.erase(session);
}

RequestResult InferenceClient::generate(std::uint64_t session, const std::string& prompt,
    int max_tokens) {
    SessionState& state = require_session(session);
    RequestResult result = provider_owned::generate(stages_, route_.hidden, session,
        state.next_request++, state.position, prompt, max_tokens, true, nullptr, 4,
        ring_return_.get());
    state.position = result.position;
    return result;
}

RequestResult InferenceClient::generate_once(const std::string& prompt, int max_tokens) {
    const std::uint64_t session = create_session();
    const std::uint64_t request = sessions_.at(session).next_request++;
    RequestResult result = provider_owned::generate(stages_, route_.hidden, session, request, 0,
        prompt, max_tokens, false, nullptr, 4, ring_return_.get());
    destroy_session(session);
    return result;
}

std::vector<std::string> InferenceClient::stage_metrics() {
    std::vector<std::string> output;
    for (Connection* stage : stages_) output.push_back(worker_metrics(*stage));
    return output;
}

std::uint64_t InferenceClient::activations_received() const {
    std::uint64_t total = ring_return_ ? ring_return_->activations_received() : 0;
    for (const auto& connection : connections_) total += connection->activations_received();
    return total;
}

void InferenceClient::shutdown_stages() {
    for (Connection* stage : stages_) shutdown_stage(*stage);
}

} // namespace dan::provider_owned
