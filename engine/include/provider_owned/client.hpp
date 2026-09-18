#pragma once

// Shared inference client: drives an already-resolved route of stage workers.
// It knows stage endpoints only -- never how the route was found (no PeerIDs,
// DHT, or libp2p). Both dan-client and the provider-owned coordinator use it,
// so there is exactly one token loop.

#include "provider_owned/protocol.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dan::provider_owned {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start);

// Opens a TCP connection to "host:port"; throws on failure.
socket_t connect_endpoint(std::string_view endpoint);
// Send and receive timeout for a socket (0 = none).
void set_socket_timeout(socket_t socket, std::uint32_t milliseconds);

class Connection {
public:
    explicit Connection(socket_t socket);
    explicit Connection(std::string_view endpoint);
    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    void set_timeout(std::uint32_t milliseconds = 30000);
    // Sends one frame and waits for its reply; returns the reply and the round-trip time.
    std::pair<Frame, std::uint64_t> exchange(const Frame& input);
    void send(const Frame& input);
    // Raw bytes, e.g. the "DAN-P2P/1 <PeerID>" line a sidecar writes before any frame.
    void send_text(std::string_view text);
    Frame receive();
    // Any frame, error frames included (receive() throws on those); throws only when the
    // connection fails.
    Frame receive_frame();
    bool receive_peer_id(std::string& peer_id);
    // Intermediate activations (not commit rows) this connection has received.
    std::uint64_t activations_received() const { return activations_received_; }

private:
    Frame read_reply();

    socket_t socket_ = invalid_socket;
    std::uint64_t activations_received_ = 0;
};

socket_t listen_on(std::string_view endpoint);
// Ring direct-return: accepts until the tail stage completes the ring handshake and, when
// expected_peer is set, its sidecar-authenticated PeerID matches. The listener stays open
// (the caller owns it); closing it from another thread aborts the wait.
std::unique_ptr<Connection> accept_ring_return(socket_t listener,
    const std::string& expected_peer = {});
// Same, on a new listener that is closed afterwards.
std::unique_ptr<Connection> accept_ring_return(const std::string& endpoint,
    const std::string& expected_peer = {});

using StageConnections = std::vector<Connection*>;

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
    bool cancelled = false;
    RequestMetrics metrics;
};

using TokenSink = std::function<bool(std::string_view)>;

struct Activation {
    Frame frame;
    std::uint64_t compute_ns = 0;
};

struct Result {
    std::uint32_t token = 0;
    std::uint64_t compute_ns = 0;
    bool eog = false;
    std::string text;
    std::uint32_t position = 0;
};

struct SpeculativeResult {
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint64_t> compute_ns;
    std::uint64_t network_ns = 0;
    std::uint32_t position = 0;
};

// Speculative draft source. The coordinator's llama.cpp draft model implements it;
// dan-client passes none, so speculative decoding stays off there.
class Drafter {
public:
    virtual ~Drafter() = default;
    virtual std::uint32_t start(std::string_view prompt) = 0;
    virtual std::vector<std::uint32_t> propose(std::uint32_t current, std::size_t count) = 0;
    virtual void rollback(std::uint32_t position) = 0;
    virtual std::string piece(std::uint32_t token) const = 0;
    virtual bool is_eog(std::uint32_t token) const = 0;
};

bool append_token(RequestResult& output, std::uint32_t token, const std::string& piece,
    const TokenSink& sink);
void require_ack(const Frame& frame, const Frame& input, bool allow_timing = false);
Activation require_activation(Frame frame, const Frame& input, std::uint32_t hidden,
    Type expected = Type::activation);
Result require_result(const Frame& frame, const Frame& input);
// A streamed token (client_chunk or the final result) of this session and request.
Result require_streamed(const Frame& frame, std::uint64_t session, std::uint64_t request);

void control_all(const StageConnections& stages, Type type, std::uint64_t session,
    std::uint64_t request = 0);
void rollback_all(const StageConnections& stages, std::uint64_t session,
    std::uint64_t request, std::uint32_t position);
SpeculativeResult route_speculative(const StageConnections& stages, Frame input,
    std::uint32_t hidden, RequestMetrics& metrics);
Result route_step(const StageConnections& stages, const Frame& input, std::uint32_t hidden,
    RequestMetrics& metrics, bool decode, Connection* ring_return = nullptr);
void commit_final_token(const StageConnections& stages, std::uint64_t session,
    std::uint64_t request, std::uint32_t position, std::uint32_t token, std::uint32_t hidden);

// Loop mode: the client sends the prompt and a token budget once, then only reads tokens.
// The last stage feeds each token back to the first stage itself.
RequestResult generate_loop(const StageConnections& stages, std::uint64_t session,
    std::uint64_t request, std::uint32_t position, const std::string& prompt, int token_limit,
    bool preserve_session, Connection& ring_return, std::uint32_t hidden,
    const TokenSink& sink = {});

// Replica mode: one request through a persistent replica's owner. The client sends the token
// budget (stream_prompt) and the prompt, reads client_chunk frames and the final result, then
// the owner's ack, whose position is the session's position once the owner has committed the
// answer on every member. Cancelling sends cancel_request and keeps reading.
RequestResult generate_replica(Connection& owner, std::uint64_t session, std::uint64_t request,
    std::uint32_t position, const std::string& prompt, int token_limit,
    const TokenSink& sink = {});

// The token loop: prompt, then one token per step until the limit or end-of-generation.
RequestResult generate(const StageConnections& stages, std::uint32_t hidden,
    std::uint64_t session, std::uint64_t request, std::uint32_t position,
    const std::string& prompt, int token_limit, bool preserve_session,
    Drafter* draft = nullptr, int draft_tokens = 4, Connection* ring_return = nullptr,
    const TokenSink& sink = {});

std::string worker_metrics(Connection& connection);
void shutdown_stage(Connection& connection);

// An ordered, already-resolved route: stage_endpoints[0] holds layer 0.
// The optional ring fields make stages send activations to each other directly
// (client -> A -> B -> C -> client). Targets and peer IDs are opaque strings here.
struct InferenceRoute {
    std::vector<std::string> stage_endpoints;
    std::uint32_t hidden = 0;
    // Ring mode is on when return_listen is set.
    std::vector<std::string> ring_targets;  // per stage: how its predecessor reaches it ([0] unused)
    std::vector<std::string> peer_ids;      // per stage (libp2p mode) or empty (direct mode)
    std::string return_listen;              // local host:port where the last stage's output arrives
    std::string return_target;              // how the last stage reaches return_listen
    // Loop mode: how the last stage reaches the first one, so decoding continues without the
    // client (one internet round trip per request instead of one per token). Empty = off.
    std::string loop_target;
    // stage_endpoints[0] is a persistent replica's owner (its front door), not a worker: the
    // owner runs the ring and relays the answer. No ring fields.
    bool replica = false;
};

class InferenceClient {
public:
    explicit InferenceClient(const InferenceRoute& route);
    // Uses already-open control connections (e.g. leased by place_route), first stage first.
    InferenceClient(const InferenceRoute& route,
        std::vector<std::unique_ptr<Connection>> connections);
    ~InferenceClient();
    InferenceClient(const InferenceClient&) = delete;
    InferenceClient& operator=(const InferenceClient&) = delete;

    // In ring mode this also configures each stage's next hop, last stage first. If any
    // stage fails, the partial route is torn down and the client becomes unusable.
    std::uint64_t create_session();
    void reset_session(std::uint64_t session);
    void destroy_session(std::uint64_t session);
    // Continues the session's conversation; its position advances.
    // `sink` receives each generated piece as it arrives; returning false stops generation.
    RequestResult generate(std::uint64_t session, const std::string& prompt, int max_tokens,
        const TokenSink& sink = {});
    // One-off request in a temporary session that is destroyed afterwards.
    RequestResult generate_once(const std::string& prompt, int max_tokens);

    std::vector<std::string> stage_metrics();
    void shutdown_stages();
    const StageConnections& stages() const { return stages_; }
    // Where the last stage's output arrives once a ring session exists (null before).
    Connection* ring_return() const { return ring_return_.get(); }
    std::uint32_t hidden() const { return route_.hidden; }
    bool ring() const { return !route_.return_listen.empty(); }
    // Whether this route decodes without the client in the loop.
    bool loops() const { return !route_.loop_target.empty(); }
    // Intermediate activations that came back to this client (0 in ring mode).
    std::uint64_t activations_received() const;
    // Time the first ring session took to link every stage (0 before that, or in hub mode).
    double route_setup_ms() const { return route_setup_ms_; }

private:
    struct SessionState {
        std::uint32_t position = 0;
        std::uint64_t next_request = 1;
    };

    void validate_route() const;
    SessionState& require_session(std::uint64_t session);
    std::uint64_t new_session_id() const;
    void create_ring_session(std::uint64_t session);
    void close_route();

    InferenceRoute route_;
    std::vector<std::unique_ptr<Connection>> connections_;
    StageConnections stages_;
    std::unordered_map<std::uint64_t, SessionState> sessions_;
    socket_t return_listener_ = invalid_socket;
    std::unique_ptr<Connection> ring_return_;
    double route_setup_ms_ = 0;
};

} // namespace dan::provider_owned
