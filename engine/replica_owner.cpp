// Persistent replica owner (replica.hpp). One process per provider node: it forms a replica
// around this node's worker with the ordinary placement code, keeps it alive, and serves
// client sessions on it until any member fails; then it forms again.

#include "provider_owned/replica.hpp"
#include "provider_owned/formation.hpp"
#include "provider_owned/manifest.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dan::provider_owned {
namespace {

// Frames buffered for one client. A client that falls this far behind its own answer is
// dropped, so it can never slow the ring down.
constexpr std::size_t max_outbox = 4096;

std::int64_t unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void sleep_ms(std::uint32_t milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

std::uint32_t jitter(std::uint32_t low, std::uint32_t high) {
    std::random_device device;
    return low + device() % (high - low + 1);
}

void shutdown_socket(socket_t socket) {
#ifdef _WIN32
    ::shutdown(socket, SD_BOTH);
#else
    ::shutdown(socket, SHUT_RDWR);
#endif
}

Frame ack_frame(std::uint64_t session, std::uint64_t request, std::uint32_t position = 0) {
    Frame frame;
    frame.type = Type::ack;
    frame.session = session;
    frame.request = request;
    frame.position = position;
    return frame;
}

std::string short_id(const std::string& id) { return id.substr(0, 8); }

// PeerIDs all start alike ("12D3KooW"); their end tells them apart in logs.
std::string short_peer(const std::string& peer) {
    return peer.size() > 6 ? peer.substr(peer.size() - 6) : peer;
}

// The replica cannot go on (a member or the ring failed): dissolve it.
struct Dissolve : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// This node's own worker cannot be reached.
struct SelfGone : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One conversation of one client, as the members know it.
struct SessionState {
    std::uint64_t internal = 0;    // the session ID members see; clients never choose it
    std::uint32_t position = 0;
    std::uint64_t next_request = 1;
};

// One client connection on the front door. A reader thread takes its frames, a writer thread
// sends what the owner posts; the owner's executor thread alone touches `sessions`.
struct Link {
    socket_t socket = invalid_socket;
    std::string peer;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Frame> outbox;
    bool closing = false;                  // send what is queued, then close
    std::atomic<bool> gone{false};         // disconnected or dropped: send nothing more
    std::atomic<int> threads{2};           // the last of reader/writer closes the socket
    std::atomic<std::uint64_t> cancel_session{0};
    std::atomic<std::uint64_t> cancel_request{0};
    std::unordered_map<std::uint64_t, std::uint32_t> budgets;  // reader only
    std::unordered_map<std::uint64_t, SessionState> sessions;  // executor only

    bool post(Frame frame) {
        std::lock_guard lock(mutex);
        if (closing || gone) return false;
        if (outbox.size() >= max_outbox) {
            std::fprintf(stderr, "replica: client %s cannot keep up with its answer; dropping it\n",
                short_peer(peer).c_str());
            gone = true;
            shutdown_socket(socket);
            wake.notify_all();
            return false;
        }
        outbox.push_back(std::move(frame));
        wake.notify_all();
        return true;
    }

    void close_with(Frame last) {
        std::lock_guard lock(mutex);
        if (!closing && !gone) outbox.push_back(std::move(last));
        closing = true;
        wake.notify_all();
    }

    bool cancelled(std::uint64_t session, std::uint64_t request) const {
        return gone || (cancel_session == session && cancel_request == request);
    }

    void thread_done() {
        if (--threads == 0) close_socket(socket);
    }
};

struct Job {
    enum class Kind { create, reset, destroy, request, cleanup };
    Kind kind = Kind::cleanup;
    std::shared_ptr<Link> link;
    Frame frame;
    std::uint32_t budget = 0;
};

struct Member {
    std::string peer;
    std::string worker;
    int begin = 0;
    int end = 0;
    std::uint32_t rtt_ms = 0;   // from the owner
    bool relayed = false;
};

struct Edge {
    std::string from;
    std::string to;
    std::uint32_t rtt_ms = 0;
    bool relayed = false;
    std::string problem;        // empty = usable
};

class Owner {
public:
    explicit Owner(const ReplicaOwnerOptions& options) : options_(options) {}

    int run() {
        const socket_t listener = listen_on(options_.session_listen);
        std::thread([this, listener] { accept_clients(listener); }).detach();
        std::thread([this] {
            for (;;) {
                write_status();
                sleep_ms(5000);
            }
        }).detach();
        std::fprintf(stderr, "replica: owner ready; front door %s\n", options_.session_listen.c_str());
        int self_failures = 0;
        for (;;) {
            try {
                if (form()) {
                    self_failures = 0;
                    serve();
                    sleep_ms(jitter(3000, 8000));
                } else {
                    sleep_ms(jitter(5000, 15000));
                }
            } catch (const SelfGone& failure) {
                set_event(std::string("own worker unreachable: ") + failure.what());
                if (++self_failures >= 12) {
                    std::fprintf(stderr, "replica: own worker is gone (%s); stopping\n", failure.what());
                    set_state("stopped");
                    write_status();
                    return 1;
                }
                sleep_ms(10000);
            } catch (const std::exception& failure) {
                std::fprintf(stderr, "replica: formation failed: %s\n", failure.what());
                set_event(std::string("formation failed: ") + failure.what());
                set_state("free");
                sleep_ms(jitter(5000, 15000));
            }
        }
    }

private:
    // ---- Formation ----

    // This node's worker state, straight from its greeting (the worker, not the DHT, knows).
    std::string self_state(std::uint64_t& offered_mib) {
        try {
            Connection connection(options_.self_control);
            connection.set_timeout(5000);
            connection.send_text("DAN-P2P/1 " + self_peer_ + "\n");
            const Frame hello = connection.receive();
            ProviderCapability capability;
            const std::string text(hello.payload.begin(), hello.payload.end());
            if (hello.type != Type::provider_available || !parse_available(text, capability)) {
                throw std::runtime_error("invalid greeting");
            }
            offered_mib = capability.offered_vram_mib;
            return capability.state;
        } catch (const std::exception& failure) {
            throw SelfGone(failure.what());
        }
    }

    // Measures every link of a planned ring (each hop and the loop back to the first stage)
    // from the sending side. Returns the control endpoint of a candidate to leave out, or
    // empty when every link is good enough.
    std::string check_edges(const std::vector<PlacementCandidate>& route) {
        std::vector<Edge> edges;
        if (route.size() > 1) {
            for (std::size_t index = 0; index < route.size(); ++index) {
                edges.push_back({route[index].peer_id, route[(index + 1) % route.size()].peer_id});
            }
        }
        std::vector<std::thread> probes;
        for (Edge& edge : edges) {
            probes.emplace_back([&, this] {
                try {
                    const LinkProbe probe = probe_link(options_.discover, edge.from, edge.to);
                    edge.rtt_ms = probe.rtt_ms;
                    edge.relayed = probe.relayed;
                    if (probe.rtt_ms > options_.max_edge_rtt_ms) {
                        edge.problem = "round trip " + std::to_string(probe.rtt_ms) + " ms";
                    } else if (probe.relayed && !options_.allow_relay_edges) {
                        edge.problem = "relayed";
                    }
                } catch (const std::exception& failure) {
                    edge.problem = failure.what();
                }
            });
        }
        for (std::thread& probe : probes) probe.join();
        std::string leave_out;
        for (std::size_t index = 0; index < edges.size(); ++index) {
            const Edge& edge = edges[index];
            std::fprintf(stderr, "replica: link %s -> %s %u ms %s%s%s\n", short_peer(edge.from).c_str(),
                short_peer(edge.to).c_str(), edge.rtt_ms, edge.relayed ? "relay" : "direct",
                edge.problem.empty() ? "" : "  REJECTED: ", edge.problem.c_str());
            if (edge.problem.empty() || !leave_out.empty()) continue;
            // Keep the head (this node); drop the other end of the bad link.
            const std::size_t next = (index + 1) % route.size();
            leave_out = route[next == 0 ? index : next].control;
        }
        {
            std::lock_guard lock(status_mutex_);
            edges_ = edges;
        }
        return leave_out;
    }

    // One formation attempt. False when this node cannot lead a replica right now (its worker
    // is busy); throws when an attempt fails.
    bool form() {
        release_route();
        std::uint64_t self_mib = 0;
        if (self_peer_.empty()) {
            // The sidecar knows this node's PeerID; the first discovery tells us.
            self_peer_ = discover_candidates(options_.discover, model_shas()).self_peer;
        }
        const std::string state = self_state(self_mib);
        if (state != "available") {
            if (state != last_skip_) {
                std::fprintf(stderr, "replica: own worker is %s; not forming\n", state.c_str());
            }
            last_skip_ = state;
            set_event("own worker is " + state + "; not forming");
            set_state("free");
            return false;
        }
        last_skip_.clear();
        std::fprintf(stderr, "replica: forming (attempt %llu)\n",
            static_cast<unsigned long long>(attempts_ + 1));
        const auto started = Clock::now();
        {
            std::lock_guard lock(status_mutex_);
            ++attempts_;
        }
        set_state("forming");
        Discovery discovery = discover_candidates(options_.discover, model_shas());
        if (discovery.return_prefix.empty() || discovery.return_listen.empty()) {
            throw std::runtime_error("the sidecar has no replica return listener (-return-inbound)");
        }
        std::vector<PlacementCandidate> candidates;
        std::size_t rank = 0;
        for (const PlacementCandidate& candidate : discovery.candidates) {
            if (candidate.peer_id == self_peer_) continue;
            candidates.push_back(candidate);
            if (candidate.offered_mib > self_mib
                || (candidate.offered_mib == self_mib && candidate.peer_id < self_peer_)) ++rank;
        }
        if (options_.rank_delay && rank > 0) {
            // Bigger free peers go first; this is only to make collisions rare.
            const std::uint32_t wait = std::min<std::uint32_t>(static_cast<std::uint32_t>(rank) * 3000, 30000);
            std::fprintf(stderr, "replica: %zu free peer(s) rank above this node; waiting %u ms\n",
                rank, wait);
            sleep_ms(wait + jitter(0, 1000));
        }
        PlacementCandidate self{options_.self_control, self_peer_, 0, false, self_mib, true};
        candidates.insert(candidates.begin(), self);

        PlacementRequest request = options_.request;
        request.head = options_.self_control;
        request.attempts = std::max(request.attempts, 5);
        request.check_route = [this](const std::vector<PlacementCandidate>& route) {
            return check_edges(route);
        };
        PlacementTimings timings;
        request.report = &timings;
        PlacedRoute placed;
        try {
            placed = place_route(candidates, request);
        } catch (...) {
            count_timings(timings);
            throw;
        }
        count_timings(timings);
        if (placed.stages.empty() || placed.stages.front().peer_id != self_peer_) {
            throw std::runtime_error("placement did not put this node first");
        }
        InferenceRoute route = placed.route;
        route.return_listen = discovery.return_listen;
        route.return_target = discovery.return_prefix + "/p2p/" + self_peer_;
        client_ = std::make_unique<InferenceClient>(route, std::move(placed.connections));
        for (Connection* stage : client_->stages()) stage->set_timeout(30000);
        replica_id_ = placed.route_id;
        {
            std::lock_guard lock(status_mutex_);
            model_sha_ = lowercase_hex(placed.manifest.sha256);
            model_id_ = placed.manifest.model_id;
            draft_sha_ = placed.draft_sha256;
            context_ = request.context != 0 ? request.context : placed.manifest.context;
            members_.clear();
            for (const PlacedStage& stage : placed.stages) {
                members_.push_back({stage.peer_id, stage.worker_id, stage.begin, stage.end,
                    stage.rtt_ms, stage.relayed});
            }
            load_ms_ = timings.load_ms;
        }
        std::string layout;
        for (const PlacedStage& stage : placed.stages) {
            layout += " " + short_peer(stage.peer_id) + "[" + std::to_string(stage.begin) + ","
                + std::to_string(stage.end) + ")";
        }
        std::fprintf(stderr, "replica %s: members%s; linking and warming up\n",
            replica_id_.c_str(), layout.c_str());

        // Link the ring and push one short request all the way around it: the replica is
        // READY only once a real token has made the whole trip.
        const auto warm_started = Clock::now();
        try {
            SessionState warm;
            warm.internal = client_->create_session();
            client_->ring_return()->set_timeout(120000);
            std::string problem;
            const std::uint32_t position = run_request(warm, "Hello", 0,
                static_cast<std::uint32_t>(options_.warmup_tokens), nullptr, 0, 0, problem);
            (void) position;
            if (!problem.empty()) throw std::runtime_error("warm-up request failed: " + problem);
            client_->destroy_session(warm.internal);
        } catch (const std::exception& failure) {
            {
                std::lock_guard lock(status_mutex_);
                ++warmup_failures_;
            }
            release_route();
            throw std::runtime_error(std::string("warm-up: ") + failure.what());
        }
        const double warm_ms = elapsed_ns(warm_started) / 1e6;
        {
            std::lock_guard lock(status_mutex_);
            ++formations_;
            warmup_ms_ = warm_ms;
            formation_ms_ = elapsed_ns(started) / 1e6;
            ready_since_ms_ = unix_ms();
            sessions_in_use_ = 0;
        }
        set_state("ready");
        set_event("replica " + short_id(replica_id_) + " ready");
        std::fprintf(stderr, "replica %s READY: model %s%s, %zu stage(s),%s formed in %.0f ms "
            "(load %.0f ms, link + warm-up %.0f ms)\n", replica_id_.c_str(), model_id_.c_str(),
            draft_sha_.empty() ? "" : " + draft", placed.stages.size(), layout.c_str(),
            formation_ms_, timings.load_ms, warm_ms);
        write_status();
        return true;
    }

    std::vector<std::string> model_shas() const {
        std::vector<std::string> shas;
        for (const ModelOption& option : options_.request.models) shas.push_back(option.manifest.sha256);
        return shas;
    }

    static std::string lowercase_hex(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
            [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
        return value;
    }

    void count_timings(const PlacementTimings& timings) {
        std::lock_guard lock(status_mutex_);
        races_lost_ += timings.refusals;
        rejected_links_ += timings.rejected_links;
    }

    // Closes every member connection: each worker's lease is released and its weights stay
    // loaded (the next formation with the same range reuses them).
    void release_route() {
        ready_ = false;
        client_.reset();
    }

    // ---- Serving ----

    void serve() {
        {
            std::lock_guard lock(jobs_mutex_);
            jobs_.clear();
        }
        ready_ = true;
        for (;;) {
            Job job;
            bool have = false;
            {
                std::unique_lock lock(jobs_mutex_);
                jobs_wake_.wait_for(lock, std::chrono::milliseconds(options_.keepalive_ms),
                    [this] { return !jobs_.empty(); });
                if (!jobs_.empty()) {
                    job = std::move(jobs_.front());
                    jobs_.pop_front();
                    have = true;
                }
            }
            try {
                if (have) execute(job);
                else check_members();
            } catch (const Dissolve& failure) {
                dissolve(failure.what());
                return;
            } catch (const std::exception& failure) {
                dissolve(std::string("unexpected: ") + failure.what());
                return;
            }
        }
    }

    // Every member answers (the worker's metrics frame): also keeps each connection's idle
    // timeout from ending the route while no one chats.
    void check_members() {
        const std::string problem = health();
        if (!problem.empty()) throw Dissolve(problem);
    }

    std::string health() {
        const auto& stages = client_->stages();
        for (std::size_t index = 0; index < stages.size(); ++index) {
            try {
                worker_metrics(*stages[index]);
            } catch (const std::exception& failure) {
                std::lock_guard lock(status_mutex_);
                const Member& member = members_[index];
                return "member " + short_peer(member.peer) + " (layers " + std::to_string(member.begin)
                    + "-" + std::to_string(member.end - 1) + ", " + (member.relayed ? "relay" : "direct")
                    + ") stopped answering: " + failure.what();
            }
        }
        return {};
    }

    // A member call failed. If every member still answers, it was this session's problem:
    // report it to the client. Otherwise the replica is broken.
    void member_failed(const std::exception& failure) {
        const std::string problem = health();
        if (!problem.empty()) throw Dissolve(problem);
        std::fprintf(stderr, "replica: session error: %s\n", failure.what());
    }

    void execute(Job& job) {
        Link& link = *job.link;
        const Frame& frame = job.frame;
        switch (job.kind) {
        case Job::Kind::cleanup:
            for (auto& [client_session, state] : link.sessions) {
                try { client_->destroy_session(state.internal); }
                catch (const std::exception& failure) { member_failed(failure); }
                change_sessions(-1);
            }
            if (!link.sessions.empty()) {
                std::fprintf(stderr, "replica: client %s left; %zu session(s) freed\n",
                    short_peer(link.peer).c_str(), link.sessions.size());
            }
            link.sessions.clear();
            return;
        case Job::Kind::create: {
            if (link.gone) return;
            if (link.sessions.contains(frame.session)) {
                link.post(error_frame(frame, "session already exists"));
                return;
            }
            if (sessions_in_use() >= options_.request.sessions) {
                link.post(error_frame(frame, "replica_full"));
                return;
            }
            SessionState state;
            try {
                state.internal = client_->create_session();
            } catch (const std::exception& failure) {
                // Creating a session on a linked route only fails when the route broke.
                throw Dissolve(std::string("creating a session failed: ") + failure.what());
            }
            link.sessions[frame.session] = state;
            change_sessions(+1);
            std::fprintf(stderr, "replica: client %s opened a session (%u/%u in use)\n",
                short_peer(link.peer).c_str(), sessions_in_use(), options_.request.sessions);
            link.post(ack_frame(frame.session, 0));
            return;
        }
        case Job::Kind::reset:
        case Job::Kind::destroy: {
            const auto found = link.sessions.find(frame.session);
            if (found == link.sessions.end()) {
                link.post(error_frame(frame, "unknown session ID"));
                return;
            }
            try {
                if (job.kind == Job::Kind::reset) {
                    client_->reset_session(found->second.internal);
                    found->second.position = 0;
                } else {
                    client_->destroy_session(found->second.internal);
                    link.sessions.erase(found);
                    change_sessions(-1);
                }
            } catch (const std::exception& failure) {
                member_failed(failure);
                link.post(error_frame(frame, failure.what()));
                return;
            }
            link.post(ack_frame(frame.session, 0));
            return;
        }
        case Job::Kind::request: {
            const auto found = link.sessions.find(frame.session);
            if (found == link.sessions.end()) {
                link.post(error_frame(frame, "unknown session ID"));
                return;
            }
            if (link.gone) return;
            SessionState& state = found->second;
            if (frame.position != state.position) {
                link.post(error_frame(frame, "session position mismatch"));
                return;
            }
            std::string problem;
            const std::string prompt(frame.payload.begin(), frame.payload.end());
            const std::uint32_t position = run_request(state, prompt, frame.position, job.budget,
                &link, frame.session, frame.request, problem);
            if (!problem.empty()) link.post(error_frame(frame, problem));
            else link.post(ack_frame(frame.session, frame.request, position));
            return;
        }
        }
    }

    // One request around the ring: budget to the tail, prompt to the head, then every token
    // from the tail's return link is relayed to the client (never waiting on it). At the end
    // the answer is committed (or rolled back, if cancelled) on every member. Returns the
    // session's new position; `problem` is set when the request failed, in which case the
    // session is reset. Throws Dissolve when the replica itself broke.
    std::uint32_t run_request(SessionState& state, const std::string& prompt, std::uint32_t position,
        std::uint32_t budget, Link* link, std::uint64_t client_session, std::uint64_t client_request,
        std::string& problem) {
        const StageConnections& stages = client_->stages();
        Connection& ring = *client_->ring_return();
        const std::uint64_t request = state.next_request++;
        try {
            Frame limit;
            limit.type = Type::stream_prompt;
            limit.session = state.internal;
            limit.request = request;
            limit.rows = budget;
            auto [accepted, ignored] = stages.back()->exchange(limit);
            (void) ignored;
            require_ack(accepted, limit);
            Frame input;
            input.type = Type::prompt;
            input.session = state.internal;
            input.request = request;
            input.position = position;
            input.payload.assign(prompt.begin(), prompt.end());
            stages.front()->send(input);
        } catch (const std::exception& failure) {
            member_failed(failure);
            problem = failure.what();
            return state.position;
        }
        bool cancel_sent = false;
        Result last;
        for (;;) {
            Frame frame;
            try {
                frame = ring.receive_frame();
            } catch (const std::exception& failure) {
                throw Dissolve(std::string("the ring stopped returning tokens: ") + failure.what());
            }
            if (frame.session != state.internal || frame.request != request) {
                std::fprintf(stderr, "replica: ignoring a stray frame from the ring\n");
                continue;
            }
            if (frame.type == Type::error) {
                problem.assign(frame.payload.begin(), frame.payload.end());
                break;
            }
            try {
                last = require_streamed(frame, state.internal, request);
            } catch (const std::exception& failure) {
                throw Dissolve(std::string("the ring returned an invalid frame: ") + failure.what());
            }
            if (link) {
                Frame relayed = frame;
                relayed.session = client_session;
                relayed.request = client_request;
                link->post(std::move(relayed));
            }
            if (frame.type == Type::result) break;
            if (!cancel_sent && link && link->cancelled(client_session, client_request)) {
                cancel_sent = true;
                try {
                    Frame cancel;
                    cancel.type = Type::cancel_request;
                    cancel.session = state.internal;
                    cancel.request = request;
                    auto [stopped, ignored] = stages.back()->exchange(cancel);
                    (void) ignored;
                    require_ack(stopped, cancel);
                } catch (const std::exception& failure) {
                    member_failed(failure);
                }
            }
        }
        if (!problem.empty()) {
            // Leave nothing half-done on any member: end the request where it started, then
            // start the conversation over. The replica itself carries on.
            std::fprintf(stderr, "replica: request failed: %s\n", problem.c_str());
            for (Connection* stage : stages) {
                Frame end;
                end.type = Type::end_request;
                end.session = state.internal;
                end.request = request;
                try { stage->exchange(end); } catch (const std::exception&) {}
            }
            try {
                control_all(stages, Type::reset_session, state.internal);
            } catch (const std::exception& failure) {
                member_failed(failure);
            }
            state.position = 0;
            return 0;
        }
        std::uint32_t next = position;
        try {
            if (cancel_sent) {
                rollback_all(stages, state.internal, request, position);
            } else {
                commit_final_token(stages, state.internal, request, last.position, last.token,
                    client_->hidden());
                next = last.position + 1;
            }
            control_all(stages, Type::end_request, state.internal, request);
        } catch (const std::exception& failure) {
            member_failed(failure);
            problem = failure.what();
            try { control_all(stages, Type::reset_session, state.internal); }
            catch (const std::exception& reset_failure) { member_failed(reset_failure); }
            state.position = 0;
            return 0;
        }
        state.position = next;
        {
            std::lock_guard lock(status_mutex_);
            ++requests_served_;
        }
        return next;
    }

    void dissolve(const std::string& reason) {
        ready_ = false;
        std::fprintf(stderr, "replica %s dissolved: %s\n", replica_id_.c_str(), reason.c_str());
        {
            std::lock_guard lock(status_mutex_);
            ++dissolutions_;
            last_dissolution_ = reason;
            sessions_in_use_ = 0;
        }
        set_state("dissolved");
        set_event("dissolved: " + reason);
        write_status();
        Frame notice;
        notice.type = Type::error;
        const std::string text = "replica dissolved: " + reason;
        notice.payload.assign(text.begin(), text.end());
        {
            std::lock_guard lock(links_mutex_);
            for (const std::weak_ptr<Link>& weak : links_) {
                if (const std::shared_ptr<Link> link = weak.lock()) {
                    link->sessions.clear();
                    link->close_with(notice);
                }
            }
            links_.clear();
        }
        {
            std::lock_guard lock(jobs_mutex_);
            jobs_.clear();
        }
        release_route();
    }

    // ---- Front door ----

    void accept_clients(socket_t listener) {
        for (;;) {
            const socket_t accepted = accept(listener, nullptr, nullptr);
            if (accepted == invalid_socket) {
                sleep_ms(100);
                continue;
            }
            auto link = std::make_shared<Link>();
            link->socket = accepted;
            // The sidecar names the client's authenticated PeerID first.
            set_socket_timeout(accepted, 5000);
            if (!recv_peer_id(accepted, link->peer)) {
                close_socket(accepted);
                continue;
            }
            set_socket_timeout(accepted, 0);
            {
                std::lock_guard lock(links_mutex_);
                std::erase_if(links_, [](const std::weak_ptr<Link>& weak) { return weak.expired(); });
                links_.push_back(link);
            }
            std::thread([link] { write_client(link); }).detach();
            std::thread([this, link] { read_client(link); }).detach();
        }
    }

    static void write_client(const std::shared_ptr<Link>& link) {
        for (;;) {
            Frame frame;
            {
                std::unique_lock lock(link->mutex);
                link->wake.wait(lock, [&] { return !link->outbox.empty() || link->closing || link->gone; });
                if (link->gone || link->outbox.empty()) break;
                frame = std::move(link->outbox.front());
                link->outbox.pop_front();
            }
            std::string error;
            if (!send_frame(link->socket, frame, error)) {
                link->gone = true;
                break;
            }
        }
        shutdown_socket(link->socket);
        link->thread_done();
    }

    void push(Job job) {
        std::lock_guard lock(jobs_mutex_);
        jobs_.push_back(std::move(job));
        jobs_wake_.notify_all();
    }

    void read_client(const std::shared_ptr<Link>& link) {
        for (;;) {
            Frame frame;
            std::string error;
            if (!recv_frame(link->socket, frame, error)) break;
            if (!ready_) {
                link->post(error_frame(frame, "replica_not_ready"));
                continue;
            }
            switch (frame.type) {
            case Type::create_session:
            case Type::reset_session:
            case Type::destroy_session:
                if (frame.session == 0 || frame.request != 0 || !empty_control(frame)) {
                    link->post(error_frame(frame, "invalid session frame"));
                    break;
                }
                push({frame.type == Type::create_session ? Job::Kind::create
                    : frame.type == Type::reset_session ? Job::Kind::reset : Job::Kind::destroy,
                    link, frame});
                break;
            case Type::stream_prompt:
                if (frame.session == 0 || frame.request == 0 || frame.rows == 0 || !frame.payload.empty()) {
                    link->post(error_frame(frame, "bad stream_prompt frame"));
                    break;
                }
                link->budgets[frame.session] = frame.rows;
                link->post(ack_frame(frame.session, frame.request));
                break;
            case Type::prompt: {
                const auto budget = link->budgets.find(frame.session);
                if (frame.session == 0 || frame.request == 0 || budget == link->budgets.end()) {
                    link->post(error_frame(frame, "send stream_prompt before the prompt"));
                    break;
                }
                const std::uint32_t rows = budget->second;
                link->budgets.erase(budget);
                push({Job::Kind::request, link, frame, rows});
                break;
            }
            case Type::cancel_request:
                link->cancel_request = frame.request;
                link->cancel_session = frame.session;
                break;
            default:
                link->post(error_frame(frame, "unexpected frame for a replica"));
                break;
            }
        }
        link->gone = true;
        link->wake.notify_all();
        push({Job::Kind::cleanup, link, {}});
        link->thread_done();
    }

    // ---- Status ----

    std::uint32_t sessions_in_use() {
        std::lock_guard lock(status_mutex_);
        return sessions_in_use_;
    }

    void change_sessions(int delta) {
        {
            std::lock_guard lock(status_mutex_);
            sessions_in_use_ = static_cast<std::uint32_t>(static_cast<int>(sessions_in_use_) + delta);
        }
        // Clients choose replicas by free sessions: publish the change now, not in 5 s.
        write_status();
    }

    void set_state(const std::string& state) {
        std::lock_guard lock(status_mutex_);
        state_ = state;
    }

    void set_event(const std::string& event) {
        std::lock_guard lock(status_mutex_);
        last_event_ = event;
    }

    // The replica status file: the sidecar advertises dan/replica/1/<sha> while it says
    // "ready" and is fresh, and answers live replica queries from it.
    void write_status() {
        if (options_.status_file.empty()) return;
        std::ostringstream json;
        {
            std::lock_guard lock(status_mutex_);
            const bool ready = state_ == "ready";
            json << "{\"protocol_version\":1,\"state\":\"" << state_ << "\""
                 << ",\"owner\":\"" << self_peer_ << "\""
                 << ",\"replica_id\":\"" << (ready ? replica_id_ : "") << "\""
                 << ",\"model_sha256\":\"" << (ready ? model_sha_ : "") << "\""
                 << ",\"model_id\":\"" << json_escape(ready ? model_id_ : "") << "\""
                 << ",\"draft_sha256\":\"" << (ready ? draft_sha_ : "") << "\""
                 << ",\"runtime_abi\":\"" << json_escape(options_.request.runtime_abi) << "\""
                 << ",\"context\":" << (ready ? context_ : 0)
                 << ",\"sessions_max\":" << options_.request.sessions
                 << ",\"sessions_in_use\":" << sessions_in_use_
                 << ",\"members\":[";
            for (std::size_t index = 0; ready && index < members_.size(); ++index) {
                const Member& member = members_[index];
                json << (index ? "," : "") << "{\"peer\":\"" << member.peer << "\",\"worker\":\""
                     << json_escape(member.worker) << "\",\"begin\":" << member.begin
                     << ",\"end\":" << member.end << ",\"rtt_ms\":" << member.rtt_ms
                     << ",\"path\":\"" << (member.relayed ? "relay" : "direct") << "\"}";
            }
            json << "],\"edges\":[";
            for (std::size_t index = 0; index < edges_.size(); ++index) {
                const Edge& edge = edges_[index];
                json << (index ? "," : "") << "{\"from\":\"" << edge.from << "\",\"to\":\"" << edge.to
                     << "\",\"rtt_ms\":" << edge.rtt_ms << ",\"path\":\""
                     << (edge.relayed ? "relay" : "direct") << "\",\"problem\":\""
                     << json_escape(edge.problem) << "\"}";
            }
            json << "],\"formation_attempts\":" << attempts_ << ",\"formations\":" << formations_
                 << ",\"races_lost\":" << races_lost_ << ",\"rejected_links\":" << rejected_links_
                 << ",\"warmup_failures\":" << warmup_failures_ << ",\"dissolutions\":" << dissolutions_
                 << ",\"requests_served\":" << requests_served_
                 << ",\"formation_ms\":" << static_cast<std::int64_t>(formation_ms_)
                 << ",\"load_ms\":" << static_cast<std::int64_t>(load_ms_)
                 << ",\"warmup_ms\":" << static_cast<std::int64_t>(warmup_ms_)
                 << ",\"ready_since_unix_ms\":" << (ready ? ready_since_ms_ : 0)
                 << ",\"last_event\":\"" << json_escape(last_event_) << "\""
                 << ",\"last_dissolution\":\"" << json_escape(last_dissolution_) << "\""
                 << ",\"updated_unix_ms\":" << unix_ms() << "}\n";
        }
        std::lock_guard write(write_mutex_);
        const std::filesystem::path path(options_.status_file);
        const std::filesystem::path temporary = path.string() + ".tmp";
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            if (!file) return;
            file << json.str();
        }
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
    }

    const ReplicaOwnerOptions options_;
    std::string self_peer_;
    std::unique_ptr<InferenceClient> client_;   // executor thread only
    std::string replica_id_;
    std::string last_skip_;                     // why the last attempt was skipped
    std::atomic<bool> ready_{false};

    std::mutex jobs_mutex_;
    std::condition_variable jobs_wake_;
    std::deque<Job> jobs_;
    std::mutex links_mutex_;
    std::vector<std::weak_ptr<Link>> links_;

    std::mutex write_mutex_;
    std::mutex status_mutex_;
    std::string state_ = "free";
    std::string model_sha_, model_id_, draft_sha_;
    std::uint32_t context_ = 0;
    std::uint32_t sessions_in_use_ = 0;
    std::vector<Member> members_;
    std::vector<Edge> edges_;
    std::uint64_t attempts_ = 0, formations_ = 0, races_lost_ = 0, rejected_links_ = 0,
        warmup_failures_ = 0, dissolutions_ = 0, requests_served_ = 0;
    double formation_ms_ = 0, load_ms_ = 0, warmup_ms_ = 0;
    std::int64_t ready_since_ms_ = 0;
    std::string last_event_, last_dissolution_;
};

} // namespace

int run_replica_owner(const ReplicaOwnerOptions& options) {
    if (options.discover.empty() || options.self_control.empty() || options.session_listen.empty()
        || options.request.models.empty()) {
        throw std::runtime_error("a replica owner needs --discover, --self-control, "
            "--session-listen and a model");
    }
    // Lives for the whole process: client threads may outlast any one replica.
    auto* owner = new Owner(options);
    return owner->run();
}

} // namespace dan::provider_owned
