#include "provider_owned/placement.hpp"
#include "provider_owned/lease.hpp"
#include "provider_owned/route.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

namespace dan::provider_owned {
namespace {

constexpr std::size_t max_planned_candidates = 8;

struct Worker {
    PlacementCandidate candidate;
    std::unique_ptr<Connection> connection;
    ProviderCapability hello;
    std::string order_key;  // PeerID when known, else the worker's own ID
    bool recheck = false;   // refused as busy: reconnect for a fresh greeting
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
    return value;
}

std::string peer_of(std::string_view target) {
    const std::size_t found = target.rfind("/p2p/");
    return found == std::string_view::npos ? std::string{} : std::string(target.substr(found + 5));
}

// Why this worker cannot take part, or empty.
std::string unusable(const Worker& worker, const PlacementRequest& request) {
    const ProviderCapability& hello = worker.hello;
    if (hello.state != "available") return "state " + (hello.state.empty() ? "unknown" : hello.state);
    if (hello.runtime_abi != request.runtime_abi) return "runtime ABI " + hello.runtime_abi;
    const std::string sha = lowercase(request.manifest.sha256);
    if (std::none_of(hello.models.begin(), hello.models.end(),
            [&](const std::string& model) { return lowercase(model) == sha; })) return "model not in catalog";
    if (hello.max_context < request.context) return "context limit";
    if (hello.max_sessions < request.sessions) return "session limit";
    if (worker.candidate.peer_id.empty()) {
        if (!valid_private_endpoint(hello.ring_endpoint)) return "invalid direct ring endpoint";
    } else if (!valid_p2p_target(hello.ring_endpoint)
        || peer_of(hello.ring_endpoint) != worker.candidate.peer_id) {
        // The ring address must name the peer we actually reached, or a worker could
        // redirect the ring to someone else.
        return "ring address does not match the authenticated PeerID";
    }
    return {};
}

double milliseconds_since(Clock::time_point start) {
    return elapsed_ns(start) / 1e6;
}

bool open_worker(const PlacementCandidate& candidate, Worker& worker, std::uint32_t timeout_ms) {
    try {
        worker.candidate = candidate;
        worker.connection = std::make_unique<Connection>(candidate.control);
        worker.connection->set_timeout(timeout_ms);
        const Frame hello = worker.connection->receive();
        const std::string text(hello.payload.begin(), hello.payload.end());
        if (hello.type != Type::provider_available || !parse_available(text, worker.hello)) {
            throw std::runtime_error("invalid worker greeting");
        }
        worker.order_key = candidate.peer_id.empty() ? worker.hello.id : candidate.peer_id;
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "placement: skipping %s: %s\n", candidate.control.c_str(), error.what());
        return false;
    }
}

// Runs `task(index)` for every index in parallel; returns each one's error ("" = success).
template <typename Task>
std::vector<std::string> run_all(std::size_t count, Task task) {
    std::vector<std::string> errors(count);
    std::vector<std::thread> threads;
    for (std::size_t index = 0; index < count; ++index) {
        threads.emplace_back([&, index] {
            try { task(index); }
            catch (const std::exception& error) { errors[index] = error.what(); }
        });
    }
    for (std::thread& thread : threads) thread.join();
    return errors;
}

Frame stage_frame(Type type, const StageRequest& request) {
    Frame frame;
    frame.type = type;
    const std::string text = stage_request_message(request);
    frame.payload.assign(text.begin(), text.end());
    return frame;
}

} // namespace

std::string random_route_id() {
    std::random_device device;
    static constexpr char digits[] = "0123456789abcdef";
    std::string id;
    for (int index = 0; index < 32; ++index) id += digits[device() & 15];
    return id;
}

Discovery discover_candidates(const std::string& api_endpoint, const std::string& model_sha256) {
    if (!valid_private_endpoint(api_endpoint) || !api_endpoint.starts_with("127.")) {
        throw std::runtime_error("the candidate API must be a loopback host:port");
    }
    if (!hex_string(model_sha256, 64)) throw std::runtime_error("invalid model SHA-256");
    const socket_t socket = connect_endpoint(api_endpoint);
    // The sidecar searches the DHT and queries every candidate before answering.
    set_socket_timeout(socket, 120000);
    const std::string query = "DAN-CANDIDATES/1 " + lowercase(model_sha256) + "\n";
    std::string text;
    bool sent = send_all(socket, query.data(), query.size());
    for (char buffer[4096]; sent && text.size() < 1024 * 1024;) {
        const int count = recv(socket, buffer, sizeof(buffer), 0);
        if (count <= 0) break;
        text.append(buffer, static_cast<std::size_t>(count));
        if (text.ends_with("END\n") || text.starts_with("ERR ")) break;
    }
    close_socket(socket);
    if (!sent) throw std::runtime_error("could not query the candidate API");

    Discovery discovery;
    bool complete = false;
    for (std::string_view rest = text; !rest.empty() && !complete;) {
        const std::size_t newline = rest.find('\n');
        if (newline == std::string_view::npos) break;
        const std::string_view line = rest.substr(0, newline);
        rest.remove_prefix(newline + 1);
        std::vector<std::string_view> fields;
        for (std::string_view remaining = line; !remaining.empty();) {
            const std::size_t space = remaining.find(' ');
            fields.push_back(remaining.substr(0, space));
            if (space == std::string_view::npos) break;
            remaining.remove_prefix(space + 1);
        }
        if (fields.empty()) continue;
        if (fields[0] == "ERR") {
            throw std::runtime_error("candidate API: " + std::string(line.substr(line.find(' ') + 1)));
        } else if (fields[0] == "SELF" && fields.size() == 2 && valid_peer_id(fields[1])) {
            discovery.self_peer = fields[1];
        } else if (fields[0] == "RETURN" && fields.size() == 2 && valid_private_endpoint(fields[1])) {
            discovery.return_listen = fields[1];
        } else if (fields[0] == "CANDIDATE" && fields.size() >= 3 && valid_peer_id(fields[1])
            && valid_private_endpoint(fields[2]) && fields[2].starts_with("127.")) {
            discovery.candidates.push_back({std::string(fields[2]), std::string(fields[1])});
        } else if (fields[0] == "END") {
            complete = true;
        } else {
            throw std::runtime_error("candidate API sent an invalid line");
        }
    }
    if (!complete || discovery.self_peer.empty()) {
        throw std::runtime_error("candidate API reply was incomplete");
    }
    return discovery;
}

PlacedRoute place_route(const std::vector<PlacementCandidate>& candidates,
    const PlacementRequest& request) {
    if (candidates.empty() || request.manifest.hidden == 0 || request.context == 0
        || request.sessions == 0) {
        throw std::runtime_error("placement needs candidates, the model hidden size, context and sessions");
    }
    if (std::any_of(candidates.begin(), candidates.end(), [&](const PlacementCandidate& candidate) {
            return candidate.peer_id.empty() != candidates.front().peer_id.empty();
        })) {
        throw std::runtime_error("candidates must be all libp2p or all direct");
    }
    const auto started = Clock::now();
    // Greet every candidate at once: over a relay each connection can take seconds.
    std::vector<Worker> opened(candidates.size());
    std::vector<char> usable(candidates.size(), 0);
    run_all(candidates.size(), [&](std::size_t index) {
        usable[index] = open_worker(candidates[index], opened[index], request.connect_timeout_ms);
    });
    std::vector<Worker> workers;
    for (std::size_t index = 0; index < opened.size(); ++index) {
        if (usable[index]) workers.push_back(std::move(opened[index]));
    }
    PlacementTimings timings;
    timings.greeting_ms = milliseconds_since(started);
    for (int attempt = 0; attempt < request.attempts; ++attempt) {
        ++timings.attempts;
        for (Worker& worker : workers) {
            if (!worker.recheck) continue;
            worker.recheck = false;
            const PlacementCandidate candidate = worker.candidate;
            if (!open_worker(candidate, worker, request.connect_timeout_ms)) {
                worker.connection.reset();
            }
        }
        // v1 preselection: usable workers, most offered memory first, PeerID breaks ties.
        std::vector<Worker*> pool;
        for (Worker& worker : workers) {
            if (!worker.connection) continue;
            const std::string reason = unusable(worker, request);
            if (reason.empty()) pool.push_back(&worker);
            else std::fprintf(stderr, "placement: skipping %s: %s\n",
                worker.candidate.control.c_str(), reason.c_str());
        }
        std::sort(pool.begin(), pool.end(), [](const Worker* left, const Worker* right) {
            return left->hello.offered_vram_mib != right->hello.offered_vram_mib
                ? left->hello.offered_vram_mib > right->hello.offered_vram_mib
                : left->order_key < right->order_key;
        });
        if (pool.size() > max_planned_candidates) pool.resize(max_planned_candidates);
        std::vector<std::uint64_t> offered;
        for (const Worker* worker : pool) offered.push_back(worker->hello.offered_vram_mib);
        const auto plan_started = Clock::now();
        auto plan = pool.size() >= request.minimum_stages
            ? plan_stages(request.model, offered, request.context, request.sessions,
                request.minimum_stages)
            : std::nullopt;
        if (!plan) throw std::runtime_error("no placement fits the available workers");
        // Prefer a split the workers already hold: no download, same number of hops.
        std::vector<std::vector<std::pair<int, int>>> cached(pool.size());
        const std::string sha = lowercase(request.manifest.sha256);
        for (std::size_t index = 0; index < pool.size(); ++index) {
            for (const CachedRange& range : pool[index]->hello.cached) {
                if (lowercase(range.model_sha256) == sha) {
                    cached[index].emplace_back(range.begin, range.end);
                }
            }
        }
        bool from_cache = false;
        if (const auto reuse = plan_from_cache(request.model, offered, cached, request.context,
                request.sessions, request.minimum_stages, plan->size())) {
            plan = reuse;
            from_cache = true;
        }

        StageRequest base;
        base.route_id = random_route_id();
        base.model_sha256 = lowercase(request.manifest.sha256);
        base.context = request.context;
        base.sessions = request.sessions;
        const auto stage_request = [&](std::size_t index) {
            StageRequest stage = base;
            stage.begin = (*plan)[index].begin;
            stage.end = (*plan)[index].end;
            return stage;
        };
        std::fprintf(stderr, "placement: route %s plan%s", base.route_id.c_str(),
            from_cache ? " (cached layers)" : "");
        for (const StageAssignment& stage : *plan) {
            std::fprintf(stderr, " %s[%d,%d)", pool[stage.provider]->hello.id.c_str(),
                stage.begin, stage.end);
        }
        std::fprintf(stderr, "\n");

        // Reserve every planned worker; the first lease a worker accepts wins.
        timings.plan_ms += milliseconds_since(plan_started);
        const auto reserve_started = Clock::now();
        const std::vector<std::string> reserved = run_all(plan->size(), [&](std::size_t index) {
            StageRequest stage = stage_request(index);
            stage.lease_ms = request.lease_ms;
            Worker& worker = *pool[(*plan)[index].provider];
            auto [reply, ignored] = worker.connection->exchange(stage_frame(Type::reserve, stage));
            (void) ignored;
            if (reply.type != Type::ack) throw std::runtime_error("unexpected reservation reply");
        });
        timings.reserve_ms += milliseconds_since(reserve_started);
        if (std::any_of(reserved.begin(), reserved.end(),
                [](const std::string& error) { return !error.empty(); })) {
            for (std::size_t index = 0; index < plan->size(); ++index) {
                Worker& worker = *pool[(*plan)[index].provider];
                if (!reserved[index].empty()) {
                    std::fprintf(stderr, "placement: %s refused: %s\n",
                        worker.hello.id.c_str(), reserved[index].c_str());
                    worker.connection.reset();
                    // Another client may hold it now; look again next attempt. Any other
                    // refusal means its greeting was wrong for this request: drop it.
                    worker.recheck = reserved[index] == "busy";
                    continue;
                }
                try {
                    Frame release;
                    release.type = Type::release_route;
                    release.payload.assign(base.route_id.begin(), base.route_id.end());
                    auto [reply, ignored] = worker.connection->exchange(release);
                    (void) ignored;
                    if (reply.type != Type::ack) throw std::runtime_error("unexpected release reply");
                } catch (const std::exception&) {
                    worker.connection.reset();  // closing the connection releases it too
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(
                100 + std::random_device{}() % 400));
            continue;
        }

        // Every worker agreed: load all ranges in parallel. No replanning after this point.
        const auto load_started = Clock::now();
        const std::vector<std::string> loaded = run_all(plan->size(), [&](std::size_t index) {
            Connection& connection = *pool[(*plan)[index].provider]->connection;
            connection.set_timeout(0);  // downloads can take long; the connection holds the lease
            connection.send(stage_frame(Type::assign_stage, stage_request(index)));
            const Frame ready = connection.receive();
            connection.set_timeout(request.connect_timeout_ms);
            if (ready.type != Type::stage_ready
                || std::string(ready.payload.begin(), ready.payload.end()) != base.route_id) {
                throw std::runtime_error("unexpected assignment reply");
            }
        });
        for (std::size_t index = 0; index < plan->size(); ++index) {
            if (!loaded[index].empty()) {
                throw std::runtime_error("worker " + pool[(*plan)[index].provider]->hello.id
                    + " could not load its stage: " + loaded[index]);
            }
        }

        timings.load_ms = milliseconds_since(load_started);
        PlacedRoute placed;
        placed.timings = timings;
        placed.route_id = base.route_id;
        placed.route.hidden = request.manifest.hidden;
        const bool p2p = !candidates.front().peer_id.empty();
        for (std::size_t index = 0; index < plan->size(); ++index) {
            Worker& worker = *pool[(*plan)[index].provider];
            placed.route.stage_endpoints.push_back(worker.candidate.control);
            placed.route.ring_targets.push_back(worker.hello.ring_endpoint);
            placed.route.peer_ids.push_back(worker.candidate.peer_id);
            placed.connections.push_back(std::move(worker.connection));
            placed.stages.push_back({worker.hello.id, worker.candidate.peer_id,
                (*plan)[index].begin, (*plan)[index].end});
        }
        if (!p2p) placed.route.peer_ids.clear();
        // Loop mode: the last stage sends each new token straight back to the first stage,
        // so decoding needs no client round trip.
        placed.route.loop_target = placed.route.ring_targets.front();
        placed.route.ring_targets.front().clear();  // nobody dials the first stage's ring
        return placed;
    }
    throw std::runtime_error("could not reserve a placement after "
        + std::to_string(request.attempts) + " attempts");
}

} // namespace dan::provider_owned
