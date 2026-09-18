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

// Why this worker cannot take part in a route for `option`, or empty.
std::string unusable(const Worker& worker, const PlacementRequest& request,
    const ModelOption& option, std::uint32_t context) {
    const ProviderCapability& hello = worker.hello;
    if (hello.state != "available") return "state " + (hello.state.empty() ? "unknown" : hello.state);
    if (hello.runtime_abi != request.runtime_abi) return "runtime ABI " + hello.runtime_abi;
    const std::string sha = lowercase(option.manifest.sha256);
    if (std::none_of(hello.models.begin(), hello.models.end(),
            [&](const std::string& model) { return lowercase(model) == sha; })) return "model not in catalog";
    if (hello.max_context < context) return "context limit";
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

// Latency class of a link, in 25 ms steps: small differences are noise, and a relayed
// link counts as one step worse because it also costs the relay's bandwidth.
std::uint64_t link_cost(const PlacementCandidate& candidate) {
    const std::uint64_t steps = candidate.rtt_ms / 25;
    return steps + (candidate.relayed ? 1 : 0);
}

double milliseconds_since(Clock::time_point start) {
    return elapsed_ns(start) / 1e6;
}

bool open_worker(const PlacementCandidate& candidate, Worker& worker, std::uint32_t timeout_ms) {
    try {
        worker.candidate = candidate;
        worker.connection = std::make_unique<Connection>(candidate.control);
        worker.connection->set_timeout(timeout_ms);
        if (candidate.local) worker.connection->send_text("DAN-P2P/1 " + candidate.peer_id + "\n");
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

Discovery discover_candidates(const std::string& api_endpoint,
    const std::vector<std::string>& model_sha256) {
    if (!valid_private_endpoint(api_endpoint) || !api_endpoint.starts_with("127.")) {
        throw std::runtime_error("the candidate API must be a loopback host:port");
    }
    if (model_sha256.empty()) throw std::runtime_error("no model to discover");
    std::string query = "DAN-CANDIDATES/1";
    for (const std::string& sha : model_sha256) {
        if (!hex_string(sha, 64)) throw std::runtime_error("invalid model SHA-256");
        query += " " + lowercase(sha);
    }
    query += "\n";
    const socket_t socket = connect_endpoint(api_endpoint);
    // The sidecar searches the DHT and queries every candidate before answering.
    set_socket_timeout(socket, 120000);
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
        } else if (fields[0] == "RETURN" && (fields.size() == 2
                || (fields.size() == 3 && fields[2] == "/dan-return"))
            && valid_private_endpoint(fields[1])) {
            discovery.return_listen = fields[1];
            if (fields.size() == 3) discovery.return_prefix = fields[2];
        } else if (fields[0] == "CANDIDATE" && fields.size() >= 3 && valid_peer_id(fields[1])
            && valid_private_endpoint(fields[2]) && fields[2].starts_with("127.")) {
            PlacementCandidate candidate{std::string(fields[2]), std::string(fields[1])};
            // CANDIDATE <peer> <control> <offered MiB> <abi> <rtt ms> <direct|relay>
            if (fields.size() >= 4) {
                candidate.offered_mib = std::strtoull(std::string(fields[3]).c_str(), nullptr, 10);
            }
            if (fields.size() >= 7) {
                candidate.rtt_ms = static_cast<std::uint32_t>(
                    std::strtoul(std::string(fields[5]).c_str(), nullptr, 10));
                candidate.relayed = fields[6] == "relay";
            }
            discovery.candidates.push_back(candidate);
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

namespace {

// One request to the local sidecar's line API; the reply's lines split into fields, up to
// (not including) END. Throws on ERR, an incomplete reply or a closed connection.
std::vector<std::vector<std::string>> api_request(const std::string& api_endpoint,
    const std::string& line, std::uint32_t timeout_ms) {
    if (!valid_private_endpoint(api_endpoint) || !api_endpoint.starts_with("127.")) {
        throw std::runtime_error("the sidecar API must be a loopback host:port");
    }
    const socket_t socket = connect_endpoint(api_endpoint);
    set_socket_timeout(socket, timeout_ms);
    std::string text;
    const std::string query = line + "\n";
    bool sent = send_all(socket, query.data(), query.size());
    for (char buffer[4096]; sent && text.size() < 1024 * 1024;) {
        const int count = recv(socket, buffer, sizeof(buffer), 0);
        if (count <= 0) break;
        text.append(buffer, static_cast<std::size_t>(count));
        if (text.ends_with("END\n") || (text.starts_with("ERR ") && text.ends_with("\n"))) break;
    }
    close_socket(socket);
    if (!sent) throw std::runtime_error("could not reach the sidecar API");
    if (text.starts_with("ERR ")) {
        throw std::runtime_error("sidecar: " + text.substr(4, text.find('\n') - 4));
    }
    if (!text.ends_with("END\n")) throw std::runtime_error("sidecar reply was incomplete");
    std::vector<std::vector<std::string>> lines;
    for (std::string_view rest = text; !rest.empty();) {
        const std::size_t newline = rest.find('\n');
        const std::string_view current = rest.substr(0, newline);
        rest.remove_prefix(newline + 1);
        if (current == "END") break;
        std::vector<std::string> fields;
        for (std::string_view remaining = current; !remaining.empty();) {
            const std::size_t space = remaining.find(' ');
            fields.emplace_back(remaining.substr(0, space));
            if (space == std::string_view::npos) break;
            remaining.remove_prefix(space + 1);
        }
        if (!fields.empty()) lines.push_back(std::move(fields));
    }
    return lines;
}

std::uint32_t to_u32(const std::string& text) {
    return static_cast<std::uint32_t>(std::strtoul(text.c_str(), nullptr, 10));
}

} // namespace

LinkProbe probe_link(const std::string& api_endpoint, const std::string& from,
    const std::string& to) {
    if (!valid_peer_id(from) || !valid_peer_id(to)) throw std::runtime_error("invalid PeerID");
    // A remote probe dials, may wait for hole punching, then pings: allow for all of it.
    for (const auto& fields : api_request(api_endpoint, "DAN-PROBE/1 " + from + " " + to, 60000)) {
        if (fields.size() == 3 && fields[0] == "PROBE" && (fields[2] == "direct" || fields[2] == "relay")) {
            return {to_u32(fields[1]), fields[2] == "relay"};
        }
    }
    throw std::runtime_error("sidecar sent no probe result");
}

std::vector<ReplicaCandidate> discover_replicas(const std::string& api_endpoint,
    const std::vector<std::string>& model_sha256) {
    if (model_sha256.empty()) throw std::runtime_error("no model to discover");
    std::string query = "DAN-REPLICAS/1";
    for (const std::string& sha : model_sha256) {
        if (!hex_string(sha, 64)) throw std::runtime_error("invalid model SHA-256");
        query += " " + lowercase(sha);
    }
    std::vector<ReplicaCandidate> replicas;
    for (const auto& fields : api_request(api_endpoint, query, 120000)) {
        // REPLICA <owner> <control> <rtt ms> <direct|relay> <replica id> <model sha256>
        //         <sessions free> <sessions max> <context> <stages> <draft sha256 | ->
        if (fields[0] == "SELF") continue;
        if (fields[0] != "REPLICA" || fields.size() != 12 || !valid_peer_id(fields[1])
            || !valid_private_endpoint(fields[2]) || !fields[2].starts_with("127.")
            || (fields[4] != "direct" && fields[4] != "relay") || !hex_string(fields[5], 32)
            || !hex_string(fields[6], 64) || (fields[11] != "-" && !hex_string(fields[11], 64))) {
            throw std::runtime_error("sidecar sent an invalid replica line");
        }
        ReplicaCandidate replica;
        replica.owner = fields[1];
        replica.control = fields[2];
        replica.rtt_ms = to_u32(fields[3]);
        replica.relayed = fields[4] == "relay";
        replica.replica_id = fields[5];
        replica.model_sha256 = lowercase(fields[6]);
        replica.sessions_free = to_u32(fields[7]);
        replica.sessions_max = to_u32(fields[8]);
        replica.context = to_u32(fields[9]);
        replica.stages = to_u32(fields[10]);
        if (fields[11] != "-") replica.draft_sha256 = lowercase(fields[11]);
        replicas.push_back(std::move(replica));
    }
    return replicas;
}

PlacedRoute place_route(const std::vector<PlacementCandidate>& candidates,
    const PlacementRequest& request) {
    if (candidates.empty() || request.models.empty() || request.sessions == 0
        || std::any_of(request.models.begin(), request.models.end(),
            [](const ModelOption& option) { return option.manifest.hidden == 0; })) {
        throw std::runtime_error("placement needs candidates, models with a hidden size and sessions");
    }
    if (std::any_of(candidates.begin(), candidates.end(), [&](const PlacementCandidate& candidate) {
            return candidate.peer_id.empty() != candidates.front().peer_id.empty();
        })) {
        throw std::runtime_error("candidates must be all libp2p or all direct");
    }
    const auto started = Clock::now();
    PlacementTimings timings;
    // Copied out however this ends, so a caller can count lost races even on failure.
    struct Report {
        const PlacementTimings& timings;
        PlacementTimings* out;
        ~Report() { if (out) *out = timings; }
    } report{timings, request.report};
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
        // Try the models in order and keep the first the available workers can run: that
        // is how a client asks for the largest model the network can serve right now.
        std::vector<Worker*> pool;
        std::optional<std::vector<StageAssignment>> plan;
        const ModelOption* chosen = nullptr;
        std::uint32_t context = 0;
        bool from_cache = false;
        const auto plan_started = Clock::now();
        for (const ModelOption& option : request.models) {
            const std::uint32_t option_context = request.context != 0
                ? request.context : option.manifest.context;
            if (option_context == 0) continue;
            pool.clear();
            for (Worker& worker : workers) {
                if (!worker.connection) continue;
                const std::string reason = unusable(worker, request, option, option_context);
                if (reason.empty()) pool.push_back(&worker);
                else if (request.models.size() == 1) {
                    std::fprintf(stderr, "placement: skipping %s: %s\n",
                        worker.candidate.control.c_str(), reason.c_str());
                }
            }
            // Every token crosses these links, so prefer workers this client reaches
            // quickly and directly; memory breaks near-ties, PeerID breaks exact ones.
            std::sort(pool.begin(), pool.end(), [](const Worker* left, const Worker* right) {
                const std::uint64_t left_cost = link_cost(left->candidate);
                const std::uint64_t right_cost = link_cost(right->candidate);
                if (left_cost != right_cost) return left_cost < right_cost;
                if (left->hello.offered_vram_mib != right->hello.offered_vram_mib) {
                    return left->hello.offered_vram_mib > right->hello.offered_vram_mib;
                }
                return left->order_key < right->order_key;
            });
            std::optional<std::size_t> head;
            if (!request.head.empty()) {
                // Replica formation: the owner's own worker leads the route.
                const auto found = std::find_if(pool.begin(), pool.end(), [&](const Worker* worker) {
                    return worker->candidate.control == request.head;
                });
                if (found == pool.end()) continue;
                std::rotate(pool.begin(), found, found + 1);
                head = 0;
            }
            if (pool.size() > max_planned_candidates) pool.resize(max_planned_candidates);
            std::vector<std::uint64_t> offered;
            for (const Worker* worker : pool) offered.push_back(worker->hello.offered_vram_mib);
            plan = pool.size() >= request.minimum_stages
                ? plan_stages(option.model, offered, option_context, request.sessions,
                    request.minimum_stages, head)
                : std::nullopt;
            if (!plan) continue;
            // Prefer a split the workers already hold: no download, same number of hops.
            std::vector<std::vector<std::pair<int, int>>> cached(pool.size());
            const std::string sha = lowercase(option.manifest.sha256);
            for (std::size_t index = 0; index < pool.size(); ++index) {
                for (const CachedRange& range : pool[index]->hello.cached) {
                    if (lowercase(range.model_sha256) == sha) {
                        cached[index].emplace_back(range.begin, range.end);
                    }
                }
            }
            from_cache = false;
            if (const auto reuse = plan_from_cache(option.model, offered, cached, option_context,
                    request.sessions, request.minimum_stages, plan->size(), head)) {
                plan = reuse;
                from_cache = true;
            }
            chosen = &option;
            context = option_context;
            break;
        }
        if (!plan || !chosen) throw std::runtime_error("no placement fits the available workers");

        // Speculative decoding: the smallest other model the first stage's worker offers.
        const ModelOption* draft = nullptr;
        if (request.speculate) {
            const Worker& first = *pool[(*plan)[0].provider];
            for (const ModelOption& option : request.models) {
                if (lowercase(option.manifest.sha256) == lowercase(chosen->manifest.sha256)
                    || option.model.logical_bytes >= chosen->model.logical_bytes) continue;
                const std::string sha = lowercase(option.manifest.sha256);
                if (std::none_of(first.hello.models.begin(), first.hello.models.end(),
                        [&](const std::string& model) { return lowercase(model) == sha; })) continue;
                if (!draft || option.model.logical_bytes < draft->model.logical_bytes) {
                    draft = &option;
                }
            }
        }

        StageRequest base;
        base.route_id = random_route_id();
        base.model_sha256 = lowercase(chosen->manifest.sha256);
        base.context = context;
        base.sessions = request.sessions;
        const auto stage_request = [&](std::size_t index) {
            StageRequest stage = base;
            stage.begin = (*plan)[index].begin;
            stage.end = (*plan)[index].end;
            // Only the first stage drafts: it is the one that turns tokens into activations.
            if (index == 0 && draft) stage.draft_sha256 = lowercase(draft->manifest.sha256);
            return stage;
        };
        std::fprintf(stderr, "placement: route %s model %s%s plan", base.route_id.c_str(),
            chosen->manifest.model_id.c_str(), from_cache ? " (cached layers)" : "");
        for (const StageAssignment& stage : *plan) {
            std::fprintf(stderr, " %s[%d,%d)", pool[stage.provider]->hello.id.c_str(),
                stage.begin, stage.end);
        }
        std::fprintf(stderr, "\n");
        timings.plan_ms += milliseconds_since(plan_started);

        if (request.check_route) {
            // Replica formation: the links between these particular workers must be good
            // enough before anyone is reserved. The caller measures them.
            std::vector<PlacementCandidate> ordered;
            for (const StageAssignment& stage : *plan) ordered.push_back(pool[stage.provider]->candidate);
            const std::string leave_out = request.check_route(ordered);
            if (!leave_out.empty()) {
                for (Worker& worker : workers) {
                    if (worker.candidate.control != leave_out) continue;
                    std::fprintf(stderr, "placement: leaving out %s: its link on this route "
                        "is too slow\n", worker.hello.id.c_str());
                    worker.connection.reset();
                    worker.recheck = false;
                }
                ++timings.rejected_links;
                continue;
            }
        }

        // Reserve every planned worker; the first lease a worker accepts wins.
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
                    if (worker.recheck) ++timings.refusals;
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
        placed.manifest = chosen->manifest;
        if (draft) {
            placed.draft_model_id = draft->manifest.model_id;
            placed.draft_sha256 = lowercase(draft->manifest.sha256);
        }
        placed.route.hidden = chosen->manifest.hidden;
        const bool p2p = !candidates.front().peer_id.empty();
        for (std::size_t index = 0; index < plan->size(); ++index) {
            Worker& worker = *pool[(*plan)[index].provider];
            placed.route.stage_endpoints.push_back(worker.candidate.control);
            placed.route.ring_targets.push_back(worker.hello.ring_endpoint);
            placed.route.peer_ids.push_back(worker.candidate.peer_id);
            placed.connections.push_back(std::move(worker.connection));
            placed.stages.push_back({worker.hello.id, worker.candidate.peer_id,
                (*plan)[index].begin, (*plan)[index].end, worker.candidate.rtt_ms,
                worker.candidate.relayed});
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
