#pragma once

// Client-side dynamic placement. Given candidate workers (already discovered or listed),
// the client reads their capabilities, plans stages with the shared planner, reserves the
// chosen workers (lease.hpp), assigns their ranges and returns leased connections ready for
// InferenceClient. There is no coordinator: this client is the planner for its own route.

#include "provider_owned/client.hpp"
#include "provider_owned/manifest.hpp"
#include "provider_owned/planner.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dan::provider_owned {

struct PlacementCandidate {
    std::string control;   // host:port of the worker's control connection (e.g. a sidecar forward)
    std::string peer_id;   // libp2p: PeerID that connection is authenticated to; empty = direct TCP
    // How this client reaches the worker, measured by the sidecar during discovery. Every
    // token crosses these links, so placement prefers close, directly reachable workers.
    std::uint32_t rtt_ms = 0;   // 0 = unknown
    bool relayed = false;
    std::uint64_t offered_mib = 0;  // as the discovery answer reported it (0 = unknown)
    // A worker on this machine reached over loopback, not through the sidecar: this
    // connection announces peer_id itself, as the sidecar would (a replica owner and its own
    // worker share one identity).
    bool local = false;
};

// One model the client is willing to run, with the shape the planner needs.
struct ModelOption {
    Manifest manifest;                 // model identity: sha256 (plus hidden size)
    ModelIndex model;                  // tensor sizes for the planner
};

struct PlacementTimings {
    double greeting_ms = 0;  // connect to every candidate and read its capabilities
    double plan_ms = 0;
    double reserve_ms = 0;   // all reservation rounds, including refusals
    double load_ms = 0;      // assign_stage until every stage_ready (downloads included)
    int attempts = 0;
    int refusals = 0;        // reservations another client won first
    int rejected_links = 0;  // candidates check_route left out
};
struct PlacementRequest {
    // Best first: the first model the discovered workers can actually run is used, so a
    // client asks for the largest model and falls back to smaller ones automatically.
    std::vector<ModelOption> models;
    std::uint32_t context = 0;         // 0 = each model's own manifest context
    std::uint32_t sessions = 1;
    std::size_t minimum_stages = 1;
    std::string runtime_abi;           // every stage must report exactly this
    std::uint32_t lease_ms = 30000;
    int attempts = 3;
    // Speculative decoding: ask the first stage to also load the smallest offered model as a
    // draft, so one pass through the route can commit several tokens.
    bool speculate = false;
    // Greeting, reservation and route setup replies; relayed WAN peers can be slow.
    // Stage loading (downloads) has no timeout.
    std::uint32_t connect_timeout_ms = 45000;
    // Replica formation: the candidate (by control endpoint) that must run the first stage.
    std::string head;
    // Replica formation: called with the planned stages, first stage first, before anything
    // is reserved. Returns the control endpoint of a candidate to leave out (a link on this
    // route is too slow), or empty to go ahead.
    std::function<std::string(const std::vector<PlacementCandidate>&)> check_route;
    // Filled with this attempt's timings and counts, whether or not placement succeeds.
    PlacementTimings* report = nullptr;
};

struct PlacedStage {
    std::string worker_id;
    std::string peer_id;
    int begin = 0;
    int end = 0;
    std::uint32_t rtt_ms = 0;
    bool relayed = false;
};


struct PlacedRoute {
    std::string route_id;
    Manifest manifest;                 // the model that was placed
    std::string draft_model_id;        // speculative decoding, empty when off
    std::string draft_sha256;
    InferenceRoute route;              // stage and ring fields set; return_* left to the caller
    std::vector<std::unique_ptr<Connection>> connections;  // leased, first stage first
    std::vector<PlacedStage> stages;
    PlacementTimings timings;
};

std::string random_route_id();

// What the local sidecar's candidate API (DAN-CANDIDATES/1) found for a model.
struct Discovery {
    std::string self_peer;                     // this client's PeerID
    std::string return_listen;                 // where the sidecar delivers ring returns
    // Prefix for this node's return target ("/dan-return"), when the sidecar delivers
    // returns on their own protocol because -ring-inbound belongs to a worker.
    std::string return_prefix;
    std::vector<PlacementCandidate> candidates;  // PeerIDs with local control forwards
};

// Asks the sidecar at `api_endpoint` (loopback) for workers that may serve any of these
// models. This client never talks to the DHT itself. Throws on errors.
Discovery discover_candidates(const std::string& api_endpoint,
    const std::vector<std::string>& model_sha256);

// How one peer reaches another, as `from`'s sidecar measured it just now (a libp2p ping on
// the connection it would use for a ring link). `from` may be this node itself.
struct LinkProbe {
    std::uint32_t rtt_ms = 0;
    bool relayed = false;
};
LinkProbe probe_link(const std::string& api_endpoint, const std::string& from,
    const std::string& to);

// A READY persistent replica, as its owner described it just now.
struct ReplicaCandidate {
    std::string owner;         // PeerID; clients talk only to the owner
    std::string control;       // local forward to the owner's front door
    std::uint32_t rtt_ms = 0;  // this client -> owner
    bool relayed = false;
    std::string replica_id;
    std::string model_sha256;
    std::uint32_t sessions_free = 0;
    std::uint32_t sessions_max = 0;
    std::uint32_t context = 0;
    std::uint32_t stages = 0;
    std::string draft_sha256;  // empty: no speculation
};

// Asks the local sidecar (DAN-REPLICAS/1) for READY replicas of any of these models. Every
// entry was confirmed live by its owner; DHT records alone are never trusted.
std::vector<ReplicaCandidate> discover_replicas(const std::string& api_endpoint,
    const std::vector<std::string>& model_sha256);

// Throws when no placement can be reserved and loaded.
PlacedRoute place_route(const std::vector<PlacementCandidate>& candidates,
    const PlacementRequest& request);

} // namespace dan::provider_owned
