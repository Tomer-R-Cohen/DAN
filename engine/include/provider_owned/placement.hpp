#pragma once

// Client-side dynamic placement. Given candidate workers (already discovered or listed),
// the client reads their capabilities, plans stages with the shared planner, reserves the
// chosen workers (lease.hpp), assigns their ranges and returns leased connections ready for
// InferenceClient. There is no coordinator: this client is the planner for its own route.

#include "provider_owned/client.hpp"
#include "provider_owned/manifest.hpp"
#include "provider_owned/planner.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dan::provider_owned {

struct PlacementCandidate {
    std::string control;   // host:port of the worker's control connection (e.g. a sidecar forward)
    std::string peer_id;   // libp2p: PeerID that connection is authenticated to; empty = direct TCP
};

// One model the client is willing to run, with the shape the planner needs.
struct ModelOption {
    Manifest manifest;                 // model identity: sha256 (plus hidden size)
    ModelIndex model;                  // tensor sizes for the planner
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
    // Greeting, reservation and route setup replies; relayed WAN peers can be slow.
    // Stage loading (downloads) has no timeout.
    std::uint32_t connect_timeout_ms = 45000;
};

struct PlacedStage {
    std::string worker_id;
    std::string peer_id;
    int begin = 0;
    int end = 0;
};

struct PlacementTimings {
    double greeting_ms = 0;  // connect to every candidate and read its capabilities
    double plan_ms = 0;
    double reserve_ms = 0;   // all reservation rounds, including refusals
    double load_ms = 0;      // assign_stage until every stage_ready (downloads included)
    int attempts = 0;
};

struct PlacedRoute {
    std::string route_id;
    Manifest manifest;                 // the model that was placed
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
    std::vector<PlacementCandidate> candidates;  // PeerIDs with local control forwards
};

// Asks the sidecar at `api_endpoint` (loopback) for workers that may serve any of these
// models. This client never talks to the DHT itself. Throws on errors.
Discovery discover_candidates(const std::string& api_endpoint,
    const std::vector<std::string>& model_sha256);

// Throws when no placement can be reserved and loaded.
PlacedRoute place_route(const std::vector<PlacementCandidate>& candidates,
    const PlacementRequest& request);

} // namespace dan::provider_owned
