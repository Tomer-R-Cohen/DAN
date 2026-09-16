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

struct PlacementRequest {
    Manifest manifest;                 // model identity: sha256 (plus hidden size)
    ModelIndex model;                  // tensor sizes for the planner
    std::uint32_t context = 0;
    std::uint32_t sessions = 1;
    std::size_t minimum_stages = 1;
    std::string runtime_abi;           // every stage must report exactly this
    std::uint32_t lease_ms = 30000;
    int attempts = 3;
};

struct PlacedStage {
    std::string worker_id;
    std::string peer_id;
    int begin = 0;
    int end = 0;
};

struct PlacedRoute {
    std::string route_id;
    InferenceRoute route;              // stage and ring fields set; return_* left to the caller
    std::vector<std::unique_ptr<Connection>> connections;  // leased, first stage first
    std::vector<PlacedStage> stages;
};

std::string random_route_id();

// Throws when no placement can be reserved and loaded.
PlacedRoute place_route(const std::vector<PlacementCandidate>& candidates,
    const PlacementRequest& request);

} // namespace dan::provider_owned
