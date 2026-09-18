#pragma once

// Persistent replicas. A provider node's owner process forms a replica around its own worker
// with the ordinary placement code (plan, reserve, assign, link), keeps the leased connections
// open after clients leave, advertises the replica through its sidecar, and serves client
// sessions on it. The owner is always the first stage and has authority over this replica
// only; nothing here coordinates the rest of the network.
//
// Client <-> owner ("front door", DAN frames over /dan/session/1.0.0):
//   create_session / reset_session / destroy_session  -> ack or error (e.g. replica_full)
//   stream_prompt (rows = token budget)                -> ack
//   prompt (position = the session's position)         -> client_chunk ..., result, then
//                                                         ack (position = new position)
//   cancel_request                                     -> no reply; the result still comes
// The owner runs the ring and relays the answer; the ring itself (head -> ... -> tail -> head)
// never passes through the client or waits for it.

#include "provider_owned/placement.hpp"

#include <cstdint>
#include <string>

namespace dan::provider_owned {

struct ReplicaOwnerOptions {
    PlacementRequest request;        // models (largest first), context, sessions, speculation
    std::string discover;            // this node's sidecar candidate API (loopback)
    std::string self_control;        // this node's worker control listener (loopback)
    std::string session_listen;      // front door; the sidecar forwards client sessions here
    std::string status_file;         // replica status the sidecar advertises and answers from
    std::uint32_t max_edge_rtt_ms = 150;   // every link of the ring, measured before reserving
    bool allow_relay_edges = true;
    int warmup_tokens = 2;
    std::uint32_t keepalive_ms = 15000;    // how often idle members are checked
    // Autonomous mode: before each attempt, wait longer the more free peers outrank this node
    // (more memory, then lower PeerID), so fewer proposals collide. Correctness never depends
    // on it: worker leases decide every race.
    bool rank_delay = false;
};

// Forms a replica, serves it until it dissolves, and forms again. Returns (nonzero) only when
// this node's own worker is gone.
int run_replica_owner(const ReplicaOwnerOptions& options);

} // namespace dan::provider_owned
