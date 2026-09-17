#pragma once

// Per-session ring route, carried in a create_session payload:
//   next=<target>            where this stage sends its hot-path output
//   previous_peer=<PeerID>   the only predecessor this stage accepts (libp2p mode only)
// Target forms:
//   libp2p mode: multiaddr list ending in /p2p/<PeerID>; dialed only through the local
//                sidecar ring proxy, so a worker never opens raw TCP to a remote-chosen host.
//   direct mode: numeric loopback/private IPv4 host:port (trusted LAN tests only).

#include "provider_owned/protocol.hpp"

#include <array>
#include <string>
#include <string_view>

namespace dan::provider_owned {

struct RingRoute {
    std::string next;
    std::string previous_peer;
    // Loop mode (last stage only): where the next token goes so decoding continues without
    // the client. "self" means this worker is the whole route and loops internally.
    std::string loop;
};

inline constexpr std::string_view loop_self = "self";

inline std::string route_message(const RingRoute& route) {
    std::string text = "next=" + route.next;
    if (!route.previous_peer.empty()) text += "\nprevious_peer=" + route.previous_peer;
    if (!route.loop.empty()) text += "\nloop=" + route.loop;
    return text;
}

inline bool parse_route(std::string_view text, RingRoute& route) {
    route = {};
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        const std::string_view line = text.substr(0, newline);
        const std::size_t equal = line.find('=');
        if (equal == std::string_view::npos) return false;
        const auto key = line.substr(0, equal), value = line.substr(equal + 1);
        if (key == "next" && route.next.empty()) route.next = value;
        else if (key == "previous_peer" && route.previous_peer.empty()) route.previous_peer = value;
        else if (key == "loop" && route.loop.empty()) route.loop = value;
        else return false;
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return !route.next.empty()
        && (route.previous_peer.empty() || valid_peer_id(route.previous_peer))
        && (route.loop.empty() || route.loop == loop_self || valid_ring_target(route.loop));
}

// A libp2p ring target: one or more multiaddrs, all naming the same PeerID.
inline bool valid_p2p_target(std::string_view value) {
    return !value.empty() && value.front() == '/' && valid_ring_target(value);
}

// A direct-mode target: numeric IPv4 in 127/8, 10/8, 172.16/12, 192.168/16 or 100.64/10.
inline bool valid_private_endpoint(std::string_view value) {
    if (value.empty() || value.front() == '/' || !valid_endpoint(value)) return false;
    const std::string host(value.substr(0, value.rfind(':')));
    std::array<unsigned char, 4> address{};
    if (inet_pton(AF_INET, host.c_str(), address.data()) != 1) return false;
    const unsigned a = address[0], b = address[1];
    return a == 127 || a == 10 || (a == 172 && b >= 16 && b <= 31)
        || (a == 192 && b == 168) || (a == 100 && b >= 64 && b <= 127);
}

} // namespace dan::provider_owned
