#include "provider_owned/route.hpp"

#include <cstdio>
#include <string>

namespace po = dan::provider_owned;

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "failed line %d\n", __LINE__); return __LINE__; } } while (false)

int main() {
    const std::string peer = "12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C";
    const std::string target = "/ip4/127.0.0.1/tcp/7132/p2p/" + peer;

    po::RingRoute route;
    CHECK(po::parse_route(po::route_message({target, peer}), route));
    CHECK(route.next == target && route.previous_peer == peer);
    CHECK(po::parse_route("next=127.0.0.1:7112", route));
    CHECK(route.next == "127.0.0.1:7112" && route.previous_peer.empty());

    CHECK(!po::parse_route("", route));
    CHECK(!po::parse_route("previous_peer=" + peer, route));
    CHECK(!po::parse_route("next=a\nnext=b", route));
    CHECK(!po::parse_route("next=127.0.0.1:1\nprevious_peer=not-a-peer", route));
    CHECK(!po::parse_route("next=127.0.0.1:1\nshell=true", route));

    // libp2p targets: multiaddrs that name a PeerID; never plain host:port.
    CHECK(po::valid_p2p_target(target));
    CHECK(po::valid_p2p_target(target + ",/ip4/10.0.0.2/tcp/1/p2p/" + peer));
    CHECK(!po::valid_p2p_target("127.0.0.1:7112"));
    CHECK(!po::valid_p2p_target("/ip4/127.0.0.1/tcp/7132"));
    CHECK(!po::valid_p2p_target("/ip4/127.0.0.1/tcp/7132/p2p/bad!"));

    // Direct targets: numeric private or loopback IPv4 only.
    CHECK(po::valid_private_endpoint("127.0.0.1:7112"));
    CHECK(po::valid_private_endpoint("10.1.2.3:1"));
    CHECK(po::valid_private_endpoint("172.20.0.1:1"));
    CHECK(po::valid_private_endpoint("192.168.1.5:1"));
    CHECK(po::valid_private_endpoint("100.100.1.1:1"));
    CHECK(!po::valid_private_endpoint("8.8.8.8:53"));
    CHECK(!po::valid_private_endpoint("172.32.0.1:1"));
    CHECK(!po::valid_private_endpoint("example.com:80"));
    CHECK(!po::valid_private_endpoint("localhost:80"));
    CHECK(!po::valid_private_endpoint("127.0.0.1"));
    CHECK(!po::valid_private_endpoint(target));
    return 0;
}
