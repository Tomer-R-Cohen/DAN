package main

import (
	"bufio"
	"context"
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	ma "github.com/multiformats/go-multiaddr"
)

// startHomeNode is a DHT client that is reachable only through the relay.
func startHomeNode(t *testing.T, ctx context.Context, relay peer.AddrInfo) (host.Host, *dialer, *dht.IpfsDHT) {
	t.Helper()
	h, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"),
		libp2p.ForceReachabilityPrivate(),
		libp2p.EnableAutoRelayWithStaticRelays([]peer.AddrInfo{relay}),
		libp2p.EnableHolePunching(),
		libp2p.AddrsFactory(onlyRelayAddrs),
		libp2p.ConnectionGater(relayOnlyGater{relays: map[peer.ID]bool{relay.ID: true}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Close() })
	if err := h.Connect(ctx, relay); err != nil {
		t.Fatal(err)
	}
	// AutoRelay only builds relay addresses from a relay's public addresses, and a loopback
	// relay has none. Give it one; it is never dialed because we are already connected.
	h.Peerstore().AddAddr(relay.ID, ma.StringCast("/ip4/1.1.1.1/tcp/9"), peerstore.PermanentAddrTTL)
	if !waitForRelayAddress(h, 30*time.Second) {
		t.Fatal("home node got no relay reservation")
	}
	kad, err := startDHT(ctx, h, "client", []peer.AddrInfo{relay}, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = kad.Close() })
	return h, &dialer{host: h, resolve: dhtResolver(kad), dialTimeout: 10 * time.Second,
		directWait: 500 * time.Millisecond}, kad
}

// echoEngine answers every local connection with the DAN-P2P header it received.
func echoEngine(t *testing.T) string {
	t.Helper()
	engine, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = engine.Close() })
	go func() {
		for {
			conn, err := engine.Accept()
			if err != nil {
				return
			}
			line, _ := bufio.NewReader(conn).ReadString('\n')
			_, _ = conn.Write([]byte(line))
			_ = conn.Close()
		}
	}()
	return engine.Addr().String()
}

func readStreamLine(t *testing.T, stream network.Stream) string {
	t.Helper()
	_ = stream.SetDeadline(time.Now().Add(10 * time.Second))
	line, err := bufio.NewReader(stream).ReadString('\n')
	if err != nil {
		t.Fatalf("stream read failed: %v", err)
	}
	return strings.TrimSpace(line)
}

func TestNATPeersConnectThroughRelayByPeerID(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Public infrastructure: relay + DHT server, as -infra configures it.
	infra := makeTestHost(t)
	relay, err := startRelay(infra, nil, true, 64, 5*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer relay.Close()
	infraDHT, err := startDHT(ctx, infra, "server", nil, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer infraDHT.Close()
	infraInfo := peer.AddrInfo{ID: infra.ID(), Addrs: infra.Addrs()}

	// Worker B behind "NAT": advertises a model and serves ring and control streams.
	workerB, dialerB, kadB := startHomeNode(t, ctx, infraInfo)
	status := filepath.Join(t.TempDir(), "status.json")
	writeTestStatus(t, status, "available", time.Now())
	serveCapabilities(workerB, status)
	runInbound(workerB, ringProtocol, echoEngine(t), nil, true)
	runInbound(workerB, controlProtocol, echoEngine(t), nil, true)
	go advertiseModels(ctx, kadB, status, 30*time.Second)

	// Worker A behind "NAT", holding a stale address for B: it must look B up by PeerID
	// and reach it through the relay.
	workerA, dialerA, _ := startHomeNode(t, ctx, infraInfo)
	workerA.Peerstore().AddAddr(workerB.ID(), ma.StringCast("/ip4/127.0.0.1/tcp/9"), time.Minute)
	stream, err := dialerA.open(target{id: workerB.ID()}, ringProtocol)
	if err != nil {
		t.Fatalf("A could not reach B by PeerID: %v", err)
	}
	if !isRelayAddr(stream.Conn().RemoteMultiaddr()) || !stream.Conn().Stat().Limited {
		t.Fatalf("expected a relayed connection, got %s", describeConn(stream.Conn()))
	}
	if got := readStreamLine(t, stream); got != strings.TrimSpace(peerHeader+workerA.ID().String()) {
		t.Fatalf("B saw the wrong predecessor: %q", got)
	}
	_ = stream.Close()

	// Client C behind "NAT": discovers B through the DHT and reaches its control port.
	client, dialerC, kadC := startHomeNode(t, ctx, infraInfo)
	runInbound(client, ringProtocol, echoEngine(t), nil, true)
	forwards := &forwardSet{dialer: dialerC, protocol: controlProtocol, listeners: map[peer.ID]net.Listener{}}
	config := discoveryConfig{queryTimeout: 10 * time.Second, discoveryTimeout: 15 * time.Second, addrTTL: time.Minute}
	found := waitForCandidates(t, 1, func() ([]candidate, error) {
		return findCandidates(ctx, dialerC, kadC, forwards, testModel, config)
	})
	if found[0].id != workerB.ID() {
		t.Fatalf("discovered %s, want %s", found[0].id, workerB.ID())
	}
	conn, err := net.DialTimeout("tcp", found[0].control, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.SetDeadline(time.Now().Add(15 * time.Second))
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil || strings.TrimSpace(line) != strings.TrimSpace(peerHeader+client.ID().String()) {
		t.Fatalf("control forward through the relay failed: %q %v", line, err)
	}

	// Return path: B reaches the client by PeerID over the relayed connection the client
	// opened, even though the client accepts no direct connections.
	returned, err := dialerB.open(target{id: client.ID()}, ringProtocol)
	if err != nil {
		t.Fatalf("B could not reach the client: %v", err)
	}
	if !isRelayAddr(returned.Conn().RemoteMultiaddr()) {
		t.Fatalf("expected the relayed return path, got %s", describeConn(returned.Conn()))
	}
	if got := readStreamLine(t, returned); got != strings.TrimSpace(peerHeader+workerB.ID().String()) {
		t.Fatalf("client saw the wrong peer: %q", got)
	}
	_ = returned.Close()
	_ = conn.Close()
}

func TestListenAddrsAddIPv6(t *testing.T) {
	got := strings.Join(listenAddrs("/ip4/0.0.0.0/tcp/4001", true), " ")
	want := "/ip4/0.0.0.0/tcp/4001 /ip6/::/tcp/4001 /ip4/0.0.0.0/udp/4001/quic-v1 /ip6/::/udp/4001/quic-v1"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if got := strings.Join(listenAddrs("/ip4/0.0.0.0/tcp/0", false), " "); got != "/ip4/0.0.0.0/tcp/0 /ip4/0.0.0.0/udp/0/quic-v1" {
		t.Fatalf("IPv6 not disabled: %q", got)
	}
	if got := strings.Join(listenAddrs("/ip4/127.0.0.1/tcp/5", true), " "); got != "/ip4/127.0.0.1/tcp/5 /ip4/127.0.0.1/udp/5/quic-v1" {
		t.Fatalf("a specific IPv4 address must not add IPv6: %q", got)
	}
}

func TestIPv6Listening(t *testing.T) {
	key, err := loadOrCreateKey(filepath.Join(t.TempDir(), "key"))
	if err != nil {
		t.Fatal(err)
	}
	h, err := newHost(hostOptions{key: key, listen: "/ip4/0.0.0.0/tcp/0"})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	for _, addr := range h.Network().ListenAddresses() {
		if _, err := addr.ValueForProtocol(ma.P_IP6); err != nil {
			continue
		}
		// Connect over IPv6 loopback on the same port, TCP or QUIC.
		local := ma.StringCast(strings.Replace(addr.String(), "/ip6/::/", "/ip6/::1/", 1))
		other := makeTestHost(t)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := other.Connect(ctx, peer.AddrInfo{ID: h.ID(), Addrs: []ma.Multiaddr{local}}); err != nil {
			t.Fatalf("IPv6 connection to %s failed: %v", local, err)
		}
		return
	}
	t.Skip("this machine has no IPv6 listener (IPv6 unavailable)")
}

// A bootstrap address can be a DNS name (e.g. dynamic DNS for a home node whose IPv6
// prefix changes); libp2p resolves it when connecting.
func TestBootstrapByDNSName(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	infra := makeTestHost(t)
	infraDHT, err := startDHT(ctx, infra, "server", nil, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer infraDHT.Close()
	port := ""
	for _, addr := range infra.Addrs() {
		if value, err := addr.ValueForProtocol(ma.P_TCP); err == nil {
			port = value
			break
		}
	}
	bootstrap, err := parsePeers([]string{"/dns4/localhost/tcp/" + port + "/p2p/" + infra.ID().String()})
	if err != nil {
		t.Fatal(err)
	}
	home := makeTestHost(t)
	homeDHT, err := startDHT(ctx, home, "client", bootstrap, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer homeDHT.Close()
	if homeDHT.RoutingTable().Size() == 0 {
		t.Fatal("bootstrap by DNS name did not join the DHT")
	}
}

func TestInfraAndHomeHostOptions(t *testing.T) {
	key, err := loadOrCreateKey(filepath.Join(t.TempDir(), "key"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := newHost(hostOptions{key: key, listen: "/ip4/127.0.0.1/tcp/0", reachability: "sideways"}); err == nil {
		t.Fatal("invalid reachability accepted")
	}
	announce := ma.StringCast("/ip4/203.0.113.7/tcp/4001")
	h, err := newHost(hostOptions{key: key, listen: "/ip4/127.0.0.1/tcp/0", reachability: "public",
		natService: true, announce: []ma.Multiaddr{announce}})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	found := false
	for _, addr := range h.Addrs() {
		found = found || addr.Equal(announce)
	}
	if !found {
		t.Fatalf("announced address missing from %v", h.Addrs())
	}
}
