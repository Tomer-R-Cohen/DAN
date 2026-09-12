package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	"github.com/multiformats/go-multiaddr"
)

func TestStableIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "identity.key")
	a, err := loadOrCreateKey(path)
	if err != nil {
		t.Fatal(err)
	}
	b, err := loadOrCreateKey(path)
	if err != nil {
		t.Fatal(err)
	}
	aID, _ := peer.IDFromPrivateKey(a)
	bID, _ := peer.IDFromPrivateKey(b)
	if aID != bID {
		t.Fatalf("PeerID changed: %s != %s", aID, bID)
	}
}

func TestOpaqueTunnelAllowlistAndReconnect(t *testing.T) {
	server := makeTestHost(t)
	allowed := makeTestHost(t)
	rejected := makeTestHost(t)
	engine, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	runInbound(server, controlProtocol, engine.Addr().String(), map[peer.ID]bool{allowed.ID(): true}, false)

	bad := connectTestHost(t, rejected, server)
	_, _ = bad.Write([]byte("blocked"))
	tcpEngine := engine.(*net.TCPListener)
	_ = tcpEngine.SetDeadline(time.Now().Add(300 * time.Millisecond))
	if conn, err := engine.Accept(); err == nil {
		_ = conn.Close()
		t.Fatal("rejected peer reached DAN")
	}
	_ = tcpEngine.SetDeadline(time.Time{})
	_ = bad.Close()

	check := func(payload []byte) {
		stream := connectTestHost(t, allowed, server)
		defer stream.Close()
		go func() { _, _ = stream.Write(payload) }()
		accepted, err := engine.Accept()
		if err != nil {
			t.Fatal(err)
		}
		defer accepted.Close()
		reader := bufio.NewReader(accepted)
		header, err := reader.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if header != peerHeader+allowed.ID().String()+"\n" {
			t.Fatalf("bad peer header %q", header)
		}
		got := make([]byte, len(payload))
		if _, err := io.ReadFull(reader, got); err != nil {
			t.Fatal(err)
		}
		if string(got) != string(payload) {
			t.Fatal("tunnel changed DAN bytes")
		}
	}

	check([]byte{0x44, 0x41, 0x4e, 0x31, 0, 2, 0, 6, 0xff, 0x00, 0x7f})
	check([]byte("second connection"))
}

func TestPublicPoolAcceptsAuthenticatedPeer(t *testing.T) {
	server := makeTestHost(t)
	provider := makeTestHost(t)
	engine, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	runInbound(server, controlProtocol, engine.Addr().String(), nil, true)
	stream := connectTestHost(t, provider, server)
	defer stream.Close()
	if _, err := stream.Write([]byte("D")); err != nil {
		t.Fatal(err)
	}
	accepted, err := engine.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer accepted.Close()
	header, err := bufio.NewReader(accepted).ReadString('\n')
	if err != nil || header != peerHeader+provider.ID().String()+"\n" {
		t.Fatalf("bad public-pool identity %q: %v", header, err)
	}
}

func TestRingProxyCarriesAuthenticatedPeer(t *testing.T) {
	successor := makeTestHost(t)
	predecessor := makeTestHost(t)
	engine, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	runInbound(successor, ringProtocol, engine.Addr().String(), nil, true)
	proxy, err := startRingProxy(predecessor, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer proxy.Close()
	conn, err := net.Dial("tcp", proxy.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := fmt.Fprintf(conn, "%s%s\n", ringTargetHeader,
		strings.Join(peerAddrs(successor), ",")); err != nil {
		t.Fatal(err)
	}
	if reply, err := bufio.NewReader(conn).ReadString('\n'); err != nil || reply != "OK\n" {
		t.Fatalf("proxy reply = %q, err = %v", reply, err)
	}
	payload := []byte("encrypted activation")
	if _, err := conn.Write(payload); err != nil {
		t.Fatal(err)
	}
	accepted, err := engine.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer accepted.Close()
	reader := bufio.NewReader(accepted)
	if header, err := reader.ReadString('\n'); err != nil || header != peerHeader+predecessor.ID().String()+"\n" {
		t.Fatalf("ring peer header = %q, err = %v", header, err)
	}
	got := make([]byte, len(payload))
	if _, err := io.ReadFull(reader, got); err != nil || string(got) != string(payload) {
		t.Fatalf("ring payload = %q, err = %v", got, err)
	}
}

func TestAllowlistedRelayCarriesEncryptedStream(t *testing.T) {
	relayHost := makeTestHost(t)
	destination := makeTestHost(t)
	service, err := startRelay(relayHost, map[peer.ID]bool{destination.ID(): true}, false, 1, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	relayInfo := peer.AddrInfo{ID: relayHost.ID(), Addrs: relayHost.Addrs()}
	if err := destination.Connect(ctx, relayInfo); err != nil {
		t.Fatal(err)
	}
	if _, err := relayclient.Reserve(ctx, destination, relayInfo); err != nil {
		t.Fatal(err)
	}
	rejected := makeTestHost(t)
	if err := rejected.Connect(ctx, relayInfo); err != nil {
		t.Fatal(err)
	}
	if _, err := relayclient.Reserve(ctx, rejected, relayInfo); err == nil {
		t.Fatal("relay accepted an unlisted reservation")
	}

	destination.SetStreamHandler(controlProtocol, func(stream network.Stream) {
		defer stream.Close()
		_, _ = io.Copy(stream, stream)
	})
	client := makeTestHost(t)
	circuit, err := multiaddr.NewMultiaddr(fmt.Sprintf("%s/p2p/%s/p2p-circuit/p2p/%s",
		relayHost.Addrs()[0], relayHost.ID(), destination.ID()))
	if err != nil {
		t.Fatal(err)
	}
	targetInfo, err := peer.AddrInfoFromP2pAddr(circuit)
	if err != nil {
		t.Fatal(err)
	}
	limited := network.WithAllowLimitedConn(ctx, "relay test")
	if err := client.Connect(limited, *targetInfo); err != nil {
		t.Fatal(err)
	}
	stream, err := client.NewStream(limited, destination.ID(), controlProtocol)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if !strings.Contains(stream.Conn().RemoteMultiaddr().String(), "p2p-circuit") {
		t.Fatalf("stream did not use relay: %s", stream.Conn().RemoteMultiaddr())
	}
	remote, err := peer.IDFromPublicKey(stream.Conn().RemotePublicKey())
	if err != nil || remote != destination.ID() {
		t.Fatalf("relayed stream did not authenticate destination: %s, %v", remote, err)
	}
	if _, err := stream.Write([]byte("dan relay")); err != nil {
		t.Fatal(err)
	}
	reply := make([]byte, len("dan relay"))
	if _, err := io.ReadFull(stream, reply); err != nil || string(reply) != "dan relay" {
		t.Fatalf("relay reply = %q, err = %v", reply, err)
	}
}

func makeTestHost(t *testing.T) host.Host {
	t.Helper()
	h, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Close() })
	return h
}

func connectTestHost(t *testing.T, from, to host.Host) network.Stream {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := from.Connect(ctx, peer.AddrInfo{ID: to.ID(), Addrs: to.Addrs()}); err != nil {
		t.Fatal(err)
	}
	stream, err := from.NewStream(ctx, to.ID(), controlProtocol)
	if err != nil {
		t.Fatal(err)
	}
	return stream
}
