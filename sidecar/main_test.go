package main

import (
	"bufio"
	"context"
	"io"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	lp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
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
	runInbound(server, engine.Addr().String(), map[peer.ID]bool{allowed.ID(): true}, false)

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
	runInbound(server, engine.Addr().String(), nil, true)
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
	stream, err := from.NewStream(ctx, to.ID(), lp2pprotocol.ID(protocol))
	if err != nil {
		t.Fatal(err)
	}
	return stream
}
