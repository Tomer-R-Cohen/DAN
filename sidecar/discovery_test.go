package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"dan/sidecar/capabilities"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	drouting "github.com/libp2p/go-libp2p/p2p/discovery/routing"
)

const testModel = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"

func writeTestStatus(t *testing.T, path, state string, updated time.Time) {
	t.Helper()
	status := map[string]any{
		"protocol_version": 1, "worker_id": filepath.Base(path), "runtime_abi": "dan-stage-v1/f32le/test",
		"device": "CPU", "offered_memory_mib": 2048, "max_context": 512, "max_sessions": 1,
		"state": state, "updated_unix_ms": updated.UnixMilli(),
		"models": []any{map[string]any{"sha256": testModel, "layers": 24, "hidden": 896,
			"cached": []any{map[string]any{"begin": 0, "end": 10}}}},
	}
	data, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestCapabilityFromStatus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "worker")
	now := time.Now()
	writeTestStatus(t, path, "reserved", now)
	status, err := readStatus(path)
	if err != nil {
		t.Fatal(err)
	}
	capability := capabilityFromStatus(status, now, testModel)
	if capability.State != capabilities.State_STATE_RESERVED || capability.OfferedMemoryMib != 2048 ||
		len(capability.Models) != 1 || capability.Models[0].Cached[0].End != 10 {
		t.Fatalf("unexpected capability: %v", capability)
	}
	if other := capabilityFromStatus(status, now, strings.Repeat("0", 64)); len(other.Models) != 0 {
		t.Fatal("a model filter returned another model")
	}
	if stale := capabilityFromStatus(status, now.Add(time.Minute), ""); stale.State != capabilities.State_STATE_OFFLINE {
		t.Fatal("a stale status was not offline")
	}
	if missing := capabilityFromStatus(nil, now, ""); missing.State != capabilities.State_STATE_OFFLINE {
		t.Fatal("a missing status was not offline")
	}
	if _, err := modelKey("not-a-sha"); err == nil {
		t.Fatal("invalid model key accepted")
	}
}

// A status file left by an earlier run is ignored; models appear once the worker writes a
// fresh status.
func TestAdvertiseWaitsForFreshStatus(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bootstrap := makeTestHost(t)
	bootstrapDHT, err := startDHT(ctx, bootstrap, "server", nil, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer bootstrapDHT.Close()
	bootstrapInfo := peer.AddrInfo{ID: bootstrap.ID(), Addrs: bootstrap.Addrs()}
	worker := makeTestHost(t)
	workerDHT, err := startDHT(ctx, worker, "server", []peer.AddrInfo{bootstrapInfo}, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer workerDHT.Close()
	status := filepath.Join(t.TempDir(), "status.json")
	writeTestStatus(t, status, "available", time.Now().Add(-time.Hour))
	go advertiseModels(ctx, workerDHT, status, 30*time.Second)

	key, err := modelKey(testModel)
	if err != nil {
		t.Fatal(err)
	}
	finder := drouting.NewRoutingDiscovery(bootstrapDHT)
	provided := func() bool {
		lookup, done := context.WithTimeout(ctx, 2*time.Second)
		defer done()
		peers, err := finder.FindPeers(lookup, modelNamespace+key)
		if err != nil {
			return false
		}
		for info := range peers {
			if info.ID == worker.ID() {
				return true
			}
		}
		return false
	}
	time.Sleep(3 * time.Second)
	if provided() {
		t.Fatal("a stale status was advertised")
	}
	writeTestStatus(t, status, "available", time.Now())
	deadline := time.Now().Add(20 * time.Second)
	for !provided() {
		if time.Now().After(deadline) {
			t.Fatal("the fresh status was not advertised")
		}
		time.Sleep(500 * time.Millisecond)
	}
}

type testWorker struct {
	host   host.Host
	status string
	engine net.Listener
}

func startTestWorker(t *testing.T, ctx context.Context, bootstrap peer.AddrInfo) *testWorker {
	t.Helper()
	worker := &testWorker{host: makeTestHost(t), status: filepath.Join(t.TempDir(), "status.json")}
	writeTestStatus(t, worker.status, "available", time.Now())
	serveCapabilities(worker.host, worker.status)
	d, err := startDHT(ctx, worker.host, "server", []peer.AddrInfo{bootstrap}, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Close() })
	go advertiseModels(ctx, d, worker.status, 30*time.Second)
	// The worker's control port: says which worker answered.
	worker.engine, err = net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = worker.engine.Close() })
	go func() {
		for {
			conn, err := worker.engine.Accept()
			if err != nil {
				return
			}
			line, _ := bufio.NewReader(conn).ReadString('\n')
			_, _ = conn.Write([]byte(line + worker.host.ID().String() + "\n"))
			_ = conn.Close()
		}
	}()
	runInbound(worker.host, controlProtocol, worker.engine.Addr().String(), nil, true)
	return worker
}

func waitForCandidates(t *testing.T, want int, find func() ([]candidate, error)) []candidate {
	t.Helper()
	deadline := time.Now().Add(45 * time.Second)
	for {
		found, err := find()
		if err == nil && len(found) == want {
			return found
		}
		if time.Now().After(deadline) {
			t.Fatalf("wanted %d candidates, found %d (%v)", want, len(found), err)
		}
		time.Sleep(time.Second)
	}
}

func TestDiscoveryFindsAvailableWorkersWithoutBootstrap(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bootstrap := makeTestHost(t)
	bootstrapDHT, err := startDHT(ctx, bootstrap, "server", nil, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	bootstrapInfo := peer.AddrInfo{ID: bootstrap.ID(), Addrs: bootstrap.Addrs()}
	first := startTestWorker(t, ctx, bootstrapInfo)
	second := startTestWorker(t, ctx, bootstrapInfo)
	client := makeTestHost(t)
	clientDHT, err := startDHT(ctx, client, "server", []peer.AddrInfo{bootstrapInfo}, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer clientDHT.Close()
	clientDialer := &dialer{host: client, resolve: dhtResolver(clientDHT), dialTimeout: 10 * time.Second}
	forwards := &forwardSet{dialer: clientDialer, protocol: controlProtocol, listeners: map[peer.ID]net.Listener{}}
	config := discoveryConfig{queryTimeout: 5 * time.Second, discoveryTimeout: 15 * time.Second, addrTTL: time.Minute}
	find := func() ([]candidate, error) {
		return findCandidates(ctx, clientDialer, clientDHT, forwards, testModel, config)
	}

	found := waitForCandidates(t, 2, find)
	for _, c := range found {
		if c.capability.RuntimeAbi != "dan-stage-v1/f32le/test" {
			t.Fatalf("capability not carried: %v", c.capability)
		}
		// The local forward reaches exactly the discovered peer, found by PeerID alone.
		conn, err := net.DialTimeout("tcp", c.control, 5*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		_ = conn.SetDeadline(time.Now().Add(10 * time.Second))
		reader := bufio.NewReader(conn)
		header, _ := reader.ReadString('\n')
		answer, _ := reader.ReadString('\n')
		_ = conn.Close()
		if strings.TrimSpace(header) != strings.TrimSpace(peerHeader+client.ID().String()) {
			t.Fatalf("worker did not see the client's PeerID: %q", header)
		}
		if strings.TrimSpace(answer) != c.id.String() {
			t.Fatalf("forward for %s reached %q", c.id, answer)
		}
	}

	// The bootstrap node was only an entry point: discovery keeps working without it.
	_ = bootstrapDHT.Close()
	_ = bootstrap.Close()
	waitForCandidates(t, 2, find)

	// A busy worker, or one whose worker stopped updating its status, is not a candidate.
	writeTestStatus(t, second.status, "serving", time.Now())
	found = waitForCandidates(t, 1, find)
	if found[0].id != first.host.ID() {
		t.Fatalf("expected only %s, got %s", first.host.ID(), found[0].id)
	}
	writeTestStatus(t, first.status, "available", time.Now().Add(-time.Minute))
	waitForCandidates(t, 0, find)
}
