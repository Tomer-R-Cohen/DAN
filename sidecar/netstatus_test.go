package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

func TestNetStatusFile(t *testing.T) {
	h := makeTestHost(t)
	activeStreams.Store("test", streamStatus{Peer: "peer-x", Protocol: string(ringProtocol),
		Path: "relay", Transport: "quic-v1", Relay: "relay-y", Started: time.Now().Add(-3 * time.Second)})
	defer activeStreams.Delete("test")
	path := filepath.Join(t.TempDir(), "net.json")
	ctx, cancel := context.WithCancel(context.Background())
	go writeNetStatus(ctx, h, path, 50*time.Millisecond)
	defer cancel()
	deadline := time.Now().Add(5 * time.Second)
	var status netStatus
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil && json.Unmarshal(data, &status) == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if status.PeerID != h.ID().String() || len(status.Streams) != 1 {
		t.Fatalf("unexpected status %+v", status)
	}
	stream := status.Streams[0]
	if stream.Path != "relay" || stream.Relay != "relay-y" || stream.Seconds < 3 ||
		!strings.HasSuffix(stream.Protocol, "/ring/1.0.0") {
		t.Fatalf("unexpected stream %+v", stream)
	}
}

// A peer this node has a DAN stream with is pinged, and its round trip shows up as a link
// (selection plan M3); the worker passes these links to planners.
func TestNetStatusLinks(t *testing.T) {
	h, other := makeTestHost(t), makeTestHost(t)
	if err := h.Connect(context.Background(), peer.AddrInfo{ID: other.ID(), Addrs: other.Addrs()}); err != nil {
		t.Fatal(err)
	}
	activeStreams.Store("links-test", streamStatus{Peer: other.ID().String(), Protocol: string(ringProtocol),
		Path: "direct", Transport: "tcp", Started: time.Now()})
	defer activeStreams.Delete("links-test")
	path := filepath.Join(t.TempDir(), "net.json")
	ctx, cancel := context.WithCancel(context.Background())
	go writeNetStatus(ctx, h, path, 50*time.Millisecond)
	defer cancel()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		var status netStatus
		data, err := os.ReadFile(path)
		if err == nil && json.Unmarshal(data, &status) == nil {
			for _, link := range status.Links {
				if link.Peer == other.ID().String() && link.Path == "direct" && link.RTTMs >= 0 {
					if !strings.Contains(string(data), `"links":`) ||
						strings.Index(string(data), `"links":`) > strings.Index(string(data), `"streams":`) {
						t.Fatal("links must come before streams")
					}
					return
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	data, _ := os.ReadFile(path)
	t.Fatalf("no measured link to the stream peer: %s", data)
}
