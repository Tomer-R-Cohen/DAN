package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
