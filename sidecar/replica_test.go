package main

import (
	"crypto/rand"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/peer"
)

func TestSplitReturnTarget(t *testing.T) {
	const id = "/p2p/12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C"
	if value, ok := splitReturnTarget("/dan-return" + id); !ok || value != id {
		t.Fatalf("return target not recognized: %q %v", value, ok)
	}
	if value, ok := splitReturnTarget(id); ok || value != id {
		t.Fatalf("ring target taken for a return target: %q %v", value, ok)
	}
}

// Only a fresh READY status naming this node as owner is advertised or described.
func TestReadyReplicaNeedsFreshReadyOwnStatus(t *testing.T) {
	key, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	self, err := peer.IDFromPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "replica-status.json")
	write := func(state, owner string, age time.Duration) {
		status := map[string]any{"protocol_version": 1, "state": state, "owner": owner,
			"replica_id": "0123456789abcdef0123456789abcdef", "model_sha256": sha64("e"),
			"sessions_max": 1, "updated_unix_ms": time.Now().Add(-age).UnixMilli()}
		data, _ := json.Marshal(status)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("ready", self.String(), 0)
	if readyReplica(path, self) == nil {
		t.Fatal("fresh ready status rejected")
	}
	write("forming", self.String(), 0)
	if readyReplica(path, self) != nil {
		t.Fatal("forming replica accepted")
	}
	write("ready", "12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C", 0)
	if readyReplica(path, self) != nil {
		t.Fatal("another owner's status accepted")
	}
	write("ready", self.String(), time.Minute)
	if readyReplica(path, self) != nil {
		t.Fatal("stale status accepted")
	}
}

func sha64(digit string) string {
	result := ""
	for len(result) < 64 {
		result += digit
	}
	return result
}
