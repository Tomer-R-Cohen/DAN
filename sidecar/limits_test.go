package main

import (
	"strings"
	"testing"

	"dan/sidecar/capabilities"
)

func goodCapability() *capabilities.Capability {
	return &capabilities.Capability{ProtocolVersion: 1, WorkerId: "w", Device: "gpu", RuntimeAbi: "abi",
		OfferedMemoryMib: 8192, MaxContext: 32768, MaxSessions: 4,
		Models: []*capabilities.ModelAvailability{{Sha256: strings.Repeat("ab", 32), Layers: 28,
			Cached: []*capabilities.Range{{Begin: 0, End: 14}}}}}
}

// Hostile or broken capability answers are dropped whole (selection plan M7).
func TestValidCapability(t *testing.T) {
	if err := validCapability(goodCapability()); err != nil {
		t.Fatal(err)
	}
	bad := map[string]func(*capabilities.Capability){
		"version":       func(c *capabilities.Capability) { c.ProtocolVersion = 2 },
		"long name":     func(c *capabilities.Capability) { c.WorkerId = strings.Repeat("x", 200) },
		"memory":        func(c *capabilities.Capability) { c.OfferedMemoryMib = 1 << 40 },
		"context":       func(c *capabilities.Capability) { c.MaxContext = 1 << 30 },
		"sessions":      func(c *capabilities.Capability) { c.MaxSessions = 100000 },
		"model sha":     func(c *capabilities.Capability) { c.Models[0].Sha256 = "nothex" },
		"layers":        func(c *capabilities.Capability) { c.Models[0].Layers = 100000 },
		"empty range":   func(c *capabilities.Capability) { c.Models[0].Cached[0].End = 0 },
		"range > model": func(c *capabilities.Capability) { c.Models[0].Cached[0].End = 29 },
		"nil model":     func(c *capabilities.Capability) { c.Models = append(c.Models, nil) },
		"many models": func(c *capabilities.Capability) {
			for len(c.Models) <= maxPeerModels {
				c.Models = append(c.Models, c.Models[0])
			}
		},
		"protocols": func(c *capabilities.Capability) {
			c.DataProtocols = make([]string, maxDataProtocols+1)
		},
	}
	for name, change := range bad {
		c := goodCapability()
		change(c)
		if validCapability(c) == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestValidReplicaStatus(t *testing.T) {
	good := func() *replicaStatus {
		s := &replicaStatus{ReplicaID: strings.Repeat("cd", 16), ModelSHA256: strings.Repeat("ab", 32),
			Context: 16384, SessionsMax: 2, SessionsInUse: 1, MsPerToken: 40, FirstTokenMs: 300}
		s.Members = append(s.Members, struct {
			Peer  string `json:"peer"`
			Begin uint32 `json:"begin"`
			End   uint32 `json:"end"`
		}{"p", 0, 28})
		return s
	}
	if err := validReplicaStatus(good()); err != nil {
		t.Fatal(err)
	}
	bad := map[string]func(*replicaStatus){
		"replica id":   func(s *replicaStatus) { s.ReplicaID = "short" },
		"draft":        func(s *replicaStatus) { s.DraftSHA256 = "zz" },
		"sessions":     func(s *replicaStatus) { s.SessionsInUse = 3 },
		"no sessions":  func(s *replicaStatus) { s.SessionsMax = 0 },
		"context":      func(s *replicaStatus) { s.Context = 1 << 30 },
		"no members":   func(s *replicaStatus) { s.Members = nil },
		"latency":      func(s *replicaStatus) { s.MsPerToken = 1 << 31 },
		"member range": func(s *replicaStatus) { s.Members[0].End = 0 },
	}
	for name, change := range bad {
		s := good()
		change(s)
		if validReplicaStatus(s) == nil {
			t.Errorf("%s accepted", name)
		}
	}
}
