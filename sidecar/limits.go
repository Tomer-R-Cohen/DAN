package main

import (
	"encoding/hex"
	"fmt"

	"dan/sidecar/capabilities"
)

// Bounds on what remote peers may claim (BETA_SELECTION_PLAN.md M7). A public network can
// contain hostile or broken peers: an answer outside these bounds is dropped as a whole,
// never clamped, so no planner ever works with a value that cannot be real.
const (
	maxNameBytes        = 128
	maxOfferedMiB       = 16 << 20 // 16 TiB
	maxPeerContext      = 1 << 20
	maxPeerSessions     = 256
	maxPeerModels       = 64
	maxPeerLayers       = 4096
	maxCachedRanges     = 64
	maxDataProtocols    = 8
	maxReplicaMembers   = 64
	maxReplicaLatencyMs = 10_000_000
)

func isHex(value string, bytes int) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == bytes
}

// validCapability checks a remote worker's capability answer.
func validCapability(c *capabilities.Capability) error {
	switch {
	case c == nil:
		return fmt.Errorf("no capability")
	case c.ProtocolVersion != 1:
		return fmt.Errorf("protocol version %d", c.ProtocolVersion)
	case len(c.WorkerId) > maxNameBytes || len(c.Device) > maxNameBytes || len(c.RuntimeAbi) > maxNameBytes:
		return fmt.Errorf("name too long")
	case len(c.DataProtocols) > maxDataProtocols:
		return fmt.Errorf("too many data protocols")
	case c.OfferedMemoryMib > maxOfferedMiB || c.TotalMemoryMib > maxOfferedMiB:
		return fmt.Errorf("implausible memory %d MiB", c.OfferedMemoryMib)
	case c.MaxContext > maxPeerContext || c.MaxSessions > maxPeerSessions:
		return fmt.Errorf("implausible limits: context %d, sessions %d", c.MaxContext, c.MaxSessions)
	case len(c.Models) > maxPeerModels:
		return fmt.Errorf("too many models (%d)", len(c.Models))
	}
	for _, model := range c.Models {
		if model == nil || !isHex(model.Sha256, 32) || model.Layers > maxPeerLayers ||
			len(model.Cached) > maxCachedRanges {
			return fmt.Errorf("invalid model entry")
		}
		for _, cached := range model.Cached {
			if cached == nil || cached.Begin >= cached.End || (model.Layers != 0 && cached.End > model.Layers) {
				return fmt.Errorf("invalid cached range")
			}
		}
	}
	return nil
}

// validReplicaStatus checks a remote owner's replica status (the fields clients plan with).
func validReplicaStatus(s *replicaStatus) error {
	switch {
	case s == nil:
		return fmt.Errorf("no status")
	case !isHex(s.ReplicaID, 16) || !isHex(s.ModelSHA256, 32):
		return fmt.Errorf("invalid replica or model id")
	case s.DraftSHA256 != "" && !isHex(s.DraftSHA256, 32):
		return fmt.Errorf("invalid draft id")
	case len(s.RuntimeABI) > maxNameBytes:
		return fmt.Errorf("name too long")
	case s.SessionsMax == 0 || s.SessionsMax > maxPeerSessions || s.SessionsInUse > s.SessionsMax:
		return fmt.Errorf("implausible sessions %d/%d", s.SessionsInUse, s.SessionsMax)
	case s.Context == 0 || s.Context > maxPeerContext:
		return fmt.Errorf("implausible context %d", s.Context)
	case len(s.Members) == 0 || len(s.Members) > maxReplicaMembers:
		return fmt.Errorf("implausible member count %d", len(s.Members))
	case s.MsPerToken > maxReplicaLatencyMs || s.FirstTokenMs > maxReplicaLatencyMs:
		return fmt.Errorf("implausible latency")
	}
	for _, member := range s.Members {
		if member.Begin >= member.End || member.End > maxPeerLayers {
			return fmt.Errorf("invalid member range")
		}
	}
	return nil
}
