// Network status for the node dashboard: a small JSON file the worker reads.
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
	ma "github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"
)

type streamStatus struct {
	Peer      string    `json:"peer"`
	Protocol  string    `json:"protocol"`
	Path      string    `json:"path"`
	Transport string    `json:"transport"`
	Relay     string    `json:"relay,omitempty"`
	Started   time.Time `json:"-"`
	Seconds   int64     `json:"seconds"`
}

// activeStreams holds every bridged DAN stream, keyed by its connection-manager tag.
var activeStreams sync.Map

// linkStatus is libp2p's smoothed round trip to a connected peer (BETA_SELECTION_PLAN.md M3):
// the worker passes these to planners, which use them for worker-to-worker ring links
// instead of guessing through the planning node.
type linkStatus struct {
	Peer  string `json:"peer"`
	RTTMs int64  `json:"rtt_ms"`
	Path  string `json:"path"`
}

const maxLinks = 32

type pingResult struct {
	rtt time.Duration
	at  time.Time
}

// pingedRTT holds this node's own recent pings to stream peers (peer.ID -> pingResult).
var pingedRTT sync.Map

type netStatus struct {
	PeerID         string `json:"peer_id"`
	RelayAddresses int    `json:"relay_addresses"`
	PublicIPv6     bool   `json:"public_ipv6"`
	ConnectedPeers int    `json:"connected_peers"`
	// Before "streams": the worker's dashboard reader treats every object after it as a stream.
	Links         []linkStatus   `json:"links"`
	Streams       []streamStatus `json:"streams"`
	UpdatedUnixMS int64          `json:"updated_unix_ms"`
}

func currentNetStatus(h host.Host) netStatus {
	status := netStatus{PeerID: h.ID().String(), ConnectedPeers: len(h.Network().Peers()),
		UpdatedUnixMS: time.Now().UnixMilli(), Streams: []streamStatus{}}
	for _, addr := range h.Addrs() {
		if isRelayAddr(addr) {
			status.RelayAddresses++
		}
	}
	if addrs, err := h.Network().InterfaceListenAddresses(); err == nil {
		for _, addr := range addrs {
			if _, err := addr.ValueForProtocol(ma.P_IP6); err == nil && manet.IsPublicAddr(addr) {
				status.PublicIPv6 = true
			}
		}
	}
	activeStreams.Range(func(_, value any) bool {
		stream := value.(streamStatus)
		stream.Seconds = int64(time.Since(stream.Started).Seconds())
		status.Streams = append(status.Streams, stream)
		return true
	})
	status.Links = []linkStatus{}
	for _, id := range h.Network().Peers() {
		// Our own recent ping first (it can legitimately be 0 on a fast link, which the
		// peerstore average cannot tell from "never measured"), else libp2p's average.
		rtt := h.Peerstore().LatencyEWMA(id)
		measured := rtt > 0
		if value, ok := pingedRTT.Load(id); ok && time.Since(value.(pingResult).at) < 5*time.Minute {
			rtt, measured = value.(pingResult).rtt, true
		}
		conns := h.Network().ConnsToPeer(id)
		if !measured || len(conns) == 0 {
			continue
		}
		path := "relay"
		for _, conn := range conns {
			if p, _, _ := connPath(conn); p == "direct" {
				path = "direct"
			}
		}
		status.Links = append(status.Links, linkStatus{Peer: id.String(), RTTMs: rtt.Milliseconds(), Path: path})
	}
	sort.Slice(status.Links, func(i, j int) bool { return status.Links[i].RTTMs < status.Links[j].RTTMs })
	if len(status.Links) > maxLinks {
		status.Links = status.Links[:maxLinks]
	}
	sort.Slice(status.Streams, func(i, j int) bool { return status.Streams[i].Seconds > status.Streams[j].Seconds })
	return status
}

// refreshStreamLatency pings, at most every 30 s, the few peers this node has DAN streams
// with, so their round trips stay current during ordinary traffic. libp2p's ping records
// the result in the peerstore; no survey of other peers is made.
func refreshStreamLatency(ctx context.Context, h host.Host, pinged map[peer.ID]time.Time) {
	const maxPinged = 16
	peers := map[peer.ID]bool{}
	activeStreams.Range(func(_, value any) bool {
		if id, err := peer.Decode(value.(streamStatus).Peer); err == nil && len(peers) < maxPinged {
			peers[id] = true
		}
		return true
	})
	for id := range pinged {
		if !peers[id] {
			delete(pinged, id)
		}
	}
	for id := range peers {
		if time.Since(pinged[id]) < 30*time.Second {
			continue
		}
		pinged[id] = time.Now()
		go func(id peer.ID) {
			pingCtx, cancel := context.WithTimeout(network.WithAllowLimitedConn(ctx, "dan"), 5*time.Second)
			defer cancel()
			select {
			case result := <-ping.Ping(pingCtx, h, id):
				if result.Error == nil {
					pingedRTT.Store(id, pingResult{result.RTT, time.Now()})
				}
			case <-pingCtx.Done():
			}
		}(id)
	}
}

// writeNetStatus rewrites the file every interval until ctx ends.
func writeNetStatus(ctx context.Context, h host.Host, path string, interval time.Duration) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		log.Printf("network status disabled: %v", err)
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	pinged := map[peer.ID]time.Time{}
	for {
		refreshStreamLatency(ctx, h, pinged)
		data, err := json.Marshal(currentNetStatus(h))
		if err == nil {
			temporary := path + ".tmp"
			if err = os.WriteFile(temporary, data, 0600); err == nil {
				err = os.Rename(temporary, path)
			}
		}
		if err != nil {
			log.Printf("network status write failed: %v", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}
