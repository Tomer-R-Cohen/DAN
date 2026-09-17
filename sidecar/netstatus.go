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

type netStatus struct {
	PeerID         string         `json:"peer_id"`
	RelayAddresses int            `json:"relay_addresses"`
	PublicIPv6     bool           `json:"public_ipv6"`
	ConnectedPeers int            `json:"connected_peers"`
	Streams        []streamStatus `json:"streams"`
	UpdatedUnixMS  int64          `json:"updated_unix_ms"`
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
	sort.Slice(status.Streams, func(i, j int) bool { return status.Streams[i].Seconds > status.Streams[j].Seconds })
	return status
}

// writeNetStatus rewrites the file every interval until ctx ends.
func writeNetStatus(ctx context.Context, h host.Host, path string, interval time.Duration) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		log.Printf("network status disabled: %v", err)
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
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
