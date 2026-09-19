package main

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// A slow peer (a dead one whose DHT record is still there) no longer holds up a search, and a
// peer that could not be reached is skipped by the next one.
func TestGatherStopsWaitingAndRemembersFailures(t *testing.T) {
	ids := []peer.ID{"fast", "slow", "dead", "busy"}
	lookup := func() <-chan peer.AddrInfo {
		peers := make(chan peer.AddrInfo, len(ids))
		for _, id := range ids {
			peers <- peer.AddrInfo{ID: id}
		}
		close(peers)
		return peers
	}
	var mutex sync.Mutex
	asked := map[peer.ID]int{}
	query := func(id peer.ID) (string, bool, error) {
		mutex.Lock()
		asked[id]++
		mutex.Unlock()
		switch id {
		case "slow":
			time.Sleep(3 * time.Second)
			return "slow", true, nil
		case "dead":
			return "", false, errors.New("no route")
		case "busy":
			return "", false, nil
		}
		return string(id), true, nil
	}
	failures := newRecentFailures()
	started := time.Now()
	results, providers := gather[string]("self", func(peer.AddrInfo) {}, lookup(), 200*time.Millisecond,
		failures, "test", query)
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("waited %s for a slow peer", elapsed)
	}
	if len(results) != 1 || results[0] != "fast" || providers != 4 {
		t.Fatalf("results %v providers %d", results, providers)
	}
	gather[string]("self", func(peer.AddrInfo) {}, lookup(), 200*time.Millisecond, failures, "test", query)
	mutex.Lock()
	defer mutex.Unlock()
	if asked["dead"] != 1 || asked["busy"] != 2 {
		t.Fatalf("dead asked %d times, busy %d", asked["dead"], asked["busy"])
	}
}
