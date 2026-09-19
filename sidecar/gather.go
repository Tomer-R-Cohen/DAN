package main

import (
	"log"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
)

const (
	// Once one peer has given a usable answer, how long the others still get. A peer whose
	// DHT record outlived it (up to the record's validity) then costs seconds, not the whole
	// query timeout.
	defaultGatherGrace = 3 * time.Second
	// How long a peer that could not be reached is skipped by later searches.
	failureMemory = 2 * time.Minute
	// Peers asked at the same time by one search.
	gatherParallel = 8
)

// recentFailures remembers peers this node recently failed to reach.
type recentFailures struct {
	sync.Mutex
	at map[peer.ID]time.Time
}

// keepAddrs keeps provider-record addresses (often relay addresses) for the record's
// lifetime, so later PeerID-only dials can use them.
func keepAddrs(h host.Host, ttl time.Duration) func(peer.AddrInfo) {
	return func(info peer.AddrInfo) {
		if len(info.Addrs) > 0 {
			h.Peerstore().AddAddrs(info.ID, info.Addrs, ttl)
		}
	}
}

func newRecentFailures() *recentFailures {
	return &recentFailures{at: map[peer.ID]time.Time{}}
}

func (r *recentFailures) recent(id peer.ID) bool {
	if r == nil {
		return false
	}
	r.Lock()
	defer r.Unlock()
	failed, ok := r.at[id]
	if ok && time.Since(failed) >= failureMemory {
		delete(r.at, id)
		return false
	}
	return ok
}

func (r *recentFailures) note(id peer.ID, failed bool) {
	if r == nil {
		return
	}
	r.Lock()
	defer r.Unlock()
	if failed {
		r.at[id] = time.Now()
	} else {
		delete(r.at, id)
	}
}

// gather asks every peer a DHT lookup yields (a few at a time) and returns the usable answers.
// `query` returns (value, usable, err); an error means the peer could not be reached, which
// is remembered so the next searches skip it. Waiting stops `grace` after the first usable
// answer, or when every peer has answered.
//
// `self` is skipped; `seen` is told about every other peer (to keep its record's addresses).
func gather[T any](self peer.ID, seen func(peer.AddrInfo), peers <-chan peer.AddrInfo,
	grace time.Duration, failures *recentFailures, what string,
	query func(peer.ID) (T, bool, error)) (results []T, providers int) {
	type answer struct {
		value  T
		usable bool
	}
	// Buffered for every peer a lookup can return, so a straggler never blocks after this returns.
	answers := make(chan answer, maxCandidates+1)
	slots := make(chan struct{}, gatherParallel)
	pending := 0
	var deadline <-chan time.Time
	// Peers arrive while the DHT lookup runs; answers while peers are asked. The grace
	// period covers both: a lookup still trying unreachable DHT nodes is not waited for either.
	for peers != nil || pending > 0 {
		select {
		case info, open := <-peers:
			if !open {
				peers = nil
				continue
			}
			if info.ID == self {
				continue
			}
			providers++
			seen(info)
			if failures.recent(info.ID) {
				log.Printf("%s %s skipped: unreachable in the last %s", what, info.ID, failureMemory)
				continue
			}
			pending++
			go func(id peer.ID) {
				slots <- struct{}{}
				defer func() { <-slots }()
				value, usable, err := query(id)
				failures.note(id, err != nil)
				if err != nil {
					log.Printf("%s %s skipped: %v", what, id, err)
				}
				answers <- answer{value, usable && err == nil}
			}(info.ID)
		case got := <-answers:
			pending--
			if !got.usable {
				continue
			}
			results = append(results, got.value)
			if deadline == nil {
				deadline = time.After(grace)
			}
		case <-deadline:
			if pending > 0 || peers != nil {
				log.Printf("%s: not waiting for %d slow peer(s) (lookup still running: %v)", what,
					pending, peers != nil)
			}
			return results, providers
		}
	}
	return results, providers
}
