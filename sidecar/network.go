// Connection setup shared by every DAN tunnel: host options for home and infrastructure
// nodes, PeerID resolution, direct-first stream selection, and path logging.
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/control"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/event"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	lp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
	noise "github.com/libp2p/go-libp2p/p2p/security/noise"
	ma "github.com/multiformats/go-multiaddr"
)

type hostOptions struct {
	key        crypto.PrivKey
	listen     string
	relays     []peer.AddrInfo
	advertised *advertisedAddrs // manual relay addresses (legacy, non-private mode)
	announce   []ma.Multiaddr   // extra public addresses (infrastructure behind cloud NAT)
	// "private": a home node behind NAT. AutoRelay then keeps reservations on the static
	// relays alive and advertises their relay addresses. "public": reachable infrastructure.
	reachability string
	natService   bool // answer AutoNAT reachability checks for other peers
	// Test only: behave as if behind a NAT that allows nothing but relayed connections.
	simulateNAT bool
}

func newHost(o hostOptions) (host.Host, error) {
	addrs := []string{o.listen}
	if quic := tcpToQUIC(o.listen); quic != "" {
		addrs = append(addrs, quic)
	}
	factory := func(addrs []ma.Multiaddr) []ma.Multiaddr {
		addrs = append(addrs, o.announce...)
		if o.advertised != nil {
			addrs = o.advertised.appendTo(addrs)
		}
		if o.simulateNAT {
			addrs = onlyRelayAddrs(addrs)
		}
		return addrs
	}
	opts := []libp2p.Option{
		libp2p.Identity(o.key),
		libp2p.ListenAddrStrings(addrs...),
		libp2p.Security(noise.ID, noise.New),
		libp2p.EnableHolePunching(),
		libp2p.NATPortMap(),
		libp2p.AddrsFactory(factory),
	}
	switch o.reachability {
	case "":
	case "private":
		opts = append(opts, libp2p.ForceReachabilityPrivate())
	case "public":
		opts = append(opts, libp2p.ForceReachabilityPublic())
	default:
		return nil, fmt.Errorf("-reachability must be private or public, not %q", o.reachability)
	}
	if o.natService {
		opts = append(opts, libp2p.EnableNATService())
	}
	if len(o.relays) > 0 {
		opts = append(opts, libp2p.EnableAutoRelayWithStaticRelays(o.relays))
	}
	if o.simulateNAT {
		gater := relayOnlyGater{relays: map[peer.ID]bool{}}
		for _, relay := range o.relays {
			gater.relays[relay.ID] = true
		}
		opts = append(opts, libp2p.ConnectionGater(gater))
	}
	return libp2p.New(opts...)
}

// relayOnlyGater makes a host behave like one behind NAT: it only dials its relays
// directly, and only accepts connections from a relay or through one.
type relayOnlyGater struct{ relays map[peer.ID]bool }

func (relayOnlyGater) InterceptPeerDial(peer.ID) bool { return true }
func (g relayOnlyGater) InterceptAddrDial(p peer.ID, addr ma.Multiaddr) bool {
	return g.relays[p] || isRelayAddr(addr)
}
func (relayOnlyGater) InterceptAccept(network.ConnMultiaddrs) bool { return true }
func (g relayOnlyGater) InterceptSecured(direction network.Direction, p peer.ID, conn network.ConnMultiaddrs) bool {
	return direction == network.DirOutbound || g.relays[p] || isRelayAddr(conn.RemoteMultiaddr())
}
func (relayOnlyGater) InterceptUpgraded(network.Conn) (bool, control.DisconnectReason) {
	return true, 0
}

func onlyRelayAddrs(addrs []ma.Multiaddr) []ma.Multiaddr {
	var result []ma.Multiaddr
	for _, addr := range addrs {
		if isRelayAddr(addr) {
			result = append(result, addr)
		}
	}
	return result
}

func isRelayAddr(addr ma.Multiaddr) bool {
	_, err := addr.ValueForProtocol(ma.P_CIRCUIT)
	return err == nil
}

// waitForRelayAddress waits until AutoRelay has a reservation, so the first DHT
// advertisement already carries a relay address. Returns false on timeout.
func waitForRelayAddress(h host.Host, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, addr := range h.Addrs() {
			if isRelayAddr(addr) {
				log.Printf("relay address ready address=%s", addr)
				return true
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	return false
}

// logAddressChanges reports when relay reservations or public addresses change.
func logAddressChanges(ctx context.Context, h host.Host) {
	sub, err := h.EventBus().Subscribe(new(event.EvtLocalAddressesUpdated))
	if err != nil {
		return
	}
	go func() {
		defer sub.Close()
		for {
			select {
			case <-ctx.Done():
				return
			case <-sub.Out():
				relayed, direct := 0, 0
				for _, addr := range h.Addrs() {
					if isRelayAddr(addr) {
						relayed++
					} else {
						direct++
					}
				}
				log.Printf("addresses updated relay=%d direct=%d", relayed, direct)
			}
		}
	}()
}

// describeConn says how a connection reaches the peer, for WAN test logs.
func describeConn(c network.Conn) string {
	remote := c.RemoteMultiaddr()
	state := c.ConnState()
	if isRelayAddr(remote) {
		relay, _ := remote.ValueForProtocol(ma.P_P2P)
		return fmt.Sprintf("transport=%s path=relay relay=%s address=%s", state.Transport, relay, remote)
	}
	return fmt.Sprintf("transport=%s security=%s path=direct address=%s", state.Transport, state.Security, remote)
}

// dialer opens DAN streams to a PeerID. The sidecar, not the caller, decides which
// addresses to use: known ones, then a DHT lookup; direct if possible, else the relay.
type dialer struct {
	host        host.Host
	resolve     peerResolver // nil without discovery
	dialTimeout time.Duration
	directWait  time.Duration
	// Peers whose recent direct-connection wait failed; skip waiting again for a while.
	noDirect sync.Map // peer.ID -> time.Time
}

// noDirectMemory is how long a failed hole punch makes new streams go straight to the relay.
const noDirectMemory = 10 * time.Minute

func (d *dialer) connect(ctx context.Context, t target) error {
	switch d.host.Network().Connectedness(t.id) {
	case network.Connected, network.Limited:
		return nil // reuse it, relayed or not (e.g. the ring return to a client behind NAT)
	}
	if len(t.addrs) > 0 || d.resolve == nil {
		return d.host.Connect(ctx, peer.AddrInfo{ID: t.id, Addrs: t.addrs})
	}
	if len(d.host.Peerstore().Addrs(t.id)) > 0 {
		known, cancel := context.WithTimeout(ctx, d.dialTimeout/2)
		err := d.host.Connect(known, peer.AddrInfo{ID: t.id})
		cancel()
		if err == nil {
			return nil
		}
		log.Printf("known addresses failed peer=%s, looking it up: %v", t.id, err)
	}
	info, err := d.resolve(ctx, t.id)
	if err != nil {
		return fmt.Errorf("could not find peer %s: %w", t.id, err)
	}
	return d.host.Connect(ctx, info)
}

func (d *dialer) open(t target, streamProtocol lp2pprotocol.ID) (network.Stream, error) {
	ctx, cancel := context.WithTimeout(context.Background(), d.dialTimeout+d.directWait)
	defer cancel()
	limited := network.WithAllowLimitedConn(ctx, "dan")
	if err := d.connect(limited, t); err != nil {
		return nil, err
	}
	var stream network.Stream
	failedAt, failedRecently := d.noDirect.Load(t.id)
	failedRecently = failedRecently && time.Since(failedAt.(time.Time)) < noDirectMemory
	if d.directWait > 0 && !failedRecently && d.host.Network().Connectedness(t.id) == network.Limited {
		// Only a relayed connection so far: give hole punching a moment to produce a
		// direct one, because a long-lived stream stays on the connection it opened on.
		wait, cancelWait := context.WithTimeout(ctx, d.directWait)
		direct, err := d.host.NewStream(wait, t.id, streamProtocol)
		cancelWait()
		if err == nil {
			stream = direct
			d.noDirect.Delete(t.id)
		} else {
			d.noDirect.Store(t.id, time.Now())
			log.Printf("no direct connection peer=%s after %s; using the relay for %s", t.id,
				d.directWait, noDirectMemory)
		}
	}
	if stream == nil {
		var err error
		if stream, err = d.host.NewStream(limited, t.id, streamProtocol); err != nil {
			return nil, err
		}
	}
	log.Printf("connected peer=%s protocol=%s %s", t.id, streamProtocol, describeConn(stream.Conn()))
	return stream, nil
}

var streamCounter atomic.Uint64

type countingWriter struct {
	w     io.Writer
	count atomic.Int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.count.Add(int64(n))
	return n, err
}

// bridge copies between a local connection and a libp2p stream until either side closes,
// keeps the peer's connection protected meanwhile, and logs the path and byte counts.
func bridge(h host.Host, local net.Conn, stream network.Stream) {
	remote := stream.Conn().RemotePeer()
	tag := fmt.Sprintf("dan-stream-%d", streamCounter.Add(1))
	h.ConnManager().Protect(remote, tag)
	defer h.ConnManager().Unprotect(remote, tag)
	started := time.Now()
	toPeer := &countingWriter{w: stream}
	toLocal := &countingWriter{w: local}
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(toPeer, local); done <- struct{}{} }()
	go func() { _, _ = io.Copy(toLocal, stream); done <- struct{}{} }()
	<-done
	_ = local.Close()
	_ = stream.Close()
	log.Printf("closed peer=%s protocol=%s %s sent_bytes=%d received_bytes=%d seconds=%.1f",
		remote, stream.Protocol(), describeConn(stream.Conn()), toPeer.count.Load(),
		toLocal.count.Load(), time.Since(started).Seconds())
}
