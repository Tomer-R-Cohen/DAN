// dan-sidecar gives DAN stable peer identity and encrypted network links.
// It only moves bytes between localhost TCP and libp2p streams.
package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	lp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	relayserver "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	"github.com/multiformats/go-multiaddr"
)

const controlProtocol lp2pprotocol.ID = "/dan/transport/1.0.0"
const ringProtocol lp2pprotocol.ID = "/dan/ring/1.0.0"
const peerHeader = "DAN-P2P/1 "
const ringTargetHeader = "DAN-RING/1 "

type stringsFlag []string

func (s *stringsFlag) String() string     { return strings.Join(*s, ",") }
func (s *stringsFlag) Set(v string) error { *s = append(*s, v); return nil }

type relayAllowlist map[peer.ID]bool

func (a relayAllowlist) AllowReserve(id peer.ID, _ multiaddr.Multiaddr) bool { return a[id] }
func (a relayAllowlist) AllowConnect(_ peer.ID, _ multiaddr.Multiaddr, destination peer.ID) bool {
	return a[destination]
}

func loadOrCreateKey(path string) (crypto.PrivKey, error) {
	if path == "" {
		return nil, errors.New("-key is required")
	}
	b, err := os.ReadFile(path)
	if err == nil {
		return crypto.UnmarshalPrivateKey(b)
	}
	if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, err
	}
	key, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		return nil, err
	}
	b, err = crypto.MarshalPrivateKey(key)
	if err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			b, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			return crypto.UnmarshalPrivateKey(b)
		}
		return nil, err
	}
	if _, err = file.Write(b); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return nil, err
	}
	return key, nil
}

func tcpToQUIC(addr string) string {
	i := strings.Index(addr, "/tcp/")
	if i < 0 {
		return ""
	}
	port := addr[i+5:]
	if slash := strings.IndexByte(port, '/'); slash >= 0 {
		port = port[:slash]
	}
	return addr[:i] + "/udp/" + port + "/quic-v1"
}

type advertisedAddrs struct {
	sync.Mutex
	addrs []multiaddr.Multiaddr
}

func (a *advertisedAddrs) addRelay(relay peer.AddrInfo) {
	a.Lock()
	defer a.Unlock()
	for _, addr := range relay.Addrs {
		circuit, err := multiaddr.NewMultiaddr(addr.String() + "/p2p/" + relay.ID.String() + "/p2p-circuit")
		if err == nil {
			a.addrs = append(a.addrs, circuit)
		}
	}
}

func (a *advertisedAddrs) appendTo(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
	a.Lock()
	defer a.Unlock()
	return append(addrs, a.addrs...)
}

func parsePeers(values []string) ([]peer.AddrInfo, error) {
	var result []peer.AddrInfo
	for _, value := range values {
		addr, err := multiaddr.NewMultiaddr(value)
		if err != nil {
			return nil, err
		}
		info, err := peer.AddrInfoFromP2pAddr(addr)
		if err != nil {
			return nil, err
		}
		result = append(result, *info)
	}
	return result, nil
}

type target struct {
	id    peer.ID
	addrs []multiaddr.Multiaddr
}

func parseTarget(value string) (target, error) {
	var result target
	for _, item := range strings.Split(value, ",") {
		addr, err := multiaddr.NewMultiaddr(strings.TrimSpace(item))
		if err != nil {
			return result, err
		}
		info, err := peer.AddrInfoFromP2pAddr(addr)
		if err != nil {
			return result, err
		}
		if result.id != "" && result.id != info.ID {
			return result, errors.New("peer addresses name different PeerIDs")
		}
		result.id = info.ID
		result.addrs = append(result.addrs, info.Addrs...)
	}
	if result.id == "" {
		return result, errors.New("no peer address supplied")
	}
	return result, nil
}

func startRelay(h host.Host, allowed map[peer.ID]bool, allowAny bool,
	limitMiB int64, limitDuration time.Duration) (*relayserver.Relay, error) {
	if len(allowed) == 0 && !allowAny {
		return nil, errors.New("relay service requires at least one -allow PeerID or -allow-any")
	}
	if limitMiB < 1 || limitMiB > 1<<20 || limitDuration <= 0 {
		return nil, errors.New("relay limits must be positive and at most 1 TiB per circuit")
	}
	resources := relayserver.DefaultResources()
	resources.Limit = &relayserver.RelayLimit{Duration: limitDuration, Data: limitMiB << 20}
	options := []relayserver.Option{relayserver.WithResources(resources)}
	if !allowAny {
		options = append(options, relayserver.WithACL(relayAllowlist(allowed)))
	}
	return relayserver.New(h, options...)
}

func peerAddrs(h host.Host) []string {
	suffix := multiaddr.StringCast("/p2p/" + h.ID().String())
	var result []string
	for _, addr := range h.Addrs() {
		result = append(result, addr.Encapsulate(suffix).String())
	}
	return result
}

func printAddrs(h host.Host) {
	for _, addr := range peerAddrs(h) {
		fmt.Printf("ADDR %s\n", addr)
	}
}

func writeReady(path, address string) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, []byte(address+"\n"), 0600); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func startForward(d *dialer, local, remote string, streamProtocol lp2pprotocol.ID) (net.Listener, error) {
	target, err := parseTarget(remote)
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", local)
	if err != nil {
		return nil, err
	}
	log.Printf("local forward ready address=%s peer=%s", listener.Addr(), target.id)
	go serveForward(d, listener, target, streamProtocol)
	return listener, nil
}

func serveForward(d *dialer, listener net.Listener, target target, streamProtocol lp2pprotocol.ID) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		go func() {
			stream, err := d.open(target, streamProtocol)
			if err != nil {
				log.Printf("peer connection failed peer=%s: %v", target.id, err)
				_ = conn.Close()
				return
			}
			bridge(d.host, conn, stream)
		}()
	}
}

func readLine(conn net.Conn, limit int) (string, error) {
	reader := bufio.NewReaderSize(conn, 1)
	var line strings.Builder
	for line.Len() <= limit {
		byteValue, err := reader.ReadByte()
		if err != nil {
			return "", err
		}
		if byteValue == '\n' {
			return line.String(), nil
		}
		line.WriteByte(byteValue)
	}
	return "", errors.New("line is too long")
}

func startRingProxy(d *dialer, local string) (net.Listener, error) {
	hostName, _, err := net.SplitHostPort(local)
	if err != nil || net.ParseIP(hostName) == nil || !net.ParseIP(hostName).IsLoopback() {
		return nil, errors.New("ring proxy must listen on a loopback IP")
	}
	listener, err := net.Listen("tcp", local)
	if err != nil {
		return nil, err
	}
	log.Printf("ring proxy ready address=%s", listener.Addr())
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go func() {
				_ = conn.SetDeadline(time.Now().Add(d.dialTimeout + d.directWait + 5*time.Second))
				line, err := readLine(conn, 16*1024)
				if err != nil || !strings.HasPrefix(line, ringTargetHeader) {
					_ = conn.Close()
					return
				}
				target, err := parseTarget(strings.TrimPrefix(line, ringTargetHeader))
				if err != nil {
					_ = conn.Close()
					return
				}
				if d.resolve != nil {
					// With discovery on, a remote-chosen target names only a PeerID; this
					// sidecar picks the addresses, so it never dials hosts a client chose.
					target.addrs = nil
				}
				stream, err := d.open(target, ringProtocol)
				if err != nil {
					log.Printf("ring peer connection failed peer=%s: %v", target.id, err)
					_ = conn.Close()
					return
				}
				if _, err := io.WriteString(conn, "OK\n"); err != nil {
					_ = conn.Close()
					_ = stream.Reset()
					return
				}
				_ = conn.SetDeadline(time.Time{})
				bridge(d.host, conn, stream)
			}()
		}
	}()
	return listener, nil
}

func runInbound(h host.Host, streamProtocol lp2pprotocol.ID, local string,
	allowed map[peer.ID]bool, allowAny bool) {
	h.SetStreamHandler(streamProtocol, func(stream network.Stream) {
		remote := stream.Conn().RemotePeer()
		if !allowAny && !allowed[remote] {
			log.Printf("rejected peer=%s", remote)
			_ = stream.Reset()
			return
		}
		conn, err := net.Dial("tcp", local)
		if err != nil {
			log.Printf("local DAN connection failed: %v", err)
			_ = stream.Reset()
			return
		}
		if _, err := io.WriteString(conn, peerHeader+remote.String()+"\n"); err != nil {
			_ = conn.Close()
			_ = stream.Reset()
			return
		}
		log.Printf("accepted peer=%s protocol=%s %s", remote, streamProtocol, describeConn(stream.Conn()))
		bridge(h, conn, stream)
	})
}

func main() {
	keyPath := flag.String("key", "", "identity key file")
	listen := flag.String("listen", "/ip4/0.0.0.0/tcp/0", "libp2p listen address (the QUIC port matches it)")
	infra := flag.Bool("infra", false, "public DAN infrastructure: DHT server, relay and reachability checks; no worker")
	reachability := flag.String("reachability", "", "private for home nodes behind NAT (keeps relay reservations), public for infrastructure")
	dialTimeout := flag.Duration("dial-timeout", 15*time.Second, "peer lookup plus connection setup")
	directWait := flag.Duration("direct-wait", 5*time.Second, "how long a relayed stream waits for a hole-punched direct connection")
	queryTimeout := flag.Duration("query-timeout", 10*time.Second, "capability query per candidate")
	simulateNAT := flag.Bool("simulate-nat", false, "test only: accept and make only relayed connections (except to -relay peers)")
	discoveryTimeout := flag.Duration("discovery-timeout", 15*time.Second, "DHT provider search per candidate request")
	inbound := flag.String("inbound", "", "local DAN coordinator address")
	ringInbound := flag.String("ring-inbound", "", "local DAN ring listener address")
	ringProxy := flag.String("ring-proxy", "", "loopback address for dynamic ring forwarding")
	readyFile := flag.String("ready-file", "", "write the local forward address here when ready")
	logFile := flag.String("log", "", "append logs to this file")
	showID := flag.Bool("id", false, "print the PeerID and exit")
	allowAny := flag.Bool("allow-any", false, "accept any authenticated PeerID")
	relayService := flag.Bool("relay-service", false, "run a bounded circuit-v2 relay")
	dhtMode := flag.String("dht", "", "join the private DAN DHT as server or client")
	provideValidity := flag.Duration("provide-validity", 5*time.Minute, "how long model advertisements stay valid")
	statusFile := flag.String("status-file", "", "worker status file; serves /dan/capabilities/1.0.0 and advertises its models")
	candidateAPI := flag.String("candidate-api", "", "loopback address answering DAN-CANDIDATES/1 requests")
	relayLimitMiB := flag.Int64("relay-limit-mib", 4096, "relay bytes per direction and relayed connection")
	relayLimitDuration := flag.Duration("relay-limit-duration", 2*time.Hour, "relayed connection lifetime")
	var allowValues, relayValues, forwardValues, bootstrapValues, announceValues stringsFlag
	flag.Var(&allowValues, "allow", "allowed inbound or relay-reservation PeerID; repeat for more")
	flag.Var(&relayValues, "relay", "relay peer address; repeat for more")
	flag.Var(&forwardValues, "forward", "LOCAL=PEER_ADDRESS control tunnel; repeat for more")
	flag.Var(&bootstrapValues, "bootstrap", "DHT bootstrap peer address; repeat for more")
	flag.Var(&announceValues, "announce", "extra public address to advertise, e.g. /ip4/PUBLIC_IP/tcp/4001; repeat for more")
	flag.Parse()
	if *infra {
		listenSet := false
		flag.Visit(func(f *flag.Flag) { listenSet = listenSet || f.Name == "listen" })
		if !listenSet {
			*listen = "/ip4/0.0.0.0/tcp/4001"
		}
		if *dhtMode == "" {
			*dhtMode = "server"
		}
		if *reachability == "" {
			*reachability = "public"
		}
		*relayService = true
	}
	if *logFile != "" {
		if err := os.MkdirAll(filepath.Dir(*logFile), 0700); err != nil {
			log.Fatal(err)
		}
		file, err := os.OpenFile(*logFile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0600)
		if err != nil {
			log.Fatal(err)
		}
		defer file.Close()
		log.SetOutput(file)
	}

	key, err := loadOrCreateKey(*keyPath)
	if err != nil {
		log.Fatal(err)
	}
	id, err := peer.IDFromPrivateKey(key)
	if err != nil {
		log.Fatal(err)
	}
	if *showID {
		fmt.Println(id)
		return
	}
	if *inbound == "" && len(forwardValues) == 0 && *ringInbound == "" && *ringProxy == "" && !*relayService &&
		*dhtMode == "" && *statusFile == "" {
		log.Fatal("choose at least one tunnel mode")
	}
	allowed := make(map[peer.ID]bool)
	for _, value := range allowValues {
		id, err := peer.Decode(value)
		if err != nil {
			log.Fatalf("bad -allow value: %v", err)
		}
		allowed[id] = true
	}
	if *inbound != "" && len(allowed) == 0 && !*allowAny {
		log.Fatal("at least one -allow PeerID or -allow-any is required")
	}
	relays, err := parsePeers(relayValues)
	if err != nil {
		log.Fatalf("bad -relay value: %v", err)
	}
	var announce []multiaddr.Multiaddr
	for _, value := range announceValues {
		addr, err := multiaddr.NewMultiaddr(value)
		if err != nil {
			log.Fatalf("bad -announce value: %v", err)
		}
		announce = append(announce, addr)
	}
	if *reachability == "private" && len(relays) == 0 {
		log.Printf("warning: -reachability private without -relay; peers behind NAT may not reach this node")
	}
	if *dialTimeout < time.Second || *directWait < 0 || *queryTimeout < time.Second ||
		*discoveryTimeout < time.Second {
		log.Fatal("timeouts must be at least 1s (-direct-wait may be 0)")
	}
	advertised := &advertisedAddrs{}
	h, err := newHost(hostOptions{key: key, listen: *listen, relays: relays, advertised: advertised,
		announce: announce, reachability: *reachability, natService: *infra, simulateNAT: *simulateNAT})
	if err != nil {
		log.Fatal(err)
	}
	defer h.Close()
	log.Printf("peer=%s reachability=%s", h.ID(), *reachability)
	if (len(bootstrapValues) > 0 || *candidateAPI != "") && *dhtMode == "" {
		log.Fatal("-bootstrap and -candidate-api need -dht")
	}
	if *provideValidity < 30*time.Second {
		log.Fatal("-provide-validity must be at least 30s")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	logAddressChanges(ctx, h)
	if *relayService {
		relay, err := startRelay(h, allowed, *allowAny || *infra, *relayLimitMiB, *relayLimitDuration)
		if err != nil {
			log.Fatal(err)
		}
		defer relay.Close()
		log.Printf("relay service ready limit_mib=%d limit_duration=%s", *relayLimitMiB, *relayLimitDuration)
	}
	// Relays come before the DHT, so the first model advertisement carries relay addresses.
	if *reachability == "private" {
		// AutoRelay reserves on the static relays and renews the reservations. It builds relay
		// addresses from the relay's known public addresses, so keep the configured ones for
		// good instead of the two minutes a plain dial remembers them.
		for _, relay := range relays {
			h.Peerstore().AddAddrs(relay.ID, relay.Addrs, peerstore.PermanentAddrTTL)
			connectCtx, cancel := context.WithTimeout(ctx, *dialTimeout)
			if err := h.Connect(connectCtx, relay); err != nil {
				log.Printf("relay connection failed peer=%s: %v", relay.ID, err)
			} else {
				h.ConnManager().Protect(relay.ID, "relay")
			}
			cancel()
		}
		if len(relays) > 0 && !waitForRelayAddress(h, 20*time.Second) {
			log.Printf("warning: no relay reservation yet; continuing and retrying in the background")
		}
	} else {
		for _, relay := range relays {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			if err := h.Connect(ctx, relay); err == nil {
				if _, err := relayclient.Reserve(ctx, h, relay); err == nil {
					advertised.addRelay(relay)
					h.ConnManager().Protect(relay.ID, "relay")
					log.Printf("relay ready peer=%s", relay.ID)
				} else {
					log.Printf("relay reservation failed: %v", err)
				}
			} else {
				log.Printf("relay connection failed: %v", err)
			}
			cancel()
		}
	}
	d := &dialer{host: h, dialTimeout: *dialTimeout, directWait: *directWait}
	if *statusFile != "" {
		serveCapabilities(h, *statusFile)
		log.Printf("capabilities ready status=%s", *statusFile)
	}
	if *dhtMode != "" {
		bootstrap, err := parsePeers(bootstrapValues)
		if err != nil {
			log.Fatalf("bad -bootstrap value: %v", err)
		}
		kad, err := startDHT(ctx, h, *dhtMode, bootstrap, *provideValidity)
		if err != nil {
			log.Fatal(err)
		}
		defer kad.Close()
		d.resolve = dhtResolver(kad)
		if *statusFile != "" {
			go advertiseModels(ctx, kad, *statusFile, *provideValidity)
		}
		if *candidateAPI != "" {
			listener, err := startCandidateAPI(ctx, d, kad, *candidateAPI, *ringInbound, discoveryConfig{
				queryTimeout: *queryTimeout, discoveryTimeout: *discoveryTimeout, addrTTL: *provideValidity})
			if err != nil {
				log.Fatal(err)
			}
			defer listener.Close()
		}
	}
	printAddrs(h)
	if *infra {
		log.Printf("infrastructure node ready (DHT server, relay, reachability checks); share one of:")
		for _, addr := range peerAddrs(h) {
			log.Printf("  bootstrap/relay address %s", addr)
		}
	}
	if *inbound != "" {
		runInbound(h, controlProtocol, *inbound, allowed, *allowAny)
		log.Printf("inbound tunnel ready address=%s", *inbound)
	}
	if *ringInbound != "" {
		// The stage/coordinator verifies this authenticated PeerID against its assignment.
		runInbound(h, ringProtocol, *ringInbound, nil, true)
		log.Printf("ring inbound tunnel ready address=%s", *ringInbound)
	}
	var listeners []net.Listener
	for _, forward := range forwardValues {
		parts := strings.SplitN(forward, "=", 2)
		if len(parts) != 2 {
			log.Fatal("-forward must be LOCAL=PEER_ADDRESS")
		}
		listener, err := startForward(d, parts[0], parts[1], controlProtocol)
		if err != nil {
			log.Fatal(err)
		}
		listeners = append(listeners, listener)
	}
	if *ringProxy != "" {
		listener, err := startRingProxy(d, *ringProxy)
		if err != nil {
			log.Fatal(err)
		}
		listeners = append(listeners, listener)
	}
	ready := strings.Join(append([]string{h.ID().String()}, peerAddrs(h)...), "\n")
	if *ringInbound == "" && *ringProxy == "" && len(listeners) == 1 && len(forwardValues) == 1 {
		ready = listeners[0].Addr().String()
	}
	if err := writeReady(*readyFile, ready); err != nil {
		log.Fatal(err)
	}
	select {}
}
