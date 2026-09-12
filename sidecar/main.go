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

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	lp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	relayserver "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	noise "github.com/libp2p/go-libp2p/p2p/security/noise"
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

func newHost(key crypto.PrivKey, listen string, relays []peer.AddrInfo, advertised *advertisedAddrs) (host.Host, error) {
	addrs := []string{listen}
	if quic := tcpToQUIC(listen); quic != "" {
		addrs = append(addrs, quic)
	}
	opts := []libp2p.Option{
		libp2p.Identity(key),
		libp2p.ListenAddrStrings(addrs...),
		libp2p.Security(noise.ID, noise.New),
		libp2p.EnableHolePunching(),
		libp2p.NATPortMap(),
		libp2p.AddrsFactory(advertised.appendTo),
	}
	if len(relays) > 0 {
		opts = append(opts, libp2p.EnableAutoRelayWithStaticRelays(relays))
	}
	return libp2p.New(opts...)
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

func openStream(h host.Host, target target, streamProtocol lp2pprotocol.ID) (network.Stream, error) {
	ctx := network.WithAllowLimitedConn(context.Background(), "dan")
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := h.Connect(ctx, peer.AddrInfo{ID: target.id, Addrs: target.addrs}); err != nil {
		return nil, err
	}
	stream, err := h.NewStream(ctx, target.id, streamProtocol)
	if err == nil {
		state := stream.Conn().ConnState()
		path := "direct"
		if strings.Contains(stream.Conn().RemoteMultiaddr().String(), "p2p-circuit") {
			path = "relay"
		}
		log.Printf("connected peer=%s security=%s path=%s address=%s", target.id, state.Security, path, stream.Conn().RemoteMultiaddr())
	}
	return stream, err
}

func pipe(a, b io.ReadWriteCloser) {
	done := make(chan struct{}, 2)
	copyOne := func(dst io.Writer, src io.Reader) { _, _ = io.Copy(dst, src); done <- struct{}{} }
	go copyOne(a, b)
	go copyOne(b, a)
	<-done
	_ = a.Close()
	_ = b.Close()
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

func startForward(h host.Host, local, remote string, streamProtocol lp2pprotocol.ID) (net.Listener, error) {
	target, err := parseTarget(remote)
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", local)
	if err != nil {
		return nil, err
	}
	log.Printf("local forward ready address=%s peer=%s", listener.Addr(), target.id)
	go serveForward(h, listener, target, streamProtocol)
	return listener, nil
}

func serveForward(h host.Host, listener net.Listener, target target, streamProtocol lp2pprotocol.ID) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		go func() {
			stream, err := openStream(h, target, streamProtocol)
			if err != nil {
				log.Printf("peer connection failed: %v", err)
				_ = conn.Close()
				return
			}
			pipe(conn, stream)
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

func startRingProxy(h host.Host, local string) (net.Listener, error) {
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
				_ = conn.SetDeadline(time.Now().Add(30 * time.Second))
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
				stream, err := openStream(h, target, ringProtocol)
				if err != nil {
					log.Printf("ring peer connection failed: %v", err)
					_ = conn.Close()
					return
				}
				if _, err := io.WriteString(conn, "OK\n"); err != nil {
					_ = conn.Close()
					_ = stream.Reset()
					return
				}
				_ = conn.SetDeadline(time.Time{})
				pipe(conn, stream)
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
		state := stream.Conn().ConnState()
		log.Printf("accepted peer=%s security=%s", remote, state.Security)
		pipe(conn, stream)
	})
}

func main() {
	keyPath := flag.String("key", "", "identity key file")
	listen := flag.String("listen", "/ip4/0.0.0.0/tcp/0", "libp2p listen address")
	inbound := flag.String("inbound", "", "local DAN coordinator address")
	forward := flag.String("forward", "", "local address=coordinator peer address")
	ringInbound := flag.String("ring-inbound", "", "local DAN ring listener address")
	ringProxy := flag.String("ring-proxy", "", "loopback address for dynamic ring forwarding")
	readyFile := flag.String("ready-file", "", "write the local forward address here when ready")
	logFile := flag.String("log", "", "append logs to this file")
	showID := flag.Bool("id", false, "print the PeerID and exit")
	allowAny := flag.Bool("allow-any", false, "accept any authenticated PeerID")
	relayService := flag.Bool("relay-service", false, "run a bounded circuit-v2 relay")
	relayLimitMiB := flag.Int64("relay-limit-mib", 1024, "relay bytes per direction and circuit")
	relayLimitDuration := flag.Duration("relay-limit-duration", 30*time.Minute, "relay circuit lifetime")
	var allowValues, relayValues stringsFlag
	flag.Var(&allowValues, "allow", "allowed inbound or relay-reservation PeerID; repeat for more")
	flag.Var(&relayValues, "relay", "relay peer address; repeat for more")
	flag.Parse()
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
	if *inbound == "" && *forward == "" && *ringInbound == "" && *ringProxy == "" && !*relayService {
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
	advertised := &advertisedAddrs{}
	h, err := newHost(key, *listen, relays, advertised)
	if err != nil {
		log.Fatal(err)
	}
	defer h.Close()
	log.Printf("peer=%s", h.ID())
	printAddrs(h)
	if *relayService {
		relay, err := startRelay(h, allowed, *allowAny, *relayLimitMiB, *relayLimitDuration)
		if err != nil {
			log.Fatal(err)
		}
		defer relay.Close()
		log.Printf("relay service ready limit_mib=%d limit_duration=%s", *relayLimitMiB, *relayLimitDuration)
	}
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
	if len(relays) > 0 {
		printAddrs(h)
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
	if *forward != "" {
		parts := strings.SplitN(*forward, "=", 2)
		if len(parts) != 2 {
			log.Fatal("-forward must be LOCAL=PEER_ADDRESS")
		}
		listener, err := startForward(h, parts[0], parts[1], controlProtocol)
		if err != nil {
			log.Fatal(err)
		}
		listeners = append(listeners, listener)
	}
	if *ringProxy != "" {
		listener, err := startRingProxy(h, *ringProxy)
		if err != nil {
			log.Fatal(err)
		}
		listeners = append(listeners, listener)
	}
	ready := strings.Join(append([]string{h.ID().String()}, peerAddrs(h)...), "\n")
	if *ringInbound == "" && *ringProxy == "" && len(listeners) == 1 && *forward != "" {
		ready = listeners[0].Addr().String()
	}
	if err := writeReady(*readyFile, ready); err != nil {
		log.Fatal(err)
	}
	select {}
}
