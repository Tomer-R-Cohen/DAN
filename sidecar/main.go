// dan-sidecar gives DAN stable peer identity and encrypted network links.
// It only moves bytes between localhost TCP and libp2p streams.
package main

import (
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
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	noise "github.com/libp2p/go-libp2p/p2p/security/noise"
	"github.com/multiformats/go-multiaddr"
)

const protocol = "/dan/transport/1.0.0"
const peerHeader = "DAN-P2P/1 "

type stringsFlag []string

func (s *stringsFlag) String() string     { return strings.Join(*s, ",") }
func (s *stringsFlag) Set(v string) error { *s = append(*s, v); return nil }

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

func openStream(h host.Host, target target) (network.Stream, error) {
	ctx := network.WithAllowLimitedConn(context.Background(), "dan")
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := h.Connect(ctx, peer.AddrInfo{ID: target.id, Addrs: target.addrs}); err != nil {
		return nil, err
	}
	stream, err := h.NewStream(ctx, target.id, protocol)
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

func runForward(h host.Host, local, remote, readyFile string) error {
	target, err := parseTarget(remote)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", local)
	if err != nil {
		return err
	}
	if err := writeReady(readyFile, listener.Addr().String()); err != nil {
		_ = listener.Close()
		return err
	}
	log.Printf("local forward ready address=%s peer=%s", listener.Addr(), target.id)
	for {
		conn, err := listener.Accept()
		if err != nil {
			return err
		}
		go func() {
			stream, err := openStream(h, target)
			if err != nil {
				log.Printf("peer connection failed: %v", err)
				_ = conn.Close()
				return
			}
			pipe(conn, stream)
		}()
	}
}

func runInbound(h host.Host, local string, allowed map[peer.ID]bool, allowAny bool) {
	h.SetStreamHandler(protocol, func(stream network.Stream) {
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
	readyFile := flag.String("ready-file", "", "write the local forward address here when ready")
	logFile := flag.String("log", "", "append logs to this file")
	showID := flag.Bool("id", false, "print the PeerID and exit")
	allowAny := flag.Bool("allow-any", false, "accept any authenticated PeerID")
	var allowValues, relayValues stringsFlag
	flag.Var(&allowValues, "allow", "allowed provider PeerID; repeat for more")
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
	if (*inbound == "") == (*forward == "") {
		log.Fatal("choose exactly one of -inbound or -forward")
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
		runInbound(h, *inbound, allowed, *allowAny)
		info := append([]string{h.ID().String()}, peerAddrs(h)...)
		if err := writeReady(*readyFile, strings.Join(info, "\n")); err != nil {
			log.Fatal(err)
		}
		log.Printf("inbound tunnel ready address=%s", *inbound)
		select {}
	}
	parts := strings.SplitN(*forward, "=", 2)
	if len(parts) != 2 {
		log.Fatal("-forward must be LOCAL=PEER_ADDRESS")
	}
	if err := runForward(h, parts[0], parts[1], *readyFile); err != nil {
		log.Fatal(err)
	}
}
