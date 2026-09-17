// Decentralized discovery: a private Kademlia DHT (/dan/kad/1.0.0) finds peers that may
// serve a model; /dan/capabilities/1.0.0 asks each one what it can do right now. The
// DHT only says "this peer may serve model M"; the capability answer is the live check.
package main

import (
	"bufio"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"dan/sidecar/capabilities"

	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p-kad-dht/records"
	"github.com/libp2p/go-libp2p/core/discovery"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	lp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
	drouting "github.com/libp2p/go-libp2p/p2p/discovery/routing"
	"github.com/libp2p/go-msgio/pbio"
)

const (
	dhtPrefix            lp2pprotocol.ID = "/dan"
	capabilitiesProtocol lp2pprotocol.ID = "/dan/capabilities/1.0.0"
	modelNamespace                       = "dan/model/1/"
	candidatesHeader                     = "DAN-CANDIDATES/1 "
	maxCapabilityMessage                 = 64 << 10
	statusStaleAfter                     = 15 * time.Second
	maxCandidates                        = 64
	maxQueriedModels                     = 8
)

// peerResolver finds addresses for a PeerID (through the DHT); nil without discovery.
type peerResolver func(ctx context.Context, id peer.ID) (peer.AddrInfo, error)

// dhtResolver asks the DHT for a peer's addresses. A DHT server answers FIND_PEER with the
// addresses it knows even for DHT clients, so a home node connected to public
// infrastructure can be found through its relay addresses.
func dhtResolver(d *dht.IpfsDHT) peerResolver {
	return d.FindPeer
}

type discoveryConfig struct {
	queryTimeout     time.Duration
	discoveryTimeout time.Duration
	addrTTL          time.Duration // how long provider-record addresses stay usable
}

func modelKey(sha string) (string, error) {
	sha = strings.ToLower(sha)
	if decoded, err := hex.DecodeString(sha); err != nil || len(decoded) != 32 {
		return "", errors.New("model must be a 64-digit SHA-256")
	}
	return sha, nil
}

func startDHT(ctx context.Context, h host.Host, mode string, bootstrap []peer.AddrInfo,
	validity time.Duration) (*dht.IpfsDHT, error) {
	var modeOption dht.ModeOpt
	switch mode {
	case "server":
		modeOption = dht.ModeServer
	case "client":
		modeOption = dht.ModeClient
	default:
		return nil, fmt.Errorf("-dht must be server or client, not %q", mode)
	}
	d, err := dht.New(h, dht.Mode(modeOption), dht.ProtocolPrefix(dhtPrefix),
		dht.BootstrapPeers(bootstrap...),
		dht.ProviderManagerOpts(records.ProvideValidity(validity), records.ProviderAddrTTL(validity)))
	if err != nil {
		return nil, err
	}
	for _, info := range bootstrap {
		connectCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		if err := h.Connect(connectCtx, info); err != nil {
			log.Printf("bootstrap connection failed peer=%s: %v", info.ID, err)
		} else {
			log.Printf("bootstrap connected peer=%s", info.ID)
		}
		cancel()
	}
	if err := d.Bootstrap(ctx); err != nil {
		_ = d.Close()
		return nil, err
	}
	// kad-dht adds already-connected peers (e.g. the relay, which a home node dials first)
	// asynchronously; wait briefly so the first lookup does not find an empty table.
	for wait := 0; len(bootstrap) > 0 && d.RoutingTable().Size() == 0 && wait < 100; wait++ {
		time.Sleep(100 * time.Millisecond)
	}
	if len(bootstrap) > 0 && d.RoutingTable().Size() == 0 {
		log.Printf("warning: DHT routing table is still empty; retrying in the background")
	}
	log.Printf("dht ready mode=%s prefix=%s provide_validity=%s", mode, dhtPrefix, validity)
	return d, nil
}

// workerStatus is the file dan-stage-worker --status-file writes.
type workerStatus struct {
	ProtocolVersion  uint32 `json:"protocol_version"`
	WorkerID         string `json:"worker_id"`
	RuntimeABI       string `json:"runtime_abi"`
	Device           string `json:"device"`
	OfferedMemoryMiB uint64 `json:"offered_memory_mib"`
	MaxContext       uint32 `json:"max_context"`
	MaxSessions      uint32 `json:"max_sessions"`
	State            string `json:"state"`
	Models           []struct {
		SHA256 string `json:"sha256"`
		Layers uint32 `json:"layers"`
		Hidden uint32 `json:"hidden"`
		Cached []struct {
			Begin uint32 `json:"begin"`
			End   uint32 `json:"end"`
		} `json:"cached"`
	} `json:"models"`
	Assignment *struct {
		RouteID     string `json:"route_id"`
		ModelSHA256 string `json:"model_sha256"`
		Begin       uint32 `json:"begin"`
		End         uint32 `json:"end"`
	} `json:"assignment"`
	UpdatedUnixMS int64 `json:"updated_unix_ms"`
}

func readStatus(path string) (*workerStatus, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var status workerStatus
	if err := json.Unmarshal(data, &status); err != nil {
		return nil, err
	}
	if status.ProtocolVersion != 1 {
		return nil, fmt.Errorf("unsupported worker status version %d", status.ProtocolVersion)
	}
	return &status, nil
}

var stateByName = map[string]capabilities.State{
	"available": capabilities.State_STATE_AVAILABLE,
	"reserved":  capabilities.State_STATE_RESERVED,
	"loading":   capabilities.State_STATE_LOADING,
	"serving":   capabilities.State_STATE_SERVING,
}

// capabilityFromStatus converts the worker's status; a missing or stale file is OFFLINE.
func capabilityFromStatus(status *workerStatus, now time.Time, model string) *capabilities.Capability {
	result := &capabilities.Capability{
		ProtocolVersion: 1,
		DataProtocols:   []string{string(controlProtocol), string(ringProtocol)},
	}
	if status == nil {
		return result
	}
	result.WorkerId = status.WorkerID
	result.RuntimeAbi = status.RuntimeABI
	result.Device = status.Device
	result.OfferedMemoryMib = status.OfferedMemoryMiB
	result.MaxContext = status.MaxContext
	result.MaxSessions = status.MaxSessions
	if now.Sub(time.UnixMilli(status.UpdatedUnixMS)) <= statusStaleAfter {
		result.State = stateByName[status.State]
	}
	for _, available := range status.Models {
		if model != "" && !strings.EqualFold(available.SHA256, model) {
			continue
		}
		entry := &capabilities.ModelAvailability{Sha256: strings.ToLower(available.SHA256),
			Layers: available.Layers, Hidden: available.Hidden}
		for _, cached := range available.Cached {
			entry.Cached = append(entry.Cached, &capabilities.Range{Begin: cached.Begin, End: cached.End})
		}
		result.Models = append(result.Models, entry)
	}
	if status.Assignment != nil {
		result.Assignment = &capabilities.Assignment{RouteId: status.Assignment.RouteID,
			ModelSha256: status.Assignment.ModelSHA256, Begin: status.Assignment.Begin,
			End: status.Assignment.End}
	}
	return result
}

func serveCapabilities(h host.Host, statusPath string) {
	h.SetStreamHandler(capabilitiesProtocol, func(stream network.Stream) {
		defer stream.Close()
		_ = stream.SetDeadline(time.Now().Add(10 * time.Second))
		var request capabilities.CapabilityRequest
		if err := pbio.NewDelimitedReader(stream, maxCapabilityMessage).ReadMsg(&request); err != nil {
			_ = stream.Reset()
			return
		}
		model := ""
		if request.ModelSha256 != "" {
			key, err := modelKey(request.ModelSha256)
			if err != nil {
				_ = stream.Reset()
				return
			}
			model = key
		}
		status, err := readStatus(statusPath)
		if err != nil {
			status = nil
		}
		if err := pbio.NewDelimitedWriter(stream).WriteMsg(capabilityFromStatus(status, time.Now(), model)); err != nil {
			_ = stream.Reset()
		}
	})
}

func queryCapabilities(ctx context.Context, d *dialer, id peer.ID, model string,
	timeout time.Duration) (*capabilities.Capability, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	limited := network.WithAllowLimitedConn(ctx, "dan")
	if err := d.connect(limited, target{id: id}); err != nil {
		return nil, err
	}
	stream, err := d.host.NewStream(limited, id, capabilitiesProtocol)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	if deadline, ok := ctx.Deadline(); ok {
		_ = stream.SetDeadline(deadline)
	}
	if err := pbio.NewDelimitedWriter(stream).WriteMsg(&capabilities.CapabilityRequest{ModelSha256: model}); err != nil {
		return nil, err
	}
	if err := stream.CloseWrite(); err != nil {
		return nil, err
	}
	var reply capabilities.Capability
	if err := pbio.NewDelimitedReader(stream, maxCapabilityMessage).ReadMsg(&reply); err != nil {
		return nil, err
	}
	return &reply, nil
}

// advertiseModels keeps a provider record for every catalog model in the worker status.
// The status is re-read every second, and only fresh statuses count: a file left by an
// earlier run must not decide what this worker advertises.
func advertiseModels(ctx context.Context, d *dht.IpfsDHT, statusPath string, validity time.Duration) {
	routing := drouting.NewRoutingDiscovery(d)
	interval := validity / 3
	running := map[string]context.CancelFunc{}
	defer func() {
		for _, cancel := range running {
			cancel()
		}
	}()
	for {
		if status, err := readStatus(statusPath); err == nil &&
			time.Since(time.UnixMilli(status.UpdatedUnixMS)) <= statusStaleAfter {
			wanted := map[string]bool{}
			for _, model := range status.Models {
				key, err := modelKey(model.SHA256)
				if err != nil {
					log.Printf("not advertising invalid model %q", model.SHA256)
					continue
				}
				namespace := modelNamespace + key
				wanted[namespace] = true
				if running[namespace] == nil {
					modelCtx, cancel := context.WithCancel(ctx)
					running[namespace] = cancel
					go advertise(modelCtx, routing, namespace, validity, interval)
				}
			}
			for namespace, cancel := range running {
				if !wanted[namespace] {
					cancel()
					delete(running, namespace)
					log.Printf("stopped advertising %s", namespace)
				}
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Second):
		}
	}
}

func advertise(ctx context.Context, routing *drouting.RoutingDiscovery, namespace string,
	validity, interval time.Duration) {
	announced := false
	for {
		// Bounded: a provide that hangs (no DHT server reachable for a while) must not stop
		// every later refresh, or this node's records quietly expire and nobody finds it.
		call, done := context.WithTimeout(ctx, interval)
		_, err := routing.Advertise(call, namespace, discovery.TTL(validity))
		done()
		wait := interval
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			// Usually an empty routing table right after start; retry soon.
			log.Printf("advertise %s failed: %v", namespace, err)
			wait = 5 * time.Second
		} else if !announced {
			log.Printf("advertising %s every %s", namespace, interval)
			announced = true
		} else if os.Getenv("DAN_DEBUG_ADVERTISE") != "" {
			log.Printf("re-advertised %s", namespace)
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
}

// forwardSet opens one local control forward per discovered peer and keeps it.
type forwardSet struct {
	sync.Mutex
	dialer    *dialer
	listeners map[peer.ID]net.Listener
}

func (f *forwardSet) get(id peer.ID) (string, error) {
	f.Lock()
	defer f.Unlock()
	if listener, ok := f.listeners[id]; ok {
		return listener.Addr().String(), nil
	}
	if len(f.listeners) >= 4*maxCandidates {
		return "", errors.New("too many forwards")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	f.listeners[id] = listener
	go serveForward(f.dialer, listener, target{id: id}, controlProtocol)
	return listener.Addr().String(), nil
}

type candidate struct {
	id         peer.ID
	control    string
	capability *capabilities.Capability
}

func findCandidates(ctx context.Context, d *dialer, kad *dht.IpfsDHT, forwards *forwardSet,
	model string, config discoveryConfig) ([]candidate, error) {
	h := d.host
	findCtx, cancel := context.WithTimeout(ctx, config.discoveryTimeout)
	defer cancel()
	peers, err := drouting.NewRoutingDiscovery(kad).FindPeers(findCtx, modelNamespace+model,
		discovery.Limit(maxCandidates))
	if err != nil {
		return nil, err
	}
	var (
		mutex  sync.Mutex
		result []candidate
		group  sync.WaitGroup
	)
	limit := make(chan struct{}, 8)
	providers := 0
	for info := range peers {
		if info.ID == h.ID() {
			continue
		}
		providers++
		if len(info.Addrs) > 0 {
			// Keep provider-record addresses (often relay addresses) for the record's
			// lifetime, so later PeerID-only dials can use them.
			h.Peerstore().AddAddrs(info.ID, info.Addrs, config.addrTTL)
		}
		group.Add(1)
		go func(id peer.ID) {
			defer group.Done()
			limit <- struct{}{}
			defer func() { <-limit }()
			capability, err := queryCapabilities(ctx, d, id, model, config.queryTimeout)
			if err != nil {
				log.Printf("candidate %s skipped: %v", id, err)
				return
			}
			if capability.State != capabilities.State_STATE_AVAILABLE || len(capability.Models) == 0 {
				log.Printf("candidate %s skipped: state=%s models=%d", id, capability.State, len(capability.Models))
				return
			}
			control, err := forwards.get(id)
			if err != nil {
				log.Printf("candidate %s skipped: %v", id, err)
				return
			}
			mutex.Lock()
			result = append(result, candidate{id: id, control: control, capability: capability})
			mutex.Unlock()
		}(info.ID)
	}
	group.Wait()
	log.Printf("providers model=%s peers=%d usable=%d", model, providers, len(result))
	sort.Slice(result, func(i, j int) bool { return result[i].id < result[j].id })
	return result, nil
}

// startCandidateAPI answers "DAN-CANDIDATES/1 <sha256> [<sha256> ...]" on loopback with
//
//	SELF <this PeerID>
//	RETURN <local ring return address>     (when -ring-inbound is set)
//	CANDIDATE <PeerID> <local control address> <offered MiB> <runtime ABI>
//	END
//
// or "ERR <reason>". Candidates are AVAILABLE peers that list one of the models right now;
// asking about several models at once is how a client picks the largest one the network can
// run. A peer serving more than one of them appears once.
func startCandidateAPI(ctx context.Context, d *dialer, kad *dht.IpfsDHT, local, ringInbound string,
	config discoveryConfig) (net.Listener, error) {
	h := d.host
	hostName, _, err := net.SplitHostPort(local)
	if err != nil || net.ParseIP(hostName) == nil || !net.ParseIP(hostName).IsLoopback() {
		return nil, errors.New("candidate API must listen on a loopback IP")
	}
	listener, err := net.Listen("tcp", local)
	if err != nil {
		return nil, err
	}
	forwards := &forwardSet{dialer: d, listeners: map[peer.ID]net.Listener{}}
	log.Printf("candidate API ready address=%s", listener.Addr())
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				_ = conn.SetDeadline(time.Now().Add(60 * time.Second))
				line, err := bufio.NewReader(conn).ReadString('\n')
				if err != nil || !strings.HasPrefix(line, candidatesHeader) {
					return
				}
				var models []string
				for _, field := range strings.Fields(strings.TrimPrefix(line, candidatesHeader)) {
					model, err := modelKey(field)
					if err != nil {
						fmt.Fprintf(conn, "ERR %v\n", err)
						return
					}
					models = append(models, model)
				}
				if len(models) == 0 || len(models) > maxQueriedModels {
					fmt.Fprintf(conn, "ERR ask about 1 to %d models\n", maxQueriedModels)
					return
				}
				// One lookup per model, in parallel; a peer serving several is listed once.
				type result struct {
					found []candidate
					err   error
				}
				results := make([]result, len(models))
				var wait sync.WaitGroup
				for index, model := range models {
					wait.Add(1)
					go func(index int, model string) {
						defer wait.Done()
						results[index].found, results[index].err =
							findCandidates(ctx, d, kad, forwards, model, config)
					}(index, model)
				}
				wait.Wait()
				var found []candidate
				seen := map[peer.ID]bool{}
				for index, r := range results {
					if r.err != nil {
						fmt.Fprintf(conn, "ERR %v\n", r.err)
						return
					}
					log.Printf("candidates model=%s found=%d", models[index], len(r.found))
					for _, c := range r.found {
						if !seen[c.id] {
							seen[c.id] = true
							found = append(found, c)
						}
					}
				}
				var reply strings.Builder
				fmt.Fprintf(&reply, "SELF %s\n", h.ID())
				if ringInbound != "" {
					fmt.Fprintf(&reply, "RETURN %s\n", ringInbound)
				}
				for _, c := range found {
					abi := c.capability.RuntimeAbi
					if abi == "" || strings.ContainsAny(abi, " \r\n") {
						abi = "-"
					}
					fmt.Fprintf(&reply, "CANDIDATE %s %s %d %s\n", c.id, c.control,
						c.capability.OfferedMemoryMib, abi)
				}
				reply.WriteString("END\n")
				_, _ = conn.Write([]byte(reply.String()))
			}()
		}
	}()
	return listener, nil
}
