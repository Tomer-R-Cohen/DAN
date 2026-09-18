// Persistent replicas: link probes for formation, replica advertisements, live replica
// queries and the client-side replica search. A replica owner (dan-client --form) shares
// this sidecar with the node's worker; see engine/include/provider_owned/replica.hpp.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/discovery"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	lp2pprotocol "github.com/libp2p/go-libp2p/core/protocol"
	drouting "github.com/libp2p/go-libp2p/p2p/discovery/routing"
	"github.com/libp2p/go-libp2p/p2p/protocol/ping"
)

const (
	// Client sessions on a replica owner's front door.
	sessionProtocol lp2pprotocol.ID = "/dan/session/1.0.0"
	// The ring's last stage returning tokens to a replica owner. A separate protocol because
	// the owner's node already uses /dan/ring/1.0.0 for its own worker.
	returnProtocol lp2pprotocol.ID = "/dan/return/1.0.0"
	// "How do you reach peer X?": a replica owner checks each planned ring link from the
	// side that would send on it.
	probeProtocol lp2pprotocol.ID = "/dan/probe/1.0.0"
	// Live description of the replica this node owns.
	replicaInfoProtocol lp2pprotocol.ID = "/dan/replica/1.0.0"
	replicaNamespace                    = "dan/replica/1/"
	// Ring targets starting with this go to a node's return listener, not its worker.
	returnTargetPrefix = "/dan-return"
	probesHeader       = "DAN-PROBE/1 "
	replicasHeader     = "DAN-REPLICAS/1 "
	maxReplicaMessage  = 64 << 10
	probePings         = 3
)

// ---- Link probes ----

// measureLink reports how this node reaches a peer: connect (relayed if that is all there
// is), give hole punching the same chance a ring stream gets, then take the best of a few
// libp2p pings on the connection that remains.
func measureLink(ctx context.Context, d *dialer, id peer.ID) (link, error) {
	var result link
	if id == d.host.ID() {
		return link{path: "direct"}, nil
	}
	limited := network.WithAllowLimitedConn(ctx, "dan")
	if err := d.connect(limited, target{id: id}); err != nil {
		return result, err
	}
	if d.directWait > 0 && d.host.Network().Connectedness(id) == network.Limited {
		deadline := time.Now().Add(d.directWait)
		for time.Now().Before(deadline) && d.host.Network().Connectedness(id) == network.Limited {
			time.Sleep(100 * time.Millisecond)
		}
	}
	pings := ping.Ping(limited, d.host, id)
	for index := 0; index < probePings; index++ {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		case reply := <-pings:
			if reply.Error != nil {
				return result, reply.Error
			}
			if result.rtt == 0 || reply.RTT < result.rtt {
				result.rtt = reply.RTT
			}
		}
	}
	result.path = "relay"
	for _, conn := range d.host.Network().ConnsToPeer(id) {
		if !isRelayAddr(conn.RemoteMultiaddr()) {
			result.path = "direct"
		}
	}
	return result, nil
}

// serveProbes answers "<PeerID>\n" with "OK <rtt ms> <direct|relay>\n": how this node
// reaches that peer right now. Bounded, so nobody can make this node dial the world.
func serveProbes(d *dialer, timeout time.Duration) {
	slots := make(chan struct{}, 4)
	d.host.SetStreamHandler(probeProtocol, func(stream network.Stream) {
		defer stream.Close()
		_ = stream.SetDeadline(time.Now().Add(timeout + 5*time.Second))
		line, err := bufio.NewReader(io.LimitReader(stream, 128)).ReadString('\n')
		if err != nil {
			_ = stream.Reset()
			return
		}
		id, err := peer.Decode(strings.TrimSpace(line))
		if err != nil {
			fmt.Fprintf(stream, "ERR bad PeerID\n")
			return
		}
		select {
		case slots <- struct{}{}:
			defer func() { <-slots }()
		default:
			fmt.Fprintf(stream, "ERR busy\n")
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		measured, err := measureLink(ctx, d, id)
		if err != nil {
			fmt.Fprintf(stream, "ERR %s\n", oneLine(err))
			return
		}
		log.Printf("probe for %s: %s rtt=%s path=%s", stream.Conn().RemotePeer(), id, measured.rtt, measured.path)
		fmt.Fprintf(stream, "OK %d %s\n", measured.rtt.Milliseconds(), measured.path)
	})
}

// probeFrom asks peer `from` how it reaches `to` (or measures it here when `from` is this node).
func probeFrom(d *dialer, from, to peer.ID, timeout time.Duration) (link, error) {
	if from == d.host.ID() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		return measureLink(ctx, d, to)
	}
	stream, err := d.open(target{id: from}, probeProtocol)
	if err != nil {
		return link{}, err
	}
	defer stream.Close()
	_ = stream.SetDeadline(time.Now().Add(timeout + 10*time.Second))
	if _, err := fmt.Fprintf(stream, "%s\n", to); err != nil {
		return link{}, err
	}
	reply, err := bufio.NewReader(io.LimitReader(stream, 512)).ReadString('\n')
	if err != nil {
		return link{}, err
	}
	var milliseconds int64
	var path string
	if _, err := fmt.Sscanf(reply, "OK %d %s", &milliseconds, &path); err != nil ||
		(path != "direct" && path != "relay") {
		return link{}, fmt.Errorf("probe at %s: %s", from, strings.TrimSpace(reply))
	}
	return link{rtt: time.Duration(milliseconds) * time.Millisecond, path: path}, nil
}

func oneLine(err error) string {
	return strings.NewReplacer("\n", " ", "\r", " ").Replace(err.Error())
}

// ---- The replica this node owns ----

// replicaStatus is the file dan-client --form writes (only the fields the sidecar uses).
type replicaStatus struct {
	ProtocolVersion uint32 `json:"protocol_version"`
	State           string `json:"state"`
	Owner           string `json:"owner"`
	ReplicaID       string `json:"replica_id"`
	ModelSHA256     string `json:"model_sha256"`
	DraftSHA256     string `json:"draft_sha256"`
	RuntimeABI      string `json:"runtime_abi"`
	Context         uint32 `json:"context"`
	SessionsMax     uint32 `json:"sessions_max"`
	SessionsInUse   uint32 `json:"sessions_in_use"`
	Members         []struct {
		Peer  string `json:"peer"`
		Begin uint32 `json:"begin"`
		End   uint32 `json:"end"`
	} `json:"members"`
	UpdatedUnixMS int64 `json:"updated_unix_ms"`
}

func readReplicaStatus(path string) (*replicaStatus, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var status replicaStatus
	if err := json.Unmarshal(data, &status); err != nil {
		return nil, err
	}
	if status.ProtocolVersion != 1 {
		return nil, fmt.Errorf("unsupported replica status version %d", status.ProtocolVersion)
	}
	return &status, nil
}

// readyReplica is the status when it describes a fresh READY replica this node owns.
func readyReplica(path string, self peer.ID) *replicaStatus {
	status, err := readReplicaStatus(path)
	if err != nil || status.State != "ready" || status.Owner != self.String() ||
		time.Since(time.UnixMilli(status.UpdatedUnixMS)) > statusStaleAfter {
		return nil
	}
	if _, err := modelKey(status.ModelSHA256); err != nil {
		return nil
	}
	return status
}

// advertiseReplica keeps dan/replica/1/<sha> advertised while this node owns a READY replica
// of that model. A record outlives a dissolved replica by up to its validity; clients
// always ask the owner before using one.
func advertiseReplica(ctx context.Context, kad *dht.IpfsDHT, self peer.ID, statusPath string,
	validity time.Duration) {
	routing := drouting.NewRoutingDiscovery(kad)
	var namespace string
	var cancel context.CancelFunc
	defer func() {
		if cancel != nil {
			cancel()
		}
	}()
	for {
		wanted := ""
		if status := readyReplica(statusPath, self); status != nil {
			wanted = replicaNamespace + strings.ToLower(status.ModelSHA256)
		}
		if wanted != namespace {
			if cancel != nil {
				cancel()
				cancel = nil
				log.Printf("stopped advertising %s", namespace)
			}
			namespace = wanted
			if namespace != "" {
				replicaCtx, stop := context.WithCancel(ctx)
				cancel = stop
				go advertise(replicaCtx, routing, namespace, validity, validity/3)
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Second):
		}
	}
}

// serveReplicaInfo answers /dan/replica/1.0.0 with the owner's live replica status (JSON),
// or {"state":"none"} when this node owns no READY replica.
func serveReplicaInfo(d *dialer, statusPath string) {
	d.host.SetStreamHandler(replicaInfoProtocol, func(stream network.Stream) {
		defer stream.Close()
		_ = stream.SetDeadline(time.Now().Add(10 * time.Second))
		status := readyReplica(statusPath, d.host.ID())
		if status == nil {
			status = &replicaStatus{ProtocolVersion: 1, State: "none"}
		}
		data, _ := json.Marshal(status)
		_, _ = stream.Write(append(data, '\n'))
	})
}

func queryReplica(ctx context.Context, d *dialer, id peer.ID, timeout time.Duration) (*replicaStatus, link, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	limited := network.WithAllowLimitedConn(ctx, "dan")
	var reach link
	if err := d.connect(limited, target{id: id}); err != nil {
		return nil, reach, err
	}
	stream, err := d.host.NewStream(limited, id, replicaInfoProtocol)
	if err != nil {
		return nil, reach, err
	}
	defer stream.Close()
	reach.path, _, _ = connPath(stream.Conn())
	asked := time.Now()
	if deadline, ok := ctx.Deadline(); ok {
		_ = stream.SetDeadline(deadline)
	}
	data, err := io.ReadAll(io.LimitReader(stream, maxReplicaMessage))
	if err != nil {
		return nil, reach, err
	}
	reach.rtt = time.Since(asked)
	var status replicaStatus
	if err := json.Unmarshal(data, &status); err != nil {
		return nil, reach, err
	}
	return &status, reach, nil
}

type foundReplica struct {
	owner   peer.ID
	control string
	status  *replicaStatus
	reach   link
}

// findReplicas looks up owners advertising a READY replica of the model and asks each one
// directly. Only a live answer from the owner itself counts.
func findReplicas(ctx context.Context, d *dialer, kad *dht.IpfsDHT, sessions *forwardSet,
	model string, config discoveryConfig) ([]foundReplica, error) {
	h := d.host
	findCtx, cancel := context.WithTimeout(ctx, config.discoveryTimeout)
	defer cancel()
	peers, err := drouting.NewRoutingDiscovery(kad).FindPeers(findCtx, replicaNamespace+model,
		discovery.Limit(maxCandidates))
	if err != nil {
		return nil, err
	}
	var (
		mutex  sync.Mutex
		result []foundReplica
		group  sync.WaitGroup
	)
	limit := make(chan struct{}, 8)
	for info := range peers {
		if len(info.Addrs) > 0 {
			h.Peerstore().AddAddrs(info.ID, info.Addrs, config.addrTTL)
		}
		group.Add(1)
		go func(id peer.ID) {
			defer group.Done()
			limit <- struct{}{}
			defer func() { <-limit }()
			var status *replicaStatus
			var reach link
			var err error
			if id == h.ID() {
				err = errors.New("own replica")
			} else {
				status, reach, err = queryReplica(ctx, d, id, config.queryTimeout)
			}
			if err != nil {
				log.Printf("replica owner %s skipped: %v", id, err)
				return
			}
			// The answer must come from the owner it names, about this model, fresh and ready.
			if status.State != "ready" || status.Owner != id.String() ||
				!strings.EqualFold(status.ModelSHA256, model) || len(status.ReplicaID) != 32 ||
				time.Since(time.UnixMilli(status.UpdatedUnixMS)) > time.Minute {
				log.Printf("replica owner %s skipped: state=%s", id, status.State)
				return
			}
			control, err := sessions.get(id)
			if err != nil {
				log.Printf("replica owner %s skipped: %v", id, err)
				return
			}
			mutex.Lock()
			result = append(result, foundReplica{owner: id, control: control, status: status, reach: reach})
			mutex.Unlock()
		}(info.ID)
	}
	group.Wait()
	log.Printf("replicas model=%s ready=%d", model, len(result))
	return result, nil
}

func hexOrDash(value string) string {
	if _, err := modelKey(value); err != nil {
		return "-"
	}
	return strings.ToLower(value)
}

// answerReplicas handles "DAN-REPLICAS/1 <sha256> [...]" on the candidate API:
//
//	SELF <this PeerID>
//	REPLICA <owner> <local session forward> <rtt ms> <direct|relay> <replica id> <model sha256>
//	        <sessions free> <sessions max> <context> <stages> <draft sha256 | ->
//	END
func answerReplicas(ctx context.Context, conn net.Conn, line string, d *dialer, kad *dht.IpfsDHT,
	sessions *forwardSet, config discoveryConfig) {
	var models []string
	for _, field := range strings.Fields(strings.TrimPrefix(line, replicasHeader)) {
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
	results := make([][]foundReplica, len(models))
	errs := make([]error, len(models))
	var wait sync.WaitGroup
	for index, model := range models {
		wait.Add(1)
		go func(index int, model string) {
			defer wait.Done()
			results[index], errs[index] = findReplicas(ctx, d, kad, sessions, model, config)
		}(index, model)
	}
	wait.Wait()
	var reply strings.Builder
	fmt.Fprintf(&reply, "SELF %s\n", d.host.ID())
	for index := range models {
		if errs[index] != nil {
			fmt.Fprintf(conn, "ERR %s\n", oneLine(errs[index]))
			return
		}
		for _, found := range results[index] {
			s := found.status
			free := uint32(0)
			if s.SessionsMax > s.SessionsInUse {
				free = s.SessionsMax - s.SessionsInUse
			}
			fmt.Fprintf(&reply, "REPLICA %s %s %d %s %s %s %d %d %d %d %s\n", found.owner, found.control,
				found.reach.rtt.Milliseconds(), found.reach.path, strings.ToLower(s.ReplicaID),
				strings.ToLower(s.ModelSHA256), free, s.SessionsMax, s.Context, len(s.Members),
				hexOrDash(s.DraftSHA256))
		}
	}
	reply.WriteString("END\n")
	_, _ = conn.Write([]byte(reply.String()))
}

// answerProbe handles "DAN-PROBE/1 <from PeerID> <to PeerID>" on the candidate API:
// "PROBE <rtt ms> <direct|relay>\nEND\n".
func answerProbe(conn net.Conn, line string, d *dialer, timeout time.Duration) {
	fields := strings.Fields(strings.TrimPrefix(line, probesHeader))
	if len(fields) != 2 {
		fmt.Fprintf(conn, "ERR usage: DAN-PROBE/1 <from> <to>\n")
		return
	}
	from, err1 := peer.Decode(fields[0])
	to, err2 := peer.Decode(fields[1])
	if err1 != nil || err2 != nil {
		fmt.Fprintf(conn, "ERR bad PeerID\n")
		return
	}
	measured, err := probeFrom(d, from, to, timeout)
	if err != nil {
		fmt.Fprintf(conn, "ERR %s\n", oneLine(err))
		return
	}
	fmt.Fprintf(conn, "PROBE %d %s\nEND\n", measured.rtt.Milliseconds(), measured.path)
}

// ---- Returns to a replica owner on this node ----

// localBridge connects a ring proxy caller to a listener on this same node (a node cannot
// dial itself over libp2p): the one-stage replica whose worker returns tokens to its owner.
func localBridge(conn net.Conn, local string, self peer.ID) {
	target, err := net.Dial("tcp", local)
	if err != nil {
		log.Printf("local ring bridge failed: %v", err)
		_ = conn.Close()
		return
	}
	if _, err := io.WriteString(target, peerHeader+self.String()+"\n"); err != nil {
		_ = conn.Close()
		_ = target.Close()
		return
	}
	if _, err := io.WriteString(conn, "OK\n"); err != nil {
		_ = conn.Close()
		_ = target.Close()
		return
	}
	_ = conn.SetDeadline(time.Time{})
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(target, conn); done <- struct{}{} }()
	go func() { _, _ = io.Copy(conn, target); done <- struct{}{} }()
	<-done
	_ = conn.Close()
	_ = target.Close()
}

// splitReturnTarget recognizes "/dan-return/p2p/<PeerID>" (a replica owner's return listener).
func splitReturnTarget(value string) (string, bool) {
	if strings.HasPrefix(value, returnTargetPrefix+"/") {
		return strings.TrimPrefix(value, returnTargetPrefix), true
	}
	return value, false
}
