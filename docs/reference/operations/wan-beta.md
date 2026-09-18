# WAN beta: DAN across home networks

Goal: a friend behind ordinary home NAT or CGNAT starts `dan-provider`, joins through the
DAN VPS, and serves inference. Nobody opens router ports. There is no coordinator.

```text
Public VPS (infrastructure, not a coordinator)
    dan-sidecar -infra: DHT bootstrap + DHT server + relay + reachability checks

Home nodes (outbound connections only)
    dan-provider network=dht: sidecar (DHT client, relay reservation) + serve-mode worker
    Start-DAN-Client.ps1 / start-dan-client.sh: client sidecar + dan-client --discover
```

Connections go direct when hole punching succeeds and through the VPS relay otherwise.
Both are correct; the direct path is only faster.

**Current deployment (since 2026-09-17):** Oracle Cloud Always Free VM (`il-jerusalem-1`,
`VM.Standard.E2.1.Micro`, Ubuntu 24.04), address
`/ip4/82.70.213.202/tcp/4001/p2p/12D3KooWGDp2QL2wBSwbE6jyzU8CH13KT8Tvmzeq4caErzVvuU35`.
Friends get `build\installer\DAN-Setup-1.1.0.exe` from
`scripts\build_installer.ps1 -Bootstrap <that address>` (see [PROJECT.md](../../PROJECT.md) §12;
current build 2026-09-18).
On Oracle Ubuntu images, also open the port in iptables (the image rejects everything
but SSH by default):
`sudo iptables -I INPUT 5 -p tcp --dport 4001 -j ACCEPT; sudo iptables -I INPUT 5 -p udp --dport 4001 -j ACCEPT; sudo sh -c "iptables-save > /etc/iptables/rules.v4"`.

## 1. VPS (small public Linux x86-64, no GPU)

Open **TCP 4001 and UDP 4001** inbound, for IPv4 and (if the VPS has it) IPv6. Nothing else.

Every sidecar that listens on `/ip4/0.0.0.0/...` (the default, and `-infra`) also listens on
`/ip6/::/...` with the same port; `-ipv6=false` turns that off. Many CGNAT connections still
have a global IPv6 address, so two such nodes can often connect directly over IPv6. A home
router may block unsolicited inbound IPv6; hole punching usually still works there.

```bash
bash scripts/build_sidecar.sh                     # or use build/sidecar/dan-sidecar-linux-amd64
sudo useradd --system --home /var/lib/dan dan
sudo install -D -m 0755 build/sidecar/dan-sidecar-linux-amd64 /opt/dan/dan-sidecar
sudo cp deploy/dan-infra.service /etc/systemd/system/
sudo sed -i 's/203.0.113.10/<PUBLIC_IP>/' /etc/systemd/system/dan-infra.service
sudo systemctl daemon-reload && sudo systemctl enable --now dan-infra
grep "bootstrap/relay address" /var/log/dan/infra.log
```

Share the printed `/ip4/<PUBLIC_IP>/tcp/4001/p2p/<PeerID>` address. The key in
`/var/lib/dan/infra.key` keeps that PeerID stable; back it up.

`-infra` means: DHT server, relay for any authenticated peer, AutoNAT service, public
reachability, listen on 4001. Relay limits per relayed connection default to 4 GiB / 2 h
(`-relay-limit-mib`, `-relay-limit-duration`). The node never runs a worker and never
plans, reserves or schedules anything; losing it stops new relayed connections, not the
DHT (which other peers keep).

## 2. A friend's worker (Windows or Linux, NVIDIA GPU)

Normally the friend just runs the installer and opens **DAN Node**. The installer is built
from a DHT package; to build only the package (Windows):

```powershell
.\scripts\package_provider.ps1 -BuildDirectory <cuda build> -Sidecar <dan-sidecar.exe> `
    -Gateway <dan-api-gateway.exe> -RuntimeDll <...> `
    -Bootstrap /ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID>
```

It writes `config\provider.conf` (and `provider-dht.conf`), listing every bundled model:

```text
network=dht
bootstrap=/ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID>
catalog=config\provider-owned-qwen2.5-0.5b-q4km.json
catalog=config\provider-owned-qwen2.5-1.5b-q4km.json
catalog=config\provider-owned-qwen2.5-14b-q8.json
catalog=config\provider-owned-qwen2.5-32b-q5km.json
stage_worker=runtime\dan-stage-worker.exe
sidecar=runtime\dan-sidecar.exe
reserve_vram_mib=1536
```

The friend double-clicks `dan-provider.exe`, or runs
`dan-provider --network dht --bootstrap <addr> --catalog <manifest>`. Optional keys:
`relay=` (default: the bootstrap nodes), `max_context=` (4096), `max_sessions=` (1),
`listen_port=` (0 = random), `device=`, `reserve_vram_mib=`, `state_dir=`, `cache_dir=`.
A Linux GPU node is built and started as in PROJECT.md §12 ("Linux GPU node").

What it starts: a persistent identity, the sidecar as a DHT client with
`-reachability private` (AutoRelay keeps a renewed reservation on the VPS and advertises
the relay address; hole punching is on), capability publication and model
advertisements, and the worker bound to loopback. "Joined the DAN network. Relay
addresses: N" should show N ≥ 1.

Then the node dashboard takes over the window: identity, relay/IPv6/peer count, status
(waiting, downloading with progress, loading, serving), the assigned layers, the
previous and next node with the link type (`direct tcp`, `direct quic`, `relay`), a
tokens/s sparkline, totals and recent events. `--verbose` shows plain logs instead.
When output is redirected it prints one status line per change. Logs:
`<state_dir>\logs\stage-worker.log` and `sidecar.log`
(default state_dir: `%LOCALAPPDATA%\DAN`). The sidecar writes its live network state to
`<state_dir>\network-status.json` (`-net-status-file`), which the dashboard reads.

**Dynamic DNS instead of a fixed IP.** Any bootstrap/relay address may use a DNS name,
resolved at every start:
`/dns4/dan.example.org/tcp/4001/p2p/<VPS_PEERID>` (IPv4) or `/dns6/...` (IPv6),
`/dns/...` for both. The PeerID still authenticates the node, so a wrong DNS answer
cannot impersonate it. The infra node itself still announces its IP (`PUBLIC_IP`); update
it if the IP changes.

## 3. Running inference from a home PC

Normally: open **DAN Chat**. It passes every installed model and the client picks the
largest one the online GPUs can run. By hand:

```powershell
.\Start-DAN-Client.ps1 -Bootstrap /ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID> `
    -Manifest config\provider-owned-qwen2.5-14b-q8.json, config\provider-owned-qwen2.5-0.5b-q4km.json -- `
    --prompt "The capital of France is" --tokens 32 --require-direct
```

Useful `dan-client` options after `--`:

| Option | Effect |
|---|---|
| `--chat` | interactive conversation (see below) |
| `--min-stages N` | force a split across at least N GPUs (tests; otherwise the fewest that fit) |
| `--speculate` | the first stage drafts with the smallest offered model; ~1.8–1.9× faster on 14B |
| `--no-loop` | old per-token path through the client (comparisons only) |
| `--context N` | context per session (default: the manifest's, 512) |

```bash
./start-dan-client.sh --bootstrap /ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID> \
    --manifest config/provider-owned-qwen2.5-0.5b-q4km.json -- --prompt "Hello" --tokens 32
```

Interactive chat: replace the prompt options with `--chat` (optional `--tokens`, default
256). It keeps one session on the route, streams tokens, prints tokens/s after each
answer, and understands `/new` (forget the conversation) and `/quit`. When the context
fills up it starts a new conversation automatically. Scripted input:
`Start-DAN-Client.ps1 ... -InputFile lines.txt -- --chat`.

The client has its own identity (`%LOCALAPPDATA%\DAN\client`), separate from a worker on
the same PC. `--require-direct` means "no activations through the client", not "no relay".
Several `-Manifest` files may be given; each one's shape is read once from its GGUF header
on Hugging Face and then kept in `%LOCALAPPDATA%\DAN\model-index`.

## 4. Timeouts

| Setting | Default | Covers |
|---|---|---|
| sidecar `-dial-timeout` | 15 s | peer lookup + connection setup |
| sidecar `-direct-wait` | 5 s | ring streams only: wait for a hole-punched connection before using the relay (skipped for 10 min after it fails for a peer) |
| sidecar `-query-timeout` | 10 s | capability query per candidate |
| sidecar `-discovery-timeout` | 15 s | DHT provider search |
| worker `--connect-timeout-ms` | 20000 | establishing a route's next hop |
| dan-client `--connect-timeout-ms` | 45000 | greeting, reservation and route setup replies |
| stage loading | none | model downloads can be long; the control connection holds the lease |

## 5. What to check in every WAN test

- `dan-client` output: `discovered candidates=`, `model=<chosen>` (and `draft=<model>` with
  `--speculate`), one `route=` line per stage with the worker, its layers and
  `link=direct|relay <rtt>ms`, `timing metadata_ms discovery_ms capabilities_ms plan_ms
  reserve_ms load_ms` (`load_ms` near zero when layers were cached or already loaded), per
  answer `tokens | tok/s | first token` (chat) or `ttft_ms decode_tok_s route_setup_ms`,
  and `client_activations_received=0`. On stderr, `placement: route … (cached layers) plan`
  means no download was needed.
- sidecar logs (`logs\sidecar.log`) for every hop:
  `connected|accepted peer=... protocol=/dan/... transport=... path=direct|relay relay=...`
  and on close `sent_bytes=... received_bytes=... seconds=...`.
  "no direct connection ... using the relay" means hole punching did not produce a
  direct connection in time. Do not count a hop as direct unless it logs `path=direct`.
- `addresses updated relay=N direct=M` shows the node's advertised addresses.
- Client sidecar: `providers model=<sha> peers=N usable=M` (DHT records found vs peers that
  answered as available) and `candidate <PeerID> rtt=… path=…` per worker;
  `connected through a relay peer=…` when a peer was reached through the relay fallback.
- Worker (`logs\stage-worker.log`, readable while running): `speculation: draft model …
  ready`, then on the last stage one `speculation: proposed=3 accepted=K` line per round;
  `speculation: draft could not follow …` means the draft fell out of step (a bug).
- Node sidecar with `DAN_DEBUG_ADVERTISE=1`: a `re-advertised …` line per model every
  100 s. If a node is missing from discovery, check this first.

## 6. Test order

| Test | Setup | Pass when | Status |
|---|---|---|---|
| A | `go test ./...` in `sidecar/` | forced-relay test passes: PeerID-only lookup, stale address fallback, relayed ring, relayed return path | passing |
| B | `scripts\Test-DAN-NatRehearsal.ps1` (one PC, all home nodes relay-only) | output matches baseline; every DAN stream `path=relay` | passing (42/42, 2026-09-18) |
| C | VPS + a public GPU host (RunPod) as a worker | discovery, placement, ring inference; paths logged | done 2026-09-17/18 |
| D | VPS + your home PC as worker and client, with a RunPod worker | split routes, loop mode, speculation, no port forwarding | done 2026-09-17/18 (see PROJECT.md §13) |
| E | you + one friend, both at home | same, from the friend's installer | **open** |

Local rehearsal:

```powershell
.\scripts\Test-DAN-NatRehearsal.ps1 -BuildDir build-client -OutDir build-client\results\nat `
    -BaselineDir build-client\results\baseline
```

## Known limits (beta)

- The VPS relay carries all traffic between peers that cannot hole-punch (RunPod pods
  usually cannot); a relayed connection is cut after 4 GiB or 2 h, which ends a route.
- Any peer can reserve a worker; relay and DHT offer no Sybil resistance
  (GO-2024-3218 is an accepted risk).
- A worker failing mid-route fails the request; there is no rerouting.
- The first run of a new model reads its header from Hugging Face (~1.5 s per model);
  later runs use the saved index.
- One route per worker: while one client chats, others cannot use that GPU.
