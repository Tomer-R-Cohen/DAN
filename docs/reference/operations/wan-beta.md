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

## 1. VPS (small public Linux x86-64, no GPU)

Open **TCP 4001 and UDP 4001** inbound. Nothing else.

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

Build a DHT package once (Windows):

```powershell
.\scripts\package_provider.ps1 -BuildDirectory <cuda build> -Sidecar <dan-sidecar.exe> `
    -Gateway <dan-api-gateway.exe> -RuntimeDll <...> `
    -Bootstrap /ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID>
```

It writes `config\provider.conf` (and `provider-dht.conf`):

```text
network=dht
bootstrap=/ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID>
catalog=config\provider-owned-qwen2.5-0.5b-q4km.json
stage_worker=runtime\dan-stage-worker.exe
sidecar=runtime\dan-sidecar.exe
reserve_vram_mib=1536
```

The friend double-clicks `dan-provider.exe`, or runs
`dan-provider --network dht --bootstrap <addr> --catalog <manifest>`. Optional keys:
`relay=` (default: the bootstrap nodes), `max_context=` (4096), `max_sessions=` (1),
`listen_port=` (0 = random), `device=`, `reserve_vram_mib=`.

What it starts: a persistent identity, the sidecar as a DHT client with
`-reachability private` (AutoRelay keeps a renewed reservation on the VPS and advertises
the relay address; hole punching is on), capability publication and model
advertisements, and the worker bound to loopback. "Joined the DAN network. Relay
addresses: N" should show N ≥ 1.

## 3. Running inference from a home PC

```powershell
.\Start-DAN-Client.ps1 -Bootstrap /ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID> `
    -Manifest config\provider-owned-qwen2.5-0.5b-q4km.json -- `
    --prompt "The capital of France is" --tokens 32 --require-direct
```

```bash
./start-dan-client.sh --bootstrap /ip4/<PUBLIC_IP>/tcp/4001/p2p/<VPS_PEERID> \
    --manifest config/provider-owned-qwen2.5-0.5b-q4km.json -- --prompt "Hello" --tokens 32
```

The client has its own identity (`%LOCALAPPDATA%\DAN\client`), separate from a worker on
the same PC. `--require-direct` means "no activations through the client", not "no relay".

## 4. Timeouts

| Setting | Default | Covers |
|---|---|---|
| sidecar `-dial-timeout` | 15 s | peer lookup + connection setup |
| sidecar `-direct-wait` | 5 s | wait for a hole-punched connection before using the relay (skipped for 10 min after it fails for a peer) |
| sidecar `-query-timeout` | 10 s | capability query per candidate |
| sidecar `-discovery-timeout` | 15 s | DHT provider search |
| worker `--connect-timeout-ms` | 20000 | establishing a route's next hop |
| dan-client `--connect-timeout-ms` | 45000 | greeting, reservation and route setup replies |
| stage loading | none | model downloads can be long; the control connection holds the lease |

## 5. What to check in every WAN test

- `dan-client` output: `discovered candidates=`, one `route=` line per stage (worker
  PeerIDs), `timing metadata_ms discovery_ms capabilities_ms plan_ms reserve_ms load_ms`,
  and per request `ttft_ms`, `decode_tok_s`, `route_setup_ms`, plus
  `client_activations_received=0`.
- sidecar logs (`logs\sidecar.log`) for every hop:
  `connected|accepted peer=... protocol=/dan/... transport=... path=direct|relay relay=...`
  and on close `sent_bytes=... received_bytes=... seconds=...`.
  "no direct connection ... using the relay" means hole punching did not produce a
  direct connection in time. Do not count a hop as direct unless it logs `path=direct`.
- `addresses updated relay=N direct=M` shows the node's advertised addresses.

## 6. Test order

| Test | Setup | Pass when |
|---|---|---|
| A | `go test ./...` in `sidecar/` | forced-relay test passes: PeerID-only lookup, stale address fallback, relayed ring, relayed return path |
| B | `scripts\Test-DAN-NatRehearsal.ps1` (one PC, all home nodes relay-only) | output matches baseline; every DAN stream `path=relay` |
| C | VPS + 2 public hosts (e.g. RunPod) as workers, client on the VPS or a public host | discovery, placement, ring inference; paths logged |
| D | VPS + your CGNAT PC as worker, then as client | works with no port forwarding; note direct vs relay |
| E | you + one friend, both at home | same, from the friend's package |

Local rehearsal:

```powershell
.\scripts\Test-DAN-NatRehearsal.ps1 -BuildDir build-client -OutDir build-client\results\nat `
    -BaselineDir build-client\results\baseline
```

## Known limits (beta)

- The VPS relay carries all traffic between peers that cannot hole-punch; a relayed
  connection is cut after 4 GiB or 2 h, which ends a route.
- Any peer can reserve a worker; relay and DHT offer no Sybil resistance
  (GO-2024-3218 is an accepted risk).
- A worker failing mid-route fails the request; there is no rerouting.
- The client re-reads the model header from Hugging Face on each run.
