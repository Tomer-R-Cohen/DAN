# DAN — Project Guide

**Read this first.** This is the single source of truth for what DAN is, why it exists,
how it works, where it stands, and what comes next. It is written for people and for
coding agents. Other documents hold detail and history; when they disagree with this
file, this file wins (and the other file should be fixed).

*Last updated: 2026-09-17.*

---

## 1. What DAN is, in one minute

DAN (**D**ecentralized **A**I **N**etwork) runs large language models across ordinary
people's GPUs over the internet.

A model too big for one GPU is cut into contiguous blocks of transformer layers
("stages"). Each volunteer GPU loads only its block. A user's prompt flows through
the GPUs in order, like a relay race, and the answer comes back to the user:

```text
your PC ──prompt──▶ GPU A (layers 0–9) ──▶ GPU B (layers 10–17) ──▶ GPU C (layers 18–23) ──answer──▶ your PC
```

There is **no central server that schedules work**. Each user's own client finds GPUs,
plans the split, reserves the GPUs and builds the chain itself. A small public
"network node" only helps machines find and reach each other; it never sees a plan
and never decides anything.

**Why:** big models need more memory than consumer GPUs have, and renting data-center
GPUs is expensive and centralized. Many idle gaming GPUs, combined, can run models that
none of them could run alone — if the system needs no central operator, works behind
home routers, and is a one-click install.

---

## 2. Status at a glance

| Area | State |
|---|---|
| Split inference across GPUs (provider-owned stages) | **Works.** Proven physically: Qwen2.5-32B split across RTX 2070 + RTX A5000 (2026-09-08). |
| No coordinator: discovery, planning, reservation, direct ring | **Works.** Private DHT + client-side planner + leases + direct A→B→C→client ring. |
| Home networks / NAT / CGNAT | **Works.** Relay + hole punching; tested with forced-relay rehearsal and over the real internet. |
| Public network node | **Running** on Oracle Cloud (Always Free), `82.70.213.202`. |
| One-click install (Windows) | **Works.** `DAN-Setup-1.1.0.exe` with *DAN Node* and *DAN Chat* shortcuts. |
| Node dashboard (TUI) and chat | **Works.** |
| Two GPUs on different networks | **Works.** 2026-09-17: Qwen2.5-14B Q8 split RunPod RTX 4000 Ada (layers 0–35, Linux) + owner's RTX 2070 (36–47, Windows install), through the Oracle relay: 4.6–5.2 tok/s, correct output. |
| A real friend's PC | **Not yet tested.** |
| Payments, reputation, Sybil resistance, verification, failover, privacy | **Not started** (deliberately deferred, see §14). |

Verified on 2026-09-17: the installed package on the owner's RTX 2070 joined through the
Oracle node, reserved a relay slot, and served a chat forced through the relay
(every DAN stream `path=relay`). Local chat on the same GPU: 68–145 tok/s.

---

## 3. Goals, non-goals, and standing rules

### Goal (current phase: "friends beta")
A friend behind any home router (NAT or CGNAT) installs DAN, double-clicks, and their
GPU is used for inference — no port forwarding, no accounts, no central scheduler.

### Design principles (all implemented; do not break them)
1. **Infrastructure has no scheduling authority.** The public node is DHT bootstrap +
   relay + reachability checks. It never plans, reserves, or routes work.
2. **The client plans its own route.** Using the same planner code everywhere.
3. **Workers decide for themselves.** A worker accepts work only if it is idle, the model
   is in *its own* catalog, and the stage fits *its own* memory.
4. **Clients never supply download URLs.** A client names a model only by SHA-256; the
   worker resolves it in its local catalog. No worker downloads from a client-chosen URL.
5. **PeerIDs come from authenticated connections**, never from what a peer claims.
6. **No raw TCP next hops in libp2p mode.** A worker reaches its next stage only through
   its local sidecar, by PeerID.
7. **Bootstrap nodes are entry points only.** Once routing tables fill, losing them does
   not stop discovery (tested).
8. **The public relay is resource-limited** (default 4 GiB / 2 h per relayed connection).
9. **Reuse battle-tested libraries** (go-libp2p, go-libp2p-kad-dht, llama.cpp) instead of
   inventing networking or inference code.

### Explicitly deferred (do not implement unless asked)
Payments/crypto, reputation, Sybil resistance, consensus, result verification, failover
mid-request, latency-aware placement, multi-client workers. (Also see the older working
rule in `PROGRESS.md`: no blockchain/marketplace work until requested.)

### Rules for agents working in this repo
- **Never run `git commit`.** The owner commits. Prepare changes and suggest a message.
- Output to the owner must be **short, clear, simple** — no filler.
- Inspect code before changing it; build and test after; update docs (this file first).
- Windows is the main dev machine; see §15 for PowerShell pitfalls that have bitten us.

---

## 4. Glossary

| Term | Meaning |
|---|---|
| **Stage** | A contiguous range of transformer layers `[begin, end)` loaded by one worker. The first stage also owns the token embedding; the last owns the final norm, output head, and sampling. |
| **Route / ring** | The ordered chain of stages for one client: client → A → B → C → client. |
| **Worker** | `dan-stage-worker`: the C++ process that loads a stage on a GPU and computes. |
| **Node** | A friend's machine running `dan-provider` (= launcher + worker + sidecar). |
| **Sidecar** | `dan-sidecar`: the Go libp2p process next to every worker and client. Owns identity, DHT, NAT traversal, encryption, and tunnels. |
| **Client** | `dan-client`: plans a route, reserves workers, runs the token loop. |
| **Network node / infra node** | `dan-sidecar -infra` on a public machine: DHT bootstrap + relay. |
| **PeerID** | A node's libp2p identity (hash of its public key), e.g. `12D3KooW…`. |
| **Catalog** | A worker's local list of model manifests it is willing to serve. |
| **Manifest** | `config/provider-owned-*.json`: a pinned GGUF (URL, revision, SHA-256, layers, hidden size, context). |
| **Lease** | A worker's single reservation slot; first client to reserve wins. |
| **Runtime ABI** | `dan-stage-v1/f32le/<patch hash>`; stages combine only if equal. |
| **Provider-owned** | Workers own and load their weights; nothing central ever holds the full model. |

---

## 5. System overview

### Processes on each machine

```text
FRIEND'S PC (GPU node)                                  PUBLIC NETWORK NODE (Oracle VM)
┌───────────────────────────────────────────────┐      ┌──────────────────────────────┐
│ DAN Node shortcut → installer/DAN.ps1 node     │      │ dan-sidecar -infra           │
│   dan-provider.exe  (launcher, config, GPU     │      │   DHT server (bootstrap)     │
│                      detection, supervision)   │◀────▶│   circuit-v2 relay           │
│     ├─ dan-sidecar  (DHT client, NAT, relay    │ libp2p│   AutoNAT service            │
│     │               reservation, tunnels,      │      │ no worker, no scheduling     │
│     │               capabilities, ads)         │      └──────────────────────────────┘
│     └─ dan-stage-worker (serve mode, GPU,      │
│                          node dashboard TUI)   │
└───────────────────────────────────────────────┘

USER'S PC (chat)
┌───────────────────────────────────────────────┐
│ DAN Chat shortcut → DAN.ps1 chat               │
│   Start-DAN-Client.ps1                         │
│     ├─ dan-sidecar (own identity, candidate    │
│     │               API, ring-return inbound)  │
│     └─ dan-client --discover … --chat          │
└───────────────────────────────────────────────┘
```

The C++ side (worker, client) **never speaks libp2p**. It talks plain TCP to
**loopback** ports that the local sidecar exposes; the sidecar carries the bytes over
encrypted, authenticated libp2p streams. This keeps the inference code simple and puts
all internet-facing code in one audited Go process built on go-libp2p.

### Components and languages

| Component | Language | Source | Role |
|---|---|---|---|
| `dan-stage-worker` | C++23 + patched llama.cpp | `engine/stage_worker.cpp` | Loads a stage, computes activations, serves leases, draws the node dashboard. |
| `dan-client` | C++23 | `engine/dan_client.cpp` + `client.cpp`, `placement.cpp`, `planner.cpp` | Placement + token loop + chat. |
| `dan-provider` | C++23 | `engine/provider_launcher.cpp` | Friend-facing launcher: reads config, detects NVIDIA GPU, starts sidecar + worker. |
| `dan-sidecar` | Go (go-libp2p v0.48.0, kad-dht v0.42.0) | `sidecar/*.go` | Networking: identity, DHT, NAT, relay, tunnels, capabilities, candidate API, net status. |
| Node dashboard | C++23 | `ui/node_dashboard.cpp`, `ui/provider_ui.cpp` | Terminal UI. |
| Installer | Inno Setup + PowerShell | `installer/`, `scripts/build_installer.ps1` | `DAN-Setup-x.y.z.exe`. |
| Platform layer | C++23 | `platform/` | Windows/POSIX process, GPU, console, hashing helpers. |

---

## 6. Repository map

```text
CMakeLists.txt            C++ build (targets below); links patched llama.cpp from
                          DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR
engine/                   MAIN PATH (provider-owned)
  stage_worker.cpp        dan-stage-worker (static/generic/serve modes, ring, leases, TUI feed)
  client.cpp              shared token loop + InferenceClient (ring setup, sessions)
  dan_client.cpp          dan-client CLI (static / --candidate / --discover, --chat)
  placement.cpp           client-side placement (greet, preselect, plan, reserve, load)
  planner.cpp             plan_stages / stage_fits (shared by client, worker, coordinator)
  range_model.cpp         GGUF metadata over HTTP ranges + sparse partial downloads
  manifest.cpp            manifest JSON loader
  provider_launcher.cpp   dan-provider
  coordinator.cpp         OLDER: metadata-only coordinator (still builds, not the direction)
  include/provider_owned/ protocol.hpp (frames), route.hpp, lease.hpp, planner.hpp,
                          placement.hpp, formation.hpp (hello/assignment text),
                          client.hpp, manifest.hpp, range_model.hpp, fair_queue.hpp
  tests/                  C++ unit tests + lease_integration.py
sidecar/                  Go module "dan/sidecar"
  main.go                 flags, host/relay/DHT startup, tunnels, ring proxy
  network.go              host options, NAT/relay, dialer (direct-first), stream bridge
  discovery.go            DHT, capabilities protocol, advertisements, candidate API
  netstatus.go            -net-status-file writer (dashboard input)
  capabilities/           capabilities.proto + generated Go
  cmd/dan-api-gateway/    OLDER: OpenAI-style HTTP gateway for the coordinator path
ui/                       provider_ui (runner + classic view), node_dashboard, admin_dashboard
platform/                 platform.hpp + windows/posix implementations
installer/                DAN.ps1 (launcher for shortcuts), dan.iss (Inno Setup script)
scripts/                  build, package, test, and launcher scripts (see §12, §13)
config/                   model manifests (provider-owned-qwen2.5-{0.5b,1.5b,14b,32b}.json) + legacy examples
deploy/                   dan-infra.service (systemd), prometheus alerts (older path)
patches/                  llama-provider-owned.patch (applied to llama.cpp 95ef7fc1)
legacy/                   OLDEST: whole-model providers + llama.cpp RPC path (reference only)
docs/                     this file + detail docs (§16)
```

Build outputs (git-ignored): `build-client/` (CPU dev build + tests), `build-cuda/`
(portable CUDA build used for the installer), `build/provider-owned-v0/llama.cpp`
(llama.cpp source with the patch), `build/gateway/`, `build/installer/`.

---

## 7. How a node joins (startup sequence)

`DAN Node` shortcut → `powershell -File DAN.ps1 node` → `dan-provider --config <app>\config\provider.conf`.

`provider.conf` (written by the packager):
```text
network=dht
bootstrap=/ip4/82.70.213.202/tcp/4001/p2p/12D3KooWGDp2QL2wBSwbE6jyzU8CH13KT8Tvmzeq4caErzVvuU35
catalog=config\provider-owned-qwen2.5-0.5b-q4km.json     (repeatable)
stage_worker=runtime\dan-stage-worker.exe
sidecar=runtime\dan-sidecar.exe
reserve_vram_mib=1536
```
Optional keys: `relay=` (default: the bootstrap nodes), `max_context=` (4096),
`max_sessions=` (1), `listen_port=` (0 = random), `device=`, `simulate_nat=` (tests only).

`dan-provider` then:
1. Queries `nvidia-smi`, picks a GPU, offers `total VRAM − reserve_vram_mib`.
2. Creates/loads a persistent identity (`%LOCALAPPDATA%\DAN\identity.key`) → PeerID.
3. Picks free loopback ports and starts **the sidecar**:
   `-dht client -reachability private -bootstrap … -relay … -inbound <control> -allow-any
    -ring-inbound <ring> -ring-proxy <proxy> -status-file worker-status.json
    -net-status-file network-status.json`
   The sidecar connects to the relay (addresses kept permanently), waits up to 20 s for a
   relay reservation (continues with a warning if none), starts the DHT and waits up to
   10 s for a routing-table entry, then advertises every catalog model.
4. Starts **the worker in serve mode**:
   `--control-listen --ring-listen --peer-header --ring-proxy --ring-target /p2p/<own PeerID>
    --status-file --net-status-file --provider-id <PeerID> --gpu --vram-mib --cache-dir
    --ctx <max_context> --max-sessions --catalog … [--tui]`
   The worker loads nothing yet; it only advertises resources.
5. Prints "Joined the DAN network. Relay addresses: N" and the dashboard takes over.

Files on a node (`%LOCALAPPDATA%\DAN`): `identity.key`, `models\` (sparse weight cache),
`worker-status.json`, `network-status.json`, `logs\sidecar.log`, `logs\stage-worker.log`,
`client\` (the chat client's separate identity and logs).

**Advertising:** every catalog model is published as a DHT provider record under
`dan/model/1/<gguf sha256>`, valid 5 min (`-provide-validity`) and refreshed every third
of that. A record only means "this peer may serve the model"; live state comes from the
capabilities protocol.

---

## 8. How a chat request works, end to end

`DAN Chat` → `DAN.ps1 chat` → `Start-DAN-Client.ps1` starts the client's own sidecar
(`-dht client -reachability private -candidate-api 127.0.0.1:P -ring-inbound 127.0.0.1:R`)
and runs `dan-client --manifest <model> --discover 127.0.0.1:P --chat`.

### Step 1 — Model metadata
`dan-client` reads the manifest, then reads the GGUF **header only** via HTTP range
requests (`inspect_range_model`) to get every tensor's byte size. (~6 s today.)

### Step 2 — Discovery (sidecar)
`dan-client` sends `DAN-CANDIDATES/1 <sha256>\n` to its sidecar. The sidecar:
1. Finds providers of `dan/model/1/<sha>` in the DHT (up to 64, `-discovery-timeout` 15 s),
   storing their addresses.
2. Asks each one over `/dan/capabilities/1.0.0` (protobuf, `-query-timeout` 10 s).
3. Keeps `AVAILABLE` ones, opens a **local control forward** (loopback port → that peer)
   for each, and replies:
   ```text
   SELF <client PeerID>
   RETURN 127.0.0.1:R
   CANDIDATE <PeerID> 127.0.0.1:<forward port> …
   END
   ```
`dan-client` never touches the DHT itself.

### Step 2b — Choosing the model (client)
`dan-client` takes several `--manifest` files. It reads each model's shape from its GGUF
header (in parallel), asks the sidecar about all of them in one query
(`DAN-CANDIDATES/1 <sha> <sha> …`, up to 8), and keeps the **largest model the available
workers can actually run**: models are tried biggest first and the first with a workable
plan wins (`model=<id>` is printed). DAN Chat passes every installed manifest, so the
network's capacity picks the model rather than a fixed setting.

### Step 3 — Greeting and preselection (client)
The client connects to every candidate forward **in parallel**. Each worker immediately
sends a `provider_available` frame (text `key=value` lines: `id, gpu, vram_mib, ring,
state, abi, max_context, max_sessions, model=<sha>…`). A candidate is dropped unless:
state is `available`; ABI equals the client's; the model SHA is in its catalog; its limits
cover the request; and its `ring` address is `/p2p/<PeerID>` **of the peer the connection
actually authenticated** (prevents redirecting the ring). The rest are sorted by offered
memory (desc), PeerID breaks ties, and at most 8 are kept.

### Step 4 — Planning (client, `plan_stages` / `plan_from_cache`)
See §9.1. Output: which worker gets which `[begin, end)`. If the candidates already hold
cached ranges that tile the model in no more stages than the ordinary plan, that split is
used instead, so nothing has to be downloaded (`placement: … plan (cached layers)`).

### Step 5 — Reservation (leases)
The client sends `reserve` to every chosen worker in parallel with
`route_id` (random 32 hex), `model_sha256`, `begin`, `end`, `context`, `sessions`,
`lease_ms` (30 s default, max 60 s). A worker says `ack` only if its lease is free, the
model is in its catalog, the range exists, context/sessions are within its limits, and
`stage_fits` passes on **its own** memory; otherwise it replies with an error
(`busy`, `unknown_model`, `limits_exceeded`, `insufficient_memory`, …).
If any refuse: the client releases the others (`release_route`), marks "busy" workers
for a fresh greeting, waits 100–500 ms at random, and replans (3 attempts).

### Step 6 — Loading
`assign_stage` to all in parallel (no timeout — downloads can be long). Each worker
downloads **only its own layers** from the URL in **its own** catalog (§9.5), loads
them on the GPU, and replies `stage_ready <route_id>`. The lease moves
`RESERVED → LOADING → SERVING`. Weights stay cached on disk and in memory for later routes.

### Step 7 — Linking the ring (per session)
`create_session` is sent **last stage first**. Its payload:
```text
next=<where this stage sends its output>
previous_peer=<the only PeerID it may accept input from>   (not for the first stage)
```
- Last stage: `next` = the client's return target; it connects back to the client while
  handling the message (the client accepts on its sidecar's ring-inbound and checks the
  PeerID).
- Other stages: `next` = `/p2p/<next worker PeerID>`; each dials through its sidecar's
  ring proxy (`DAN-RING/1 <target>`), and the receiving sidecar prefixes
  `DAN-P2P/1 <authenticated PeerID>` so the next worker can check `previous_peer`.
If any step fails, the client destroys what it created and the route closes (no reroute).

### Step 8 — Token loop (`generate_loop` / `generate` in `engine/client.cpp`)

**Loop mode (default).** The client sends the token budget to the last stage
(`stream_prompt`) and the prompt to the first stage, then only reads tokens. The last
stage streams every token to the client (`client_chunk`) and feeds it straight back to
the first stage, so decoding costs no client round trip; the final token arrives as the
usual `result` frame. A single-stage route loops inside the worker
(`loop=self`), with no network hop per token at all. The loop connection is dialed on the
first token, because the first stage only learns to expect it after its own session is
set up. Stopping: budget reached, end-of-text, or the client's `cancel_request`.
`dan-client --no-loop` keeps the old per-token path for comparison.
Measured 2026-09-17 (14B on a remote RunPod GPU, relayed): 6.9 tok/s without, 19.7 with.

The per-token path, used by `--no-loop` and the older coordinator:
```text
prefill:  client ──prompt(text)──▶ A → B → C ──result(token,text)──▶ client
decode:   client ──token(id)────▶ A → B → C ──result(token,text)──▶ client   (repeat)
finish:   commit_token through every stage (keeps the session's KV consistent),
          then end_request to every stage
```
- The first stage tokenizes the prompt and runs the embedding; middle stages receive FP32
  activations (`rows × hidden`); the last stage applies the head and **greedy** sampling
  and returns the token id + text piece.
- In ring mode intermediate activations never touch the client
  (`--require-direct` enforces it; `client_activations_received=0`).
- Chat keeps one session (the model sees the whole conversation through KV cache).
  Prompts use the Qwen chat template. `/new` resets the session. When the context fills,
  the client resets and retries the message.

### Step 9 — Ending
Closing the client's control connection releases every lease (weights stay cached).
An idle served route is dropped after 10 minutes; a connection that has not reserved
is dropped after 60 s.

---

## 9. Algorithms

### 9.1 Planner (`engine/planner.cpp`)
Input: model tensor sizes, each candidate's offered MiB (≤ 8), context, sessions,
minimum stage count.
- **Fit test (`stage_fits`)** for layers `[b, e)` on a worker offering `M`:
  `reserve = max(1 GiB, 15% of M)`;
  `weights = bytes of all tensors in those layers (+ embedding/head where owned)`;
  `kv = context × sessions × (e−b) × (hidden/heads) × kv_heads × 2 × 2 bytes` (F16 K and V);
  fits iff `weights ≤ M − reserve` and `kv ≤ M − reserve − weights`.
- **Search:** for `count = min_stages … candidates`, try orderings of `count` candidates
  (depth-first); split layers so each stage's share ≈ its share of the remaining offered
  memory, trying cut points nearest that target first, backtracking on misfit. Return the
  **first plan with the fewest stages**.
- Only dense Qwen2 GGUFs are accepted (`compatible_dense_qwen2`).
- **Cache-aligned plans (`plan_from_cache`).** Workers report the ranges they already have
  on disk in their greeting (`cached=<sha>:<begin>-<end>`, scanned from the cache folder, so
  a restart still knows). If those ranges tile 0..layers with each stage fitting its worker,
  and in no more stages than the ordinary plan, the client uses that split: longest range
  first, so hops stay few. Measured 2026-09-17 (14B across two machines): 14 s to load from
  cache versus 201 s when the split shifted by one layer and both sides re-downloaded.
- The same code is used by the client, by workers (checking a reservation), and by the
  older coordinator. It matched the previous implementation on 3,000 random cases.

### 9.1b Speculative decoding (`--speculate`)
The first stage also loads the smallest model the client offers that the worker has in its
own catalog (`draft_sha256` in the reservation). Each round the draft proposes the next
three tokens, the real stage verifies all four positions in one batch, and every correct
guess is a token committed without another pass; the first wrong guess ends the round, and
the stage's KV is truncated by the next frame's lower position (implicit rollback). The
draft mirrors the session's create/reset/destroy and prompts, so it continues the same
text, and it catches up by one token when every proposal was accepted. The acceptance rule
lives in `provider_owned/speculation.hpp` and is unit-tested. Measured 2026-09-18 (14B on
a remote GPU): 19.7 → 36.8 tok/s, 49% of proposals accepted, 2.46 tokens per round, same
text as plain decoding.

**Split routes.** The first stage drafts when the token comes around the ring, runs all
positions as one batch, and appends the guessed token ids after the activations
(`rows - 1` ids, 4 bytes each, only on `speculative_activation`). Middle stages pass that
tail through untouched; the last stage reads it, applies the same acceptance rule, streams
the committed tokens and sends the next one around the ring. The first stage learns how
many were accepted from the next token's position, and its draft catches up when all were
(also right before a commit or the next prompt, since no further round may follow).

The draft follows the conversation exactly: create/reset/destroy, prompts, end of request
and the final-token commit are mirrored to it, it forgets its sessions whenever the route's
client disconnects, and a reused stage re-attaches its draft for the next route.

Measured 2026-09-18, Qwen2.5-14B Q8 split across the owner's RTX 2070 (layers 0–10, draft
0.5B) and a RunPod RTX 3090 (11–47), relayed through Oracle (73–135 ms):

| Mode | tok/s |
|---|---|
| per-token client loop (`--no-loop`) | 9.7 |
| loop mode | 9.6 (the client sits beside stage 0 here, so the loop saves nothing) |
| loop + `--speculate` | 17.1–17.9 (first answer), 13.0–13.1 (follow-up) |

127 rounds: 44% of guesses accepted, 2.31 tokens per trip around the ring.

**Output is not always byte-identical to plain decoding.** Verifying several positions in
one batch gives slightly different floating-point results than one at a time (documented in
`reference/design/speculative-decoding.md`), so a near-tie can flip. Tested 2026-09-18 on
1.5B split in two with a 0.5B draft: one of three outputs diverged at one word
("capital of France" vs "capital of Italy"); the single-worker and split speculative paths
produced exactly the same text, so the difference is the batching, not the ring.
`scripts/Test-DAN-Placement.ps1 -DraftManifest <smaller model>` runs this test.

### 9.2 Lease state machine (`engine/include/provider_owned/lease.hpp`)
```text
AVAILABLE ──reserve (first wins)──▶ RESERVED ──assign_stage (same stage)──▶ LOADING ──▶ SERVING
    ▲            expires after lease_ms ◀┘                                             │
    └──────────── release_route(route_id) / control connection closed ◀────────────────┘
```
One lease per worker. No expiry while loading. Every transition names its `route_id`.

### 9.2b Latency-aware placement
The sidecar times each capability query and records whether the peer is reached directly or
through a relay, and reports both in the candidate list. The client orders candidates by
that link cost (25 ms steps, a relayed link counting one step worse), then by offered
memory. Every token crosses these links, so a close direct worker beats a slightly larger
distant one; cache-aligned plans still win over both, because a download costs minutes.

### 9.3 Direct-first dialing (`sidecar/network.go`, `dialer`)
To reach a PeerID: reuse an existing connection (direct or relayed) → try addresses given
explicitly → try known addresses (half the dial timeout) → DHT `FindPeer`. If only a
relayed ("limited") connection exists, wait up to `-direct-wait` (5 s) for hole punching
(DCUtR) to produce a direct one; if that fails, remember it for 10 minutes and use the
relay immediately next time. Every stream logs `path=direct|relay`, transport, bytes,
and duration.

### 9.4 NAT traversal
- **Reaching DHT clients.** Every home node runs as a DHT client, so no routing table lists
  it and a peer lookup can fail once its provider record's addresses age out. The dialer
  then dials `<relay>/p2p-circuit/p2p/<peer>` through the relays it already uses, which is
  how a CGNAT node stays reachable.
- **Advertisement refresh.** Each model's provider record is re-published every third of
  its validity, and each call is bounded by that interval: a provide that hangs (no DHT
  server reachable) must not stop later refreshes, or the node silently disappears from
  discovery.
- Home nodes: `-reachability private`, static relay = the network node. AutoRelay keeps a
  renewed reservation and advertises `/p2p-circuit` addresses. Configured relay addresses
  are stored permanently (AutoRelay only uses relays with a *public* address; the VPS
  announces its public IP with `-announce`).
- Hole punching (DCUtR) upgrades relayed connections to direct ones when possible.
- IPv6: nodes also listen on `/ip6/::/…` (TCP + QUIC). A global IPv6 address works
  directly if the router allows inbound (the owner's router does not).
- Bootstrap/relay addresses may use DNS: `/dns4/…`, `/dns6/…`, `/dns/…` (PeerID still
  authenticates).
- The network node serves as relay with limits: 4096 MiB / 2 h per relayed connection
  (`-relay-limit-mib`, `-relay-limit-duration`).

### 9.5 Range-backed weights (`engine/range_model.cpp`)
Workers never download whole models. Given its stage, a worker reads the GGUF header,
computes the byte ranges of its tensors, and downloads only those (HTTP `Range`, `curl`
with retries) into a **sparse file** the size of the full GGUF. Every response must carry
the manifest's pinned revision (`x-repo-commit`) and full-file SHA-256 (`x-linked-etag`,
Hugging Face) and the expected `Content-Range`. Each stored range's SHA-256 is recorded in
an index next to the cache and checked when the cache is reused across restarts and routes.
(The full file is never downloaded, so its hash is trusted from the pinned revision.)

### 9.6 Stage computation (`patches/llama-provider-owned.patch`)
llama.cpp (pinned commit `95ef7fc16054e63b427a3ef00188e055ef7586d8`) is patched so a
model can be loaded with `dan_stage_start/dan_stage_end`: only those layers (plus
embedding on the first stage and norm/head on the last) are allocated, and a middle stage
can take FP32 embeddings as direct input. Each session is a llama.cpp sequence with its
own KV cache. The patch hash is part of the runtime ABI.

### 9.7 Worker dashboard feed
The sidecar writes `network-status.json` every 2 s (PeerID, relay count, public IPv6,
connected peers, active DAN streams with path/transport). The worker's dashboard thread
reads it each second, adds lease/load events, throughput samples, totals and uptime,
and redraws in place (`ui/node_dashboard.cpp`; ASCII fallback; compact under 60 columns;
one status line per change when output is redirected).

---

## 10. Wire protocols

### 10.1 DAN frames (C++ ⇄ C++, over loopback + sidecar tunnels)
`engine/include/provider_owned/protocol.hpp`. 48-byte big-endian header + payload
(≤ 64 MiB):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `0x44414e31` ("DAN1") |
| 4 | 2 | version `2` |
| 6 | 2 | type |
| 8 | 8 | session |
| 16 | 8 | request |
| 24 | 4 | position |
| 28 | 4 | rows |
| 32 | 4 | cols |
| 36 | 2 | dtype (`0` none, `1` f32le) |
| 38 | 2 | reserved (must be 0) |
| 40 | 8 | payload size |

Types: `error 0, create_session 1, reset_session 2, destroy_session 3, prompt 4, token 5,
activation 6, result 7, end_request 8, ack 9, metrics 10, shutdown 11, commit_token 12,
commit_activation 13, cancel_request 14, client_result 15, provider_available 16,
assign_stage 17, stage_ready 18, unload_stage 19, speculative_activation 20, rollback 21,
prompt_chunk 22, stream_prompt 23, client_chunk 24, reserve 25, release_route 26`.

Text payloads (newline-separated `key=value`):
- `provider_available`: `id gpu vram_mib ring state abi max_context max_sessions model…`
- `reserve` / `assign_stage` (serve mode): `route_id model_sha256 begin end context sessions [lease_ms]`
- `create_session` (ring): `next [previous_peer]`
- `release_route` / `stage_ready`: the route_id

### 10.2 Sidecar protocols (libp2p)
| Protocol | Use |
|---|---|
| `/dan/transport/1.0.0` | Control tunnel: client forward → worker `-inbound`. |
| `/dan/ring/1.0.0` | Ring tunnel: worker ring proxy → next worker / client `-ring-inbound`. |
| `/dan/capabilities/1.0.0` | One uvarint-delimited protobuf request/reply (`sidecar/capabilities/capabilities.proto`): protocol version, worker id, ABI, data protocols, device, total/offered memory, limits, models (+ cached ranges, hint only), state, assignment. A status file older than 15 s reads as OFFLINE. |
| kad-dht with prefix `/dan` | Private DHT (not IPFS). |
| circuit v2, DCUtR, AutoNAT, identify | Standard libp2p. |

### 10.3 Local text lines (loopback only)
- `DAN-P2P/1 <PeerID>\n` — written by a sidecar before an inbound stream's bytes; tells
  the worker/client which authenticated peer is on the other side.
- `DAN-RING/1 <target>\n` — written by a worker to its ring proxy to name the next hop.
- `DAN-CANDIDATES/1 <sha256>\n` → `SELF`, `RETURN`, `CANDIDATE …`, `END` (or `ERR …`).

---

## 11. Security model

**What is protected**
- All traffic between machines is libp2p-encrypted and authenticated (Noise/TLS, QUIC).
- Identities are keys; PeerIDs are checked on every control, ring, and return connection.
- Workers accept ring input only from the `previous_peer` the client named; ring
  addresses must match the authenticated PeerID.
- Workers download only from their own catalog URLs, pinned to a revision and file hash.
- Workers and clients listen on loopback only; the sidecar is the only network listener.
- The relay is rate/size limited. `govulncheck` gates every package build.

**What is NOT protected (known, accepted for the friends beta)**
- **Honesty:** a node can return wrong results or lie about its GPU. No verification.
- **Sybil/eclipse:** Kademlia's weakness (GO-2024-3218, no fixed version).
  `scripts/collect_go_licenses.ps1` accepts exactly that advisory with a warning; any
  other reachable vulnerability fails the build.
- **Privacy:** the first stage sees the prompt text, the last sees the output, middle
  stages see activations (which can often be inverted). Transport encryption does not
  help because nodes must compute on the data. Practical next step: run the embedding
  and head on the user's own PC. (Encrypted computation — HE/MPC — is far too slow today.)
- **Any authenticated peer** can take a worker's single lease (`-allow-any`).
- **Local processes** are trusted (loopback ports, `DAN-P2P/1` lines).
- A worker exits when its launcher disappears (parent-process check) or 3 s after a stop
  signal, so a killed `dan-provider` never leaves a worker holding GPU memory.
- **Installer** is unsigned (Windows SmartScreen may warn).

---

## 12. Build, package, deploy

### Build (Windows, Visual Studio 2026, CMake, CUDA 13.3, Go 1.25.x)
```powershell
# CPU dev build with tests
cmake -S . -B build-client -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=D:/Desktop/DAN/build/provider-owned-v0/llama.cpp
cmake --build build-client --config Release --parallel 8
ctest --test-dir build-client -C Release

# Portable CUDA build (installer): any AVX2-era CPU, RTX 20xx–50xx
cmake -S . -B build-cuda -G "Visual Studio 18 2026" -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=… `
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF "-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;120"
cmake --build build-cuda --config Release --parallel 8 --target dan-provider dan-stage-worker dan-client dan-provider-owned-coordinator
# (~45 min, mostly CUDA kernels)

# Sidecar (from sidecar/, GOTOOLCHAIN=go1.25.7)
go build -o ../build-client/sidecar/dan-sidecar.exe .
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o ../build-client/sidecar/dan-sidecar-linux-amd64 .
go test ./...
```
llama.cpp: clone, check out `95ef7fc1…`, apply `patches/llama-provider-owned.patch`
(`scripts/build_provider_owned.ps1` does this).

### Installer
```powershell
.\scripts\build_installer.ps1 -Bootstrap /ip4/82.70.213.202/tcp/4001/p2p/12D3KooWGDp2QL2wBSwbE6jyzU8CH13KT8Tvmzeq4caErzVvuU35
# → build\installer\DAN-Setup-1.1.0.exe (+ .sha256)
.\scripts\build_installer.ps1          # no -Bootstrap: "local test" build that runs its own network node
```
It calls `package_provider.ps1` (stages files, runtime DLLs, licenses, `govulncheck` gate),
then Inno Setup (`installer/dan.iss`). Install is per-user
(`%LOCALAPPDATA%\Programs\DAN`), no admin, with Start-menu/desktop shortcuts and an
uninstaller. Users need only Windows 10/11 x64, an AVX2 CPU, and an NVIDIA RTX 20-series
or newer GPU with a driver recent enough for CUDA 13 (about R580+). Everything else
(CUDA runtime, MSVC runtime, llama.cpp, sidecar) is bundled.

### Public network node (Oracle Cloud, Always Free)
- VM `dan-network`, region `il-jerusalem-1`, `VM.Standard.E2.1.Micro` (1 GB RAM,
  Ubuntu 24.04, x86-64), public IP `82.70.213.202`.
- Service `dan-infra` (`deploy/dan-infra.service`, `PUBLIC_IP` set), binary
  `/opt/dan/dan-sidecar`, identity `/var/lib/dan/infra.key`, log `/var/log/dan/infra.log`.
- Ports TCP+UDP 4001 open in the Oracle security list **and** iptables
  (`/etc/iptables/rules.v4`).
- SSH: `ssh -i ~/.ssh/dan_oracle ubuntu@82.70.213.202` (key on the owner's PC).
- Uses ~24 MB RAM. **If `infra.key` is lost, the PeerID changes and every installer must
  be rebuilt.**

---

## 13. Testing

| Layer | Command | Checks |
|---|---|---|
| C++ unit | `ctest --test-dir build-client -C Release` | protocol, range model, formation, client, route, lease, placement, UI, platform (13 tests). Run `provider_owned_formation_test` in Debug too (asserts). |
| Go unit | `cd sidecar; go test ./...` | tunnels, relay/NAT by PeerID, IPv6, DNS bootstrap, capabilities, discovery, net status. |
| Static ring | `scripts/Test-DAN-Client-Static.ps1 -Transport hub\|ring\|libp2p [-WrongPredecessor]` | identical output vs baseline; wrong PeerID refused. |
| Placement | `scripts/Test-DAN-Placement.ps1 -Transport direct\|libp2p [-LeaseChecks] [-Race]` | leases, races, identical output. |
| Discovery | `scripts/Test-DAN-Discovery.ps1` | DHT end to end incl. killing the bootstrap and a worker. |
| NAT rehearsal | `scripts/Test-DAN-NatRehearsal.ps1 -BuildDir build-client -OutDir … -BaselineDir build-client\results\baseline` | infra + 3 `dan-provider` nodes with `simulate_nat`; every DAN stream must be relayed; output identical; dashboard + scripted chat. Last run: 36/36 relayed, PASS. |
| Real internet | install `DAN-Setup-1.1.0.exe`, run Node + Chat (`Start-DAN-Client.ps1 -SimulateNat` forces the relay) | done 2026-09-17 through the Oracle node. |
| Two real GPUs | a Linux GPU pod (build on the pod: clone llama.cpp at the pinned commit, apply the patch with LF endings, `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch>`), `dan-provider --config` with `network=dht`, then `dan-client --min-stages 2` from the Windows install | done 2026-09-17 (RunPod, see §2). |

Baseline output reports: `build-client/results/baseline`. Test model:
Qwen2.5-0.5B-Instruct Q4_K_M (24 layers, hidden 896).

### Measured numbers (for orientation)
- NAT rehearsal (one PC, CPU, 3 stages, all relayed): metadata 6 s, capabilities 5 s,
  first route setup 15 s, later routes 5 s, decode ≈ 32–37 tok/s.
- Real GPU, one stage, chat on same PC: 68–145 tok/s, first token 14–78 ms.
- Real internet relay (Oracle, Jerusalem): 13–20 tok/s on very short answers.
- Loop mode, 14B alone on the remote GPU: 19.7 tok/s (6.9 tok/s with `--no-loop`),
  first token 0.3 s.
- Two GPUs on different networks via the relay (RunPod ↔ owner's PC): Qwen2.5-1.5B 6.5 tok/s;
  Qwen2.5-14B Q8 4.6–5.2 tok/s, first token 0.4–0.8 s, 4 min to download and load both
  ranges. RunPod's network blocked hole punching, so the ring used the relay.
- Older coordinator-path results (32B split, WAN speculative decoding up to 6.3×) are in
  `TESTS_AND_STATS.md`.

---

## 14. Limitations and roadmap

### Current limits
- One client route per worker (`max_sessions` 1); no queueing.
- A GPU leaving mid-chat fails the chat (no failover); the client must start over.
- Placement ignores latency and bandwidth; picks by memory.
- Manifests set `context_size` 512 → chat conversations reset often.
- Packages ship all four model configs and chat picks the largest that fits; there is no
  `/model` override yet, and a node still serves only models in its own catalog.
- Dense Qwen2 GGUF only, greedy sampling only.
- One public network node; no auto-update; Windows-only GPU installer; unsigned.
- Speculative decoding is opt-in (`--speculate`) because batched verification can flip a
  near-tie. Pipelining exists only on the older coordinator path.

### Roadmap (roughly in order)
1. **Real second machine test** (friend or laptop on a phone hotspot).
2. **Automatic model choice:** all manifests in the installer catalog; chat picks the
   largest model the online network can place; `/model` command.
3. **Privacy step:** embedding + head on the user's PC.
4. **More entry points:** several independent network nodes; public nodes volunteering as
   relays; DNS name for the bootstrap.
5. **Robustness:** failover to a spare mid-request, multi-client workers, queueing,
   latency-aware placement, larger contexts.
6. **Trust:** result verification (spot checks against a trusted copy), reputation,
   Sybil resistance.
7. **Incentives:** usage accounting, then payments/crypto.
8. **Governance:** how the network agrees on models and protocol versions.
9. Linux GPU installer, signed installer, auto-update, port speculative decoding to the
   decentralized path.

### Where decentralization stands
Done: no coordinator; DHT discovery; client-side planning; direct GPU-to-GPU data path;
worker-owned catalogs and decisions; key-based identities.
Still central: the single entry node (existing nodes survive its loss, new ones can't
join, CGNAT users lose their relay); the installer and its built-in address; the model
list. Not solved: trust, incentives, governance.

---

## 15. History and older paths (why the repo contains more than the above)

1. **Legacy path** (`legacy/`): whole-model providers and llama.cpp RPC groups under a
   poll-based C++ coordinator. Proved GPUs could participate, but the coordinator held the
   full model. Reference only. See `reference/design/legacy-path.md`.
2. **Provider-owned + coordinator** (`engine/coordinator.cpp`, `dan-api-gateway`): workers
   own their layers; a metadata-only coordinator formed replicas, routed activations,
   ran speculative decoding/pipelining, and exposed an HTTP API; Tailscale or libp2p
   transport; Windows v1.0.1 packages. Proved 32B aggregate-VRAM inference and WAN speedups.
   Still builds; not the direction (a single trusted operator).
3. **Decentralized path (current, 2026-09-16/17):** commits `2bf5bd6` (shared client +
   `dan-client`) → `408ad71` (dynamic placement, direct ring) → `6955109` (DHT discovery,
   capabilities) → `9d9212f` (NAT/relay) → `22bcaec` (WAN beta ops, `dan-provider` DHT
   mode) → `47c6446` (dashboard, chat, IPv6, DNS) → `c9dc9e2` (installer).

### Environment pitfalls (Windows dev machine)
- PowerShell 5.1: native stderr becomes an error under `$ErrorActionPreference='Stop'`
  when output is redirected → wrap native calls with `'Continue'` and check `$LASTEXITCODE`.
- Splatting arrays mangles `--flags` → pass a plain array.
- `Start-Process -ArgumentList` does not quote → quote paths with spaces
  (the owner's profile is `C:\Users\Halakim Family`).
- `$x = if (…) { @(…) }` unwraps one-element arrays → wrap the whole expression in `@()`.
- Advanced scripts (`[CmdletBinding()]`) reject piped input; PowerShell piping to native
  programs adds a UTF-8 BOM and CRLF.
- Git Bash rewrites `/ip4/...` arguments → `MSYS_NO_PATHCONV=1`.
- The CUDA build dir must be configured at the current path (moving the repo breaks it).

---

## 16. Document map

| Document | Contents |
|---|---|
| `docs/PROJECT.md` | **This file** — start here. |
| `docs/reference/operations/wan-beta.md` | Operating the WAN beta: VPS, friend package, client, timeouts, checks. |
| `docs/reference/operations/friend-readme.txt` | README shipped to friends. |
| `docs/ARCHITECTURE.md` | Detailed architecture incl. older paths. |
| `docs/PROGRESS.md` | Historical progress log and the old coordinator-path beta gate. |
| `docs/TESTS_AND_STATS.md` | Physical test evidence (mostly coordinator path). |
| `docs/OPERATIONS.md` | Operating the coordinator-path service. |
| `docs/reference/design/*` | Deep dives: p2p transport, range storage, replica formation, runtime v1/v2, speculative decoding, decisions, legacy. |
| `docs/reference/tests/*` | Test receipts. |
