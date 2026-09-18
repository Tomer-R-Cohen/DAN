# DAN — Project Guide

**Read this first.** This is the single source of truth for what DAN is, why it exists,
how it works, where it stands, and what comes next. It is written for people and for
coding agents. Other documents hold detail and history; when they disagree with this
file, this file wins (and the other file should be fixed).

*Last updated: 2026-09-18.*

---

## 1. What DAN is, in one minute

DAN (**D**ecentralized **A**I **N**etwork) runs large language models across ordinary
people's GPUs over the internet.

A model too big for one GPU is cut into contiguous blocks of transformer layers
("stages"). Each volunteer GPU loads only its block. A user's prompt flows through the
GPUs in order, like a relay race; the last GPU produces each new token, sends it to the
user and straight back to the first GPU for the next step:

```text
your PC ──prompt──▶ GPU A (layers 0–9) ──▶ GPU B (layers 10–17) ──▶ GPU C (layers 18–23)
   ▲                    ▲                                                   │
   └──── tokens ────────┴──────────────────── next token ◀──────────────────┘
```

There is **no central server that schedules work**. Each user's own client finds GPUs,
picks the largest model they can run together, plans the split, reserves the GPUs and
links them into a ring itself. A small public "network node" only helps machines find and
reach each other; it never sees a plan and never decides anything.

**Why:** big models need more memory than consumer GPUs have, and renting data-center
GPUs is expensive and centralized. Many idle gaming GPUs, combined, can run models that
none of them could run alone — if the system needs no central operator, works behind
home routers, and is a one-click install.

---

## 2. Status at a glance

| Area | State |
|---|---|
| Split inference across GPUs (provider-owned stages) | **Works.** Qwen2.5-32B split across RTX 2070 + RTX A5000 (2026-09-08); Qwen2.5-14B across RTX 2070 + RunPod GPUs (2026-09-17/18). |
| No coordinator: discovery, planning, reservation, ring | **Works.** Private DHT, client-side planner, worker leases, direct GPU-to-GPU ring. |
| Home networks / NAT / CGNAT | **Works.** Relay + hole punching, relay fallback for unlisted peers; forced-relay rehearsal and real internet. |
| Public network node | **Running** on Oracle Cloud (Always Free), `82.70.213.202`. |
| One-click install (Windows) | **Works.** `DAN-Setup-1.1.0.exe` rebuilt 2026-09-18 with everything below; a node and a chat run from its files through the public network. |
| Node dashboard (TUI) and chat | **Works.** |
| Client out of the token loop ("loop mode") | **Works, default.** 14B on a remote GPU: 6.9 → 19.7 tok/s. |
| Automatic model choice | **Works.** Chat picks the largest model the online GPUs can run. |
| Cache-aware and latency-aware placement | **Works.** Reuses layers already on disk (201 s → 14 s to be ready); prefers close, direct links. |
| Speculative decoding (`--speculate`) | **Works, opt-in.** One GPU: 19.7 → 36.8 tok/s. Split across two networks: 9.6 → 17.4 tok/s. |
| Persistent self-forming replicas (§9.8) | **Works, off by default** (`replica=auto`). Providers form a replica with no client, keep it, and clients chat on it: first token ~0.1 s instead of route setup + warm-up (~6 s). Tested locally and once over the internet (2070 + RunPod 3090, 14B). Next: speed-aware formation (§14, 1a). |
| A real friend's PC | **Not yet tested** (RunPod pods have stood in). |
| Payments, reputation, Sybil resistance, verification, failover, privacy | **Not started** (deferred, §3). |

---

## 3. Goals, non-goals, and standing rules

### Goal (current phase: "friends beta")
A friend behind any home router (NAT or CGNAT) installs DAN, double-clicks, and their GPU
is used for inference — no port forwarding, no accounts, no central scheduler.

### Design principles (all implemented; do not break them)
1. **Infrastructure has no scheduling authority.** The public node is DHT bootstrap +
   relay + reachability checks. It never plans, reserves, or routes work.
2. **Whoever needs a route plans it**, with the same planner code everywhere: a replica
   owner (a provider node, for its own replica only) or, as a fallback, the chat client.
   Nobody plans for the network as a whole.
3. **Workers decide for themselves.** A worker accepts work only if it is idle, the model
   is in *its own* catalog, and the stage fits *its own* memory.
4. **Clients never supply download URLs.** A client names a model (and a draft model) only
   by SHA-256; the worker resolves it in its local catalog.
5. **PeerIDs come from authenticated connections**, never from what a peer claims.
6. **No raw TCP next hops in libp2p mode.** A worker reaches another stage only through its
   local sidecar, by PeerID.
7. **Bootstrap nodes are entry points only.** Once routing tables fill, losing them does
   not stop discovery (tested).
8. **The public relay is resource-limited** (default 4 GiB / 2 h per relayed connection).
9. **Reuse battle-tested libraries** (go-libp2p, go-libp2p-kad-dht, llama.cpp) instead of
   inventing networking or inference code.

### Explicitly deferred (do not implement unless asked)
Payments/crypto, reputation, Sybil resistance, consensus, result verification, failover
mid-request, multi-client workers. (Also see the older working rule in `PROGRESS.md`: no
blockchain/marketplace work until requested.)

### Rules for agents working in this repo
- **Never run `git commit`.** The owner commits. Prepare changes and suggest a message.
- Output to the owner must be **short, clear, simple** — no filler.
- Inspect code before changing it; build and test after; update docs (this file first).
- Windows is the main dev machine; see §15 for pitfalls that have bitten us.

---

## 4. Glossary

| Term | Meaning |
|---|---|
| **Stage** | A contiguous range of transformer layers `[begin, end)` loaded by one worker. The first stage also owns the token embedding; the last owns the final norm, output head, and sampling. |
| **Route / ring** | The stages serving one client: client → A → B → C, with C feeding each token back to A (loop mode) and streaming it to the client. |
| **Loop mode** | Decoding without the client in the loop: the last stage feeds each token straight back to the first. Default. |
| **Worker** | `dan-stage-worker`: the C++ process that loads a stage on a GPU and computes. |
| **Node** | A machine running `dan-provider` (= launcher + worker + sidecar). |
| **Sidecar** | `dan-sidecar`: the Go libp2p process next to every worker and client. Owns identity, DHT, NAT traversal, encryption, tunnels, link measurement. |
| **Client** | `dan-client`: picks the model, plans a route, reserves workers, reads tokens. |
| **Network node / infra node** | `dan-sidecar -infra` on a public machine: DHT bootstrap + relay. |
| **PeerID** | A node's libp2p identity (hash of its public key), e.g. `12D3KooW…`. |
| **Catalog** | A worker's local list of model manifests it is willing to serve. |
| **Manifest** | `config/provider-owned-*.json`: a pinned GGUF (URL or `hf_repo`+file, revision, SHA-256, context; layers/hidden optional — read from the GGUF header when absent). |
| **Lease** | A worker's single reservation slot; the first client to reserve wins. |
| **Draft model** | A small model the first stage runs to guess the next tokens (speculative decoding). |
| **Runtime ABI** | `dan-stage-v1/f32le/<patch hash>`; stages combine only if equal. |
| **Provider-owned** | Workers own and load their weights; nothing central ever holds the full model. |
| **Replica** | A complete, linked, warmed-up route (all layers, ring and loop connected) that stays up after clients leave. Its ID is the route ID the members' leases carry. |
| **Replica owner** | The provider node whose formation attempt won the leases. It is always the first stage, holds the members' leases, advertises the replica and is its only front door. Authority over that replica only. |
| **Session** | One conversation on a replica: a KV sequence on every member, created and destroyed by the owner. Weights outlive sessions; KV never outlives its session. |

---

## 5. System overview

### Processes on each machine

```text
FRIEND'S PC (GPU node)                                  PUBLIC NETWORK NODE (Oracle VM)
┌───────────────────────────────────────────────┐      ┌──────────────────────────────┐
│ DAN Node shortcut → installer/DAN.ps1 node     │      │ dan-sidecar -infra           │
│   dan-provider  (launcher: config, GPU check;  │      │   DHT server (bootstrap)     │
│                  on Linux it execs the worker) │◀────▶│   circuit-v2 relay           │
│     ├─ dan-sidecar  (DHT client, NAT, relay    │ libp2p│   AutoNAT service            │
│     │   reservation, tunnels, capabilities,    │      │ no worker, no scheduling     │
│     │   model advertisements, net status)      │      └──────────────────────────────┘
│     └─ dan-stage-worker (serve mode: leases,   │
│         stage + optional draft model, decode   │
│         loop, node dashboard)                  │
└───────────────────────────────────────────────┘

USER'S PC (chat)
┌───────────────────────────────────────────────┐
│ DAN Chat shortcut → DAN.ps1 chat               │
│   Start-DAN-Client.ps1                         │
│     ├─ dan-sidecar (own identity, candidate    │
│     │   API with link measurement, ring return)│
│     └─ dan-client --discover … --chat          │
│         (every installed manifest)             │
└───────────────────────────────────────────────┘
```

The C++ side (worker, client) **never speaks libp2p**. It talks plain TCP to **loopback**
ports that the local sidecar exposes; the sidecar carries the bytes over encrypted,
authenticated libp2p streams. This keeps the inference code simple and puts all
internet-facing code in one Go process built on go-libp2p.

### Components and languages

| Component | Language | Source | Role |
|---|---|---|---|
| `dan-stage-worker` | C++23 + patched llama.cpp | `engine/stage_worker.cpp` | Loads a stage (+ draft), computes, serves leases, runs the decode loop and speculation, feeds the dashboard. |
| `dan-client` | C++23 | `engine/dan_client.cpp` + `client.cpp`, `placement.cpp`, `planner.cpp`, `replica_owner.cpp` | Model choice, placement, token streaming, chat; `--replica` uses persistent replicas; `--form` is a node's replica owner. |
| `dan-provider` | C++23 | `engine/provider_launcher.cpp` | Friend-facing launcher: reads config, detects the NVIDIA GPU, starts sidecar + worker (+ replica owner with `replica=auto`). |
| `dan-sidecar` | Go (go-libp2p v0.48.0, kad-dht v0.42.0) | `sidecar/*.go` | Identity, DHT, NAT, relay, tunnels, capabilities, candidate API, link measurement, net status. |
| Node dashboard | C++23 | `ui/node_dashboard.cpp`, `ui/provider_ui.cpp` | Terminal UI. |
| Installer | Inno Setup + PowerShell | `installer/`, `scripts/build_installer.ps1` | `DAN-Setup-x.y.z.exe`. |
| Platform layer | C++23 | `platform/` | Windows/POSIX process, GPU, console, hashing helpers. |

---

## 6. Repository map

```text
CMakeLists.txt            C++ build; links patched llama.cpp from DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR
engine/                   MAIN PATH (provider-owned)
  stage_worker.cpp        dan-stage-worker: static/generic/serve modes, ring, leases, decode
                          loop, speculative rounds, draft model, dashboard feed
  client.cpp              shared client: InferenceClient, generate_loop (loop mode), generate
                          (per-token path), ring setup, sessions
  dan_client.cpp          dan-client CLI (--provider / --candidate / --discover, --chat,
                          --speculate, --no-loop, --replica, --form), model choice
  replica_owner.cpp       persistent replica owner (--form): formation loop, link checks,
                          warm-up, front door, request relay, keepalive, status file
  placement.cpp           client-side placement: greet, rank by link, pick model, plan, reserve,
                          load; candidate API parsing
  planner.cpp             plan_stages, plan_from_cache, stage_fits (shared by client, worker,
                          coordinator)
  range_model.cpp         GGUF metadata over HTTP ranges + sparse partial downloads
  manifest.cpp            manifest JSON loader (full or hf_repo form)
  provider_launcher.cpp   dan-provider
  coordinator.cpp         OLDER: metadata-only coordinator (still builds, not the direction)
  include/provider_owned/ protocol.hpp (frames), route.hpp (ring/loop route), lease.hpp
                          (reservations, draft request), planner.hpp, placement.hpp,
                          formation.hpp (greeting/assignment text, cached ranges),
                          speculation.hpp (acceptance rule), replica.hpp (owner options,
                          front-door protocol), client.hpp, manifest.hpp,
                          range_model.hpp, fair_queue.hpp
  tests/                  C++ unit tests + lease_integration.py
sidecar/                  Go module "dan/sidecar"
  main.go                 flags, host/relay/DHT startup, tunnels, ring proxy
  network.go              host options, NAT/relay, dialer (direct-first, relay fallback), bridge
  discovery.go            DHT, capabilities protocol, advertisements, candidate API, RTT
  replica.go              link probes, replica advertisement + live query, DAN-REPLICAS,
                          return-target bridging
  netstatus.go            -net-status-file writer (dashboard input)
  capabilities/           capabilities.proto + generated Go
  cmd/dan-api-gateway/    OLDER: OpenAI-style HTTP gateway for the coordinator path
ui/                       provider_ui (runner + classic view), node_dashboard, admin_dashboard
platform/                 platform.hpp + windows/posix implementations
installer/                DAN.ps1 (launcher for shortcuts), dan.iss (Inno Setup script)
scripts/                  build, package, test, and launcher scripts (§12, §13)
config/                   model manifests provider-owned-qwen2.5-{0.5b,1.5b,14b,32b}*.json
                          (+ legacy examples)
deploy/                   dan-infra.service (systemd); prometheus/ is for the older path
patches/                  llama-provider-owned.patch (applied to llama.cpp 95ef7fc1)
legacy/                   OLDEST: whole-model providers + llama.cpp RPC path (reference only)
docs/                     this file + detail docs (§16)
AGENTS.md, CLAUDE.md      short rules for coding agents, pointing here
```

Build outputs (git-ignored): `build-client/` (CPU dev build, tests, sidecar binaries in
`build-client/sidecar/`), `build-cuda/` (portable CUDA build for the installer),
`build/provider-owned-v0/llama.cpp` (patched llama.cpp source), `build/gateway/`,
`build/installer/`.

---

## 7. How a node joins (startup sequence)

`DAN Node` shortcut → `powershell -File DAN.ps1 node` → `dan-provider --config <app>\config\provider.conf`.

`provider.conf` (written by `package_provider.ps1`):
```text
network=dht
bootstrap=/ip4/82.70.213.202/tcp/4001/p2p/12D3KooWGDp2QL2wBSwbE6jyzU8CH13KT8Tvmzeq4caErzVvuU35
catalog=config\provider-owned-qwen2.5-0.5b-q4km.json     (one line per model; packages
catalog=config\provider-owned-qwen2.5-1.5b-q4km.json      built from 2026-09-17 on list
catalog=config\provider-owned-qwen2.5-14b-q8.json         all four)
catalog=config\provider-owned-qwen2.5-32b-q5km.json
stage_worker=runtime\dan-stage-worker.exe
sidecar=runtime\dan-sidecar.exe
reserve_vram_mib=1536
```
Optional keys: `relay=` (default: the bootstrap nodes), `max_context=` (4096),
`max_sessions=` (1), `listen_port=` (0 = random), `device=`, `state_dir=`, `cache_dir=`,
`simulate_nat=` (tests only).

`dan-provider` then:
1. Queries `nvidia-smi`, picks a GPU, offers `total VRAM − reserve_vram_mib`.
2. Creates/loads a persistent identity (`<state_dir>\identity.key`) → PeerID.
3. Picks free loopback ports and starts **the sidecar**:
   `-dht client -reachability private -bootstrap … -relay … -inbound <control> -allow-any
    -ring-inbound <ring> -ring-proxy <proxy> -status-file worker-status.json
    -net-status-file network-status.json`
   The sidecar connects to the relay (its addresses kept permanently), waits up to 20 s
   for a relay reservation (continues with a warning if none), starts the DHT and waits up
   to 10 s for a routing-table entry, then advertises the catalog.
4. Starts **the worker in serve mode** (on Linux it `exec`s into it, so the worker *is*
   the launcher process):
   `--control-listen --ring-listen --peer-header --ring-proxy --ring-target /p2p/<own PeerID>
    --status-file --net-status-file --provider-id <PeerID> --gpu --vram-mib --cache-dir
    --ctx <max_context> --max-sessions --catalog … [--tui]`
   The worker loads nothing yet; it only advertises resources.
5. Prints "Joined the DAN network. Relay addresses: N" and the dashboard takes over.

Files on a node (`%LOCALAPPDATA%\DAN` by default): `identity.key`, `models\` (sparse
weight cache, named `<model_id>-<begin>-<end>.gguf` + `.ranges` index; the draft model as
`<model_id>-draft.gguf`), `worker-status.json`, `network-status.json`,
`logs\sidecar.log`, `logs\stage-worker.log` (readable while the node runs), and
`client\` (the chat client's separate identity and logs).

**Advertising.** The sidecar re-reads the worker's status file every second and keeps one
DHT provider record per catalog model under `dan/model/1/<gguf sha256>`, valid 5 min
(`-provide-validity`) and refreshed every third of that. Only a *fresh* status file counts
(a leftover from an earlier run must not decide what is advertised), and each refresh call
is time-bounded (a hanging provide once silently stopped all refreshes). A record only
means "this peer may serve the model"; live state comes from the capabilities protocol.
`DAN_DEBUG_ADVERTISE=1` logs every refresh.

**Stopping.** Ctrl+C or closing the window stops the node. A worker exits at most 3 s
after a stop signal even if it is blocked in a socket call, so it never keeps GPU memory.
On Windows the launcher's children live in a job object that dies with it.

---

## 8. How a chat request works, end to end

`DAN Chat` → `DAN.ps1 chat` → `Start-DAN-Client.ps1` starts the client's own sidecar
(`-dht client -reachability private -candidate-api 127.0.0.1:P -ring-inbound 127.0.0.1:R`)
and runs `dan-client --manifest <every installed model> --discover 127.0.0.1:P --chat`.

### Step 1 — Model shapes
For every manifest, `dan-client` needs every tensor's byte size, the layer count and hidden
size. The first time, it reads the GGUF **header only** via HTTP range requests
(`inspect_range_model`, all models in parallel, ~6.5 s for four) and saves the parsed index
as `<data dir>\model-index\<sha256>.index`. A manifest pins its file by full SHA-256, so
that header can never change: later runs load the index from disk in milliseconds.

### Step 2 — Discovery (sidecar)
`dan-client` sends one query for all its models:
`DAN-CANDIDATES/1 <sha256> [<sha256> …]\n` (up to 8). For each model, in parallel, the
sidecar:
1. Finds providers of `dan/model/1/<sha>` in the DHT (up to 64, `-discovery-timeout` 15 s)
   and stores their addresses.
2. Asks each one over `/dan/capabilities/1.0.0` (protobuf, `-query-timeout` 10 s),
   **timing the round trip** and noting whether the connection is direct or relayed.
   Control streams (this query, the greeting, reservations, the prompt) never wait for a
   direct path; only ring streams do (§9.4).
3. Keeps `AVAILABLE` peers, opens a **local control forward** (loopback port → that peer)
   for each, merges peers found under several models (keeping the best round trip), and
   replies:
   ```text
   SELF <client PeerID>
   RETURN 127.0.0.1:R
   CANDIDATE <PeerID> 127.0.0.1:<forward port> <offered MiB> <runtime ABI> <rtt ms> <direct|relay>
   END
   ```
`dan-client` never touches the DHT itself.

### Step 3 — Greeting (client)
The client connects to every candidate forward **in parallel**. Each worker immediately
sends a `provider_available` frame (text `key=value` lines: `id, gpu, vram_mib, ring,
state, abi, max_context, max_sessions, model=<sha>…, cached=<sha>:<begin>-<end>…`).

### Step 4 — Choosing the model and the plan (client)
Models are tried **largest first**; the first one the workers can run wins
(`model=<id>` is printed). For each model:
1. **Filter.** Keep a worker only if it is `available`, has the same runtime ABI, lists the
   model, covers the context and session limits, and its `ring` address is `/p2p/<PeerID>`
   **of the peer the connection actually authenticated** (prevents redirecting the ring).
2. **Rank.** Order by link cost (round trip in 25 ms steps, a relayed link one step worse),
   then offered memory, then PeerID; keep at most 8. Every token crosses these links.
3. **Plan** with `plan_stages` (§9.1): the fewest stages that fit.
4. **Prefer cached layers.** If the workers already hold ranges that tile the model in no
   more stages (`plan_from_cache`), use that split instead — nothing to download
   (`placement: … (cached layers) plan`).
5. **Draft model** (`--speculate` only): the smallest other offered model that the first
   stage's worker lists in its own catalog.

### Step 5 — Reservation (leases)
The client sends `reserve` to every chosen worker in parallel: `route_id` (random 32 hex),
`model_sha256`, `begin`, `end`, `context`, `sessions`, `lease_ms` (30 s, max 60 s), and for
the first stage optionally `draft_sha256`. A worker says `ack` only if its lease is free,
the model is in its catalog, the range exists, the limits fit, and `stage_fits` passes on
**its own** memory; otherwise it replies with an error (`busy`, `unknown_model`,
`limits_exceeded`, `insufficient_memory`, …). If any refuse, the client releases the
others (`release_route`), re-greets "busy" workers, waits 100–500 ms at random, and
replans (3 attempts).

### Step 6 — Loading
`assign_stage` to all in parallel (no timeout — downloads can be long). Each worker
downloads **only its own layers** from the URL in **its own** catalog (§9.5), loads them on
the GPU and replies `stage_ready <route_id>`; the lease moves `RESERVED → LOADING → SERVING`.
The first stage also loads the draft model when asked (never fatal: without it the route
just decodes one token per pass). Weights stay cached on disk and, while the same range
is reused, in GPU memory.

### Step 7 — Linking the ring (per session)
`create_session` is sent to **every stage at once**. Its payload:
```text
next=<where this stage sends its output>
previous_peer=<the only PeerID it may accept input from>
loop=<first stage's ring address | self>        (last stage only, loop mode)
```
- Last stage: `next` = the client's return target (it connects back while handling the
  message; the client checks the PeerID); `loop` = the first stage, or `self` when one
  worker is the whole route.
- Other stages: `next` = `/p2p/<next worker PeerID>`, dialed through the sidecar's ring
  proxy (`DAN-RING/1 <target>`); the receiving sidecar prefixes
  `DAN-P2P/1 <authenticated PeerID>` so the worker can check `previous_peer`.
- In loop mode the first stage's `previous_peer` is the last stage, and the last stage
  dials that loop link **in the background** as soon as its own route is set, so the first
  token finds it ready.
- Order does not matter: a stage that dials a neighbour which has not been told to expect
  it yet is refused and retries every second (within the worker's 20 s connect budget).
  Route setup therefore costs the slowest link, not the sum of all links (15 s → 5 s on
  three relayed links). Dialing the loop link *synchronously* during setup would deadlock
  — the first stage may not expect it yet — which is why it runs in the background, and a
  first token that arrives earlier waits for that dial rather than opening a second link.
If any step fails, the client destroys what it created and the route closes (no reroute).

### Step 8 — Generating
**Loop mode (default for placed routes).** The client sends the token budget to the last
stage (`stream_prompt`, `rows` = budget) and the prompt to the first stage, then only
reads. The prompt flows A → B → C; the last stage samples a token, streams it to the client
(`client_chunk`) and sends it back to the first stage as the next input; the final token
arrives as an ordinary `result`. Stopping: budget, end-of-text, or the client's
`cancel_request` (the current token ends the request). A single-worker route loops inside
the worker with no network hop per token.

**With `--speculate`** each pass commits several tokens (§9.2).

**The per-token path** (`--no-loop`, fixed `--provider` routes, the older coordinator):
the client sends each token to the first stage and receives each result itself.

In every mode: the first stage tokenizes the prompt and runs the embedding; middle stages
receive FP32 activations (`rows × hidden`); the last stage applies the head and **greedy**
sampling. Intermediate activations never touch the client (`--require-direct` enforces
it; `client_activations_received=0`). After each answer the client commits the final token
through every stage (`commit_token`) so the session's KV stays consistent, then sends
`end_request`. Chat keeps one session (the model sees the whole conversation through its
KV cache), uses the Qwen chat template, `/new` resets, and a full context resets and
retries the message.

### Step 9 — Ending
Closing the client's control connection releases every lease and clears every session,
including the draft model's (weights stay cached). An idle served route is dropped after
10 minutes; a connection that has not reserved is dropped after 60 s.

---

## 9. Algorithms

### 9.1 Planner (`engine/planner.cpp`)
Input: model tensor sizes, each candidate's offered MiB (≤ 8), context, sessions, minimum
stage count.
- **Fit test (`stage_fits`)** for layers `[b, e)` on a worker offering `M`:
  `reserve = max(1 GiB, 15% of M)`;
  `weights = bytes of all tensors in those layers (+ embedding/head where owned)`;
  `kv = context × sessions × (e−b) × (hidden/heads) × kv_heads × 2 × 2 bytes` (F16 K and V);
  fits iff `weights ≤ M − reserve` and `kv ≤ M − reserve − weights`.
- **Search:** for `count = min_stages … candidates`, try orderings of `count` candidates
  (depth-first, in ranked order); split layers so each stage's share ≈ its share of the
  remaining offered memory, nearest cut points first, backtracking on misfit. The **first
  plan with the fewest stages** wins, so earlier (closer) candidates are preferred.
- **Cache-aligned plans (`plan_from_cache`).** From each candidate's cached ranges, build a
  tiling of `0..layers` where each range fits its worker, longest range first, at most as
  many stages as the ordinary plan. Measured 2026-09-17 (14B across two machines): 14 s to
  be ready from cache versus 201 s when the split shifted by one layer and both sides
  re-downloaded.
- Only dense Qwen2 GGUFs are accepted (`compatible_dense_qwen2`).
- The same code runs in the client, in workers (checking a reservation), and in the older
  coordinator; it matched the previous implementation on 3,000 random cases.
- Not modeled: the draft model's memory (a draft that does not fit simply fails to load
  and the route runs without it).

### 9.2 Speculative decoding (`--speculate`)
**Idea.** A small draft model guesses the next 3 tokens; the real model checks all 4
positions (the current token + 3 guesses) in **one** batch. Sample *i* is what really
follows the first *i+1* inputs, so sample 0 is always right and every later sample is
right only while the guess in front of it matched. The commit stops at the first wrong
guess (`provider_owned/speculation.hpp`, unit-tested). A wrong guess costs only the draft's
time; the stages drop the extra KV when the next frame arrives at a lower position
(implicit rollback).

**Single-worker routes.** The worker drafts and verifies locally in its decode loop.

**Split routes.** The first stage drafts when the token comes around the ring, runs the 4
positions as one batch, and appends the guessed ids after the activations of the
`speculative_activation` frame (`rows − 1` ids, 4 bytes each). Middle stages pass that tail
through untouched; the last stage reads it, applies the acceptance rule, streams every
committed token, and sends the last one around the ring. The first stage learns how many
were accepted from that token's position.

**Keeping the draft in step** (each of these was a real bug found in testing):
- It mirrors create/reset/destroy session, prompts, `end_request` and the final-token
  `commit_token`, so it always holds the same text as the real session.
- It feeds itself every guess but the last; when all guesses were accepted it catches up
  by one token (at the next round, or right before a commit or prompt).
- It forgets its sessions whenever the route's client disconnects, and a stage reused for
  the next route re-attaches (and, if the route asks for a different draft, reloads) it.

**Measured 2026-09-18** (Qwen2.5-14B Q8, 0.5B draft):

| Route | Plain loop | `--speculate` | Accepted | Tokens per pass |
|---|---|---|---|---|
| One remote GPU (RunPod RTX 4000 Ada), relayed | 19.7 tok/s | 36.8 tok/s | 49% | 2.46 |
| RTX 2070 (layers 0–10, draft) + RunPod RTX 3090 (11–47), relayed 73–135 ms | 9.6 tok/s | 17.1–17.9 tok/s (follow-ups ~13) | 44% | 2.31 |

**Why it is opt-in.** Checking several positions in one batch gives slightly different
floating-point results than one at a time (see `reference/design/speculative-decoding.md`),
so a near-tie can flip a word. In local tests (1.5B split in two, 0.5B draft) one of three
outputs changed at one word ("capital of France" vs "capital of Italy"), while the
single-worker and split speculative paths produced **exactly the same** text — the
difference is the batching, not the ring. DAN Chat does not pass `--speculate` yet.

### 9.3 Lease state machine (`engine/include/provider_owned/lease.hpp`)
```text
AVAILABLE ──reserve (first wins)──▶ RESERVED ──assign_stage (same stage)──▶ LOADING ──▶ SERVING
    ▲            expires after lease_ms ◀┘                                             │
    └──────────── release_route(route_id) / control connection closed ◀────────────────┘
```
One lease per worker. No expiry while loading. Every transition names its `route_id`.

### 9.4 Dialing and NAT traversal (`sidecar/network.go`, `dialer`)
To reach a PeerID: reuse an existing connection (direct or relayed) → addresses given
explicitly → known addresses (half the dial timeout) → DHT `FindPeer` → **through a relay
this node already uses** (`<relay>/p2p-circuit/p2p/<peer>`). The last step matters because
every home node is a DHT *client*, so no routing table lists it and a lookup can fail once
its provider record's addresses age out, while its relay reservation still reaches it.

If only a relayed ("limited") connection exists, a **ring** stream (the ones that carry
every token) waits up to `-direct-wait` (5 s) for hole punching (DCUtR) to produce a direct
one; if that fails, the peer is remembered for 10 minutes and the relay is used
immediately. Control streams never wait: they carry setup and one prompt per message, and
hole punching keeps upgrading the connection in the background for later streams. Every stream logs `path=direct|relay`,
transport, bytes, and duration.

- Home nodes: `-reachability private`, static relay = the network node. AutoRelay keeps a
  renewed reservation and advertises `/p2p-circuit` addresses. Configured relay addresses
  are stored permanently (AutoRelay only uses relays with a *public* address; the VPS
  announces its public IP with `-announce`).
- IPv6: nodes also listen on `/ip6/::/…` (TCP + QUIC). A global IPv6 address works
  directly only if the router allows inbound (the owner's router refuses it even with a
  port rule).
- Bootstrap/relay addresses may use DNS: `/dns4/…`, `/dns6/…`, `/dns/…` (the PeerID still
  authenticates).
- Relay limits: 4096 MiB / 2 h per relayed connection (`-relay-limit-mib`,
  `-relay-limit-duration`).

### 9.5 Range-backed weights (`engine/range_model.cpp`)
Workers never download whole models. Given its stage, a worker reads the GGUF header,
computes the byte ranges of its tensors, and downloads only those (HTTP `Range`, `curl`
with retries) into a **sparse file** the size of the full GGUF. Every response must carry
the manifest's pinned revision (`x-repo-commit`), full-file SHA-256 (`x-linked-etag`,
Hugging Face) and the expected `Content-Range`. Each stored range's SHA-256 is recorded in
the `.ranges` index and checked when the cache is reused. The worker reports which ranges
it holds by scanning its cache folder, so the information survives restarts. (The full file
is never downloaded, so its hash is trusted from the pinned revision.)

### 9.6 Stage computation (`patches/llama-provider-owned.patch`)
llama.cpp (pinned commit `95ef7fc16054e63b427a3ef00188e055ef7586d8`) is patched so a model
can be loaded with `dan_stage_start/dan_stage_end`: only those layers (plus embedding on the
first stage and norm/head on the last) are allocated, and a middle stage can take FP32
embeddings as direct input. Each session is a llama.cpp sequence with its own KV cache.
The patch hash is part of the runtime ABI. The draft model is an ordinary whole-model
load (`0..layers`) on the first stage's GPU.

### 9.7 Worker dashboard feed
The sidecar writes `network-status.json` every 2 s (PeerID, relay count, public IPv6,
connected peers, active DAN streams with path/transport). The worker's dashboard thread
reads it each second, adds lease/load events, throughput samples, totals and uptime, and
redraws in place (`ui/node_dashboard.cpp`; ASCII fallback; compact under 60 columns; one
status line per change when output is redirected). With `replica=auto` it also shows a
REPLICA line from the owner's status file (state, replica, stages, sessions, formed and
dissolved counts).

### 9.8 Persistent replicas (`engine/replica_owner.cpp`, `sidecar/replica.go`)
Design: `docs/DAN_replica_design.md`. Instead of each chat building and tearing down a
route, provider nodes build routes themselves and keep them.

**Who forms.** With `replica=auto`, `dan-provider` also starts `dan-client --form` (the
*owner process*) next to its worker, sharing the node's sidecar. Every such node can form a
replica, but only while its own worker is free. The node that wins the leases becomes the
replica's owner. It is always the first stage (the head) and has no authority beyond that
replica. There is no network-wide planner, election or consensus.

**Formation** (one attempt; repeated every 5–15 s while the worker is free):
1. Ask the own worker for its greeting; not `available` → skip.
2. Discover candidates as a client does (DHT → capability queries → RTT and path).
3. *Rank delay* (optimization only): wait 3 s per free peer with more memory (or equal
   memory and a lower PeerID), at most 30 s, so the biggest free node usually proposes first.
4. `place_route` with this node's worker required as the head (`plan_stages` /
   `plan_from_cache` with `head`); the reuse-first cache planning is unchanged.
5. **Before reserving anything**, measure every link of the planned ring (each hop and
   the loop from the tail back to the head) *from its sending side*: the owner asks that
   peer's sidecar over `/dan/probe/1.0.0`, which dials, gives hole punching the same 5 s a
   ring stream gets, and pings. A link over `replica_max_edge_rtt_ms` (150), or relayed when
   `replica_relay_edges=false`, drops that candidate and replans. The proposer's own RTT to a
   candidate is only the first filter; internet paths are not a metric, so each chosen link
   is checked.
6. Reserve (first reservation wins; a refusal releases the rest and replans), assign, and
   load. Workers reuse a stage that is still loaded, and cached ranges, as before.
7. Link the ring with the route's first session, then run a short warm-up request all the
   way around it. Only then is the replica **READY**; the status file says so and the
   sidecar advertises `dan/replica/1/<model sha256>`.

**Serving.** The owner keeps every member's control connection (so every lease) open. The
ring stays direct (head → … → tail → head); the tail returns tokens to the owner over
`/dan/return/1.0.0`. Clients connect only to the owner, over `/dan/session/1.0.0`, and speak
ordinary DAN frames (`replica.hpp` lists them). The owner gives each client session its own
internal session ID on the members, relays the prompt in and the streamed text out through a
bounded per-client queue (a client that falls 4096 frames behind is dropped), and commits
or rolls back the answer on every member itself, so a slow or vanished client never holds
the ring. Requests run one at a time in arrival order; sessions (KV) coexist up to
`replica_sessions`. Every 15 s the owner checks each member with a `metrics` frame, which
also keeps the workers' 10-minute idle timeout from ending an idle replica.

**Failure.** Any member, ring link or return link failing dissolves the replica: active
requests get an error, the status leaves READY (the DHT record expires; clients always ask
the owner live), and the owner closes every member connection, so their leases are released
while their layers stay loaded. The owner then forms again, and the same split re-forms from
loaded layers in seconds (new replica ID). If the owner process dies, its connections close
and every member is released the same way. A failed request (e.g. a full context) only
resets that session: on a linked route, workers now keep the connection after a
session-level error and clear the loop state, and error frames pass along the ring unchanged.

**Clients.** `dan-client --replica` (DAN Chat uses it) asks the sidecar for
`DAN-REPLICAS/1`, which looks up owners in the DHT and queries each one live
(`/dan/replica/1.0.0`). Replicas with a free session are ranked by model (largest first),
then direct before relayed, then RTT. With none, the client places its own route as
before (`--replica-only` refuses instead).

**Speculation** is unchanged on replicas: the owner picks the draft at formation
(`replica_speculate=true`), the head keeps it loaded, and the draft now has one KV sequence
per replica session. Its per-route guess state is correct because requests are serialized;
concurrent requests would need it per session.

**Observability.** `<state>/replica-status.json` (state, replica ID, members and ranges, each
measured link with its path, sessions, formation attempts, races lost, rejected links,
warm-up failures, dissolutions and the last reason, formation/load/warm-up times) and
`logs/replica-owner.log`.

Replicas use each model's manifest context, like chat routes. (The node's `max_context`
would not work: a 4096-token prompt on 14B is 83 MB of activations, over the 64 MiB frame
limit, so workers refuse it as `context_too_large`.)

**Settings** (`provider.conf`): `replica=auto|off` (off by default), `replica_sessions`
(1, ≤ `max_sessions`), `replica_max_edge_rtt_ms` (150), `replica_relay_edges` (true),
`replica_speculate` (false), `replica_min_stages` (1; tests), `replica_client` (path).

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

Payloads that matter for the decentralized path:
- `token`: 4-byte token id; `rows` = N and N ids for a speculative batch.
- `activation` / `speculative_activation`: 8-byte compute time + `rows × cols` f32. A
  speculative batch crossing the ring may add `rows − 1` guessed token ids (4 bytes each).
- `result`: token id (4) + compute ns (8) + end-of-text flag (1) + text piece. A verified
  speculative batch instead carries compute (8) + `rows` sampled ids.
- `client_chunk`: same payload as `result`; a streamed, non-final token (loop mode).
- `stream_prompt`: no payload; `rows` = token budget; sent to the last stage (loop mode).
- `cancel_request`: sent to the last stage to stop a streamed request.

Text payloads (newline-separated `key=value`):
- `provider_available`: `id gpu vram_mib ring state abi max_context max_sessions
  model=<sha>… cached=<sha>:<begin>-<end>…`
- `reserve` / `assign_stage` (serve mode): `route_id model_sha256 begin end context sessions
  [lease_ms] [draft_sha256]`
- `create_session` (ring): `next [previous_peer] [loop]`
- `release_route` / `stage_ready`: the route_id

### 10.2 Sidecar protocols (libp2p)
| Protocol | Use |
|---|---|
| `/dan/transport/1.0.0` | Control tunnel: client forward → worker `-inbound`. |
| `/dan/ring/1.0.0` | Ring tunnel: worker ring proxy → next worker, first stage, or client `-ring-inbound`. |
| `/dan/capabilities/1.0.0` | One uvarint-delimited protobuf request/reply (`sidecar/capabilities/capabilities.proto`): protocol version, worker id, ABI, data protocols, device, total/offered memory, limits, models (+ cached ranges), state, assignment. A status file older than 15 s reads as OFFLINE. The asking sidecar times it. |
| `/dan/session/1.0.0` | Client → replica owner's front door (sidecar `-session-inbound`); DAN frames. |
| `/dan/return/1.0.0` | Replica tail → owner's return listener (`-return-inbound`). Ring targets `/dan-return/p2p/<PeerID>` select it; a target naming the node itself is bridged locally. |
| `/dan/probe/1.0.0` | "`<PeerID>`" → "`OK <rtt ms> <direct\|relay>`": how this node reaches that peer (dial, up to 5 s for hole punching, best of 3 pings). At most 4 at a time. |
| `/dan/replica/1.0.0` | The owner's live replica status (JSON) or `{"state":"none"}`. |
| kad-dht with prefix `/dan` | Private DHT (not IPFS). Keys `dan/model/1/<sha>` (worker may serve) and `dan/replica/1/<sha>` (owner has a READY replica). |
| circuit v2, DCUtR, AutoNAT, identify | Standard libp2p. |

### 10.3 Local text lines (loopback only)
- `DAN-P2P/1 <PeerID>\n` — written by a sidecar before an inbound stream's bytes; tells the
  worker/client which authenticated peer is on the other side.
- `DAN-RING/1 <target>\n` — written by a worker to its ring proxy to name the next hop.
- `DAN-CANDIDATES/1 <sha256> [<sha256> …]\n` → `SELF`, `RETURN`,
  `CANDIDATE <PeerID> <control> <offered MiB> <abi> <rtt ms> <direct|relay>`, `END`
  (or `ERR …`). Older clients reading only the first fields keep working. With
  `-return-inbound` the line is `RETURN <addr> /dan-return`.
- `DAN-REPLICAS/1 <sha256> …\n` → `SELF`, `REPLICA <owner> <session forward> <rtt ms>
  <direct|relay> <replica id> <model sha256> <free> <max sessions> <context> <stages>
  <draft sha256|->`, `END`.
- `DAN-PROBE/1 <from PeerID> <to PeerID>\n` → `PROBE <rtt ms> <direct|relay>`, `END`.

---

## 11. Security model

**What is protected**
- All traffic between machines is libp2p-encrypted and authenticated (Noise/TLS, QUIC).
- Identities are keys; PeerIDs are checked on every control, ring, loop and return
  connection.
- Workers accept ring input only from the `previous_peer` the client named; ring addresses
  must match the authenticated PeerID.
- Workers download only from their own catalog URLs, pinned to a revision and file hash —
  also for draft models, which the client names only by hash.
- Workers and clients listen on loopback only; the sidecar is the only network listener.
- The relay is rate/size limited. `govulncheck` gates every package build.

**What is NOT protected (known, accepted for the friends beta)**
- **Honesty:** a node can return wrong results or lie about its GPU. No verification.
- **Sybil/eclipse:** Kademlia's weakness (GO-2024-3218, no fixed version).
  `scripts/collect_go_licenses.ps1` accepts exactly that advisory with a warning; any other
  reachable vulnerability fails the build.
- **Privacy:** the first stage sees the prompt text, the last sees the output, middle stages
  see activations (which can often be inverted). Transport encryption does not help because
  nodes must compute on the data. Practical next step: run the embedding and head on the
  user's own PC. (Encrypted computation — HE/MPC — is far too slow today.)
- **Any authenticated peer** can take a worker's single lease (`-allow-any`).
- **Local processes** are trusted (loopback ports, `DAN-P2P/1` lines).
- **Installer** is unsigned (Windows SmartScreen may warn).

---

## 12. Build, package, deploy

### Build (Windows, Visual Studio 2026, CMake, CUDA 13.3, Go 1.25.x)
```powershell
# CPU dev build with tests
cmake -S . -B build-client -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=D:/Desktop/DAN/build/provider-owned-v0/llama.cpp
cmake --build build-client --config Release --parallel 8
ctest --test-dir build-client -C Release

# Portable CUDA build (installer): any AVX2 CPU, RTX 20xx–50xx
cmake -S . -B build-cuda -G "Visual Studio 18 2026" -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=… `
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF "-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;120"
cmake --build build-cuda --config Release --parallel 8 --target dan-provider dan-stage-worker dan-client dan-provider-owned-coordinator
# first build ~45 min (CUDA kernels); later rebuilds of the worker take a minute

# Sidecar (from sidecar/, GOTOOLCHAIN=go1.25.7)
go build -o ../build-client/sidecar/dan-sidecar.exe .
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o ../build-client/sidecar/dan-sidecar-linux-amd64 .
go test ./...
```
llama.cpp: clone, check out `95ef7fc1…`, apply `patches/llama-provider-owned.patch`
(`scripts/build_provider_owned.ps1` does this).

### Linux GPU node (how the RunPod tests were run)
```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git checkout 95ef7fc16054e63b427a3ef00188e055ef7586d8
git apply /root/dan/patches/llama-provider-owned.patch     # the patch must have LF endings
cd /root/dan
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=/root/llama.cpp \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<86 for RTX 30xx, 89 for RTX 40xx/Ada>
cmake --build build -j 64 --target dan-provider dan-stage-worker dan-client
# copy dan-sidecar-linux-amd64 to build/dan-sidecar, write a provider.conf with
# network=dht, bootstrap, catalog lines, stage_worker, sidecar, state_dir, cache_dir, then:
setsid nohup build/dan-provider --config provider.conf >> provider.out 2>&1 < /dev/null &
```
Get the source there with `git -c core.autocrlf=input archive HEAD` (a plain `git archive`
on Windows converts the patch to CRLF and it no longer applies). Stop it with
`pkill -x dan-stage-worke; pkill -x dan-sidecar` (process names are cut to 15 characters;
`pkill -f` would also match the SSH command running it). A stopped RunPod pod loses its
container disk, so it needs a rebuild (~10 min on 64+ cores).

### Installer
```powershell
.\scripts\build_installer.ps1 -Bootstrap /ip4/82.70.213.202/tcp/4001/p2p/12D3KooWGDp2QL2wBSwbE6jyzU8CH13KT8Tvmzeq4caErzVvuU35
# → build\installer\DAN-Setup-1.1.0.exe (+ .sha256)
.\scripts\build_installer.ps1          # no -Bootstrap: "local test" build with its own network node
```
It calls `package_provider.ps1` (stages files, runtime DLLs, licenses, `govulncheck` gate,
all four model configs in the catalog), then Inno Setup (`installer/dan.iss`). Install is
per-user (`%LOCALAPPDATA%\Programs\DAN`), no admin, with Start-menu/desktop shortcuts and
an uninstaller. Users need only Windows 10/11 x64, an AVX2 CPU, and an NVIDIA RTX 20-series
or newer GPU with a driver recent enough for CUDA 13 (about R580+). Everything else (CUDA
runtime, MSVC runtime, llama.cpp, sidecar) is bundled.

Current build: `DAN-Setup-1.1.0.exe`, 2026-09-18, SHA-256
`87da16baf62ab788249ec6aadaa289c270444dce8763f14f74ff62d8acc2d8fb`, bootstrap = the Oracle
node. Checked by running a node and a chat straight from `build\installer-stage` with a
separate data folder (chat picked the model itself and answered correctly). Rebuild the
CUDA targets first whenever engine code changes; the package takes whatever is in
`build-cuda\Release`. The version number was not bumped for this build.

### Public network node (Oracle Cloud, Always Free)
- VM `dan-network`, region `il-jerusalem-1`, `VM.Standard.E2.1.Micro` (1 GB RAM,
  Ubuntu 24.04, x86-64), public IP `82.70.213.202`.
- Service `dan-infra` (`deploy/dan-infra.service`, `PUBLIC_IP` set), binary
  `/opt/dan/dan-sidecar`, identity `/var/lib/dan/infra.key`, log `/var/log/dan/infra.log`.
- Ports TCP+UDP 4001 open in the Oracle security list **and** iptables
  (`/etc/iptables/rules.v4`).
- SSH: `ssh -i ~/.ssh/dan_oracle ubuntu@82.70.213.202` (key on the owner's PC).
- Uses ~24 MB RAM. **If `infra.key` is lost, the PeerID changes and every installer must be
  rebuilt.** Its sidecar binary is from 2026-09-17; the infra role has not changed since.

---

## 13. Testing

| Layer | Command | Checks |
|---|---|---|
| C++ unit | `ctest --test-dir build-client -C Release` | protocol, range model, formation (incl. greeting cached ranges and the speculation acceptance rule), client (incl. the replica front-door protocol), route, lease, placement (cache plans, model fallback, link ranking, required head, route rejection before reserving, lost-race counts), UI, platform — 13 tests. Run `provider_owned_formation_test` in **Debug** too (it uses `assert`). |
| Go unit | `cd sidecar; go test ./...` | tunnels, relay/NAT by PeerID, IPv6, DNS bootstrap, capabilities, discovery, stale-status advertising, net status. |
| Static ring | `scripts/Test-DAN-Client-Static.ps1 -Transport hub\|ring\|libp2p [-WrongPredecessor]` | identical output vs baseline; wrong PeerID refused. |
| Placement | `scripts/Test-DAN-Placement.ps1 -Transport direct\|libp2p [-LeaseChecks] [-Race] [-Manifest M -DraftManifest D -OfferedMib … -MinStages N]` | leases, races, identical output; with a draft model, speculation on a local split. |
| Discovery | `scripts/Test-DAN-Discovery.ps1` | DHT end to end incl. killing the bootstrap and a worker. |
| NAT rehearsal | `scripts/Test-DAN-NatRehearsal.ps1 -BuildDir build-client -OutDir … -BaselineDir build-client\results\baseline` | infra + 3 `dan-provider` nodes with `simulate_nat`; every DAN stream relayed; output identical; dashboard + scripted chat. Last run 2026-09-18: 42/42 relayed, PASS. |
| Replica rehearsal | `scripts/Test-DAN-Replica.ps1 -BuildDir build-client -OutDir … -BaselineDir build-client\results\baseline` | all relayed (`simulate_nat`). (1) A, B, C start free; A's owner forms A→B→C with no client; two clients reuse the same replica ID; no worker logs a new reserve, load or ring link; outputs identical; chat. (2) B killed mid-answer: client error, dissolution, A and C released with layers loaded; B back → new replica from loaded layers; owner killed → every member released. (3) four owners race for three GPUs: one replica, one node free. `-Speculation -SpeculationBaselineDir build-client\results\spec-draft2`: 1.5B on two nodes with the 0.5B draft, output identical to the placed speculating route. Last run 2026-09-18: all PASS. |
| Real internet | owner's install + a RunPod GPU pod (§12), `Start-DAN-Client.ps1 … -- --chat [--min-stages 2] [--speculate] [--no-loop]` | 2026-09-17/18, see §2 and §9.2. Replicas 2026-09-19: below. |
| Replicas, real internet | owner's install with `replica=auto` (+ `replica_speculate=true`), RunPod RTX 3090 node with `replica=off`, DAN Chat | 2026-09-19: the owner's node formed 14B (2070 layers 0–10, 3090 11–47) with no client; both ring links measured 64 ms direct; formed in 200 s (the pod's first 12 GB download), link + warm-up 1 s; re-formed in 29 s with the pod reusing its loaded layers; chat used the replica with no route setup. Plain 9 tok/s (same as a placed route); with speculation 2.2–3.5 tokens per ring trip, ~19–30 tok/s. |

Baseline output reports: `build-client/results/baseline` (plain decoding; speculative runs
are compared with each other, not with it). Test model: Qwen2.5-0.5B-Instruct Q4_K_M
(24 layers, hidden 896); speculation tests use 1.5B with a 0.5B draft.

### Measured numbers (for orientation)
| Setup | Result |
|---|---|
| NAT rehearsal (one PC, CPU, 3 stages, all relayed) | before 2026-09-18: metadata 6.3 s, greeting 5.0 s, route setup 15.0 s; now 0.002 s, 0.01 s, 5.0 s (fixed start-up ~26 s → ~5 s); decode 30–37 tok/s |
| Owner's RTX 2070 alone, chat on the same PC | 68–145 tok/s, first token 14–78 ms |
| 14B on one remote GPU, relayed | `--no-loop` 6.9, loop 19.7, loop + `--speculate` 36.8 tok/s; first token ~0.3 s |
| 14B split RTX 2070 + RunPod RTX 3090, relayed | `--no-loop` 9.7, loop 9.6, loop + `--speculate` 17.1–17.9 tok/s |
| 14B split RTX 2070 + RunPod RTX 4000 Ada, relayed (before speculation) | 4.6–6.0 tok/s; 14 s to be ready from cached layers vs 201 s re-downloading |
| Link measurement | owner's PC ↔ own node ~1 ms direct; ↔ RunPod 73–141 ms relayed |
| Replica rehearsal (one PC, CPU, 3 stages, all relayed) | formation ~21 s (load 4–5 s from cache, link + warm-up 11 s); a client on the READY replica: first token 75–160 ms, decode 26–36 tok/s (placed route: 5 s route setup + ~6 s first-request warm-up); re-formation from loaded layers: no download, no reload |
| Replica with speculation (1.5B on 2 CPU stages + 0.5B draft) | 9.3 / 13.6 tok/s vs 8.8 / 13.1 on a placed speculating route; first token 0.2 s vs 10.8 s |

Older coordinator-path results (32B split, WAN speculative decoding up to 6.3×) are in
`TESTS_AND_STATS.md`.

---

## 14. Limitations and roadmap

### Current limits
- One client route per worker (`max_sessions` 1); no queueing — a busy network fails with
  "no placement fits".
- A GPU leaving mid-chat fails the chat (no failover); the client must start over.
- Manifests set `context_size` 512, so chat conversations reset often.
- Chat picks the largest model automatically; there is no `/model` override yet, and a
  node serves only the models in its own catalog.
- Speculation is opt-in and not exposed in DAN Chat; the draft's memory is not planned for.
- Discovery waits for every peer's capability answer, so a node that went offline in the
  last 5 minutes (its DHT record is still there) delays chat start by up to the 10 s
  query timeout (seen: 9 s).
- Chat start-up still includes one 5 s wait for a direct path on relayed token links
  (skipped for 10 minutes after it fails for a peer), and a freshly loaded CPU stage pays a
  one-time warm-up on its first request (~6 s in the local rehearsal; 0.3–0.9 s on GPUs).
- Replicas (§9.8): off by default; tested on one PC and once over the internet (§13).
  Formation ignores speed, so an owner may form a slower split than a free GPU could run
  alone (roadmap 1a). A replica runs one request
  at a time (sessions wait their turn); it dissolves on any member or link failure (no
  repair) and whenever a relayed link hits the relay's 2 h / 4 GiB cap; its GPUs stay
  reserved while it is idle; `replica_sessions` is fixed, not optimized against model fit;
  the owner must be the head, so a node whose cached range is not layer 0 cannot reuse it
  as the owner.
- Activations cross the network as FP32 (a 512-token prompt on 14B is ~10.5 MB per hop).
- Dense Qwen2 GGUF only, greedy sampling only.
- One public network node; no auto-update; Windows-only GPU installer; unsigned.

### Roadmap (roughly in order)
1. **A real friend test** with the rebuilt installer.
1a. **Replicas: speed-aware formation** (found in the first real-network test, §13). Today
   an owner always puts itself in its own replica as the head, and nothing in formation
   knows speed. So the owner's RTX 2070 plus a RunPod 3090 formed a 14B split at ~9 tok/s
   (32 ms on the 2070 + 16 ms on the 3090 + 64 ms of network per token), while the 3090 alone
   holds 14B and would run ~40+ tok/s. To implement:
   - an estimated time per token for a plan: each stage's compute (from a per-GPU speed
     figure in the greeting, e.g. measured ms per layer, or VRAM bandwidth as a first
     proxy) plus the measured RTT of each ring link, including the loop;
   - a node forms a replica only if no plan over the free peers *without* it is clearly
     faster (with a margin, so equal plans still form); otherwise it stays free and lets a
     better-placed node lead;
   - a node that could lead a single-GPU replica of the chosen model does so (a fast GPU's
     own owner should not wait for a slower node to include it);
   - clients rank READY replicas by that estimate plus their own RTT to the owner, not
     only by model and link;
   - the owner stays the head of its own replica (prompt and draft run there).
1b. **Replicas, then:** `replica=auto` by default in the installer (after 1a); concurrent
   requests per replica (per-request loop and draft state); demand signals (form when all
   replicas are full, dissolve long-idle duplicates); sessions vs model capacity; disk
   prefetch of scarce layers if downloads dominate formation.
2. **Discovery that does not wait for dead peers:** stop waiting for stragglers shortly after
   enough candidates answered, or remember peers that just failed.
3. **Start-up latency:** done 2026-09-18 (model index cache, no direct-path wait on control
   streams, parallel ring linking, background loop dial). Left: a warm-up pass while loading.
4. **Smaller activations:** FP16 first, then INT8 with per-row scale for prompts.
5. **Chunked prefill** on the decentralized path (the worker already supports it).
6. **Speculation by default** once its output drift is accepted, plus a `--speculate`
   switch in DAN Chat; account for the draft's memory in planning.
7. **Chat quality of life:** larger contexts, `/model`, sampling options; an HTTP/agent API
   or a tool-calling harness on top of `dan-client`.
8. **Privacy step:** embedding + head on the user's PC.
9. **More entry points:** several independent network nodes; public nodes volunteering as
   relays; a DNS name for the bootstrap.
10. **Robustness:** failover to a spare mid-request, multi-client workers, queueing.
11. **Trust:** result verification (spot checks against a trusted copy), reputation, Sybil
    resistance.
12. **Incentives:** usage accounting, then payments/crypto.
13. **Governance:** how the network agrees on models and protocol versions.
14. Linux GPU installer, signed installer, auto-update, version bump per release.

### Where decentralization stands
Done: no coordinator; DHT discovery; client-side model choice and planning; direct
GPU-to-GPU data path with the client out of the loop; worker-owned catalogs and decisions;
key-based identities; provider nodes forming persistent replicas themselves (each owner
only for its own replica; leases settle races; no election or consensus).
Still central: the single entry node (existing nodes survive its loss, new ones can't join,
CGNAT users lose their relay); the installer and its built-in address; the model list.
Not solved: trust, incentives, governance.

---

## 15. History and older paths (why the repo contains more than the above)

1. **Legacy path** (`legacy/`): whole-model providers and llama.cpp RPC groups under a
   poll-based C++ coordinator. Proved GPUs could participate, but the coordinator held the
   full model. Reference only. See `reference/design/legacy-path.md`.
2. **Provider-owned + coordinator** (`engine/coordinator.cpp`, `dan-api-gateway`): workers
   own their layers; a metadata-only coordinator formed replicas, routed activations, ran
   speculative decoding/pipelining, and exposed an HTTP API; Tailscale or libp2p transport;
   Windows v1.0.1 packages. Proved 32B aggregate-VRAM inference and WAN speed-ups. Still
   builds; not the direction (a single trusted operator).
3. **Decentralized path (current)**, in commit order:
   `2bf5bd6` shared client + `dan-client` → `408ad71` dynamic placement, direct ring →
   `6955109` DHT discovery, capabilities → `9d9212f` NAT/relay → `22bcaec` WAN beta ops,
   `dan-provider` DHT mode → `47c6446` dashboard, chat, IPv6, DNS → `c9dc9e2` installer →
   `a4f2cbb` this guide → `9faa783` loop mode → `3ec24bb` cache-aware planning →
   `0b2846c` automatic model choice, relay-fallback dialing, bounded advertisement refresh →
   `fb41d56` latency-aware placement, single-worker speculation → `77d16af` speculation on
   split routes → `1c81a9d` draft model kept in step with the conversation →
   `4e657f3` start-up ~26 s → ~5 s → persistent self-forming replicas (§9.8).

### Environment pitfalls (Windows dev machine)
- PowerShell 5.1: native stderr becomes an error under `$ErrorActionPreference='Stop'` when
  output is redirected → wrap native calls with `'Continue'` and check `$LASTEXITCODE`.
- Splatting arrays mangles `--flags`, and splatting an empty array passes a stray `-` →
  build a plain array.
- `Start-Process -ArgumentList` does not quote → quote paths with spaces (the owner's
  profile is `C:\Users\Halakim Family`; in Git Bash `$HOME` splits too, use `~`).
- `$x = if (…) { @(…) }` unwraps one-element arrays → wrap the whole expression in `@()`.
- Advanced scripts (`[CmdletBinding()]`) reject piped input; PowerShell piping to native
  programs adds a UTF-8 BOM and CRLF (`dan-client --chat` strips both).
- Git Bash rewrites `/ip4/...` arguments → `MSYS_NO_PATHCONV=1`.
- Heredocs in the agent's Bash tool turn `\n` inside strings into real newlines → write
  edit scripts with the file-writing tool instead.
- `git archive` on Windows converts line endings → `git -c core.autocrlf=input archive`.
- The CUDA build dir must be configured at the current path (moving the repo breaks it).
- A long-running node keeps `dan-stage-worker.exe` and `dan-sidecar.exe` locked; stop the
  node before copying new binaries into an install.

---

## 16. Document map

| Document | Contents |
|---|---|
| `docs/PROJECT.md` | **This file** — start here. |
| `docs/DAN_replica_design.md` | Persistent replica design (report + the v1 decisions); `DAN_local_replica_formation.md` is the original idea it replaced (fixed-range coverage). |
| `README.md` | Short introduction and how it works. |
| `docs/reference/operations/wan-beta.md` | Operating the WAN beta: VPS, friend package, client, timeouts, checks. |
| `docs/reference/operations/friend-readme.txt` | README shipped to friends. |
| `docs/ARCHITECTURE.md` | Detailed architecture incl. older paths. |
| `docs/PROGRESS.md` | Historical progress log and the old coordinator-path beta gate. |
| `docs/TESTS_AND_STATS.md` | Physical test evidence (mostly coordinator path). |
| `docs/OPERATIONS.md` | Operating the coordinator-path service. |
| `docs/reference/design/*` | Deep dives: p2p transport, range storage, replica formation, runtime v1/v2, speculative decoding (incl. the batching caveat), decisions, legacy. |
| `docs/reference/tests/*` | Test receipts. |
