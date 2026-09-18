# DAN

DAN runs large language models across volunteers' GPUs over the internet, with no
central scheduler.

```text
your PC ──prompt──▶ GPU A (layers 0–9) ──▶ GPU B (layers 10–17) ──▶ GPU C (layers 18–23)
   ▲                    ▲                                                   │
   └──── tokens ────────┴──────────────────── next token ◀──────────────────┘
```

- Each GPU downloads and loads only its own block of layers.
- The user's client finds GPUs through a private DHT, picks the largest model they can run
  together, plans the split, reserves the GPUs and links them into a ring. No coordinator.
- The last GPU feeds each new token straight back to the first one and streams it to the
  user, so the user's PC is not in the per-token loop.
- Home routers and CGNAT work through libp2p relays and hole punching.
- Friends install it with one Windows installer and get two shortcuts: **DAN Node**
  (share a GPU, with a live dashboard) and **DAN Chat**.

**Full guide: [docs/PROJECT.md](docs/PROJECT.md)** — goals, current state, structure,
algorithms, protocols, security, build and deploy, tests, and roadmap.

## How it works

### Architecture

Every machine runs two kinds of processes:

- **C++ inference processes** — `dan-stage-worker` on GPU nodes, `dan-client` on the
  user's PC. They speak a small binary frame protocol over loopback TCP only.
- **A Go network sidecar** — `dan-sidecar`, built on go-libp2p. It owns the node's
  identity (a key → PeerID), the DHT, NAT traversal, encryption, link measurement, and the
  tunnels that carry the C++ frames between machines.

A small public **network node** (`dan-sidecar -infra`) is the entry point and relay.
It never plans, reserves or routes work.

### Provider-owned execution

No machine ever holds the whole model. A worker is told a layer range `[begin, end)` and
downloads only those tensors from the GGUF with HTTP range requests into a sparse file
(pinned revision and file hash), which it keeps for later. A patched llama.cpp loads just
those layers:

```text
tokens ─▶ first stage (embedding + layers) ─▶ FP32 activations ─▶ middle stage(s)
       ─▶ FP32 activations ─▶ last stage (layers + head + greedy sampling) ─▶ token
```

Each conversation is a session with its own KV cache on every stage.

### A request, step by step

1. **Discover.** Workers advertise each model they serve in a private Kademlia DHT
   under `dan/model/1/<gguf sha256>`. The client's sidecar asks about all the client's
   models at once, queries each peer's live capabilities (memory, limits, state, runtime
   ABI, cached layers) over `/dan/capabilities/1.0.0`, and measures how fast and how
   directly it reaches each one.
2. **Choose the model.** Models are tried largest first; the first one the available
   workers can run is used.
3. **Filter and rank.** Keep workers that are idle, have the model in their own catalog,
   run the same runtime ABI, meet the limits, and whose ring address names the PeerID the
   connection actually authenticated. Rank by link (round trip, direct before relayed),
   then memory; keep up to 8.
4. **Plan.** Find the fewest stages that fit (below). If the workers already hold layers
   that tile the model in no more stages, use that split, so nothing is downloaded.
5. **Reserve.** Ask every chosen worker for a short lease. Each worker checks the request
   against its **own** catalog and memory; the first reservation wins. If any refuses,
   release the rest, wait a random 100–500 ms, and plan again (3 attempts).
6. **Load.** Workers download (only if needed) and load their ranges in parallel.
7. **Link the ring.** Create the session on the last stage first. Each stage learns where
   to send its output (`next`, a PeerID) and the single PeerID it may accept input from
   (`previous_peer`). The last stage connects back to the client and loops to the first.
8. **Generate.** The client sends the prompt to the first stage and a token budget to the
   last. Activations flow A → B → C; the last stage samples a token, streams it to the
   client, and sends it back to the first stage for the next step. At the end the final
   token is committed through every stage so the conversation can continue.

### Persistent replicas (`replica=auto`, off by default)

Instead of every chat building its own route, provider nodes can build routes themselves
and keep them:

- Each node with `replica=auto` also runs a *replica owner*. While its GPU is free, the
  owner discovers free peers and plans a route with itself as the first stage, using the
  same planner as a client.
- Before reserving anything, it measures every link of that route from the sending side,
  including the loop back to itself. A slow link leaves that peer out.
- It then reserves (the first reservation wins, so racing owners sort themselves out),
  loads, links the ring, and sends one warm-up token around it.
- Only then is the replica READY. It is advertised in the DHT under
  `dan/replica/1/<sha256>`.

Clients (`dan-client --replica`, which DAN Chat uses) find READY replicas, ask each owner
live, and chat through the owner. The owner relays the prompt in and the text out, while the
tokens loop directly GPU to GPU. When a client leaves, only its session is dropped and the
replica stays up for the next one. If a member fails, the replica dissolves and the other
GPUs keep their layers loaded, so it re-forms in seconds.

### Speculative decoding (`--speculate`)

The first stage also loads a small draft model that guesses the next 3 tokens; the real
model checks all 4 positions in one pass and keeps every correct guess. On a split route
the guesses travel with the activations so the last stage can check them. Measured on a
14B model: 19.7 → 36.8 tok/s on one remote GPU, 9.6 → 17.4 tok/s split across two
networks. It is opt-in because checking several positions in one batch can, rarely, flip
a word compared with one-at-a-time decoding.

### Planning algorithm

A stage `[b, e)` fits a worker offering `M` memory if, after keeping back
`max(1 GiB, 15% of M)`, there is room for the weights of those layers and for their KV
cache (`context × sessions × layers × head_dim × kv_heads × 2 × 2 bytes`).
The planner tries 1 stage, then 2, 3, … (up to the number of candidates). For each count
it tries orderings of the candidates and splits the layers so each worker's share is
about its share of the remaining memory, trying the nearest cut points first and
backtracking when a stage does not fit. The first plan with the fewest stages wins.
Clients, workers (checking a reservation) and the tests all use the same code.

### Leases

```text
AVAILABLE ──reserve (first wins)──▶ RESERVED ──assign──▶ LOADING ──▶ SERVING
    ▲        (expires after ≤ 60 s) ◀┘                                   │
    └──────────── release, or the client's connection closes ◀───────────┘
```

A worker holds one lease at a time and never downloads from a URL a client sends: the
client names the model (and any draft model) only by SHA-256.

### Networking

- Home nodes keep a reservation on the network node's relay and advertise relayed
  addresses, so anyone can reach them without port forwarding.
- A connection to a PeerID reuses an existing link, then known addresses, then a DHT
  lookup, then a relay this node already uses. If only a relayed link exists, the sidecar
  waits up to 5 s for hole punching to produce a direct one, and remembers failures for
  10 minutes.
- Nodes also listen on IPv6; bootstrap addresses may be DNS names.
- The relay is limited to 4 GiB / 2 h per relayed connection.
- Once routing tables are filled, losing the network node does not stop discovery among
  connected nodes.

## Status (2026-09-18)

- Working: decentralized discovery, automatic model choice, cache- and latency-aware
  placement, the GPU ring with the client out of the loop, NAT traversal, speculative
  decoding (opt-in).
- Proven across networks: Qwen2.5-14B split between a home RTX 2070 and rented cloud GPUs,
  through the public relay.
- Public network node (bootstrap + relay) running on Oracle Cloud.
- Installer `DAN-Setup-1.1.0.exe` rebuilt 2026-09-18 with all of the above.
- Chat start-up overhead cut from ~26 s to ~5 s (cached model headers, parallel ring
  setup, no direct-path wait on control traffic).
- Persistent self-forming replicas: working in the local rehearsal (first token ~0.1 s on a
  ready replica); off by default until tested on a real network.
- Next: a real friend test, replicas on a real network, smaller activations.
- Not started: payments, reputation, result verification, failover, privacy protection.

## Quick start

Users: run `DAN-Setup-1.1.0.exe`, then open **DAN Node** or **DAN Chat**.
Requires Windows 10/11 x64 and an NVIDIA RTX 20-series or newer GPU with a current driver.

Developers (Windows, Visual Studio 2026, CUDA 13.3, Go):

```powershell
cmake -S . -B build-client -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=<patched llama.cpp>
cmake --build build-client --config Release --parallel 8
ctest --test-dir build-client -C Release
cd sidecar; go test ./...; cd ..
.\scripts\build_installer.ps1 -Bootstrap <network node address>
```

See [docs/PROJECT.md](docs/PROJECT.md) §12–13 for the CUDA build, Linux GPU nodes, the
patched llama.cpp, and the test scripts.

## Security

Traffic between machines is encrypted and authenticated (libp2p), and nodes check each
other's identities. Nodes are not verified: a node can return wrong results, and nodes on
a route can read the prompt and the answer. Use the beta only with people you trust.

## License

DAN is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Bundled dependencies and separately downloaded model weights keep their own licenses.
