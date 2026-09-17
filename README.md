# DAN

DAN runs large language models across volunteers' GPUs over the internet, with no
central scheduler.

```text
your PC ──prompt──▶ GPU A (layers 0–9) ──▶ GPU B (layers 10–17) ──▶ GPU C (layers 18–23) ──answer──▶ your PC
```

- Each GPU downloads and loads only its own block of layers.
- The user's client finds GPUs through a private DHT, plans the split, reserves the GPUs,
  and links them into a direct ring. No coordinator.
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
  identity (a key → PeerID), the DHT, NAT traversal, encryption, and the tunnels that
  carry the C++ frames between machines.

A small public **network node** (`dan-sidecar -infra`) is the entry point and relay.
It never plans, reserves or routes work.

### Provider-owned execution

No machine ever holds the whole model. A worker is told a layer range `[begin, end)` and
downloads only those tensors from the GGUF with HTTP range requests into a sparse file
(pinned revision and file hash). A patched llama.cpp loads just those layers:

```text
tokens ─▶ first stage (embedding + layers) ─▶ FP32 activations ─▶ middle stage(s)
       ─▶ FP32 activations ─▶ last stage (layers + head + greedy sampling) ─▶ token
```

Each conversation is a session with its own KV cache on every stage. Loaded weights
stay cached for later routes.

### A request, step by step

1. **Discover.** Workers advertise each model they serve in a private Kademlia DHT
   under `dan/model/1/<gguf sha256>`. The client's sidecar finds those peers, asks each
   for live capabilities (memory, limits, state, runtime ABI) over
   `/dan/capabilities/1.0.0`, and hands the available ones to `dan-client`.
2. **Filter.** Keep workers that are idle, have the model in their own catalog, run the
   same runtime ABI, meet the context/session limits, and whose ring address names the
   PeerID the connection actually authenticated. Sort by offered memory; keep up to 8.
3. **Plan.** Find the fewest stages that fit (algorithm below).
4. **Reserve.** Ask every chosen worker for a short lease. Each worker checks the request
   against its **own** catalog and memory; the first reservation wins. If any worker
   refuses, release the rest, wait a random 100–500 ms, and plan again (3 attempts).
5. **Load.** Workers download and load their ranges in parallel.
6. **Link the ring.** Create the session on the last stage first. Each stage learns
   where to send its output (`next`, a PeerID) and which single PeerID it may accept
   input from (`previous_peer`). The last stage connects back to the client.
7. **Generate.** The client sends the prompt to the first stage; activations flow
   A → B → C; the last stage returns each token to the client, which sends it back to
   the first stage for the next step. Intermediate activations never pass through the
   client. At the end, the final token is committed through every stage so the session
   can continue.

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
client names the model only by SHA-256.

### Networking

- Home nodes keep a reservation on the network node's relay and advertise relayed
  addresses, so anyone can reach them without port forwarding.
- A connection to a PeerID reuses an existing link, then known addresses, then a DHT
  lookup. If only a relayed link exists, the sidecar waits up to 5 s for hole punching to
  produce a direct one, and remembers failures for 10 minutes.
- Nodes also listen on IPv6; bootstrap addresses may be DNS names.
- The relay is limited to 4 GiB / 2 h per relayed connection.
- Once routing tables are filled, losing the network node does not stop discovery among
  connected nodes.

## Status (2026-09-17)

- Decentralized discovery, placement, direct GPU ring, NAT traversal: working.
- Public network node (bootstrap + relay) running on Oracle Cloud.
- One-click installer `DAN-Setup-1.1.0.exe`: working; a chat through the public relay
  passed on a real RTX 2070.
- Next: a second machine on another network, automatic model choice.
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

See [docs/PROJECT.md](docs/PROJECT.md) §12–13 for the CUDA build, the patched llama.cpp,
and the test scripts.

## Security

Traffic between machines is encrypted and authenticated (libp2p), and nodes check each
other's identities. Nodes are not verified: a node can return wrong results, and nodes on
a route can read the prompt and the answer. Use the beta only with people you trust.

## License

DAN is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Bundled dependencies and separately downloaded model weights keep their own licenses.
