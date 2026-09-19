# DAN: current technical state and performance review

Reviewed 2026-09-19. DAN source snapshot: `14bf461f50351f40e2ed94e8d1d9d3168a068e27` (initially clean worktree).

This report describes the implementation, not a proposed replacement architecture. No inference, networking, configuration, or test code was changed. Recommendations are proposals, not completed work.

Evidence convention:

- **Source-confirmed** means the relevant executable path was traced. It does not establish performance or deployment success.
- **Local evidence** means tests or receipts from one PC, including multiple processes and forced relay. It is not a physical LAN or WAN test.
- **WAN evidence** distinguishes available raw client receipts from historical results recorded in the project guide. Historical GPU runs were not repeated for this review.
- **Measured** numbers retain their original workload and topology. Formulas and proposed gains are explicitly estimates.

The review covered the repository layout, active C++ engine and headers, Go sidecar, launcher/platform boundaries, build and packaging configuration, test scripts and receipts, and current/historical documentation. Older coordinator, gateway, and legacy paths were inspected to distinguish their capabilities from the active path; this is not an exhaustive security audit of every legacy implementation or every upstream kernel.

The local patched llama.cpp checkout is `95ef7fc16054e63b427a3ef00188e055ef7586d8`. External comparisons use freshly fetched public source snapshots:

- [Shard `fcf728096948c7686bcf0897e9acb75d1abda1d5`](https://github.com/leyten/shard/tree/fcf728096948c7686bcf0897e9acb75d1abda1d5).
- [c0mpute `fcd4690fb6b19af51e801069ce6d35ae93b6db7a`](https://github.com/leyten/c0mpute/tree/fcd4690fb6b19af51e801069ce6d35ae93b6db7a).

## 1. Executive technical summary

DAN currently has a working **dense-Qwen2 layer pipeline**, with provider-owned sparse GGUF ranges, encrypted libp2p transport, client/provider-side placement, exclusive worker leases, and optional persistent replicas. Infrastructure supplies discovery and connectivity; it does not assign inference work. A replica owner is a provider-local process with authority over its own leased route.

The main performance limitation of **one conversation** is the dependent traversal: head compute, intermediate compute, tail compute, and the token returning to the head happen in sequence. Speculation amortizes a traversal across up to four committed tokens; it does not keep multiple speculative rounds of that conversation in flight. Multiple sessions already interleave independent frames and can use otherwise idle stages. They are not combined into a GPU batch.

The numerical boundary is **FP32 host activations**, optionally encoded as FP16 or scaled FP8 for the wire, then expanded to FP32 before the next stage. This does **not** mean every GPU operation computes or accumulates in FP32. Quantized GGUF weights remain quantized, KV defaults to FP16, and llama.cpp dispatches mixed-precision kernels according to hardware, tensor type, shape and backend settings. CUDA graphs and asynchronous backend operations already exist underneath DAN, but DAN synchronizes before consuming host outputs.

Two easy-to-miss request costs are present:

1. The head mirrors a prompt into its draft **before forwarding** the main stage's final prefill output.
2. The final-token commit traverses stages through **owner/client control exchanges**, carrying commit activations through that process, rather than using the autonomous decode ring.

Therefore, “the owner never handles activations” is too broad. It avoids ordinary prefill/decode activations, but handles final-token commit activations. Reported `client_activations_received=0` is not proof that no activation bytes ever reach an owner/client.

Recent WAN receipts support useful two-session throughput on one two-stage 14B replica. They do not establish broad scalability, hostile-network robustness, or production readiness. Replicas remain off by default, idle smaller replicas cannot upgrade when new capacity appears, and draft memory/tokenizer compatibility are incompletely planned or validated.

The immediate direction should be: **measure the existing path, fix resource/compatibility gaps, reduce prefill and control-path costs, then make placement use measurements more accurately**. A new scheduler, expert-parallel execution, consensus or repair system is not necessary to pursue these improvements.

## 2. Current architecture

### 2.1 Components and source map

| Subsystem | Important implementation | Actual responsibility |
|---|---|---|
| Provider launcher | [provider_launcher.cpp](../engine/provider_launcher.cpp), `Options`, `main`, `start_sidecar`, GPU/config helpers | Select reported GPU/capacity, prepare state, launch sidecar and optionally owner, launch/exec worker |
| Worker | [stage_worker.cpp](../engine/stage_worker.cpp), `Stage`, `Session`, `ServeContext`, `RingState` | Model/context lifetime, leases, frame validation, inference, ring, draft, status |
| Common client | [client.cpp](../engine/client.cpp), `Connection`, `InferenceClient`, `generate_loop`, `generate_replica`, `commit_final_token` | Socket framing, sessions, ring setup, generation and commit |
| User CLI | [dan_client.cpp](../engine/dan_client.cpp), `read_models`, `open_replica`, `main` | Manifest/index loading, model ordering, replica selection, fallback placement, Qwen chat loop |
| Placement | [placement.cpp](../engine/placement.cpp), `place_route`, `open_worker`, discovery/probe parsers | Live greetings, candidate selection, plan, reserve/load, race handling |
| Capacity planner | [planner.cpp](../engine/planner.cpp), `compatible_dense_qwen2`, `stage_fits`, `plan_stages`, `plan_from_cache`, `decode_bytes`; [planner.hpp](../engine/include/provider_owned/planner.hpp), `estimate_token_ms` | Dense-model fit and contiguous layer assignment; rough decode estimate |
| Replica owner | [replica_owner.cpp](../engine/replica_owner.cpp), `Owner::form`, `serve`, `run_request`, `health`, `release_route` | Form one route, warm it, retain leases, serve sessions, dissolve on failure |
| Model files | [manifest.cpp](../engine/manifest.cpp), [range_model.cpp](../engine/range_model.cpp), `inspect_range_model`, `prepare_range_model` | Local catalog, GGUF metadata, pinned HTTP ranges, sparse files, verification/resumption |
| Lease/protocol | [lease.hpp](../engine/include/provider_owned/lease.hpp), `WorkerLease`; [protocol.hpp](../engine/include/provider_owned/protocol.hpp), `Frame`, `send_frame`, `recv_frame`; [route.hpp](../engine/include/provider_owned/route.hpp) | Reservations, wire contract, route validation |
| Codecs/spec acceptance | [activations.hpp](../engine/include/provider_owned/activations.hpp); [speculation.hpp](../engine/include/provider_owned/speculation.hpp) | FP8 packing, activation byte counts, greedy prefix acceptance |
| Networking | [main.go](../sidecar/main.go), `runInbound`, `startForward`, `startRingProxy`; [network.go](../sidecar/network.go), `newHost`, `dialer`, `bridge` | Identity, encrypted streams, local tunnels, NAT and relay |
| Discovery | [discovery.go](../sidecar/discovery.go), `startDHT`, `advertiseModels`, `findCandidates`, `startCandidateAPI`; [gather.go](../sidecar/gather.go), `gather`, `recentFailures` | Private provider discovery, live capability checks, bounded collection after a usable answer |
| Replica network services | [replica.go](../sidecar/replica.go), `measureLink`, `advertiseReplica`, `findReplicas`, `localBridge` | Edge probes, replica announcements/live status, owner return path |
| Observability/UI | [netstatus.go](../sidecar/netstatus.go), `run_dashboard`, `update_speed`, `status_json`; `ui/node_dashboard.cpp`, `ui/provider_ui.cpp` | Status files, activity/throughput display, connection path and lifetime byte logging |
| Backend | [llama-provider-owned.patch](../patches/llama-provider-owned.patch) | Qwen2 range load/graph, direct embeddings input, stage-local KV, sparse mapping behavior |
| Delivery | [CMakeLists.txt](../CMakeLists.txt), `scripts/build_provider_owned.ps1`, `build_sidecar.ps1`, `package_provider.ps1`, `build_installer.ps1`, `installer/DAN.ps1`, `installer/dan.iss` | C++/Go builds, portable CUDA bundle, manifests, shortcuts, installer |

`engine/coordinator.cpp` and `sidecar/cmd/dan-api-gateway` remain buildable older paths. `legacy/` contains the earlier whole-model/RPC system. Their pipelining, API, queueing and failover features must not be credited to the decentralized replica path without an actual call path connecting them.

### 2.2 Provider lifecycle

The installed node shortcut enters `installer/DAN.ps1 node`, then `dan-provider --config ...`.

1. Parse configuration, resolve installed executable/catalog paths, choose an NVIDIA GPU from `nvidia-smi` output, and calculate offered memory as total VRAM minus `reserve_vram_mib` (default 1536 MiB).
2. Load/create persistent sidecar identity and state/cache folders. This is not a fresh network identity on each restart.
3. Choose three loopback ports, or six with replica formation enabled. These are selected then released before the respective processes bind; they are not a permanent atomic reservation.
4. Start the sidecar with DHT client mode, forced private reachability, configured relays/bootstrap peers, control/ring tunnels, and status files. With replicas, also enable candidate API, owner front door, and owner return listener.
5. Start `dan-client --form` when `replica=auto`, sharing this sidecar. The owner can start before the worker becomes reachable and retries.
6. Launch the worker in serve mode. On POSIX `replace_with_provider` uses `execv`; on Windows the platform process machinery retains child processes/job cleanup.
7. The worker inspects each catalog's GGUF metadata, validates dense-Qwen2 compatibility, starts listeners/status/dashboard, but loads no stage until assigned.
8. A stop signal gives the worker a short grace period, followed by `std::_Exit` after three seconds even if socket I/O is blocked. This bounds process shutdown; it is not a guarantee of graceful request completion.

Source caveat: provider GPU selection determines reported identity and offered capacity, but the DHT launch arguments do not pass the chosen device index/UUID into the worker or set `CUDA_VISIBLE_DEVICES`. The worker leaves llama.cpp device selection at its default. Treat multi-GPU device pinning as **incomplete**, not verified by the single-GPU tests. The planner also uses configured offered memory, not a continuously refreshed free-VRAM reservation.

### 2.3 Sidecar, identity, NAT and DHT

C++ speaks ordinary loopback TCP. The Go sidecar owns network identity and transport. `newHost` explicitly enables Noise, hole punching, NAT port mapping and IPv4/IPv6 listeners; QUIC transport has its own secure connection machinery. This is not application-level end-to-end encryption against participating workers.

Inbound streams obtain identity from `stream.Conn().RemotePeer()`. The sidecar writes `DAN-P2P/1 <PeerID>` to the local receiver. That header is trusted **only because local processes are trusted**. `-allow-any` means any authenticated network identity may reach the worker, not that identities are unauthenticated.

The private Kademlia prefix is `/dan`. Provider namespaces are `dan/model/1/<sha>` and `dan/replica/1/<sha>`. A DHT record means “ask this peer,” not “this worker is available.” Worker capability replies come from the sidecar's status file; C++ placement subsequently performs a live greeting and reservation.

The configured five-minute provider validity and refresh every third of that are distinct from the worker status freshness check (15 seconds). Candidate queries run per model in parallel, up to eight models and 64 advertised peers per model; `gather` permits eight simultaneous live queries per collection. Recent failures are remembered in memory for two minutes.

**The three-second discovery number is conditional:** `gather` starts its grace timer after its first usable answer. A lookup with no usable answers can wait for discovery/query timeouts. Several per-model collections are joined, so a slow empty collection can still delay the combined response. It is incorrect to advertise a universal three-second discovery deadline.

Dialing reuses an existing direct or limited connection, otherwise tries addresses/peerstore, DHT resolution, then circuit addresses through configured relays. Explicit-address connection failures return from their branch rather than always traversing every fallback. Only new `/dan/ring/1.0.0` streams wait up to five seconds for a direct path; failed direct attempts are remembered for ten minutes. Existing streams stay on their original connection even if a later direct connection appears.

The infra relay starts from go-libp2p default resources and overrides duration/data limits: default two hours and 4 GiB **per direction per relayed circuit**, not a network-wide bandwidth guarantee. Reservation renewal does not extend an already-open circuit's lifetime indefinitely. Relay exhaustion can break a persistent replica.

### 2.4 Leasing, model files and reuse

`WorkerLease` has one route slot: `available -> reserved -> loading -> serving`. Reservation is mutex-protected and first-wins. A reservation expires; loading/serving do not use that expiry. Assignment must match the reserved route/model/range/context/session count/draft/activation format. Before assignment a matching release frees it; after assignment the held control connection determines lifetime. Serve connections use a ten-minute receive timeout; owners keep them active with metrics calls.

Clients specify SHA-256 identifiers, never model URLs. Workers resolve identifiers in their own catalogs and independently validate range, context, sessions and memory fit. The frame-size check is based on FP32-equivalent prompt bytes even when the route asks for compressed activations.

`prepare_range_model` reads GGUF metadata, selects `blk.<layer>.*`, embedding on the head, and norm/output on the tail. Tied embeddings also belong to a tail lacking a distinct output tensor. It creates a full-logical-size sparse GGUF and downloads only required byte ranges, coalesced and split into at most 256 MiB chunks. HTTP fetches use `curl`, retry, pinned revision/hash headers and `Content-Range` validation. An incomplete range index supports resumption; completed ranges have local SHA-256 checks.

The full remote GGUF hash is **not recomputed from bytes never downloaded**. Initial artifact identity relies on pinned origin metadata over HTTPS; local range hashes detect subsequent corruption. This is not a content-addressed P2P weight distribution system.

Cache reuse has three levels:

- Client model indexes: parsed `.index` files avoid repeated header fetches in `dan_client.cpp`.
- Disk stage ranges: exact model/range files plus `.ranges`; reusing them hashes the stored ranges. *(Fixed after this review:* `hash_range` now hashes each range in place via `sha256_file_range`; no temporary copy is written. The disk still reads every cached byte on reuse.*)*
- GPU residency: identical model/range/context/session-count assignments reuse `Stage`; activation format can change without reloading. Draft reuse has its own check.

There is no eviction budget, per-layer shared cache, overlapping-range deduplication or peer fetch. `cached_ranges` advertises filename/index existence as a hint; actual reuse validation happens later. Provider catalog initialization still calls `inspect_range_model` for each manifest; the CLI's parsed-index cache is not automatically shared with that startup path.

### 2.5 Planner and placement

Models are ordered by GGUF **logical byte size**, not parameter count or a quality score. “Largest model” therefore has that concrete meaning.

For offered bytes `M`, `stage_fits` subtracts `max(1 GiB, 15% of M)` in addition to the launcher's prior VRAM reserve. It charges selected tensor bytes and FP16 K/V:

`KV = context * sessions * stage_layers * (hidden / heads) * kv_heads * 2 * 2 bytes`.

The index uses tensor offset spans, including padding, rather than reconstructing each kernel allocation. This is a conservative file-size-based model with slack, not exact peak VRAM accounting. Draft weights/KV, graph workspaces, backend repacking and loading transients are not explicitly modeled.

Workers are filtered by availability, ABI, catalog, limits and authenticated ring target. Placement ranks by RTT buckets of 25 ms, a one-bucket relay penalty, memory and deterministic identity. At most eight candidates enter the planner. `plan_stages` searches minimum stage count first, candidate orders next, and cuts near proportional remaining memory, backtracking when necessary. Cache tilings replace a plan only within its stage-count limit.

Replica formation adds an own-worker-required head, alternative-owner speed comparison, and actual proposed-edge probes before reservation. Its speed estimate adds weight-byte cost at measured microseconds/GiB to half-RTTs. It does not optimize all cuts for throughput; it does not model acceptance, draft cost, upload bandwidth or concurrent-session service time. A measured edge pass/fail does not re-fit the estimate with the entire measured edge matrix.

Reservations and loads run in parallel across chosen workers. A busy reservation triggers release/re-greet/jitter/replanning; load failure aborts placement rather than repairing a partly loaded route.

### 2.6 Replica lifecycle and client attachment

`Owner::form` checks its own worker, discovers free peers, optionally delays behind larger peers, then calls the shared placement code. It may stand aside for an alternative owner estimated more than 25% faster. After two consecutive deferrals it stops deferring, avoiding endless mutual yielding. This is a heuristic, not election or consensus.

It probes every ring edge from the sender, including tail-to-head. Default limits allow relays and reject RTT over 150 ms. It reserves/loads, builds the ring, runs an eight-token warm-up, destroys the warm-up session, and only then marks the replica ready. Leases and weights remain resident between chats.

The owner advertises status via its sidecar. `findReplicas` queries announced owners live, checks owner identity/model/readiness and freshness, and creates local session forwards. `open_replica` orders usable replicas by model preference, measured token time, direct connection and owner RTT. The actual create-session request arbitrates remaining capacity; advertised free slots can race. With no chosen replica the ordinary client placement path remains available.

The front door maps each client's session IDs to internal worker sequence IDs. Reader/writer threads per client separate socket output from inference. The outgoing queue drops a client after 4096 queued frames. Requests run on owner threads, one return-reader demultiplexes by internal session, and a common member mutex serializes groups of control exchanges. A one-stage replica deliberately has one active request slot; a multistage replica can have up to its configured session count.

The queues are not uniformly bounded: front-door jobs, per-request inboxes and some per-client bookkeeping can grow independently of the outgoing frame cap. This is a friends-beta resource boundary, not hardened admission control.

## 3. Current implementation state

### 3.1 Verification performed during this review

- Release CPU build completed with the current checkout and patched llama.cpp.
- After that build, CTest passed **13/13** on Windows.
- Debug `provider_owned_formation_test` was rebuilt and executed successfully, retaining its `assert` checks. The build emitted a missing `pwsh.exe` diagnostic but returned success and produced a runnable test.
- `go test -count=1 ./...` passed, including the sidecar and older API gateway packages. These are local tests; their relay/NAT simulations do not establish physical WAN behavior.
- No new GPU benchmark, physical LAN test, WAN deployment, installer execution, race sanitizer or CUDA profiler run was performed.

### 3.2 Feature classification

| Feature | Classification | Evidence and limit |
|---|---|---|
| Dense-Qwen2 contiguous range inference | **WORKING AND TESTED** | Local tests and historical physical two-machine GPU runs. 32B September 8 evidence belongs to the older coordinator path; current decentralized 14B WAN tests are separately documented |
| DHT, capability query, live greeting, leases | **WORKING AND TESTED** | Current local Go/C++ tests and discovery/placement rehearsals; WAN use recorded in current guide |
| Forced relay ring | **WORKING AND TESTED** | Local `nat13.txt`, `replica11.txt` and scripts; historical real-internet relayed 14B tests. Local forced relay is not CGNAT hardware |
| Persistent self-forming replica, warm attachment, re-formation | **WORKING AND TESTED** | `replica11.txt` passes ordinary/failure/restart/race scenarios; guide records PC + RunPod WAN formation |
| Multiple sessions on multistage replicas | **WORKING AND TESTED** | `multi2/` local outputs and `wan-multi/client-{1,2}.out`; two sessions, not arbitrary scale |
| Loop mode and split speculation | **WORKING AND TESTED** | Local speculative baselines, `replica-spec2.txt`; historical WAN throughput measurements |
| FP32 wire | **WORKING AND TESTED** | Default path and controlled output equality checks; not universal hardware-independent bit identity |
| FP16/FP8 wire | **EXPERIMENTAL** | Implemented and locally exercised; FP8 WAN outcome recorded in guide. Lossy, opt-in; quality validation is limited |
| Speed-aware formation | **WORKING AND TESTED** | Unit/rehearsal evidence, guide's WAN estimate/measured comparison. Heuristic remains approximate |
| IPv6/DNS behavior | **IMPLEMENTED BUT ONLY LOCALLY TESTED** for the relevant automated paths | Local tests exist; owner's router did not establish a successful general IPv6 WAN deployment |
| Worker chunked prefill | **IMPLEMENTED BUT INCOMPLETE** in product integration | Worker accepts `--prefill-chunk`; older fixed-ring evidence exists; default zero and provider config does not expose it |
| Replica smaller-to-larger upgrade | **PLANNED / DOCUMENTATION ONLY** | No idle-upgrade search/dissolve transition in owner serve loop |
| Draft resource/tokenizer contract | **IMPLEMENTED BUT INCOMPLETE** | *(Updated after review:)* the head now checks draft weights + KV at load (estimate; reservation still ignores the draft) and a full tokenizer-equivalence check, falling back to plain decoding. Owner status still reports the configured draft, not the worker's actual state |
| Single-GPU provider selection | **WORKING AND TESTED** in exercised deployments | Does not prove configured GPU pinning on a machine with multiple GPUs |
| Multi-GPU device pinning | **IMPLEMENTED BUT INCOMPLETE** | Launcher selection/reporting is not propagated into DHT worker backend selection |
| Pipelined speculative rounds for one conversation | **NOT IMPLEMENTED** in active replicas | Older `engine/coordinator.cpp` has a distinct experimental path; current head/tail loop waits for round return |
| Cross-session GPU batching | **NOT IMPLEMENTED** | Sessions interleave separate `llama_decode` calls |
| Windows installer | **WORKING AND TESTED**, historical artifact | Guide identifies September 18 1.1.0 build; later September 19 source changes cannot be assumed present in it |
| Dedicated physical LAN coverage | **NOT ESTABLISHED** | One-host loopback/direct tests are not two-PC LAN receipts |
| Public friends deployment / production readiness | **NOT ESTABLISHED** | RunPod stood in for another machine; no real friend installation evidence, broad soak, adversarial hardening or operational SLO |
| Cache eviction/dedup/P2P weights | **NOT IMPLEMENTED** | Range caching and resume exist; those features do not |
| MoE/other model architectures/BF16 wire | **NOT IMPLEMENTED** in active DAN protocol/runtime | Upstream llama.cpp support alone is insufficient; patch rejects non-Qwen2 |
| Mid-request failover, result verification, reputation, Sybil resistance, payments, consensus | **NOT IMPLEMENTED** in active path; explicitly deferred | Older gateway retry and coordinator experiments are not replica repair |

### 3.3 Historical evidence quality

The rehearsal helper `Same-Outputs` compares rendered output strings, not activation/logit bytes or necessarily token-ID sequences. Both saved `multi2` outputs also matched the saved baseline when checked during this review. This supports output correctness for those cases, not universal numerical identity.

`build-client/results/replica10.txt` records a failed four-owner race with all nodes free. The later `replica11.txt` passes that scenario; current source contains the two-deferral escape. Preserve both receipts rather than claiming every run succeeded.

The two raw `wan-multi` client receipts name the same replica and report **13.45 and 13.79 tok/s**, TTFT **665 and 689 ms**, for 96 output tokens each. They establish successful client completions. Hardware, interleaving counts and edge topology need the accompanying worker evidence/guide; a client's `link=direct 0ms` describes its local owner link, not every GPU edge. `mode=hub` in these receipts describes the client's single front-door connection, not a hub carrying ordinary inter-stage inference.

The guide reports 2070 + RTX 2000 Ada, 79 ms direct edges, 131 ms measured versus 134 ms estimated per plain traversal, and 72 session switches in 90 head steps. It also reports FP8 reducing long-prompt TTFT from 3.2 seconds to 1.5–1.7 seconds while decode fell from 12.0 to 10.4 tok/s. These remain **historical reported measurements**, not results reproduced in this review.

## 4. Request/data-flow trace

For `client -> owner -> A -> B -> C`, the owner and A share a physical provider but are separate processes. “Local” still includes socket handling and CPU scheduling.

### 4.1 Attach and prefill

1. Client discovers a ready owner, connects its session tunnel and creates a session. Owner creates corresponding sessions on all members; each reserves a sequence within an already allocated llama context.
2. Client sends a token budget (`stream_prompt`) and prompt. Owner stores the budget and queues the request.
3. Under `members_mutex_`, the request thread exchanges the budget with C and sends prompt bytes to A. That budget acknowledgement adds an owner-to-tail RTT before A starts.
4. A tokenizes with the main model tokenizer, creates a batch with positions/sequence IDs, and calls `llama_decode` for the entire prompt by default. The graph runs embedding and A's layers, fills A's KV, and copies boundary hidden states to host.
5. A synchronizes and packs an activation. In `serve_control`, it also feeds the prompt into the draft synchronously, under the same stage mutex. Only after that does it forward the main activation to B.
6. B receives the whole frame, allocates a batch, expands/copies hidden states into FP32 embeddings, executes its layers, synchronizes, packs and forwards. C performs the same input work, then final norm/head and greedy sampling.
7. C streams the first sampled token toward the owner, then sends it to A on the loop edge if more output is requested. The owner return-reader queues the frame to the request; the request remaps IDs and queues it to the client writer.

Default prefill is serial across stages at whole-prompt granularity. With worker chunking explicitly enabled, earlier chunks can execute downstream while A computes later chunks. The sender still performs blocking sends, and A's control handler holds the stage mutex through its chunk loop and draft mirror.

### 4.2 Plain decode

For each subsequent token:

`C token -> A receive -> A layers/KV -> activation -> B layers/KV -> activation -> C layers/KV/head/sample -> token to owner, token back to A`.

One token of one session depends on completing this loop. C sends toward the owner before its feedback token, so a blocked return send can delay the next round. Owner/client forwarding usually carries tiny payloads and should be a small cost on a healthy local head, but it has not been isolated by measurement.

### 4.3 Speculative decode

A receives current token at position `p`, catches up the draft if needed, generates up to three proposals serially, then runs `[current, guess0, guess1, guess2]` as one main-model batch. The activation frame carries all rows and the guessed IDs. B verifies its layers on those rows; C produces one greedy prediction per row and accepts the matching prefix. C streams accepted tokens separately, feeds only the last accepted token back to A, and the next frame's position drives correction of speculative KV.

### 4.4 End, commit and cleanup

C ends at EOG, token limit or cancellation and sends `result`. Owner then either rolls back the cancelled answer to its starting position or commits the final sampled token so every stage's KV contains the completed conversation.

`commit_final_token` performs **sequential exchanges over each member control connection**: A returns `commit_activation` to owner; owner passes it to B; B returns its commit activation; owner passes it to C. `is_hot_path` deliberately excludes commit frame types. Thus this final pass differs from the decode ring. `end_request` is also acknowledged sequentially and checks position agreement. The owner sends its final acknowledgement only after cleanup/commit.

A successful next chat turn reuses the session KV and sends only the incremental formatted text. Reset/destroy removes that sequence's KV. Closing the client frees its sessions, not the replica's weights. A failed request can reset its own session while other sessions continue. A route/member failure instead aborts connections, wakes and joins request threads, releases leases and triggers new formation; no answer continuation is repaired automatically.

## 5. Numerical/dtype path

### 5.1 Exact application boundaries

| Boundary | Actual type and conversion | Source |
|---|---|---|
| Weights on disk | GGUF per-tensor types; not one dtype for the entire model. Catalog labels: 0.5B/1.5B Q4_K_M, 14B Q8_0, 32B Q5_K_M | `config/provider-owned-*.json`, GGUF loader |
| Loaded weights | Quantized tensors retained where backend supports them; backend may repack or convert for a chosen operation | `Stage` constructor, llama model loader/CUDA dispatch |
| Prompt/token IDs | `llama_token` integer IDs; token payload uses big-endian 32-bit fields | `tokenize`, `token_frame`, `run_first` |
| Main graph activations | FP32 external embeddings/residual boundary; internal attention/matmul operands can differ | patched `qwen2.cpp`, `build_inp_embd_direct`, ggml |
| KV | K and V default `GGML_TYPE_F16`; stage-local layers, sequence-specific positions | `llama_context_default_params`, range KV filter |
| Non-tail stage output | Host `float*` from `llama_get_embeddings_ith`, FP32 | `put_hidden` |
| FP32 wire | Raw little-endian FP32 bytes copied from host output | `put_hidden`, `get_hidden` |
| FP16 wire | CPU `ggml_fp32_to_fp16_row`; raw little-endian half bytes; receiver CPU `ggml_fp16_to_fp32_row` | same |
| FP8 wire | CPU row maximum/scale, FP32 divide, round to e4m3fn; scale stored as **big-endian FP32 bits**, then byte codes; receiver expands code and multiplies scale into FP32 | `pack_fp8_row`, `unpack_fp8_row` |
| Next-stage input | FP32 `llama_batch.embd`; copied/uploaded into direct input tensor | `get_hidden`, patched graph input |
| Tail logits | Host FP32 logits; CPU greedy sampler reads them | llama context output extraction, `llama_sampler_sample` |
| Draft | Its own GGUF quantization, llama context, FP16 KV and FP32 logits; same `Stage` class, full layer range | `load_draft_model`, `draft_proposals` |

Observed loader logs for the 1.5B Q4_K_M test show **F32, Q4_K and Q6_K tensors**. Q4_K_M is a mixed quantization recipe, not “all tensors are four-bit.” The planner discards individual GGUF type fields and sizes tensors by file offsets; it cannot currently produce a trustworthy per-tensor dtype inventory by itself.

### 5.2 Compute and accumulation are operation-dependent

The local llama.cpp backend establishes the following, which is more precise than “FP32 compute”:

- Quantized matrix-vector/matrix-matrix dispatch chooses MMVQ/MMQ, float kernels or cuBLAS based on type, dimensions and GPU architecture. Quantized dot paths can quantize input rows to Q8_1 and use integer products with floating-point scaling/reduction, returning FP32 tensors.
- CUDA cuBLAS traits include FP32 operands/FP32 accumulation, BF16 operands/FP32 accumulation, and FP16 operands/FP16 accumulation. Operation precision flags and `GGML_CUDA_CUBLAS_COMPUTE_TYPE` influence selection. These alternatives are not all necessarily exercised by one DAN run.
- Attention graph construction casts relevant operands for flash attention and requests FP32 precision for flash attention/QK accumulation. KV storage remains FP16; storage dtype is not accumulator dtype.
- The final API-visible logits and stage embeddings are FP32 even when earlier products used reduced precision.
- CPU and CUDA paths select different kernels. Batch width, context length, graph reuse, hardware generation, compiler and backend version can change arithmetic order.

Relevant local backend files: `src/llama-context.cpp` (defaults, output extraction, synchronization), `src/llama-graph.cpp` (input and attention precision), `src/models/qwen2.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmvq.cu` and quantization kernels under the pinned checkout. Runtime kernel selection and accumulation for **each actual GPU invocation** require a profiler/graph trace; no single global compute dtype can honestly be inferred from DAN's wire enum.

### 5.3 Serialization contract

`protocol.hpp` encodes a 48-byte big-endian header: magic, version 2, frame type, session, request, position, rows, columns, dtype, zero reserved field, payload length. Maximum payload is 64 MiB. Header and payload are sent in two `send_all` calls; receiver reads the complete header, resizes a vector, then reads the complete payload.

Activation payload: **8-byte big-endian compute nanoseconds**, then encoded rows. A split speculative activation appends `(rows - 1)` big-endian token IDs. Each stage replaces the timing prefix with **its own** compute time; it is not an accumulated per-stage trace. Ordinary token feedback is four payload bytes. Streamed result payload is token ID (4), compute ns (8), EOG flag (1), then token text. There is no BF16 wire enum or general lossless compression layer.

For 14B hidden size 5120:

| Format | Activation values/scales per token | Complete ordinary activation frame | 512-row prompt values/scales |
|---|---:|---:|---:|
| FP32 | 20,480 B | 20,536 B | 10,485,760 B |
| FP16 | 10,240 B | 10,296 B | 5,242,880 B |
| FP8 | 5,124 B | 5,180 B | 2,623,488 B |

Headers, timing prefixes, guesses, feedback, token text, libp2p/TCP/QUIC framing and encryption are additional to the values/scales columns. A four-row FP32 speculative frame with three guesses is 81,988 B; FP8 is 20,564 B. Bytes per **committed** token depend on acceptance.

### 5.4 Numerical identity

| Comparison | Appropriate expectation |
|---|---|
| FP32 serialization round trip | Lossless bit preservation on supported little-endian machines |
| Matched CPU reference vs tested FP32 split ring | Exact output/token equality is a useful regression requirement and has controlled-test evidence |
| Arbitrary CPU/GPU/split/hardware reference | No blanket bit-identical guarantee; kernel dispatch and arithmetic ordering differ |
| FP16 or FP8 vs FP32 | Legitimate numerical and token drift, including during prefill |
| Plain decode vs batched verification | Can drift even with identical weights and FP32 wire because matrix shapes select different arithmetic |
| Speculating replica vs matched speculating placed route | Strong controlled regression target; tested receipts exist |
| Same row encoded alone vs in FP8 multirow frame | Codec is row-local; identical incoming row produces identical encoding. This does not force upstream computed rows to be identical |

`speculation.hpp`'s “exactly what plain decoding would have produced” is an algorithmic claim that omits finite-precision batch differences. Likewise, `activations.hpp`'s “stages always compute in f32” describes the boundary, not all kernels. FP8 only during prompts would still alter KV and potentially all subsequent output; it would not restore FP32-reference semantics.

## 6. Performance analysis

### 6.1 What dominates in each operating regime

For one plain session, approximate traversal time is:

`L = sum(stage service times) + sum(edge one-way transit + bytes / bandwidth) + host/queue overhead`.

Decode rate is approximately `1000 / L_ms`. For serial speculation with mean committed tokens `g`:

`rate ~= 1000 * g / (draft_ms + sum(verify_service_ms) + wire_ms + overhead_ms)`.

The denominator includes costs of rejected candidates. Increasing width is helpful only if added committed tokens outweigh extra drafting, verification and payload cost.

The guide's two-stage 32+16 ms compute and roughly 64 ms round-network example suggests about 112 ms before other overhead, consistent in scale with roughly nine plain tokens/s. This is an illustrative decomposition of that historical setup, not a current measured breakdown. With 79 ms RTT between two nodes, the two opposite one-way edges together cost about 79 ms under symmetry, not 158 ms; using RTT per edge without halving would double-count.

For cold routes, downloads/reloads dominate startup by seconds to minutes. For a ready replica, that cost is removed from request TTFT. For long prompts over a residential uplink, activation bytes dominate; for short decode frames on long RTT links, latency usually dominates. Local GPU-only decoding can instead be memory bandwidth, launch, logits-copy or CPU-sampling limited.

### 6.2 Concrete costs and priority

| Cost in current source | Primary impact | Relative importance / evidence |
|---|---|---|
| Metadata lookup, discovery, rank delay, reserve/load/link/warm-up | Cold startup, TTFT | Seconds/minutes historically; amortized by replicas, not removed from node startup |
| Whole-prompt stage-by-stage prefill | TTFT, prefill throughput, GPU idle | Potentially large on long prompts; worker chunking exists but is not exposed by provider |
| Draft prompt computed before head forwards | TTFT, head utilization | Definite serial dependency; draft share unmeasured |
| Tail budget acknowledgement before head send | TTFT | One remote control RTT; meaningful for warm short requests |
| Dependent whole-ring decode | tok/s, WAN scalability, GPU utilization | Structural single-session limit; measured speculation and multi-session gains support importance |
| Serial draft calls under stage mutex | tok/s, head contention | Three draft steps plus occasional catch-up; timing not isolated |
| `llama_batch_init/free` and vector construction each frame | CPU overhead, memory churn | Certain allocations; likely secondary on high-RTT WAN, more relevant to fast local/small models |
| Backend input upload and output readback | tok/s, prefill, memory traffic | Required by host boundary. Wire compression currently does not reduce these FP32 boundary transfers |
| `llama_synchronize` plus getters/sampler synchronization | CPU/GPU overlap, launch overhead | Prevents DAN-level asynchronous handoff; duplicate calls may be cheap after first completion, so deleting them blindly is not justified |
| Full vocabulary logits readback for every verified position | tok/s, draft/verify cost | About 0.61 MB per 151,936-entry FP32 row; four rows about 2.43 MB per verifier pass, plus draft readbacks. Device-side greedy sampling may help; needs measurement |
| FP8 CPU maximum/division/scalar encoding; FP16 conversion | CPU cost vs WAN payload | Clear byte reduction, conversion timing missing. FP8 decode regressed in reported WAN comparison |
| `run_middle` copies input frame before replacing its payload | Prefill CPU/memory traffic | *(Fixed after review:* header-only copy via `frame_header`; the gain was not measured)* |
| Zero-initializing output vectors then overwriting data | Prefill memory traffic | Avoidable in some paths, but safe lifetime/bounds must remain |
| Two frame writes, loopback and libp2p copies | Per-frame overhead, bandwidth | Ordinary buffered I/O, no GPU-direct network path. Significance unmeasured |
| Blocking send before reading next ring frame | Multi-session throughput, backpressure | Worker cannot consume next upstream frame during a stalled downstream send |
| Tail return send before feedback | tok/s under owner/return congestion | Owner decouples the final client, but return-link stalls still affect loop |
| Common owner member mutex and sequential control exchanges | Session start/end latency, multi-session contention | Can serialize one request's commit/health with another's start; bulk decoding is outside this mutex |
| Final-token owner-mediated commit and end acknowledgements | Completion latency, next-turn TTFT | Extra computation and WAN exchanges every answer; short responses suffer proportionally more |
| Synchronous detailed `fprintf` per frame | CPU/I/O, jitter | Can perturb measurements; cost not isolated |
| Hashing cached ranges through temporary files | Formation/restart, disk wear/space | Potentially substantial for many GB; absent when exact loaded Stage is reused |
| Session-count-scaled KV allocation | Memory/model fit | Definite up-front cost; increasing session count can prevent a larger model fitting |

The most defensible initial gains are eliminating known unnecessary host copies, exposing existing prefill chunking, avoiding repeated startup work, and improving draft/placement accounting. None needs an estimate of GPU occupancy to be understood. Their actual end-to-end gains still require A/B measurement.

## 7. Speculative decoding analysis

### 7.1 State transitions

The draft is a complete smaller catalog model on the **head's device/runtime**, held as a second `Stage`. Placement chooses the smallest other model by logical bytes that the head advertises. It does not select for measured acceptance, trained compatibility or best tokens-per-second.

`draft_width=4` means up to three proposed future tokens. `draft_proposals` feeds current/previous guess to the draft one at a time, sampling greedily after each synchronous call. The last guessed token has not yet been fed back into draft KV.

The target then verifies current plus guesses. Every stage mutates its KV for every verified position, including rejected ones. C compares proposed tokens with target samples using `accept_speculation`: sample zero is always emitted, then later samples only while the preceding guesses matched. This is greedy acceptance, not probability-ratio speculative sampling for nonzero temperature.

After full acceptance, `catch_up_draft` feeds the last proposal that the draft had not consumed. After partial acceptance, the next frame starts at a lower absolute position; `Stage::execute` removes KV from there onward before running. Main and draft sessions mirror create/reset/destroy/end/prompt/final commit. Split rounds track guesses **per session** in `RingState::guessed`.

At a token-budget/EOG boundary C may emit fewer tokens than a successful verification produced; the final commit's lower position also truncates excess KV. Near context exhaustion, however, draft/verify width is not explicitly clamped to remaining main context or output budget. A speculative batch can overflow remaining context even when a plain step might fit. *(Fixed after review: width is now limited to the remaining context (`speculation_room`); real-model runs with a 20-token context confirmed that plain and speculative routes both reach position 19. The output budget is still not clamped; the tail truncates excess tokens as before.)*

### 7.2 Transport and bubbles

No extra draft RPC traverses the WAN. The head performs drafting locally, adds row payloads and a small proposal-ID suffix, and C returns one feedback token per round. It emits separate result frames for committed output tokens. There is no next-round work from that conversation in flight during verification.

Consequently speculation saves network round trips and weight traversal per committed token, but leaves bubbles between rounds. The draft does not overlap the head's target verification, and the current code does not opportunistically draft a future round while downstream verifies. Other sessions can occupy stages during those gaps, subject to the single worker mutex and socket ordering.

Verification uses a real multi-token `llama_batch`, with logits requested for all verification rows. This is substantially different from calling target decode four times. It may still select small-batch kernels rather than the most efficient large GEMM; only actual profiling can establish utilization. llama.cpp CUDA graphs can reduce launches, but synchronous output consumption remains.

### 7.3 Correctness and throughput limitations

- **Tokenizer compatibility is assumed:** raw target token IDs are fed into the draft without a vocab/tokenizer equivalence check. Current catalog family matching is convention, not validation. *(Fixed after review: full table + metadata + corpus-tokenization check, fail closed.)*
- **Memory is incomplete:** main stage may fit while draft weights/KV/workspace do not. Draft load exceptions fall back, but backend OOM may not be cleanly recoverable in every case. *(Partly fixed after review: weights and both KV caches are charged at draft load; workspace is covered only by the reserve, and reservation does not charge the draft.)*
- **Failure handling differs:** split `draft_round` catches proposal failures and disables drafting for the route; local `local_decode` catches a failure around the combined step and returns an error. “Every draft problem silently falls back” is false.
- **Resident draft metadata is optimistic:** the selected draft SHA is recorded in placement/replica status even if the worker could not load it or later disables it. Status is not proof that speculation is active.
- **Acceptance statistics are incomplete:** logs report proposed and accepted guesses, but counters are not structured per model/session/phase. The first guaranteed token must not be counted as an accepted guess.
- **Shape changes alter numerics:** a correct acceptance algorithm cannot guarantee serial-reference equality when verifier logits differ because of batching.

Semantics-preserving opportunities include reducing draft logits readback/CPU sampling overhead, ensuring reusable draft graph shapes work, eliminating redundant frame copies, reporting effective draft state, and avoiding unnecessary speculative work at the output/context boundary. Adaptive width is algorithmically compatible with greedy verification, but can still change numerical output by changing verifier batch shapes; benchmark correctness against the stated numerical contract.

## 8. Pipeline/GPU utilization analysis

For one session, the stages alternate compute and waiting. During A's work B/C wait upstream; during B's work A waits for feedback and C waits upstream; during C's work A/B wait. Network transit and host synchronization introduce additional gaps. “Waiting downstream” is visible when a blocking send cannot progress; otherwise upstream/feedback waiting is mostly socket receive time.

No available trace measures separate GPU computing, upstream wait, downstream wait, network wait and CUDA synchronization wait. Per-step wall times cannot recover those categories uniquely. `nvidia-smi` utilization averages also cannot establish exact overlap or per-token idle fractions.

For intuition only, if a two-stage plain round takes 131 ms and a stage computes for 32 ms, its single-session compute duty would be about 24%. That is not a measured utilization result for the current 131 ms run: the 32 ms value came from a different historical setup.

With `N` independent sessions, throughput is bounded by both outstanding work and the bottleneck stage:

`aggregate plain steps/s <= min(N / L_seconds, 1 / max(stage_service_seconds))`.

This upper bound ignores queueing and communication bottlenecks. For speculation, use verified frames and committed tokens/frame; do not multiply unrelated headline speedups.

| Mechanism | Current reality | Practical assessment |
|---|---|---|
| Independent sessions around the ring | Implemented and two-session WAN-tested | Best existing source of useful overlap; quantify scaling to 3–4 sessions before changing scheduler |
| Several sessions in one GPU pass | Absent; explicitly deferred | Could improve weight reuse, but KV batching, fairness, cancellation and numerical testing are real work; do not add now |
| Speculative batch | Implemented, one round/session in flight | Amortizes traversal, not full pipeline filling |
| Chunked pipelined prefill | Worker mechanism exists | Realistic incremental integration; tune chunk size and fairness under concurrent decode |
| Dedicated receive/send buffering | Absent at worker frame level | Could overlap next frame reception/current send with GPU work from another session; requires bounded queues and single-writer rules |
| Async H2D/D2H and event handoff | Backend primitives exist, DAN host interface waits | Requires runtime/API and buffer-lifetime changes; no benefit from replacing sync with unsafe pointer use |
| Additional CUDA streams | Backend already uses streams | Do not add an application stream abstraction without evidence of overlap opportunity and context safety |
| Multiple speculative rounds in flight | Absent in current route | New execution semantics for stale work/epochs/rollback; later experiment, not a quick optimization |

When increasing sessions, budget their KV and draft KV first. A replica with more reserved sessions can form a smaller model or require more stages, offsetting throughput gains. On one-stage replicas, extra session capacity currently means resident conversations taking turns, not simultaneous generation.

## 9. WAN/network analysis

### 9.1 What the network carries

Every inter-stage edge carries prompt rows and subsequent single/speculative rows. Tail-to-head carries token feedback; tail-to-owner carries output tokens; owner-to-members carries setup, health, session management and final commits. Infra is absent from direct paths but can carry the entire data stream on circuit relay paths.

`bridge` runs `io.Copy` in both directions and logs lifetime bytes at closure. There is no remote GPU memory transport, tensor-aware congestion control, frame-level network timestamp, or guaranteed communication/compute overlap. Encryption/framing remain sidecar responsibilities.

A 512-row FP32 14B prompt costs about 10.5 MB per stage edge. At 10 Mbit/s usable sender upload, payload transmission alone is about 8.39 seconds per such edge; FP8 reduces that arithmetic bound to about 2.10 seconds. These are bandwidth calculations, not a DAN benchmark. For a 20 KB decode frame on that link, transmission is about 16 ms; RTT may still dominate. Planning only with pings cannot distinguish these workloads.

### 9.2 Current limitations and useful improvements

- Edge probes use best-of-three ping RTT and classify direct if a direct connection exists. That is not a measured bulk throughput or necessarily the path of every long-lived inference stream.
- Peer RTT measured through capability queries includes application response work and is not interchangeable with ping RTT without qualification.
- Worker discovery filters busy nodes. An idle small replica reserves all its members, so another owner cannot see that capacity as available to form a larger model.
- Freshness uses wall-clock timestamps. Clock skew can admit future-dated status or reject otherwise live remote replica status. Durations/profiling should use monotonic clocks locally.
- `advertiseModels` only reconciles running advertisement goroutines inside the fresh-status branch. If an already advertised worker's file becomes stale while its sidecar lives, existing model advertisements can continue refreshing. Live capability filtering prevents placement onto an offline worker, but the guide's implied expiration behavior is too strong.
- Relay data/duration caps are intentional resource policy. Removing them to keep replicas alive would violate the present bounded-infrastructure design. Measure defaults such as relay buffer size independently of caps.
- Control-plane health and return-stream liveness have different failure budgets. An idle ring can lose a middle link before the owner proves that loss through an actual traversal. “Dissolves immediately on any link failure” is not a general guarantee: request return timeout is 120 seconds; member operations have their own timeouts.

Useful near-term WAN work is directional payload-aware measurement, removing repeated setup costs, preserving loaded replicas, and selecting better feasible splits. Unlimited relay traffic, frequent global mesh scans and automatic repair are not prerequisites.

## 10. Shard comparison

Shard is not one uniform execution engine. The current checkout moves model engines under `engines/minimax_m25`, `engines/kimi_k3` and `engines/deepseek_v4`; many docs still name their old `phase0/` locations. `shard/node.py` defines `ModelRuntime` methods that raise `NotImplementedError`, and `shard/specdec.py` contains a stub generic speculative loop. Real serving implementations live elsewhere. The deployed `shard.stage`/`shard.coordinate` entrypoints import the MiniMax engine, rather than dispatching arbitrary models through the generic interface. [Runtime interface](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/node.py), [stage entrypoint](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/stage.py).

| Area | DAN now | Shard/c0mpute implementation | Implication |
|---|---|---|---|
| Runtime boundary | One patched llama.cpp dense-Qwen2 stage | Separate tuned model engines plus incomplete generic interface | Do not replace a working runtime with an interface-shaped promise |
| Model support | Qwen2/Qwen2.5 dense GGUF | MiniMax MoE, Kimi and V4 engines; other research runs | Wider support reflects substantial model-specific work |
| Layer/range ownership | Contiguous GGUF tensor ranges in sparse file | Contiguous layers with safetensors/checkpoint mappings, engine-specific boundaries | Same basic partitioning idea; file/layout mechanics differ |
| MoE | Not supported by DAN patch | Local routed-expert execution using vLLM/custom/reference kernels | MoE is not synonymous with remote expert parallelism |
| Draft | Independent small full model on head | MiniMax EAGLE/n-gram variants; V4 DSpark depends on tail-layer taps | No universal “move draft to tail” rule |
| Speculation | One multirow verify traversal, then feedback | MiniMax chunk/tree and per-stream paths; V4 pipelined single-position frames with epoch fencing | Porting pipelining needs new stale-work semantics |
| Activation precision | FP32 boundary, f32/f16/fp8 wire | Predominantly BF16 model boundaries; codecs permit f32/f16/bf16 and FP8 forms | DAN already has FP8 wire, but conversion placement differs |
| FP8 scaling | Per row, FP32 scale, CPU conversion | MiniMax activation scale per tensor; V4 per position/residual stream with BF16 scale | V4's row-local principle is already represented in DAN; MiniMax codec is not identical |
| Wire format | Compact fixed binary header and typed payload | JSON metadata plus tensor blobs; PSK wire or libp2p message layer | Copy codec ideas, not the entire message format |
| CUDA graphs | Inherited ggml support, enabled in local CUDA build | Explicit model/row/batch/whole-layer graph runners | DAN needs graph-use measurement, not an automatic graph rewrite |
| Streams/synchronization | Sync at host output boundary | Graph capture streams and explicit synchronization; CPU tensor transport remains | Shard also pays host materialization and sync costs |
| Kernels | ggml quantized CPU/CUDA dispatch, optional flash attention | vLLM NVFP4 FusedMoE, PyTorch attention, custom Triton/CUDA kernels, model-specific tuning | Kernel gains depend on model/hardware, especially sm120 |
| KV | llama sequences, FP16 storage, absolute-position truncation | Engine-specific static/per-row caches; optional quantization; V4/Kimi state differs | Recurrent/compressed state needs more than simple KV truncation |
| Multiple sessions | Persistent conversations, independent frames | MiniMax lockstep batch and default de-lockstep paths exist; coordinator CLI remains single-job | An available batched engine is not proof the product entrypoint schedules it |
| Prefill | Whole prompt by default; worker chunk option | MiniMax coordinator chunks/pipelines; V4 has different prefill limits and OOM history | Borrow measured chunk integration, not a generic claim that all Shard prefills pipeline |
| Persistence | Providers autonomously form/reserve/warm replicas | Warm swarms with control-plane lifecycle and stage supervision | Similar amortization, different authority |
| Topology | Fewest stages, memory cuts, cache reuse, RTT rank, rough speed yield | Topology module includes directed latency ordering and compute/upload-aware profiles | Directional bandwidth and actual service times are useful incremental inputs |
| Weights/cache | Pinned origin HTTP ranges; no eviction/P2P | Verified manifest/shard fetch providers, HTTP resume, local and libp2p sources | P2P fetch is a larger supply feature; first fix local reuse costs |
| NAT/relay | Go sidecar, Noise/QUIC, auto-relay, bounded circuit resources | Similar sidecar; explicit reservation behavior; relay buffers increased and circuit limits removed | Buffer tuning may transfer; unlimited limits do not |
| Failure | Session reset or whole replica dissolution/re-form | Supervised edges, watchdogs, warm adoption/healing and orchestration paths | Useful diagnostics transfer; repair policy is outside current scope |
| Instrumentation | Compute prefixes/logs, client metrics, status | Per-stage spans, in-flight/waste statistics, workload receipts | Strongest thing to borrow immediately |

Sources for these distinctions: [MiniMax engine](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/minimax_m25/m25_stage.py), [MiniMax pipeline](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/minimax_m25/m25_pipe.py), [V4 pipeline](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_pipe.py), [transport](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/transport.py), [fetch](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/fetch.py), [topology](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/topology.py), [sidecar](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/sidecar/main.go).

### 10.1 Public claims versus executable evidence

**Generic runtime:** the newer README acknowledges separate engines; the older “one engine, every model” design document is a direction, not an implemented universal adapter. `VllmRuntime` remains described as future work. [Design document](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/docs/MODEL_RUNTIME.md).

**Decentralization:** c0mpute workers connect to an orchestrator using Socket.IO, and `SwarmManager` invokes `shard.plan` then emits `swarm:assign`. The network architecture document's DHT/gossip self-organization language does not replace this actual scheduling authority. This is incompatible with DAN's no-infrastructure-scheduler invariant. [Worker](https://github.com/leyten/c0mpute/blob/fcd4690fb6b19af51e801069ce6d35ae93b6db7a/c0mpute-worker/src/shard-worker.ts), [swarm manager](https://github.com/leyten/c0mpute/blob/fcd4690fb6b19af51e801069ce6d35ae93b6db7a/lib/orchestrator/swarm.ts), [planner seam](https://github.com/leyten/c0mpute/blob/fcd4690fb6b19af51e801069ce6d35ae93b6db7a/lib/orchestrator/swarm-seam.ts).

**Bit-identical headlines:** the V4 README's single-machine wording is stronger than the cited matrix's explicit comparisons to greedy runs on the same ring. The matrix is evidence for those configured comparisons, not proof of equality across arbitrary hardware, wire precision or standalone baselines. Its headline 30.15 tok/s is a warm workload result; other context settings/run order show substantial variation. Signed receipts authenticate a statement/output digest, not independently prove every GPU computation correct. [V4 matrix receipt](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/docs/receipts/v4-flash-matrix-20260802.json), [README](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/README.md).

**Pipeline filling:** V4's own measurements report that filling almost all available in-flight capacity can reduce useful throughput because of incorrect speculative work. Earlier and later sections describe different bottleneck regimes after successive optimizations. They cannot be combined into one timeless claim that either compute or network always dominates. Its suggestion to use best-of-N for congested boxes is useful as an optimistic capacity bound, but median and tail latency remain necessary for user-facing performance. [V4 engine measurements](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/docs/V4_FLASH_ENGINE.md).

## 11. Techniques DAN can borrow from Shard

| Technique | Classification | Reason and required adaptation |
|---|---|---|
| Resident layers, warm rings | **ALREADY IN DAN** | Replica formation already loads once and warms before READY |
| Direct stage-to-stage ring | **ALREADY IN DAN** | Ordinary decode already avoids owner-mediated activations |
| Independent session frames filling WAN gaps | **ALREADY IN DAN** | DAN's multi-session ring resembles the useful de-lockstep principle |
| FP8 row-local activation encoding | **ALREADY IN DAN** | Existing DAN codec captures the position-independent scale idea; retain its numerical tests |
| CUDA graphs | **ALREADY IN DAN** at backend level | Verify actual graph hits/rebuilds before investing in another capture layer |
| Per-stage service/compute spans and waste counters | **DIRECTLY APPLICABLE** | Start with local logs/counters, correlate offline, avoid per-token RPCs |
| Payload-aware directional upload measurement | **APPLICABLE WITH MODIFICATION** | Measure at formation or with sparse probes; retain provider/client-local planning |
| Pipelined prefill with explicit chunk/depth limits | **APPLICABLE WITH MODIFICATION** | Integrate existing worker option; ensure session fairness and draft consistency |
| Segmented buffer sends / fewer serialization copies | **APPLICABLE WITH MODIFICATION** | DAN already separates fixed header/payload; optimize its payload ownership rather than porting JSON framing |
| Static/reusable draft execution buffers | **APPLICABLE WITH MODIFICATION** | Work through llama.cpp's existing context/graph allocator rather than PyTorch graph runners |
| Draft-only lossy optimization | **APPLICABLE WITH MODIFICATION** | Target still verifies proposals, but tokenizer compatibility and state semantics remain essential |
| Bounded failure diagnostics/watchdog phases | **DIRECTLY APPLICABLE** as instrumentation | Distinguish setup, first response, prefill progress and steady decode; do not add automatic repair implicitly |
| Local cache verification/resume discipline | **ALREADY IN DAN** in principle | Improve implementation cost; do not remove hash checks to claim faster loads |

## 12. Techniques DAN should not borrow

| Technique | Classification | Why |
|---|---|---|
| Central orchestrator forming all swarms | **ARCHITECTURALLY INCOMPATIBLE** | Violates DAN's infrastructure boundary |
| Removing relay duration/data limits | **ARCHITECTURALLY INCOMPATIBLE** with current policy | Transfers unbounded costs to shared infra; buffer tuning is a separate question |
| MiniMax NVFP4 MoE-specific kernels | **MODEL-SPECIFIC** | Current DAN targets dense Qwen2 quantized GGUF and multiple NVIDIA generations |
| DeepSeek DSpark at tail | **MODEL-SPECIFIC** | Depends on MTP blocks and final-layer taps absent in current Qwen2 models |
| Kimi recurrent/AttnRes or V4 multi-residual wire state | **MODEL-SPECIFIC** | DAN's single hidden matrix/KV contract cannot represent these by changing a model name |
| Tree/deep speculative pipeline transplant | **NOT WORTH IMPLEMENTING YET** | Requires position/epoch fencing, stale-frame discard and more numerical validation; useful simpler overlap already exists |
| Broad runtime interface/factory before a second supported architecture | **NOT WORTH IMPLEMENTING YET** | Existing backend is a useful boundary; add an adapter only around a concrete second model |
| Generic JSON tensor protocol | **NOT WORTH IMPLEMENTING YET** | Would replace a small, established binary contract without demonstrated performance gain |
| Continuous multi-session GPU batching | **NOT WORTH IMPLEMENTING YET** and explicitly deferred | First measure existing independent sessions and prefill interference |
| GPU busy-work keepalive / network padding | **NOT WORTH IMPLEMENTING YET** | Adds energy/traffic; only consider after measuring clock/cwnd idle recovery on DAN |
| Signed receipts, economics, healing/election | **NOT WORTH IMPLEMENTING YET** for this performance task | Different product/security scope; not needed for profiling or current-path optimization |

## 13. DAN-specific optimization opportunities

**Make persistent replicas more valuable.** Their lifetime can amortize model-index loading, graph warm-up, link/bandwidth probes and buffers. Measure these once per replica generation rather than repeating them per request. The idle small-model trap prevents this architecture from using newly available capacity; an idle upgrade rule is a capacity feature, not a new network scheduler.

**Optimize resident reuse before distributed files.** Exact loaded-stage reuse is already far faster than reload. A bounded cache policy and direct streaming range hashes address actual disk costs without introducing a peer weight exchange. A later cache layout could share overlapping layers, but first measure how frequently splits change and how many bytes are duplicated.

**Separate single-session latency from multi-session service capacity.** For one session, minimize sum of compute and link time. For many sessions, reduce the slowest stage's service time while respecting memory and payload cost. DAN's memory-proportional cuts do neither exactly. A small measured cut-search over existing candidates is more appropriate than a global scheduling system.

**Use head/tail asymmetry.** Head cost includes embedding and draft; tail includes norm/head/logit extraction plus output/feedback. Speed records normalized only by layer bytes hide these fixed costs. Maintain a small distinction between prompt, ordinary decode and verification batch timings before adjusting splits.

**Keep prompt compression separate from decode policy.** FP8's reported WAN benefit is TTFT, while decode regressed. An explicit prompt-only format policy is plausible, but it changes prefill/KV numerics and needs frame-phase signaling; `rows > 1` is insufficient because speculative decode also has multiple rows. FP16 may be a better cost/quality tradeoff for some links.

**Improve draft correctness and resource admission before width.** Validate token-ID compatibility, charge draft memory, report whether drafting is really active, and clamp context-edge speculation. The existing automatic draft selection should not silently become a broader model compatibility promise.

**Future MoE need not change distribution granularity.** A first supported MoE can still keep all experts of a layer on that layer's worker. Memory then scales with total stored experts, while compute/traffic depend on selected experts. DAN's current bytes-to-time estimator and dense KV assumptions would need replacement for that actual model. Distributed expert parallelism adds WAN all-to-all and is a separate execution model, not the next optimization.

## 14. Missing measurements

### 14.1 Current telemetry limitations

Worker compute time includes portions of runtime execution and synchronization, but excludes most input decoding/packing and draft work. Labels based on row count can call speculative work “prefill.” Middle stages overwrite timing prefixes; owners do not receive a full per-stage breakdown. `note_step` accepts small batches and can mix commit/short-prefill/speculative observations. Single-stage speculative return bypasses the ordinary `note_step` path. A single persistent `decode-speed.txt` lacks model, precision, device and batch-width keys.

Owner `measured_ms_` starts at first token and updates after commit/end, so it includes completion bookkeeping. It also depends on concurrent load. Client `decode_tok_s` similarly needs its timing endpoints understood before comparing with pure GPU decode. Sidecar bytes are mostly emitted at stream closure, not periodically for long-lived replicas. There is no direct measurement of GPU idle time, synchronization time, owner forwarding cost or effective activation bandwidth.

### 14.2 Minimal proposed profiling layer

Use existing logs/status with one opt-in profiling flag, not a new service or telemetry dependency. Keep aggregate counters/histograms per request and stage, with a sampled detailed trace (for example one round in 64). Log replica generation, route/session/request/position, phase, rows, dtype and actual payload bytes. Never log prompt text or tensor values for profiling.

| Measurement | Minimal capture point | Interpretation |
|---|---|---|
| Stage service time | After full frame receive to after frame send | Includes CPU, GPU wait, pack and blocked send |
| Socket receive/send duration and bytes | `recv_frame` / `send_frame`, tagged edge | Receive can include upstream wait; send can stop at local buffers. Neither alone is exact WAN transit |
| Deserialize/pack time | `get_hidden` / `put_hidden` | CPU conversion/copy overhead separate from runtime |
| Main decode wall time | Around `llama_decode` plus existing sync | Comparable to current logs, clearly phase/batch labeled |
| Draft prompt, proposal, catch-up | Around corresponding `draft->handle` calls | Charge draft time separately and count generated/accepted guesses |
| Verify time and committed work | Main verify call, `verify_round` | Rows verified, guesses proposed, accepted guesses, tokens emitted, discarded KV positions |
| KV operations | Create/reset/remove/rollback/end operations | Time and logical tokens affected; do not equate logical removal with bytes physically copied |
| TTFT and token loop | Client request start/first token; head feedback entry timestamps | Separate startup, first response, steady rounds and finalization |
| Owner overhead | Return reader receive, inbox post/pop, client queue post/write | Local monotonic spans; include member-mutex wait/hold time |
| Live transport path/bytes | Extend existing sidecar stream counters, periodic low-rate snapshot | Actual stream direct/relay classification, not merely discovery label |
| Effective activation bandwidth | Bounded representative transfer/probe at formation | Directional goodput; subtract/record RTT, do not call ping “bandwidth” |
| GPU time, H2D/D2H, sync, idle | One representative Nsight Systems run plus CUDA event instrumentation if needed | Use backend stream events without inserting extra device-wide synchronization into normal inference |
| Graph effectiveness | Backend diagnostics/profiler | Capture/replay/rebuild counts per prompt/decode/verify shape |

Do not subtract timestamps from unsynchronized machines to invent one-way network time. Correlate causal frame IDs and local durations; use round-trip probes or an offline trace with known clock error. Any residual “network” bucket must be labeled **network + queues + unmeasured host overhead**.

Benchmark fixture: fixed pinned model/weights/prompt token IDs, same context and output budget, warm-up protocol, device/driver/backend settings, direct/relay topology, and workload phase. Report cold formation, disk-warm formation, GPU-resident attachment, TTFT, completion time, steady decode, aggregate/per-session throughput, acceptance, bytes/token and peak memory separately. Use interleaved control/treatment runs and medians/tails; preserve raw receipts and token hashes. Include a long prompt, short prompt, multiple chat turns, context edge, cancellation and concurrent sessions.

## 15. Prioritized performance roadmap

Benefits without a measured number below are hypotheses. Complexity/risk: L = low, M = moderate, H = high. “Numerical risk” includes changing output through different batch shape or lossy state, even when the algorithm remains valid.

| Optimization | Current bottleneck | Expected benefit / affected metrics | Complexity | Architecture risk | Correctness/numerical risk | Dependencies | Benchmark | Recommendation |
|---|---|---|---|---|---|---|---|---|
| Request/phase timing and acceptance counters | Unknown cost breakdown | Enables decisions for TTFT/tok-s/WAN/multi-session; no speed claim | L–M | L | L if sampled | Existing logs/frame IDs | Profiling off/on overhead and trace reconciliation | **Now** |
| Validate draft tokenizer and budget draft memory | Unsafe/inaccurate admission | Reliable speculation/model fit; fewer failures, no guaranteed speed gain | M | L | M; compatibility checks crucial | GGUF tokenizer identity and memory metadata | Compatible/incompatible pairs, memory ceiling, 1/2 sessions | **Now** |
| Clamp speculation at context/output boundaries | Fixed width can overrun useful capacity | Avoid request failures/wasted final work | L–M | L | M; batch shape drift | Track remaining context/budget at drafter | Last 1–4 positions, EOG, partial/full accept | **Now** |
| Remove middle-frame copy and reuse safe batch/payload buffers | Repeated allocations/copies | Prefill/memory/CPU improvement, end-to-end amount unknown | L–M | L | L–M buffer lifetime | Timing baseline | FP32 token equality, long prompt copy/alloc counters | **Soon** |
| Expose/tune existing chunked prefill | Whole-prompt sequential stages | TTFT/prefill/WAN and multi-session responsiveness; unknown gain | M | L | M shape/draft/ordering | Provider option, tests, context/frame checks | 128/256/etc chunks, 1/2 sessions, direct/relay | **Soon** |
| Forward head prefill before draft mirror when safe | Serialized draft prefill | TTFT overlap with downstream stages | M | L | M ordering/failure semantics | Trace current send/draft dependency | First/second token latency, draft state after errors | **Soon** after timing |
| Prompt-only FP16/FP8 mode | Long-prompt WAN bytes | FP8 payload ~4x smaller vs f32; TTFT gain workload-dependent | M | L | H relative to FP32 baseline | Explicit phase handling; no rows-only heuristic | Long prompt quality/token drift, subsequent decode | **Soon**, opt-in |
| Faster cache hashing and bounded cleanup | GB-scale verification/temp I/O and stale ranges | Formation time/disk use | M | L | M data-loss/corruption protection | Preserve integrity and active-file ownership | Corrupt/resumed cache, cold/disk/GPU-warm timings | **Soon** |
| Reuse parsed catalog metadata at provider startup | Repeated header HTTP work | Node startup/discovery readiness | L–M | L | L with hash-keyed validation | Existing index helpers | Offline/warm startup and invalid index fallback | **Soon** |
| Idle replica upgrade rule | Small models reserve useful capacity | Larger-model availability; not a tok/s claim | M | M | M lease races/live sessions | Busy-but-idle replica visibility | Late join, competing owners, active chats untouched | **Soon**, before default-on |
| Payload/phase-aware placement using measured edges | RTT-only ranking and memory cuts | TTFT/WAN/heterogeneous utilization | M | M | L numerically, M fit | Directional goodput and service times | Held-out 2/3-stage layouts vs existing planner | **Soon** after telemetry |
| Reduce owner control/final-commit round trips | Sequential end/start work | Completion/next-turn latency; small answers benefit most | M | M | M–H request/KV ordering | Per-request timings, failure tests | 1/8/32/128-token answers, cancellation, 2 sessions | **Soon** if measured significant |
| Draft/verify graph and logits-readback optimization | Host synchronization and vocab copies | Local/draft tok/s, GPU utilization; no measured gain | M–H | M | M sampling/tie equivalence | Profile actual kernels/copies; llama API support | Graph/eager and device/host greedy equality + WAN A/B | **Soon** investigation, implementation conditional |
| Tune speculation width/draft choice | Fixed width and smallest-draft heuristic | tok/s/WAN at an acceptance-dependent optimum | M | L | M numeric batch effects | Draft/verify/acceptance/resource telemetry | Width 1/2/4/8, matched prompts, held-out workloads | **Later** than accounting |
| Persistent failure memory | Fresh client sidecar forgets dead peers | Some startup delays; no steady decode gain | L–M | L | L stale-negative risk | TTL/identity-scoped cache | Restart with dead/recovered peers and no usable peer | **Later** unless user-visible priority |
| Relay buffer tuning within caps | Potential bulk relay throughput limit | Prefill/WAN, gain unmeasured | L | L | L | Inspect deployed relay resources | Same payload, direct vs relay, bounded buffer sizes | **Soon** measurement only |
| Bounded network I/O worker queues | Blocking send/read serializes stages | Multi-session throughput/overlap | M–H | M | H ordering/backpressure | Evidence of send stalls; ownership design | Slow edge/client, cancellation, memory bound, sessions | **Later** |
| Device-side wire conversion / async transfers | FP32 host materialization | PCIe/memory/prefill and overlap | H | M | H lifetime/precision | Backend extension and profiler proof | Transfer trace plus codec/numerical reference | **Later** |
| Quantized KV | Session/context memory | Larger contexts/session capacity, possible speed tradeoff | M | L–M | H lossy KV | Exposed backend setting and planner accounting | Quality, long context, peak VRAM, throughput | **Later** |
| One-session deep speculative pipeline | Dependent traversal bubbles | Potential WAN single-session speedup, unmeasured | H | H | H stale KV/epochs/numerics | Acceptance model and explicit new protocol design | Useful/wasted frames, depth sweep, failure injection | **Later**, separate experiment |
| Cross-session GPU batching | Repeated weight reads | Potential aggregate throughput | H | M–H | H concurrency/numerics | Explicit owner request; currently deferred | Aggregate/per-session latency and fairness | **Don't do now** |
| Tensor/expert parallelism, repair/election, payments/verification | Different execution/product goals | Not needed to optimize present loop | H | H | H | Separate requirements | Separate project | **Don't do** in this roadmap |

## 16. Technical debt / architectural risks

### 16.1 Documentation discrepancies to resolve

| Documentation assertion | Source/evidence correction |
|---|---|
| Intermediate activations never reach client/owner | Final-token commits use owner/client-mediated control exchanges; ordinary decode remains direct |
| Stages compute in FP32 | FP32 boundary; mixed internal kernels/accumulation and FP16 KV |
| Speculation is exactly plain output | Greedy acceptance is exact relative to verifier samples; batch arithmetic can change those samples |
| Draft request state serialized / concurrency would need per-session guesses | Current split head stores guesses per session and owner runs concurrent request threads |
| Dead peers cost at most 3 s | Grace starts after first usable answer, per collection; not a global deadline |
| All features are in September 18 installer | September 19 source additions require explicit artifact rebuild/version evidence |
| Replicas tested once / FP8 WAN not measured / combined WAN run still pending | Other sections contain September 19 WAN results; these passages are stale |
| One route per worker implies one conversation | One lease remains true, but a replica route can contain several resident/active sessions |
| Client startup always picks largest feasible model | Ready replica selection precedes fallback placement; an available smaller replica can be selected without searching free capacity for a larger placed route |
| Replica design at `docs/DAN_replica_design.md` | File is now under `docs/reference/design/DAN_replica_design.md` |
| Older pipelining/gateway reports describe current system | They describe separate coordinator execution paths |

### 16.2 Source-level risks requiring focused follow-up

These are review findings, not all reproduced failures:

- GPU reporting/device selection and actual backend placement can diverge on multiple-GPU machines.
- Draft resource planning and tokenizer compatibility are missing. Replacing a main stage can temporarily retain an old draft until the later draft-load step, increasing peak-memory pressure.
- `load_draft_model` reuses by draft SHA when `context.loaded` remains present; context/session changes deserve explicit tests across main-stage reuse/reload paths. Do not assume an arbitrary draft is valid solely because its hash was previously loaded.
- Status may advertise a configured draft while execution has fallen back. Speed estimators mix incomparable phases and normalized byte counts.
- Unbounded control jobs/connections/inboxes make the owner and worker unsuitable for unrestricted adversarial load despite authenticated transport and frame caps.
- Detached lifetime threads and shared ring/control state deserve race-sanitizer/lifecycle tests. The design relies on controlled ownership and process-long objects, not a general safe concurrent runtime API.
- A single worker mutex prevents concurrent llama context access, but whole-prefill calls and synchronous operations can starve other sessions. Fairness is arrival/order dependent, not a strict scheduler guarantee.
- The existing `FairQueue` abstraction is not evidence of fair scheduling in the replica owner's request path; that owner uses its own deque/thread logic.
- Worker connection/lease cleanup, ring failure propagation and owner health detection are different mechanisms; exact recovery latency varies with where a break occurs.
- Runtime ABI hashes the patch, not the full compiler/backend/GPU/numerical configuration or all engine protocol behavior. Matching ABI means declared compatibility, not identical floating-point execution.
- Catalog filenames use model IDs/ranges while validation checks revision/hash. Colliding/revised IDs can invalidate caches; cache layout and hints are not a global immutable tensor store.
- Current model shape checks are intentionally dense-Qwen2-specific and weaker than a general model schema. A new architecture requires explicit graph, tensor ownership, KV and rollback work.
- Confidentiality against model participants, Sybil/eclipsing defenses, result integrity and mid-request failover remain unsolved. They should remain explicit limitations, not be inferred from PeerIDs or TLS/Noise.

No production-readiness label is justified by these tests alone. A friend installation, mixed-version rejection tests, bounded-load tests, realistic relay-lifetime exercise and longer concurrent-session soak are still valuable before broader deployment.

## 17. Recommended next 3–5 implementation tasks

1. **Add minimal phase profiling and reproducible receipts.** Separate draft, target, packing, owner control, commit, bytes and acceptance; run one representative warm direct/relay GPU comparison with one and two sessions. This should establish the next bottleneck rather than guess it.
2. **Complete draft/device admission contracts.** Verify actual GPU selection, tokenizer compatibility, effective draft state and memory accounting; test context-edge speculation and failure fallback. These protect current performance claims from silent configuration differences.
3. **Integrate the existing chunked-prefill path and measure head draft ordering.** Keep defaults conservative; compare long-prompt TTFT and concurrent decode responsiveness. Evaluate prompt-only reduced wire precision separately because it changes numerics.
4. **Make idle replicas able to use newly available capacity.** Implement the smallest idle-only upgrade rule with lease-race tests, then test late joins on the real network before enabling replicas by default.
5. **Improve measured reuse/placement costs.** Begin with avoidable activation copies and cache verification/cleanup; use the new timings to choose between control-path finalization work and payload-aware split selection. Avoid implementing both as speculative infrastructure.

The architectural baseline remains provider-owned contiguous layers, local planning, exclusive worker leases and direct inference streams. The evidence supports improving that baseline before introducing a new execution model.
