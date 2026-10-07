# WAN GPU inference: implementation review

Checked 2026-10-06. Target: latency of a single interactive request across
independent GPUs over ordinary internet links. Throughput-only improvements do
not qualify without a latency measurement. This review compares primary papers,
project reports and DAN code; upstream performance claims are not DAN results.

## Findings

No universal best optimization exists. Measure prefill and decode separately.
For sequential layer splitting, decode must traverse the stages and return the
next token before its next step. Approximate step time is stage compute plus
one-way transit around the ring, serialization and packing. Speculation divides
round cost by the expected number of committed tokens, but adds draft and verify
compute. Optimizing acceptance alone can therefore make latency worse.

For a 3584-wide activation row, F16 is 7168 bytes; DAN FP8 is 3588 bytes including
scale. At an illustrative 20 Mbit/s bottleneck, serialization alone is 2.87 versus
1.44 ms per row. Against an 80 ms traversal, saving 1.43 ms is small. For 2000
rows it is about 5.73 versus 2.87 seconds per edge, before overlap, headers or
compute. These are arithmetic examples, not measured network results.

## Sources and applicability

| Source | What was examined | Fit and limitations |
|---|---|---|
| [Shard WAN experiments](https://github.com/leyten/shard/blob/master/docs/research/wan-speculative-decoding.md) | Actual scattered/clustered nodes, direct return, linear/tree drafting, compute breakdown | Highly relevant WAN evidence. More accepted tokens did not always improve speed; target verify compute mattered. DAN already has direct tail-to-head return and client-free decode. |
| [SpecPipe, v2](https://arxiv.org/html/2504.04104v2) | Dynamic speculative trees, step-wise pipelining, pruning, evaluation setup | Strong algorithmic reference, but testbed uses three servers and 10 GbE. Its reported gains cannot be transplanted to consumer WAN or this GPU. |
| [FlowSpec, v2](https://arxiv.org/html/2507.02620v2), [code](https://github.com/Leosang-lx/FlowSpec) | Continuous tree expansion, pruning, verification scheduling | Relevant to idle stages in a single request. Requires a substantial tree/KV implementation; underlying draft quality still limits gains. Verify code licensing and exact revision before reuse. |
| [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | Conventional draft, EAGLE-3, DFlash, DSpark, MTP and n-gram implementations, target/draft constraints | Exact pinned source already contains these algorithms; optional common helpers are disabled and unused. Prefer upstream proposal code. Target-specific methods need compatible checkpoints and distributed feature/stage integration, not a runtime upgrade merely to obtain the algorithms. |
| [DSD](https://arxiv.org/abs/2511.11733) | Parallel candidate verification to amortize communication | Relevant principle. Adaptive semantic acceptance changes the correctness question; do not equate task accuracy with exact target behavior. |
| [DiP-SD](https://arxiv.org/html/2604.20919v1) | Draft length, assignment and batching optimization | Device drafting with a centralized verifier and numerical multi-user throughput evaluation. Useful cost-model ideas; its architecture and headline throughput are not evidence for DAN single-chat latency. |
| [SpecFed](https://arxiv.org/html/2604.25777v1) | Compressed probability distributions and reconstruction bias | Different topology: workers upload vocabulary distributions for server aggregation. DAN sends boundary activations and tail-sampled token IDs. Top-K logit compression does not address DAN's current traffic. |
| [DFlash paper](https://arxiv.org/abs/2602.06036), [implementation](https://github.com/z-lab/dflash) | Parallel block drafting and target-specific checkpoints | Candidate for cheaper drafting after runtime/model compatibility work. Not a generic replacement for current Qwen2 draft model or evidence of a WAN speedup here. |
| [Petals paper](https://arxiv.org/abs/2312.08361) | Uneven devices, internet deployment, routing and block placement | Useful reference for distributed placement. DAN must retain client/provider-local planning; infrastructure scheduling and failover remain outside this change. |
| [exo](https://github.com/exo-explore/exo) | Topology-aware partitioning and Thunderbolt RDMA | Topology ideas apply; physically connected Apple RDMA results do not apply to unrelated internet GPUs. |
| [dnet](https://github.com/firstbatchxyz/dnet) | Profiling and heterogeneous topology-aware assignment | Useful profiling approach; Apple UMA and disk-overlap assumptions do not directly map to RTX 2070 WAN workers. |
| [Winsock TCP options](https://learn.microsoft.com/en-us/windows/win32/winsock/ipproto-tcp-socket-options) | Nagle/TCP_NODELAY behavior | C++ DAN sends header and body separately without explicit NODELAY. A 100-sample loopback probe found no material stall: medians 0.11–0.14 ms across tested frames, with no consistent benefit. Do not claim a WAN bug from this. |

## DAN audit and ranked implementation branches

| Order | Branches covered | Existing state | Next decision/test |
|---|---|---|---|
| 1 | Activation precision, compression quality, network profiling | F32/F16/FP8 available; FP8 output drift previously observed | Measure actual pack, send, receive, unpack and transfer sizes for prefill/decode. Keep FP8 opt-in until quality and end-to-end latency are assessed. |
| 2 | Placement, hop count, direct links, relay, NAT | Ring, hole punching, fallback and speed-aware placement implemented | Compare same model/route over direct and relay; measure stage-to-stage edges, not merely client-to-worker RTT. Prefer fewer stages when total latency falls. |
| 3 | Linear drafting, adaptive speculation, n-gram drafting | Single/split draft speculation plus session-local elapsed-cost width adaptation implemented; opt-in | Reuse pinned llama.cpp n-gram and draft proposal helpers. Preserve distributed verification and WAN round-cost adaptation; owner evaluates latency later. |
| 4 | Compute/network overlap, pipelined speculative verification | Historical coordinator pipeline implemented; decentralized ring draft verification exists | Audit decentralized overlap and rollback before porting tree algorithms. Establish linear baseline first. |
| 5 | Prefill chunking and pipeline | Provider exposes prefill_chunk; default 512; split propagation correctness checked | Already uses upstream batched compute. WAN chunk propagation is DAN-specific. Owner evaluates chunk size later; smaller is not automatically faster. |
| 6 | Persistent routes, discovery, formation, loading/cache | Persistent replicas work; local setting on; dead-peer grace and weight cache implemented | Measure cold versus warm setup separately. Do not attribute setup savings to token-transfer improvements. |
| 7 | GPU kernels, graphs, attention, KV, microbatch, context | CUDA/graphs/FA upstream implementations active; F16/1024 selected; pinned output-row bound now integrated on tails | Actual local 7B CUDA compute reservation reduced 608.00 to 296.02 MiB; focused correctness checks pass. No new WAN speed claim. KV precision does not reduce activation network traffic. |
| 8 | Target-specific drafts, upstream runtime, architectures/MoE | Pinned patched llama.cpp already has EAGLE-3/DFlash/DSpark/MTP; DAN stage support is Qwen2 and OLMoE | Reuse existing helpers with compatible checkpoints and distributed feature capture. Upgrade/port only for newer APIs, model/backend support or fixes. |
| Deferred | Concurrent sessions, continuous batching, API/sampling, constrained output, prefix reuse, streaming, other hardware | Partial support as documented in PROJECT.md | Not the immediate WAN transfer investigation; API item 2 remains untouched. Batching and retained prefix state require separate scope. |

## Future owner performance evaluation

The owner subsequently requested implementation with bug/correctness checks only
and will test across two PCs. The following is a future evaluation plan, not a
requirement to run benchmarks before authorized implementation.

Use the same model, layer split, context, generated-token budget and warm weights.
Collect first-token latency, inter-token p50/p95, total request time, committed
tokens per verify round, draft/target compute, transfer bytes, pack/unpack time,
actual edge latency and direct/relay status. Compare greedy reference outputs
for lossless changes; evaluate quantization drift explicitly for lossy formats.

First matrix: two stages, F32/F16/FP8, short and 2000-token prompts, plain decode.
Then plain versus existing linear speculation with the same wire format. Sweep
bandwidth and delay independently in a controlled simulator if a second machine
is unavailable; label simulation separately from real WAN results.

Decision: strongest candidate for RTT-bound decode is cost-aware speculation plus
latency-aware topology, not automatic FP8 everywhere. For bandwidth-bound prefill,
FP8 plus chunked overlap is the leading candidate. No production change is made
from this review alone. Owner subsequently directed implementation with bug tests
only and will perform the two-PC performance test later; no more benchmarks run.

## Implemented after review

- Provider-owned `prefill_chunk` now reaches the existing first-stage chunking
  path. Default 512, 0 disables, maximum 1024. Single-worker inference skips it.
- Worker-owned `draft_width` replaces the fixed verify width of four. Default
  four, allowed 1..32 including the current token. Context clipping and exact
  rejection logic remain in place; enabling speculation still requires a draft
  in the worker catalog and the existing speculation switch.
- `adaptive_draft=true` (default) now treats that width as a cap and learns per
  session from complete-round elapsed time per committed token. Four-round
  blocks probe ordinary decode, 2, 4, ... up to the cap; only a 5% improvement
  wins. It periodically rechecks the ordinary baseline and falls back when
  speculation costs more. `false` retains a fixed width. This is a bounded
  heuristic; measured round time includes queuing and streaming as well as
  network transit. It does not measure pure RTT or guarantee a global optimum.
- During ordinary rounds, draft inputs remain only in session RAM. Replaying
  before a probe keeps KV aligned without drafting each fallback token; its
  time is included in the probe. Text/end/rollback controls also flush deferred
  inputs, so request completion can pay replay cost. Terminal/clipped rounds
  do not train width selection; reset/destroy/disconnect clear the state.
- Placement correctness harness accepts both options. Three-stage real-model
  CPU checks passed for chunked F32 and F16, new and persistent sessions against
  the unchunked reference. Width boundary/rejection and protocol tests passed.
- Adaptive real-model correctness checks passed with cached 1.5B target / 0.5B
  draft weights: one worker and three CPU stages, new and persistent sessions,
  32-token requests matched the plain reference outputs. Logs confirm fallback
  and re-probing. Controller tests cover all caps 1..32 and expensive verification
  despite high acceptance. No FP8 quality, multi-machine, or speed claim follows
  from these checks. API item 2 is untouched.
- A forced fixed-width 32 / 40-token-context check completed clipped batches and
  persistent commits without context/session errors, but flipped one reference
  word. Old and new workers gave identical outputs in both modes, reproducing
  the already documented batching numerical drift. This case does not establish
  plain-reference parity; speculation stays opt-in.

Provider configuration example for the owner's WAN test:

```ini
prefill_chunk=512
draft_width=16   # adaptive cap; default remains 4
adaptive_draft=true
# Existing opt-ins, once a compatible smaller draft is cataloged:
# replica_speculate=true
# replica_activations=fp8
```
