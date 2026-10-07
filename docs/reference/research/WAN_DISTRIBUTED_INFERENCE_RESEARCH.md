# WAN/distributed LLM inference: evidence and architecture recommendations for DAN

Research date: **2026-09-20**. Research/design only; no inference code changed, external runtimes executed, models downloaded, or hardware provisioned.

Companion: [DeepSeek-V4.1-Flash feasibility audit](DEEPSEEK_V41_FLASH_AUDIT.md). This report widens that audit to the distributed-inference ecosystem. It does not supersede the model-specific tensor/state analysis there.

## 1. Executive recommendation

**Keep DAN's decentralized discovery, authenticated P2P transport, worker admission, persistent replicas and layer-range execution. Do not keep its present dense-model execution contract at the expense of computing V4.1 incorrectly.** Qualify an existing V4.1 implementation as the numerical/runtime foundation, then expose only the state and partition boundaries that this model actually requires.

The strongest evidence supports five conclusions:

1. **Layer pipelines over WAN are real and useful.** Petals demonstrated geographically distributed inference; DAN already has its own measured WAN result. Neither establishes that an arbitrary frontier model will be interactive on any collection of consumer cards.
2. **No inspected result establishes production-ready V4.1-Flash inference at 20–30 committed tokens/s across heterogeneous residential WAN GPUs.** There are valuable distributed V4 results and emerging single-workstation V4.1 implementations. These are different achievements.
3. **Batching, asynchronous speculation and reduced communication solve different problems.** Independent sessions fill idle stages and raise aggregate throughput. Speculation can reduce the number of dependency round trips per committed token. Compression helps when bytes, rather than propagation or compute, dominate. None is a universal multiplier.
4. **The most urgent V4.1 problems precede network optimization:** a trustworthy runtime, legal partitions of cross-layer state, hundreds of GiB of storage/residency, exact session lifecycle, and the interaction of speculative verification with expert streaming.
5. **The best architecture is a hierarchy of locality, not a global all-to-all expert swarm:** fast local compute within a worker or tightly coupled group; a small number of model-valid pipeline boundaries over measured WAN paths; several independent persistent replicas where capacity allows. This is a recommendation, not an implemented DAN capability.

Do not launch a V4.1 beta on a promise derived from OLMoE's speed. Maintain the working Qwen2/OLMoE path as the networking/control-plane regression baseline while qualifying V4.1 separately.

## 2. What was reviewed, and what the evidence means

### Evidence labels

| Label | Meaning |
|---|---|
| Code inspected | Relevant implementation was read; not a claim that every path is correct or exercised |
| Author measurement | Published numbers or receipts; not independently reproduced here |
| Actual WAN | Explicit geographically/public-network distributed execution, with caveats stated |
| LAN/local | Loopback, same-host GPUs, local Ethernet, Thunderbolt or controlled cluster |
| Emulated WAN | Bandwidth/delay restriction in a local testbed; useful but not NAT, relay, jitter and churn validation |
| Proposal/model | Algorithm, design document, fitted simulation or extrapolation; not a deployed measurement |
| DAN record | Results in the project guide or supplied by the owner; not rerun during this research |

This is a targeted source review, not an exhaustive audit of every repository. Implementation findings identify the files inspected. Paper-only findings are labeled accordingly. “Not demonstrated” means not established by the inspected evidence, not proof that nobody has ever done it.

### Snapshots

| Project | Inspected snapshot | Scope |
|---|---|---|
| DAN | `fe1f16cf80207dd361d0d861c688405657b0f420` plus existing worktree changes | Guide, planner, workers, range loading, replica/protocol context |
| DAN llama.cpp | `95ef7fc16054e63b427a3ef00188e055ef7586d8` plus DAN patch | Current backend and compatibility constraints |
| Shard | `fcf728096948c7686bcf0897e9acb75d1abda1d5` | V4/MiniMax engines, speculation, receipts/design investigations |
| enapt/SwarmLLM | `aab4fd7296e4f06bd80a024b1e294accfadb1572` | Distributed forwarding, DSD, allocation, benchmarks |
| Petals | `22afba627a7eb4fcfe9418c49472c6a51334b8ac` | Routing and session recovery; historical papers |
| Diffuse | `c7e5c5ea76f9c181c67fc832a031c4162c3429ec` | Compute/decryption/forwarding and benchmark record |
| exo | `21a54c5ea0230a3bec1e1a786d200126c7e34ec6` | Placement, MLX batch generation, remote prefill |
| Parallax | `162354a03234a28cf6e2946e2e0b2203da7c3721` | Allocation/scheduler and paper |
| BloomBee | `33cbcd1e49e55c36f699eaacac05b9960e80f436` | Pruner, compression tooling, paper |
| Helix | `8639497a4aaf1eb3b7594614cb0bbd376c1342b3` | Max-flow scheduling source and reproducibility artifact |
| PipeInfer | `123a824d9b53e56efa2a7645894d30a9a1135ed2` | Asynchronous speculative C++ example and deployment assumptions |
| ShardFlow | `57493b2701c2357988f553e16f301a26a4627aca` | Draft graph, transport, scheduler, checked-in benchmark |
| prima.cpp | `288df6a2cc2120387e12d6f40798f30cc525c9ea` | Deployment/performance documentation; not a full runtime audit |
| KTransformers | `f7607c0f66c643220ff43b822b24483c98c41392` | Active kt-kernel MoE wrapper, tests/tree and deployment documentation |

The V4.1 audit pins the official checkpoint and Vcruz, JigSaw and DwarfStar runtime versions. Those model-specific findings are reused here. Upstream heads can change after this date.

## 3. DAN's starting point

DAN is not starting from a networking prototype. It already has the expensive integration pieces: persistent provider identity, DHT discovery, leases, direct/relayed libp2p streams, ABI fencing, sparse range downloads, resident stage weights, replica ownership, token return loops and isolated session KV. The engine admits Qwen2 and OLMoE; OLMoE experts stay with their owning layer. The inference client does not need to relay activation tensors.

Important source anchors:

- [stage_worker.cpp](../engine/stage_worker.cpp): `Stage`, `run_middle`, draft proposal/verification paths, session handling and stage-local execution.
- [planner.cpp](../engine/planner.cpp): `compatible_stage_model`, `allocated_positions`, `kv_bytes`, fit/partition logic.
- [range_model.cpp](../engine/range_model.cpp): GGUF inspection, ownership and cached ranges.
- [replica_owner.cpp](../engine/replica_owner.cpp), [placement.cpp](../engine/placement.cpp): formation/attachment and assigned routes.
- [protocol.hpp](../engine/include/provider_owned/protocol.hpp): current frame/activation contract.
- [sidecar/main.go](../sidecar/main.go), [sidecar/replica.go](../sidecar/replica.go): network/discovery/replica services.
- [PROJECT.md](PROJECT.md), especially §§8–9: implementation status and measured routes.

The supplied warmed OLMoE benchmark is the correct present baseline:

| Route | Median decode | Median TTFT | Evidence boundary |
|---|---:|---:|---|
| A4500, one stage | 371.9 tok/s | 48.9 ms | One GPU, warm, 10 repetitions, 128 tokens |
| A4500, two stages | 304.7 tok/s | 49.9 ms | Same GPU/loopback; not two independent GPUs |
| A4500, three stages | 274.8 tok/s | 50.6 ms | Same GPU/loopback |
| 2070 `[0,4)` → A4500 `[4,16)` | 10.0 tok/s | 281 ms | Actual WAN, relayed link, reported 88 ms; no matched direct counterpart yet |

All use FP32 wire, no speculation, fresh request sessions and excluded model loading. The local 1→3-stage loss is about 26%, but does not isolate serialization from process scheduling or GPU contention. The WAN result strongly suggests dependency-loop latency matters; it does not establish an exact network/compute decomposition. Also, the three-stage local range 270.5–284.6 is about 5.1% of its median peak-to-peak, so the earlier blanket “spread under 2% everywhere” description is too strong.

OLMoE's two-session correctness evidence establishes isolation under interleaving/reset/destroy. It does not establish GPU-batched execution, OLMoE replica formation, or V4.1 state isolation. Existing Qwen speculation measurements are encouraging but not transferrable to a different target/draft pair.

## 4. Comparison at a glance

| System | Most useful contribution | Strongest relevant evidence inspected | Main boundary for DAN |
|---|---|---|---|
| Shard | Model-specific stage execution; bounded asynchronous speculation; honest bottleneck accounting | Executable V4 pipeline plus measured depth sweeps | V4 is not V4.1; model-specific Python/CUDA assumptions; no universal WAN speedup |
| SwarmLLM | Broad P2P integration, persistent streams, adaptive paths, failure replay | Substantial Rust implementation; detailed local benchmarks | Some headline gains are loopback; several advanced paths are conditional/partial |
| Petals | Client-side latency routing, DHT ranges, stateful recovery | Historical real geodistributed inference and mature source | Older model/runtime generation; replay costs; not frontier-MoE proof |
| Diffuse | Chained worker forwarding and transparent benchmark limitations | Public-network CPU run and compute source | Same-region workers, no relay benchmark, result predates correctness fix |
| exo | Efficient local hardware pooling, MLX batching/prefix caching | Executable MLX path; local RDMA/Thunderbolt emphasis | Fast local fabric and Apple ecosystem are not ordinary WAN |
| Parallax | Heterogeneous layer allocation and route selection | Working scheduling code; large-model author benchmark | Logical scheduler authority; network conditions underspecified in paper |
| BloomBee | Microbatch/communication optimization under limited bandwidth | Paper includes actual WAN plus emulated tests | Mainly offline, batch-32 throughput; not single-user interactive speed |
| Helix | Capacity-aware placement/routing with KV admission | Code, simulator and real-system artifact | Central optimization and serving throughput objective |
| PipeInfer | Continuous speculation, cancellation, multiple in-flight runs | C++ implementation on an old llama.cpp fork | MPI/shared cluster deployment and invasive runtime fork |
| SpecPipe / FlowSpec | Pipeline-aware speculation algorithms | Papers with local-cluster experiments | Not validated as residential-WAN frontier-MoE stacks |
| ShardFlow | Practical small-model WAN speculation/draft CUDA graphs | Code plus author WAN claims | Short runs, different checked-in receipt, transport is not end-to-end zero-copy |
| KTransformers / model-specific forks | CPU/GPU expert execution and memory-tier management | Actual expert kernels/wrappers; single-host measurements | Hardware-specific; offload/cache state changes the bottleneck |

## 5. Source-level findings

### 5.1 Shard: the most directly relevant speculation investigation

The inspected V4 engine is real stage execution, not a generic API façade. `v4_stage.py` owns a contiguous layer interval and implements model-specific input/state handling. `v4_pipe.py` contains ordinary, synchronous DSpark and `coordinate_dspark_pipelined` paths. The latter uses a bounded horizon, in-flight tracking, sender/receiver coordination and epochs. Stages recognize fenced/stale work; the settled frontier permits old snapshots to be released. The state machinery is necessary because compressed/sliding attention cannot always be undone by truncating an ordinary KV array. [Stage source][shard-stage], [pipeline source][shard-pipe]

The dated initial design discusses future improvements; the current source already implements parts of that design. Calling all pipelined speculation “planned” would therefore be wrong. Conversely, its early 10–20 tok/s ambitions are not measured outcomes.

The newer investigation reports a six-stage V4 depth sweep: greedy 1.80 tok/s; pipelined results roughly 2.49–3.26 tok/s, with the observed best near depth four. Its **fitted simulator**, not direct GPU occupancy instrumentation, estimates a busy bottleneck and roughly half the computed frames discarded at the default depth. The fitted hop latency hits a lower bound; actual RTT was not measured. It also admits poor prediction of the serial speculative path. These qualifications prevent presenting its simulated ceilings as precise hardware facts. [Efficiency investigation][shard-efficiency]

Reusable lessons:

- Optimize **committed tokens per bottleneck-second**, not busy time or proposed tokens/s.
- Keep a small, explicitly bounded speculative window. Increasing depth can lower acceptance and amplify cancellation waste.
- Fence stale replies before commit; cancel queued work before expensive execution where ordering permits it.
- Checkpoint every state mutated by verification, including overwritten ring-buffer cells and compressor state.
- Measure acceptance by draft position, not only one average acceptance percentage.

Shard's MiniMax and V4 engines also show why “MoE support” is not one capability: attention/cache semantics, output contracts, draft feature taps and kernels remain architecture-specific. No inspected Shard engine implements V4.1's Engram/CSA2 contract. Borrow its invariants and experiments, not an assumption that its V4 engine is a V4.1 backend.

### 5.2 enapt/SwarmLLM: useful breadth, uneven evidence maturity

`src/inference/pipeline/distributed.rs` has real segment forwarding, persistent-stream/chained routes and fallback paths. The distributed execution path tries specialized modes before ordinary forwarding. Failure recovery records activation history and can replay it to a replacement covering the missing segment. That consumes memory and recomputation/network time; it is not state-free failover. [Distributed source][swarm-dist]

`pipeline/dsd.rs` is conditional on the llama feature and configuration. It checks eligibility, drafts locally, sends a verification batch through the target segments, accepts greedy target agreement and piggybacks pending truncation on subsequent work. Unsupported configurations fall back rather than magically gaining distributed speculation. The source itself notes batch-kernel numerical differences: matching the greedy acceptance algorithm does not promise bit-identical floating-point evaluation. [DSD source][swarm-dsd]

`scheduler/parallax_allocator.rs` is especially instructive: comments describe a recommendation-only initial allocator, not the complete dynamic-programming Parallax system. Capacity and performance proxies should not be confused with measured per-layer/edge optimization. A feature named after a paper is not evidence of reproducing that paper's results. [Allocator][swarm-allocator]

The round-7 benchmark is explicitly **two loopback daemons** on a laptop, using a small CPU model, with important network features disabled. Q8 activation transport reduced reported payload sizes, but the run got slower and generated different token counts; the fixed-iteration comparison also slowed. The multi-segment WAN DSD benchmark remained pending in that record. Round 8 likewise documents same-box CPU/GPU tests, including substantial split overhead and historical correctness/memory issues. Do not treat those old issues as proof they persist at HEAD. [Round 7][swarm-round7], [round 8][swarm-round8]

The current project additionally reports local CPU/GPU offload and cross-node prefix-cache improvements; its 12.9× prefix-cache headline is **loopback**, not WAN. Its broader real-network deployment claims are worth reproducing, but do not supply a controlled frontier-MoE WAN throughput baseline in the evidence inspected. [Project documentation][swarm]

For DAN: reuse the stream/replay/adaptive-window design lessons. Do not replace the working Go sidecar and C++ runtime with an entire Rust/Candle stack merely to acquire a long feature list. Model support, KV serialization and backend precision would all require requalification.

### 5.3 Petals: still the strongest historical WAN foundation

Petals' `RemoteSequenceManager` constructs a latency graph using client/server and server/server RTTs, span compute estimates, serialization overhead and available cache. `_make_sequence_with_min_latency` finds a path; a separate throughput mode has a different objective. This directly supports distinguishing single-chat latency from service capacity in DAN. [Routing implementation][petals-routing]

`InferenceSession` stores span-input history. On failure, replacement sessions receive enough prefix history to rebuild their state; established sessions exchange incremental inputs. Direct server-to-server session links are also present. Recovery is possible because the required execution inputs are retained—not because KV can be ignored. [Session implementation][petals-session]

The papers demonstrate real Internet inference of large dense models, including BLOOM-176B and Llama-2-70B. They establish feasibility, not contemporary frontier-model performance. Petals' last observed push in this snapshot was in 2024; describe it as prior art rather than one of the most actively changing 2026 runtimes. [Internet-inference paper][petals-paper], [repository][petals]

For DAN: keep per-session route affinity, measure actual neighbor edges, include cache availability in admission, and borrow replay-based recovery only when its storage/recompute budget is acceptable. Do not import client-side activation storage as a mandatory dependency if DAN deliberately keeps inference traffic out of the client.

### 5.4 Diffuse: clear benchmark caveats, overstated privacy wording

`process_compute_request` and `process_chained_request` in `compute.rs` decrypt incoming activations before the local worker computes. Chained forwarding then encrypts for another hop. This is transport confidentiality, **not computation over ciphertext**. A participating worker sees the activations it processes. Keeping the embedding/head local also does not prove resistance to inversion or malicious workers. [Compute source][diffuse-compute]

The benchmark document reports approximately 1.19 tok/s for Mistral-7B BF16 across three CPU servers in Helsinki with a home client in Germany. The inter-worker links were direct, not relayed. It explicitly says the run predates a global-position/RoPE correctness fix; partial-shard downloading also did not work as intended during that run. Therefore it is transport/performance evidence with a correctness qualification, not a clean validated-model comparison against DAN. [Benchmark record][diffuse-bench]

Persistent connections and direct chained forwarding are sensible, but DAN already possesses the core pattern. No reason emerged to replace DAN's inference network with Diffuse's. The inspected repository is AGPL-3.0; assess licensing before copying implementation, independently of technical suitability.

### 5.5 exo: borrow local execution ideas, not a WAN performance assumption

Current exo placement enumerates topology cycles, filters by memory/model/backend constraints and ranks candidate placements. Its MLX path wraps `mlx_lm.BatchGenerator`, maintains request-specific state and prefix caches, and includes a remote-prefill path that ingests transferred KV into local cache objects. These are executable capabilities, not just roadmap items. [Placement][exo-placement], [batch generation][exo-batch], [remote prefill][exo-prefill]

The strongest public performance emphasis is local Apple hardware pooling, including MLX/JACCL and Thunderbolt RDMA. Such transport makes collective communication plausible locally; it says little about an 80–150 ms residential link. Backend enums containing CUDA/CPU also do not prove equal maturity on every backend. [Repository/deployment documentation][exo]

Useful to DAN: batch scheduling and cache reuse **inside a qualified local execution environment**, plus topology-aware placement. exo's cluster master/election arrangement is not a drop-in replacement for DAN's independent replica ownership. A Mac/MLX worker backend is a separate hardware-support project, not required for a V4.1 NVIDIA beta.

### 5.6 Parallax and Helix: scheduling principles worth extracting

Parallax's allocation code has greedy and dynamic-programming allocators, capacity-constrained proportional balancing, endpoint reservations and layer-latency/RTT route costs. Its scheduler coordinates materialization, joins/leaves, queueing and route dispatch. This is substantially more than SwarmLLM's similarly named initial allocator. [Layer allocation][parallax-allocation], [scheduler][parallax-scheduler]

The paper reports Qwen2.5-72B GPTQ-Int4 on two RTX 5090s and Qwen3-235B on larger homogeneous and mixed NVIDIA/Mac configurations. The displayed 72B 4K-prompt row has 40.7 ms ITL and 22.0 output tok/s including request time. However, network RTT, relay status and bandwidth are not adequately specified for a controlled ordinary-WAN comparison. The text also inconsistently refers to two GPUs where the 235B table specifies six. Treat it as promising heterogeneous-serving evidence, not a validated DAN WAN forecast. [Paper][parallax-paper]

Helix has both simulation and a real-system artifact, using placement optimization and max-flow-based routing. Its scheduling source explicitly fixes decode to the initialization route because KV is resident there, and checks expected cache capacity before admission. The artifact distinguishes generated cluster simulations from real ZeroMQ/vLLM experiments. [Scheduler source][helix-scheduler], [artifact][helix]

For DAN, the smallest useful extraction is **profiled stage costs + measured directed edges + legal cuts + KV admission**, computed by the client/replica owner. A global MILP service, large solver dependency or arbitrary per-token rerouting is not required. Service-wide max-flow becomes relevant only after multiple replicas and sustained demand make aggregate capacity the problem.

### 5.7 BloomBee: WAN throughput evidence, not single-chat speed

The paper explicitly targets offline throughput. Its primary workload is LLaMA-30B, batch 32. Most network conditions are emulated; a separate California/New Jersey/Canada setup is actual Internet execution. That setup reports 108 aggregate tok/s with compression plus microbatching versus 89 for Petals. It does **not** mean one chat runs at 108 tok/s. The paper inconsistently labels its heterogeneous GPU types between setup text and a table. [Paper, §§2 and 8][bloom-paper]

The inspected repository contains an actual neural candidate pruner and activation-layout/compression experiments. Useful ideas are small microbatches, lossless bit-layout compression and deferred accepted-prefix bookkeeping. Pruned speculation is not always selected by the paper's own experiments: extra verification traffic can lose on slow links. [Pruner source][bloom-pruner], [compression experiments][bloom-compression]

Do not add a trained pruning subsystem first. DAN should measure plain chunking and lossless compression before considering tree verification, ragged transport, training data and branch-aware KV compaction.

### 5.8 PipeInfer, SpecPipe and FlowSpec: latency hiding has implementation costs

**PipeInfer:** the C++ speculative example maintains multiple asynchronous runs, polls for cancellation, finishes/discards canceled work and manipulates sequence KV for accepted paths. This is concrete evidence that continuous speculative pipeline execution can be implemented on a llama-derived runtime. It is also a deeply modified old fork, with MPI communicators and shared-cluster deployment assumptions. Copying the whole runtime would regress DAN's backend maintenance and P2P integration. Borrow cancellation and bounded-run invariants. [C++ source][pipeinfer-code], [deployment][pipeinfer]

**SpecPipe:** the paper evaluates pipeline-aware speculation on three servers with 10 GbE, including L40/4090/3090 GPUs. Its speculative trees can increase stage traffic from 16 KiB to 1 MiB. This is meaningful pipeline evidence but is not a WAN benchmark, and the traffic expansion can reverse the gain on residential links. No full source implementation was qualified during this review. [Paper][specpipe]

**FlowSpec:** the inspected paper uses five Jetson Orin Nano devices on a LAN and reports continuous pipelined speculation improvements on 7B/13B models. Some stochastic cases lose to the comparison speculative method. Treat it as an algorithmic experiment, not proof of frontier MoE speed or a ready replacement backend. [Paper][flowspec]

The important common idea is that **a verifier need not wait for a complete speculative round before another useful unit enters the pipeline**. The difficult part is preserving causal state and bounding wasted execution after rejection. WAN delay, model-specific state and offloaded expert misses make that harder than a local dense-model demonstration.

### 5.9 ShardFlow: attractive WAN claim, inspect the data path

The project reports Qwen2.5-7B across two cloud regions through a relay: 4.92 tok/s plain and a best speculative prompt at 28.10 tok/s; other listed prompts are slower. These are author-reported short runs, not repeated variance-controlled results. Its checked-in `benchmark_cuda.json` instead records TinyLlama, three short runs and roughly 23 tok/s; that file does not substantiate the specific Qwen WAN headline. [README][shardflow], [checked-in result][shardflow-result]

The code does implement a draft `StaticCache` and CUDA graph capture with eager fallback. It also implements frame/token transport and speculative round metadata. But `transport/relay.py` calls GPU synchronization, copies to CPU, converts to NumPy bytes and uses `sendall`. Whatever the relay's forwarding implementation, the end-to-end tensor path is **not zero-copy**. [Draft implementation][shardflow-draft], [transport][shardflow-transport]

Useful experiment: reproduce graph-accelerated drafting and the fixed-cut WAN test on DAN's existing dense path. Not justified: assuming a short 7B run proves multi-stage V4.1, or importing raw relay code over DAN's authenticated libp2p transport.

### 5.10 Model-local runtimes and memory-tier execution

**KTransformers:** the active kt-kernel tree contains GPU/CPU expert masks, CPU task submission/synchronization, NUMA/threadpool parameters, quantized MoE implementations and per-commit accuracy tests. This is a credible source of local hybrid-expert execution, not WAN scheduling. CPU capability, memory channels, RAM capacity and NUMA placement are first-class requirements; “one consumer GPU” does not mean “an ordinary gaming PC.” The inspected documentation supports V4, not a qualified V4.1 path. [MoE wrapper][kt-moe], [project][kt]

**JigSaw's V4.1 fork:** an actual runtime with expert streaming and reported reference comparisons. Its single-5090 report distinguishes approximately 5.1 tok/s on new/cold content from approximately 21 tok/s on resident/repeated content. It also records bugs, expert-I/O ordering concerns and unhelpful speculative configurations. These are valuable counterexamples to assuming that a warm repeated prompt predicts new-conversation speed. Neither rate is a distributed result. [Runtime source][jiggraph], [report][jigreport]

**Vcruz's fork:** actual V4.1 graph/state work, but an explicitly incomplete hierarchical candidate-mask path with a context limitation. Embedded inputs have model-specific text/vision and Engram consequences. Do not graft DAN's dense hidden-input convention onto it without checking those branches. [Source][vcruz]

**DwarfStar:** actual V4.1 CUDA/Metal kernels and file-backed Engram row loading. A useful alternative backend candidate if its correctness and stage API prove easier to maintain than a llama.cpp fork. It is not a transparent DLL replacement: sessions, range ownership, state extraction/rollback and output contracts still need integration. [CUDA][dscuda], [Engram I/O][dsengram]

**prima.cpp:** useful documentation of heterogeneous local execution, device exclusion, offload and weight prefetch. Its deployment uses manually connected home-network ranks; this review did not deeply audit its runtime. Its large speedup claims are not evidence of frontier WAN behavior. The general lesson is strong: adding a slow peer can make inference worse, and RAM/disk characteristics belong in admission. [Repository][prima]

## 6. V4.1 changes the problem, not merely the parameter count

The dedicated audit derives these facts from the pinned official configuration, checkpoint index and reference code:

| Property | Consequence for distributed inference |
|---|---|
| 552B backbone plus roughly 196B Engram parameters | Active parameter count cannot be used as resident-weight memory |
| 40 backbone layers, hidden size 5,120, four residual streams | A single `[rows, hidden]` payload is insufficient |
| 384 routed experts/layer, six selected, plus shared expert | Keep all required weights in some qualified local memory tier; selection saves compute, not storage |
| Shared cross-layer KV, index and candidate sources | Not every layer boundary is self-contained |
| Engram tables at layers 1 and 14 | Token IDs/history and lookup metadata matter at remote stages |
| Sliding/compressed attention and carried mixing state | Rollback cannot be assumed to equal ordinary KV truncation |
| DSpark taps late decoder layers | DAN's head-local Qwen draft model arrangement does not implement DSpark |

These are not generic objections to MoE. OLMoE does not impose this combination of dependencies. [Official reference][v41-model], [configuration][v41-config], [Engram][v41-engram]

### Capacity: legal chunks matter more than total advertised VRAM

Under the audit's **dependency-closed, no-cross-boundary-shared-state** interpretation, useful internal cuts are at 2, 8, 14 and 20, leaving decoder `[20,40)` together. This is a conservative design inference from the reference dependencies, not a theorem that V4.1 can never be split more finely.

That decoder block has approximately **134.47 GiB of routed-expert storage alone** at the reference packed FP4-plus-scale accounting. Attention, shared experts, head, buffers, cache and runtime overhead are additional. The two Engram tables total approximately **188.83 GiB** in the inspected native table format. The official checkpoint is about 510 GB decimal. See the companion audit for arithmetic and artifact variants.

Therefore:

- A set of small cards with enough total VRAM is not automatically a legal all-resident placement.
- Keeping dependency groups intact can require a large-memory worker, local multi-GPU execution or hybrid CPU/RAM/NVMe execution.
- Splitting such a group over WAN requires explicitly forwarding/replicating the shared state. That is a material model-protocol extension, though not necessarily distributed expert parallelism.
- A smaller quantization may change capacity, quality and kernels; it does not erase state dependencies.

### Reference execution versus production optimization

The newly published technical paper describes CED deployment optimizations, a compact global KV representation and bounded SWA replay. Crucially, it states that bounded replay reconstructs **approximate** state; results may depend on the cache-hit position. Its production prefill savings are not automatically present in the public Python loop or third-party forks. The paper also discusses cross-stage shared-state management in its **training** infrastructure; that is not a shipped consumer-WAN inference protocol. [Official technical report, §§2–3][v41-paper]

Recommendation: establish exact full-reference-style execution first. Treat approximate bounded replay, compressed physical KV storage and reduced decoder prefill as separately validated optimization modes. Do not silently weaken DAN's reset/replay correctness gate to match a marketing throughput number.

## 7. What actually limits useful interactive performance

### 7.1 Define the target before optimizing it

Suggested **product targets to negotiate**, not measured capabilities:

- At least 15–20 committed output tokens/s for an ordinary single conversation, with a stretch target of 30.
- Report p50/p95 TTFT separately at 128, 2K, 8K and larger supported prompt sizes.
- Report p95/max gaps between committed output bursts. A five-token burst every 250 ms is not the same experience as a token every 50 ms.
- At two and four sessions, report both per-session speed and aggregate rate, plus fairness and admission failures.
- New content must be tested, not only identical warmed prompts. Report cold weights, cold experts, cold KV and warm resident cases separately.

### 7.2 Sequential decode lower bound

For a normal token dependency loop, a useful approximation is:

```text
T_token ≈ sum(stage compute + host/device transfer + serialization)
        + sum(actual forward-edge propagation + payload transfer)
        + token-return propagation + queue/synchronization delays

single-stream rate ≈ 1 / T_token
```

Count actual directed traversals, not an RTT independently for every one-way transfer. The return edge matters. Owner/client output delivery matters to user latency, but must not be charged to the model dependency loop if it is not on that loop.

A 100 ms complete token loop is approximately 10 tok/s even if each GPU computes for only a few milliseconds. This explains why DAN's fast local OLMoE does not retain hundreds of tok/s over WAN. It does **not** mean every frontier-MoE route is network-bound: Shard's V4 investigation is a concrete warning that stage compute and discarded speculation can dominate.

### 7.3 Speculation budget

For synchronous speculation:

```text
rate ≈ E[committed tokens per round]
       / (draft + target verification + network loop + rollback/commit overhead)
```

If a round costs 250 ms, a 20 tok/s target requires an average five committed tokens per round. That is a requirement, not a predicted acceptance rate. A larger proposal batch increases verification rows, expert union, bytes and checkpoint pressure.

For a continuously fed pipeline, idealized service rate is bounded by the slowest stage/edge. An approximate fill requirement is `window >= ceil(loop_latency / bottleneck_service_time)`, but useful goodput additionally depends on the fraction of work that survives rejection. Acceptance at increasing draft distance normally needs measurement; assuming a constant independent probability is only a toy model.

The edge/cloud literature often places the **entire target** remotely and only the draft at the edge. A recent analytical paper argues that this frequently loses to co-located speculation when that baseline is available. That is not a proof against DAN, where the target itself may not fit at one location; it is a reason to compare against the best feasible baseline and avoid adding a remote draft hop gratuitously. [Speculation at a Distance][distance]

### 7.4 V4.1's activation bytes

At a boundary carrying four FP32 streams plus four FP32 mixing values, the audit derives about **81,936 bytes per token row**, before token IDs and frame overhead.

- At 20 rows/s: about 1.64 MB/s, or 13.1 Mbit/s, per such edge before protocol overhead/retransmissions.
- A 512-row prefill chunk: about 40 MiB. At a sustained 20 Mbit/s uplink, serialization onto the link alone is about **16.8 seconds**; at 100 Mbit/s, about 3.36 seconds.
- A 2,048-row chunk: about 160 MiB, exceeding DAN's present 64 MiB payload cap.

These are arithmetic bounds, not performance measurements. They show why ordinary single-row decode may be propagation-bound while long prefill or speculative trees are bandwidth-bound on the same route. FP16 roughly halves the main payload, but needs numerical validation. Shared-state traffic at finer V4.1 cuts would be additional.

### 7.5 Expert memory traffic

MoE creates three distinct resource questions:

1. **Storage:** all possible experts plus tables must be available.
2. **Resident working set:** enough selected experts must be in GPU/RAM caches to avoid destructive misses.
3. **Per-step execution:** router, selected/shared experts, attention, reductions and transfer must complete fast enough.

Speculation can select more unique experts across its verification rows. A configuration that reduces WAN rounds can simultaneously increase disk reads enough to lose overall. Multi-session batching can improve expert reuse or expand the union; measure it. “Only six experts active” does not establish either capacity fit or bandwidth fit.

## 8. Techniques to reuse, adapt or reject

| Technique | Evidence source | DAN decision | Reason |
|---|---|---|---|
| Resident contiguous ranges and persistent sessions | DAN, Petals, Shard | Already present; preserve | Avoid repeated loading and session setup |
| Authenticated persistent P2P streams, relay fallback | DAN, SwarmLLM | Already present; improve observability | Replacing transport has no demonstrated payoff |
| Neighbor-edge latency plus compute-aware routing | Petals, Parallax | Directly relevant | Memory-only fit is not latency placement |
| KV-aware admission and route affinity | Petals, Helix, exo | Reuse principle | Prevent oversubscription and state migration surprises |
| Chunked prefill and decode-friendly scheduling | Established runtimes, BloomBee | Near-term controlled experiment | Reduces head-of-line stalls; chunk size is workload-dependent |
| Lossless activation layout/compression | BloomBee tooling | Test on saved real frames | Preserves values; only beneficial beyond CPU/transfer break-even |
| Reduced-precision activation wire | DAN already has options | Validate per model and phase | Routing/index decisions can amplify numerical changes |
| Draft CUDA graph capture | ShardFlow; backend techniques | Profile first | Helps local draft overhead, not WAN propagation itself |
| Bounded asynchronous speculative pipeline | Shard, PipeInfer | Research after exact rollback | Best direct route to hiding dependency latency, but high state complexity |
| Neural/tree pruning | BloomBee/SpecPipe | Later | Training, tree masks and KV compaction are unnecessary first steps |
| Continuous GPU batching | exo, serving runtimes | Later, after multi-session timing | Aggregate benefit can worsen individual latency |
| CPU/GPU expert execution and expert caching | KTransformers, V4.1 forks | Backend candidate | Necessary if resident capacity is insufficient; measure cache misses |
| Generic WAN expert all-to-all | Datacenter EP concepts | Do not use for first beta | Serial router/dispatch/combine dependencies and unreliable slow links |
| Tensor parallelism within a fast local group | exo, serving runtimes | Conditional local backend option | May satisfy an oversized dependency group without WAN collectives |
| WAN tensor parallelism | No suitable proof here | Reject as default | Layer-level collective latency compounds |
| Global max-flow/MILP scheduler | Helix | Not yet | Too much scope before profiled replica-local placement is inadequate |
| Replay-based recovery | Petals, SwarmLLM | Limited, explicit future tier | Useful but costly; V4.1 must restore all relevant state |
| Approximate bounded SWA replay | Official V4.1 deployment paper | Opt-in later research | Changes numerical/session equivalence contract |

## 9. Recommended DAN architecture

### 9.1 Preserve the control plane

Keep provider identity, DHT discovery, leases, per-replica ownership and client attachment. Infrastructure should remain a discovery/transport aid, not a global scheduling authority. Owners plan only their own replicas; workers retain final admission authority.

No inspected project provides sufficient evidence to justify replacing these working components wholesale. SwarmLLM and Petals provide useful techniques, not a reason to restart DAN in another language.

### 9.2 Replace or extend the execution backend when justified

The realistic choice is between qualifying a V4.1-capable llama.cpp fork and integrating a narrowly scoped model-specific runtime such as DwarfStar. Do not decide solely by which patch looks shorter. Compare:

- Reference correctness across prompts, context thresholds and both prefill/decode.
- Required hardware and fallback behavior on older consumer GPUs.
- Peak/resident VRAM, host RAM, pinned RAM and SSD I/O.
- Exact state reset, checkpoint, rollback and session isolation.
- Ability to load just assigned weights and expose legal stage boundaries.
- Maintenance burden, upstream prospects, license and deployment packaging.

A backend with better kernels and state APIs may justify replacing `Stage` internals. It does not imply replacing networking, leases or replica ownership. Conversely, a fast whole-model CLI without safe session/range interfaces is not yet a usable DAN backend.

Do not build a universal model-runtime framework in advance. First define a V4.1-specific stage/state contract alongside the working Qwen2/OLMoE path. Generalize only the interfaces that both actually need.

### 9.3 A few WAN boundaries between locality groups

```text
client ── attach ── replica owner
                       │ admission / lifecycle
                       ▼
       [head + legal group A] ── WAN ── [legal group B] ── WAN ── [tail + group C]
                  ▲                                                 │
                  └────────── committed token / control ─────────────┘

Within a group: local GPU, or a qualified CPU/GPU or fast multi-GPU backend.
Between groups: model-valid activation/state payloads; no remote expert dispatch by default.
```

For ordinary MoE, groups may be simple layer intervals. For V4.1 they must respect CSA2 dependency ownership, or explicitly transport the missing state. A logical local group is an option, not an assertion that DAN already supports multi-node workers.

Prefer regional/low-latency neighbors when capacity allows. Admit a far-away peer only if its capacity or compute benefit outweighs the added dependency cost. Prefer direct paths based on actual measurements, not as an absolute rule: a stable nearby relay can outperform an unstable direct route.

### 9.4 Placement objective

Use hard constraints first:

```text
model/ABI compatibility
legal state ownership and partition cuts
VRAM + host RAM + pinned RAM + disk capacity
context/session/workspace limits
actual reachability
```

Then score feasible candidates by an explicit objective:

- Single conversation: complete token-loop latency and requested TTFT.
- Multiple sessions: committed-token goodput subject to per-session latency limits.
- Formation/recovery: load/download time and already resident artifacts, separately from steady-state rate.

Start with measured ranges and a small candidate search. Do not infer MoE speed from TFLOPS, active parameter count or memory capacity alone. Include phase-specific costs and warm/cold expert behavior. Persistent replicas make this affordable: profiling can be amortized over many requests.

## 10. Scheduling, KV and speculation recommendations

### Scheduling

1. Keep one ordered state mutation stream per session.
2. Give decode small scheduling quanta; bound prefill chunks to avoid long monopolization.
3. Exploit existing independent sessions before adding GPU batch assembly.
4. If batching is added, cap waiting time and batch bytes, not only request count. Report whether it improves each user's latency or only aggregate throughput.
5. Do not migrate active sessions for a small transient route-score improvement; their state is already resident.

### KV/cache

Prefer local reuse on the resident replica first. Prefix reuse needs keys containing model/artifact revision, tokenizer/chat-template identity, precision/mode, relevant position/context settings and token sequence. Tenant isolation and memory quotas matter even when weights are shared.

Remote prefix/KV transfer is worthwhile only when transfer plus reconstruction beats local prefill. vLLM's NIXL connector exposes such state movement and failure policies; it is a building block, not an Internet-wide cache coherence service. V4.1 state is not interchangeable with a dense KV tensor. [NIXL connector guide][nixl]

Avoid full prefill/decode disaggregation over WAN initially. It adds KV/state transfer, routing and extra resident model capacity. Chunked prefill on the existing route is a smaller experiment. Disaggregation can later make sense within a local group or for measured extreme prompt workloads.

### Speculation

First keep V4.1 speculation off. Validate plain execution, sessions and state rollback before trying synchronous verification. DSpark is model-specific, and its late-layer features naturally suggest tail-local draft work; do not move it to the head merely to imitate the Qwen route.

When asynchronous work is introduced, minimum invariants are:

- Every speculative unit identifies session, absolute position, epoch and committed frontier.
- Rejected-epoch work cannot commit, even if its reply arrives late.
- Every speculative mutation is either reversible or regenerated from a known checkpoint.
- Window/checkpoint memory is bounded and included in admission.
- Cancellation cannot race with buffer reuse or destroy another session's state.
- Only verified tokens are exposed as final output; speculative UI output would be a different product contract.

Start with short linear proposals. A tree is justified only if measured extra acceptance beats its compute, state and transfer costs. Exact stochastic speculation requires the appropriate acceptance/correction algorithm; target-argmax verification is a greedy feature, not distribution-preserving arbitrary-temperature sampling.

## 11. Network and failure behavior

### Networking

Keep libp2p identity, authenticated channels, DHT, hole punching and relay support. Add observability for which stream/edge is direct or relayed, connection reuse, first-byte delay, effective payload throughput, backpressure and reconnects. Do not describe loopback as “no network”; it still includes socket/kernel processing, just no physical WAN.

Probe the actual data plane. A ping to a rendezvous server is not the latency between two workers. Measure both directions and the tail→head edge. Do not continually flood residential links with bandwidth tests; use brief formation probes and passive frame statistics.

Avoid a large prefill write indefinitely blocking decode/control traffic. Bounded chunks and explicit scheduling may suffice; separate streams can help head-of-line behavior but do not create extra uplink bandwidth. Async host/device transfers require buffers to remain live until transfer/compute completion; merely using a CUDA stream does not remove existing synchronization dependencies.

### Failure tiers

| Tier | Behavior | Recommendation |
|---|---|---|
| 0 | Fail fast, release leases/sessions, report lost replica | Keep as the initial correctness baseline |
| 1 | Reattach/retry a request before any output commit | Useful limited reliability improvement |
| 2 | Restart from retained prompt and committed token history on another complete replica | Explicit latency/cost; exactness depends on runtime/numerical contract |
| 3 | Replace a stage and reconstruct its state from checkpoints/history | Research after full state schema and capacity accounting |
| 4 | Transparent state replication/election/consensus | Not required for this beta |

Petals and SwarmLLM show replay recovery mechanisms, but no system can cheaply recover state it neither retained nor can recompute. For V4.1, recovery must include compressed/shared attention state, history-dependent inputs and all relevant epochs—not just ordinary KV. Approximate bounded replay is a separately declared mode, not exact failover.

### Security and trust

Peer authentication identifies a transport peer. It does not verify its model, arithmetic, memory handling or honesty. TLS/Noise/per-hop AEAD does not hide activations from the worker executing them. A “no raw prompt sent” property is narrower than prompt confidentiality. Keep these facts explicit in beta messaging; do not infer privacy or result verification from decentralized placement.

## 12. Combinations particularly suited to DAN

These are proposed experiments, **not claims of novelty or already demonstrated integrated systems**:

1. **Persistent replicas + measured legal-cut placement + session affinity.** Amortize load/profiling cost and select paths using actual V4.1 execution measurements. More useful than continually rebuilding routes from GPU model names.
2. **Bounded speculation + independent-session interleaving.** Use speculative work when one conversation is otherwise waiting, but reduce depth when other sessions provide useful work. Optimize committed goodput and fairness rather than occupancy.
3. **Joint window/codec selection.** A larger speculative window and a smaller activation encoding interact through acceptance, payload size and verification cost. Tune them together only after independent baselines; do not assume FP8 and larger windows multiply gains.
4. **Token-known Engram prefetch + resident stage ranges.** Engram lookup IDs depend on tokens/history rather than waiting for that layer's hidden state. Once tokens are known, local row lookup/prefetch may overlap upstream execution. Speculative token prefetch must be bounded to avoid cache pollution. This overlaps I/O, not model semantics, if values and history are exact.
5. **Local hybrid expert execution + WAN layer groups.** Use CPU/GPU/cache techniques within each owner instead of distributing individual experts across slow links. Profile PCIe/NUMA/disk costs as part of stage service time.
6. **Two cache lifetimes, without a global cache system.** Keep active-session state near its stage; retain reusable prefix state locally with a separate quota. Only add remote transfer after a measured break-even case.
7. **Regional replicas + global discovery.** Let global discovery locate a complete nearby replica rather than force every request through a globally scattered route. This needs sufficient duplicate capacity; it is not a solution when the entire network barely holds one model.

## 13. Missing measurements: the minimum useful instrumentation

Use request/session IDs, stage/edge IDs, frame sequence, absolute position, phase and epoch. Collect monotonic local durations; do not subtract unsynchronized clocks across machines and label the result one-way latency.

| Measurement | Where | Why |
|---|---|---|
| Queue wait and mutex wait | Worker entry | Separates scheduling/contending sessions from compute |
| Host packing/unpacking and allocation time | Frame construction/parsing | Finds CPU overhead without blaming WAN |
| Bytes before/after encoding | Each edge/phase | Actual activation bandwidth and compression ratio |
| D2H/H2D and GPU kernels | Sampled CUDA events plus occasional profiler trace | Identifies transfers/sync; avoid per-token global synchronization |
| Local stage service time | Input-ready → output-ready | Placement and pipeline bottleneck estimate |
| Write duration, first-byte/read completion | Persistent stream endpoints | Backpressure and transport timing |
| Complete token loop | Head send → committed return | Direct single-stream dependency metric |
| Draft and verify duration | Draft/tail/stages | Speculation cost, not merely acceptance |
| Accepted prefix length by draft index | Verifier | Correctly tunes window and detects drift |
| Computed, canceled, stale and committed rows | Every stage | Quantifies wasted GPU work |
| Checkpoint/rollback/replay bytes and time | Session state manager | Safety and cost of asynchronous speculation |
| Expert cache hits/misses, bytes and wait time | Runtime memory tier | Distinguishes resident from streaming performance |
| Engram row I/O/cache behavior | Engram implementation | Detects random-I/O stalls hidden by warm prompts |
| VRAM/host/pinned peak and resident use | Load, warm-up, prefill, decode, multi-session | Replaces guessed reserves with evidence |
| TTFT, output burst gaps, per-session rate | Owner/client | User-visible performance and fairness |

Implementation shape when authorized: bounded counters/histograms with periodic summaries and optional sampled frame traces. Use a few diagnostic NVTX ranges/CUDA-event samples rather than adding a synchronization to every layer. Publish a machine-readable run manifest with model hash, runtime commit, topology, dtype, prompt IDs, output token IDs and cache conditions. No large telemetry platform is necessary.

### Benchmark matrix

1. Full runtime on one host; one stage through DAN; same-host two/three stages.
2. Two separate LAN GPUs; controlled RTT/bandwidth/loss tests; actual WAN direct and relay on the **same cut**.
3. Short and long prompts; fresh and continued sessions; one/two/four concurrent sessions.
4. New prompts after weights are warm versus exactly repeated prompts; record expert and KV cache status separately.
5. Plain FP32 baseline; then one change at a time: chunk size, wire precision/codec, synchronous speculation, asynchronous window.
6. Failure/cancel/reset during prefill and verification; late reply injection; repeat around sliding/compression/index thresholds.

Use at least ten measured repetitions for stable short tests, with warm-up policy declared. For expert streaming, use a prompt corpus and report distribution rather than warming the exact same route through experts. Keep token budgets and EOS policy identical; do not compare shorter divergent outputs as a performance win.

## 14. Prioritized roadmap

Benefits below are hypotheses unless explicitly tied to a measured result above. There is no honest universal speedup estimate for DAN V4.1 yet.

| Priority / task | Bottleneck addressed | Expected impact | Complexity / risk | Gate and benchmark | Recommendation |
|---|---|---|---|---|---|
| 1. Reproducible phase/edge/commit telemetry | Unknown time decomposition | Enables correct choices; no direct speed promise | Low–medium; instrumentation overhead | Baseline overhead negligible relative to run variance | Now |
| 2. Qualify one V4.1 runtime, full model first | No trustworthy DAN V4.1 baseline | Correctness and feasibility | High model/runtime risk | Reference corpus, long-context thresholds, peak memory, cold/resident expert runs | Now, prerequisite |
| 3. Capacity/state/partition worksheet from real runtime | Illegal cuts and underestimated memory | Prevents infeasible deployment | Medium; model-specific | Enumerate legal two-stage cuts and actual hardware fit | Now, before hardware commitment |
| 4. Repeat current DAN direct/relay fixed-cut baseline | WAN attribution | Establishes transport opportunity | Low; connectivity variability | Same model/cut/hardware; actual path logged | Now |
| 5. Tune chunked prefill and decode scheduling | Long prompt transfers and head-of-line stalls | TTFT/prefill/fairness; may add frame overhead | Low–medium; ordering risk | Long prompts with an active second decode session | Soon on existing models |
| 6. Exact V4.1 two-stage boundary, then sessions | Model state contract | Enables distributed model; no speed guarantee | High correctness risk | Match qualified reference within declared numerical contract; reset/destroy/interleave | After 2–3 |
| 7. Measured compute/edge-aware placement | Poor cuts/slow peers/relay paths | WAN latency and utilization | Medium; planner policy only | Compare fixed candidate routes and held-out timings | Soon |
| 8. Lossless wire compression / FP16 validation | Bandwidth-heavy prefill and verification | Bytes/TTFT; little propagation benefit for tiny frames | Medium; FP16 numerical risk, lossless lower | Real saved activations, encode cost, quality, actual WAN | Soon if bytes dominate |
| 9. Reuse backend CPU/GPU expert-cache execution | Insufficient resident memory | Capacity; speed depends on misses | High hardware/memory complexity | Novel-content miss rate, p95 stalls, multi-session working set | Conditional, not automatic |
| 10. Synchronous V4.1 speculation | Token-loop round trips | Potential per-user rate gain; can regress offload path | High state/numerical risk | Target verification/rollback, accepted-prefix distribution, cache traffic | After plain sessions |
| 11. Bounded asynchronous speculation | Remaining pipeline wait | Potential WAN latency hiding | High protocol/state risk | Sweep small windows; committed goodput, stale work and p95 gaps | Research, after 10 |
| 12. Small continuous batches | Idle capacity with real demand | Aggregate throughput; possible ITL penalty | Medium–high cache/scheduling risk | Per-user SLO plus aggregate rate at 2/4 sessions | Later, demand-driven |
| 13. Local prefix reuse, then selective transfer | Repeated long prompts | TTFT and prefill savings | Medium; cache identity/isolation | Hit/miss corpus and transfer-vs-recompute break-even | Local soon; remote later |
| 14. Restart/replay recovery | Lost sessions on churn | Reliability, not steady-state speed | Medium–high; V4.1 state cost | Kill stages at defined positions, verify committed output policy | Later explicit milestone |
| 15. Fast local multi-GPU backend | Oversized dependency groups | Capacity and local compute | High deployment complexity | Compare against hybrid-offload alternative | Only if 3 requires it |
| 16. Global expert parallelism or global scheduler | Hypothetical future scale | Unproven WAN benefit | Very high architecture/risk | No present justification | Do not make beta prerequisites |

### Component replacement decisions

- **Keep:** libp2p sidecar, decentralized ownership, worker leases, content-addressed range/cache concepts, client attachment.
- **Potentially replace:** V4.1 execution internals with the best qualified external runtime; bespoke kernels with tested backend kernels; local batching/cache machinery with an existing backend's implementation where its state API is compatible.
- **Adapt:** latency placement from Petals/Parallax; KV admission from Helix; bounded speculation from Shard/PipeInfer; memory-tier execution from KTransformers/model forks.
- **Do not copy wholesale:** global orchestration, a second P2P stack, datacenter EP fabric, old llama.cpp forks, or large tree/pruner systems before simpler experiments fail.

## 15. Remaining open research problems

1. **Useful speculation horizon versus WAN delay.** Longer delay demands more outstanding work, but farther predictions become less reliable and rejection invalidates more work. A robust adaptive policy must account for stage speed, bytes and expert misses—not acceptance alone.
2. **Cross-layer compressed-state partitioning.** V4.1 reduces cache size by sharing state across layers, which creates ownership constraints. The best trade-off between larger local groups and incremental shared-state transfer remains unmeasured in DAN.
3. **Memory-tier-aware speculative verification.** Batch verification's expert union can change the memory working set discontinuously. Local cache benchmarks are not sufficient to predict pipeline behavior under several conversations.
4. **Joint TTFT and decode scheduling on weak uplinks.** Long prefill can saturate a link even when decode bandwidth is tiny. Deadline-aware chunk scheduling may outperform indiscriminate batching, but needs measured fairness and backpressure behavior.
5. **Numerical contracts across heterogeneous backends.** FP16/FP8 boundaries, expert reductions, router top-k ties and sparse index selection can change token decisions. “Looks coherent” is not sufficient; bit-identical CPU/GPU output is also an unrealistic universal requirement.
6. **Cheap exact recovery of nonordinary state.** Full replay is expensive; approximate replay changes semantics. Checkpoints/deltas need explicit bounded cost and ownership.
7. **Stable formation under changing capacity.** Persistent replicas amortize setup but can strand memory in smaller models or poor routes. Demand-aware dissolution/reformation is a control-plane policy problem, separate from per-token scheduling.
8. **Representative benchmarks.** Most public results omit some combination of relay state, long prompts, new expert working sets, multiple sessions, numerical validation and p95 latency. A reproducible DAN benchmark can itself be a valuable contribution.

## 16. Immediate next steps and stop conditions

The next engineering work should be five bounded tasks, not one network rewrite:

1. **Publish a reproducible current-DAN benchmark bundle** with per-stage/edge timing and direct/relay identity.
2. **Run an external V4.1 runtime qualification** on explicit hardware with reference outputs, novel prompts, context-threshold tests and memory/I/O measurements. Choose one backend afterward.
3. **Produce a measured legal-partition/capacity plan.** If no affordable two-stage configuration fits and meets a plausible service-time budget, stop and revise hardware/model expectations before integration.
4. **Prove exact plain two-stage V4.1 and session lifecycle locally, then over WAN.** No speculation or approximate replay to conceal a state-contract bug.
5. **Use the trace to choose one optimization:** chunking if prefill dominates; placement if edges dominate; backend/expert memory work if stage compute dominates; speculation only when its verification and rollback costs leave a credible gain.

Stop claiming a performance win if token budgets, output quality, hardware, cut, quantization or cache conditions changed without being disclosed. Stop expanding architecture when a smaller backend/configuration change solves the measured bottleneck.

**Bottom line:** DAN's WAN foundation is credible. A frontier V4.1 beta is a separate runtime-and-state qualification project with a WAN optimization phase—not a model-catalog update. The evidence favors reusing specialized execution technology while preserving DAN's working decentralized control plane.

## Sources

The links below identify the inspected implementations and primary research. Paper results and third-party benchmarks are not independently reproduced by this report.

[shard-stage]: https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_stage.py
[shard-pipe]: https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_pipe.py
[shard-efficiency]: https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/docs/V4_PIPELINE_EFFICIENCY.md
[swarm]: https://github.com/enapt/SwarmLLM/tree/aab4fd7296e4f06bd80a024b1e294accfadb1572
[swarm-dist]: https://github.com/enapt/SwarmLLM/blob/aab4fd7296e4f06bd80a024b1e294accfadb1572/src/inference/pipeline/distributed.rs
[swarm-dsd]: https://github.com/enapt/SwarmLLM/blob/aab4fd7296e4f06bd80a024b1e294accfadb1572/src/inference/pipeline/dsd.rs
[swarm-allocator]: https://github.com/enapt/SwarmLLM/blob/aab4fd7296e4f06bd80a024b1e294accfadb1572/src/inference/scheduler/parallax_allocator.rs
[swarm-round7]: https://github.com/enapt/SwarmLLM/blob/aab4fd7296e4f06bd80a024b1e294accfadb1572/docs/plans/benchmarks/round7.md
[swarm-round8]: https://github.com/enapt/SwarmLLM/blob/aab4fd7296e4f06bd80a024b1e294accfadb1572/docs/plans/benchmarks/round8_windows_cpu_gpu.md
[petals]: https://github.com/bigscience-workshop/petals/tree/22afba627a7eb4fcfe9418c49472c6a51334b8ac
[petals-routing]: https://github.com/bigscience-workshop/petals/blob/22afba627a7eb4fcfe9418c49472c6a51334b8ac/src/petals/client/routing/sequence_manager.py
[petals-session]: https://github.com/bigscience-workshop/petals/blob/22afba627a7eb4fcfe9418c49472c6a51334b8ac/src/petals/client/inference_session.py
[petals-paper]: https://arxiv.org/abs/2312.08361
[diffuse-compute]: https://github.com/diffuse-systems/diffuse/blob/c7e5c5ea76f9c181c67fc832a031c4162c3429ec/crates/diffuse-daemon/src/compute.rs
[diffuse-bench]: https://github.com/diffuse-systems/diffuse/blob/c7e5c5ea76f9c181c67fc832a031c4162c3429ec/docs/BENCHMARKS.md
[exo]: https://github.com/exo-explore/exo/tree/21a54c5ea0230a3bec1e1a786d200126c7e34ec6
[exo-placement]: https://github.com/exo-explore/exo/blob/21a54c5ea0230a3bec1e1a786d200126c7e34ec6/src/exo/master/placement.py
[exo-batch]: https://github.com/exo-explore/exo/blob/21a54c5ea0230a3bec1e1a786d200126c7e34ec6/src/exo/worker/engines/mlx/generator/batch_generate.py
[exo-prefill]: https://github.com/exo-explore/exo/blob/21a54c5ea0230a3bec1e1a786d200126c7e34ec6/src/exo/worker/engines/mlx/generator/remote_prefill.py
[parallax-allocation]: https://github.com/GradientHQ/parallax/blob/162354a03234a28cf6e2946e2e0b2203da7c3721/src/scheduling/layer_allocation.py
[parallax-scheduler]: https://github.com/GradientHQ/parallax/blob/162354a03234a28cf6e2946e2e0b2203da7c3721/src/scheduling/scheduler.py
[parallax-paper]: https://gradient.network/parallax.pdf
[helix]: https://github.com/Thesys-lab/Helix-ASPLOS25/tree/8639497a4aaf1eb3b7594614cb0bbd376c1342b3
[helix-scheduler]: https://github.com/Thesys-lab/Helix-ASPLOS25/blob/8639497a4aaf1eb3b7594614cb0bbd376c1342b3/simulator/scheduler/global_maxflow/scheduler_core.py
[bloom-paper]: https://arxiv.org/html/2604.21072v2
[bloom-pruner]: https://github.com/ai-decentralized/BloomBee/blob/33cbcd1e49e55c36f699eaacac05b9960e80f436/src/bloombee/server/speculative_pruner/adaptive_neural_pruner.py
[bloom-compression]: https://github.com/ai-decentralized/BloomBee/tree/33cbcd1e49e55c36f699eaacac05b9960e80f436/benchmarks/compression
[pipeinfer]: https://github.com/AutonomicPerfectionist/PipeInfer/tree/123a824d9b53e56efa2a7645894d30a9a1135ed2
[pipeinfer-code]: https://github.com/AutonomicPerfectionist/PipeInfer/blob/123a824d9b53e56efa2a7645894d30a9a1135ed2/examples/speculative/speculative.cpp
[specpipe]: https://arxiv.org/html/2504.04104v2
[flowspec]: https://arxiv.org/html/2507.02620v1
[shardflow]: https://github.com/rautaditya2606/Shardflow/tree/57493b2701c2357988f553e16f301a26a4627aca
[shardflow-result]: https://github.com/rautaditya2606/Shardflow/blob/57493b2701c2357988f553e16f301a26a4627aca/benchmarks/results/benchmark_cuda.json
[shardflow-draft]: https://github.com/rautaditya2606/Shardflow/blob/57493b2701c2357988f553e16f301a26a4627aca/shardflow/node/draft_model.py
[shardflow-transport]: https://github.com/rautaditya2606/Shardflow/blob/57493b2701c2357988f553e16f301a26a4627aca/shardflow/transport/relay.py
[kt]: https://github.com/kvcache-ai/ktransformers/tree/f7607c0f66c643220ff43b822b24483c98c41392
[kt-moe]: https://github.com/kvcache-ai/ktransformers/blob/f7607c0f66c643220ff43b822b24483c98c41392/kt-kernel/python/utils/moe_kernel.py
[prima]: https://github.com/OpenCPIL/prima.cpp/tree/288df6a2cc2120387e12d6f40798f30cc525c9ea
[jiggraph]: https://github.com/JigSawPT/llama.cpp/blob/3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5/src/models/deepseek4.cpp
[jigreport]: https://github.com/JigSawPT/deepseek-v41-flash-on-5090
[vcruz]: https://github.com/vcruz305/llama.cpp/blob/5210c7c5ed61dddaee6ed476623abf4b63093d16/src/models/deepseek41.cpp
[dscuda]: https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4_deepseek41_cuda.cuh
[dsengram]: https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4_engram.c
[v41-model]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
[v41-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/config.json
[v41-engram]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py
[v41-paper]: https://arxiv.org/html/2609.19969v1
[distance]: https://arxiv.org/html/2606.25091v1
[nixl]: https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
