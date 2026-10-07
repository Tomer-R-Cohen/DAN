# Optimization availability review

Reviewed 2026-10-06. Review first; no implementation selected by this document.
The initial n-gram edits made during this review were reverted before building or
deploying. Earlier project changes remain intact.

This covers the relevant optimization surfaces in DAN and the inspected llama.cpp
API/common/server/backend code, including memory, throughput, quality and startup.
It is not a claim that every possible research idea has been discovered. Benefits
below are mechanisms and candidates, not measured improvements on DAN.

## What reuse actually requires

DAN calls `llama`, not `llama-server`. Core kernels already run upstream code.
Server flags do not reach this worker. The pin contains reusable common helpers,
and the runtime now builds/links `LLAMA_BUILD_COMMON=ON` for the integrations below.

Implementation update (2026-10-06): **already active** in the rebuilt runtime are
upstream common samplers, model templates, tool/schema grammars, bounded RAM prefix
reuse and persistent API clients. Ordinary draft proposals use the upstream helper;
simple n-gram drafting and ready-frame ring batching are exposed as **opt-in**.
The tables below retain the broader audit's remaining integration/model-dependent
work. See PROJECT.md's latest integration note for tested scope and limitations.

- **Option wiring:** validate provider configuration and pass an existing API field.
- **Helper integration:** link upstream code and supply its contexts/session state.
- **Lifecycle integration:** retain and coordinate DAN sessions across requests/stages.
- **Model/backend port:** adapt the stage boundary patch and build compatible hardware support.

None of these categories authorizes rewriting an upstream algorithm. Compile the
upstream helper from the pinned checkout or link its library. Keep custom code for
the distributed boundary, worker ownership and transport that upstream does not supply.

Classification: **active** = already active/available (local selection noted);
**integration** = integration needed; **model-dependent** includes hardware-dependent
features; **deferred** = deferred / not worth changing under current requirements.

## 1. Compute and device use

| Opportunity | Classification / current state | Existing upstream code and required DAN work | Benefit / tradeoff |
|---|---|---|---|
| CUDA quantized kernels | active | Already supplied by ggml CUDA; keep upstream kernels. | GPU compute and weight bandwidth; no replacement algorithm needed. |
| Flash Attention | active locally | `flash_attn_type`: auto for F16, forced on for quantized KV. Exposing on/off/auto is option wiring, not enabling a missing local feature. | Attention memory/compute; backend/model compatibility matters. |
| CUDA graphs | active locally | Upstream CUDA graph implementation and build configuration already used. | Lower repeated launch overhead. |
| Kernel selection: MMQ/cuBLAS | deferred | Upstream dispatch and `GGML_CUDA_FORCE_MMQ/FORCE_CUBLAS` exist. Build variants only if investigating a specific GPU/workload. | Forcing either can lose upstream's shape-dependent choice; no demonstrated gain. |
| GPU layer offload | active | `n_gpu_layers` already wired; local model fully offloaded. | Avoid CPU inference; no further layer-offload gain for the current fully resident model. |
| KV/attention and host-operation offload | active defaults | `offload_kqv` and `op_offload` default true at the pin. Expose only for an actual capacity/backend need. | Disabling saves device memory in some configurations but adds CPU/transfer work. |
| CPU threads / batch threads | integration: option wiring | Set `n_threads` and `n_threads_batch`; current worker inherits core defaults. | CPU/hybrid tuning; more threads is not automatically faster. |
| CPU affinity, polling, NUMA | model-dependent | Reuse ggml threadpool/common CPU helpers; worker-owned placement of threads. | Relevant to large CPU/NUMA hosts; polling consumes CPU/power. |
| Explicit warmup | integration: helper/API use | Reuse upstream warmup pattern and `llama_set_warmup`; no explicit worker warmup found. Clear warmup state before real sessions. | Moves first-use work to startup; costs startup time and temporary memory. |
| GPU-side sampling | deferred | `llama_set_sampler` and sampler-sequence config exist. Integrate sampler lifetime, output IDs and speculative rows at the tail. | Can avoid host vocabulary-logit readback; pin also allocates backend buffers. Net latency/memory unresolved. |
| Local multi-GPU split/device selection | model-dependent | `devices`, `main_gpu`, `split_mode`, `tensor_split` plus upstream backend transfers. Expose worker settings and account for each device locally. | Capacity/compute on one host; topology and stage-patch compatibility need checks. Not WAN tensor parallelism. |
| Other backends | model-dependent | Build upstream Vulkan/Metal/SYCL/other suitable backend; verify stage extraction and packaging on that hardware. | Hardware coverage; no reason to replace working CUDA on the RTX 2070. |

## 2. Memory and context

| Opportunity | Classification / current state | Existing upstream code and required DAN work | Benefit / tradeoff |
|---|---|---|---|
| Q8 KV | active option; F16 selected locally | Existing `kv_cache=q8_0`; no new implementation. | Smaller KV allocation; does not shrink network activations. |
| Separate K/V types and more cache formats | integration: option wiring plus validation | Set independent `type_k/type_v`; upstream advertises Q4/Q5/IQ4/BF16 as well. Check backend, head dimensions, FA support and memory estimates. | More capacity; quantization quality and kernel latency vary. Never accept unsupported combinations silently. |
| Quantization-aware admission | integration | Planner currently budgets F16 KV conservatively. Use worker-advertised supported precision and upstream size calculation; workers still enforce actual fit. | Lets placement exploit KV savings; simply changing worker precision does not make the planner admit larger routes. |
| Physical/logical batch sizing | active, partly exposed | Worker exposes `prefill_batch` 256/512/1024 for `n_ubatch`; `n_batch` remains context-sized. More controls are direct API wiring with constraints. | Prefill speed versus temporary memory and scheduling granularity. |
| Final-stage output reservation | active | Already uses `n_outputs_max`; intermediate stages retain all boundary hidden rows. | Existing measured memory saving; do not reimplement. |
| Unified KV pool / slot allocation | integration | Core `kv_unified` exists, default false here. Review session guarantees and memory accounting before exposing a policy. | Pool sharing and prefix copies versus attention behavior; not synonymous with batching or PagedAttention. |
| RAM prefix caching across HTTP/tool turns | already active | Exact token-prefix reuse and upstream KV trim on every stage; one idle-expiring RAM session in local API. | Avoid repeated prefill/activation transfer; bounded cache, explicit per-chat cache key, no cache files. |
| Common-prefix sharing between sequences | integration: lifecycle | `llama_memory_seq_cp` can share prefix state within a context. Coordinate sequence identities across stages. | Less duplicated KV/prefill; requires an explicit isolation policy, not automatic sharing between unrelated users. |
| Reuse unchanged chunks after prompt edits | integration: lifecycle | Upstream server `cache-reuse` and sequence shifting patterns; follow after basic prefix reuse. | Avoid some re-prefill after history edits; positions, templates and model semantics constrain reuse. |
| RAM KV eviction/restore | integration: lifecycle | `llama_state_seq_get/set_data` and upstream bounded RAM cache; never use file save APIs for worker conversation data. | More inactive conversations than resident GPU slots; restoring adds transfer latency. |
| Context shifting / RoPE / YaRN | model-dependent | Core sequence operations and scaling parameters exist; coordinate all stages and validate model support. | Longer operation within limits; scaling is not free context, shifting discards information. |
| Sliding-window and recurrent state checkpoints | model-dependent | Upstream SWA/recurrent memory and checkpoint support. Requires a compatible architecture port and non-plain-KV rollback handling. | Architecture-specific memory efficiency. No benefit switch for the current Qwen2 model. |
| Different weight quantizations / importance-matrix quantization | model-dependent | Use existing upstream quantizers or compatible prequantized GGUFs; catalog each immutable artifact. | Weight memory/bandwidth versus quality. Changing a model artifact is not a runtime flag. |

## 3. Speculation

| Opportunity | Classification / current state | Existing upstream code and required DAN work | Benefit / tradeoff |
|---|---|---|---|
| Ordinary draft model | already active, opt-in | Upstream `common/speculative.cpp` owns proposals; DAN maps contexts/sequences and retains ring verification/rollback. Prepared API chats currently fall back to plain decode. | No automatic acceleration claim; requires compatible draft. |
| Adaptive draft width | active DAN controller | Current controller observes full-round time per committed token. Compare upstream alternatives by objective and supported method before replacement. | WAN-aware fallback; acceptance rate alone does not capture WAN/compute cost. |
| N-gram simple | already active, opt-in | Calls `common_ngram_simple_draft` with session-local history/reset/rollback and existing verifier. | No draft weights; repetition-dependent. Non-greedy/constrained chats fall back to normal decode. |
| N-gram map/cache/mod | integration: upstream helpers | Reuse `ngram-map`, `ngram-cache`, `ngram-mod` or dispatcher. Isolate state per session; no disk statistics or shared user histories. | Different lookup/memory tradeoffs; do not assume all methods should run together. |
| Chained speculation methods | integration | Upstream common dispatcher supports method selection/chaining. Integrate once with DAN rather than independent hand-written draft algorithms. | Fallback choices; more lookup/model work can negate gains. |
| EAGLE3 | model-dependent | Existing helper; trained compatible draft and selected target-layer features required. Route those features to the draft owner. | Cheaper/more accurate proposals in supported combinations; extra state and possibly WAN bytes. |
| DFlash / DSpark | model-dependent | Existing parallel draft helpers; compatible draft checkpoints and target hidden-feature plumbing. | Parallel proposals; no universal compatibility with current target. |
| MTP | model-dependent | Existing `load_mtp`/MTP contexts and helper; trained heads, correct stage ownership and hidden state required. | Draft without a separately chosen ordinary model; not available for arbitrary checkpoints. |
| Tree verification / asynchronous draft overlap | deferred | Reuse available batching/KV/speculative primitives; branch positions, cancellation and in-flight WAN rounds still need DAN coordination. | Potential overlap; large correctness/state cost and no established DAN gain. |

The latest upstream release notes also list speculative correctness changes,
including probabilistic simple/MTP sampling, truncation handling and EOG acceptance.
Audit applicable fixes before enabling non-greedy speculation. Do not assume the
older pinned helper already contains them.

## 4. Serving and useful output

| Opportunity | Classification / current state | Existing upstream code and required DAN work | Benefit / tradeoff |
|---|---|---|---|
| Persistent model/routes | active; local replicas on | DAN already owns decentralized replica formation and route lifetime. | Avoid loading and route setup. Upstream server residency does not replace the distributed lease system. |
| Persistent client across HTTP requests | already active | One retained framed-stdin client with user/cache-key scope and five-minute idle expiry. | Enables prefix reuse; release on cancellation/error/shutdown. |
| Concurrent API requests | integration | Local gateway currently permits one inference at a time with a bounded queue; engine supports several resident sessions. Connect request limits to real replica capacity. | Better concurrency; contention can hurt individual latency. Separate from GPU batching. |
| Continuous batching | already active, opt-in, limited scope | Ready ring decode frames map to distinct sequence IDs in one upstream batch; no fill delay. Prefill, local whole-model loops and serialized HTTP requests remain unbatched. | Owner approved opt-in; default off. Draft-enabled workers skip batching. |
| Prefill/decode scheduling fairness | integration alongside concurrency | Reuse upstream slot/batch scheduling patterns; retain provider-local decisions. | Prevent large prefills starving active chats; chunking already exists but is not a complete multi-request policy. |
| Sampling controls / stop / usage | already active sampling/nonstream usage; integration needed for stop/stream usage | Upstream common sampler configured per request at tail; cached/prompt/completion token counts returned. | Non-greedy speculation stays disabled. |
| Grammar and JSON-schema output | already active | Upstream grammar sampler and common schema conversion, session-local. | Only constrained requests pay grammar costs. |
| Chat templates / tool parsing | upstream templates active; general parser integration needed | Model-owned common/Jinja templates replace Go prompt formatting; Qwen2 output parser remains. | Required tool calls tested; new architectures still need adapter compatibility. |
| SSE and tool-result loops | active | Current gateway streams content; Open WebUI runs tools. | Already functional. Prefix reuse can avoid re-prefilling repeated tool context. |
| RAG / compaction / prompt size | active harness options | Existing upstream harness retrieval and compaction; tune limits and retrieval selection there. | Less context processing and memory; summaries/retrieval can omit useful information. |
| Embeddings / reranking / multimodal | model-dependent | Reuse upstream embedding/reranker/mtmd implementations; compatible models and input/stage support needed. Local document embeddings already exist in the harness. | Capability/answer quality; additional model work and memory, not automatic inference speed. |

## 5. Loading, fit and architecture

| Opportunity | Classification / current state | Existing upstream code and required DAN work | Benefit / tradeoff |
|---|---|---|---|
| Weight cache / range downloads | active | DAN downloads owned ranges from its own catalog and reuses files; llama loads those artifacts. | Faster cold provisioning; never replace worker catalog resolution with client URLs. |
| Loading mode: mmap/mlock/direct I/O | integration: option wiring | `load_mode` exists; current worker inherits auto. Check platform support, host memory and stage files. | Startup/residency tuning; locking RAM can pressure other processes. |
| Automatic local fit | integration: helper | Upstream common fit helpers can inform worker-local sizing; reconcile with DAN's declared context, lease and placement budget. | Fewer allocation failures; must not silently lower negotiated context or force CPU offload. |
| MoE tensor offload / CPU experts | model-dependent | Reuse `tensor_buft_overrides` and upstream CPU-MoE policy; update local resource accounting. | Fits larger models, often with CPU/PCIe cost. No expert-placement algorithm should be reinvented. |
| Lazy tensor loading / host-buffer bypass | model-dependent | `lazy_mode`, `no_host`, extra buffer types already exist upstream; only marked/supported tensors/backends apply. | Host memory/capacity tradeoffs; disk-on-demand can hurt latency. |
| More model architectures | model-dependent | Reuse upstream model graphs; port only DAN stage-boundary handling and architecture-specific memory estimates/rollback. Qwen2 and OLMoE currently supported. | Better model selection and potentially more efficient architectures. |
| Upstream version update | integration only for selected benefits | Port the stage patch in an isolated build; reconcile batch/context APIs and rebuild matching binaries. | New model/backend support and fixes; presence of an algorithm at the current pin does not require an upgrade. |
| Idle unloading / sleep | deferred | Upstream server sleep behavior could be reused as worker policy. | Saves idle VRAM/power but removes warm availability; conflicts with current latency-first replica use. |

## 6. WAN and host transport

| Opportunity | Classification / current state | Reuse and actual work required | Benefit / tradeoff |
|---|---|---|---|
| Activation F16 / FP8 | active opt-in | Already wired; upstream F16 conversion plus DAN row framing/compression. F32 remains default. | Fewer bytes; FP8 quality drift is documented. KV quantization is a different feature. |
| Prefill pipelining | active | Upstream decode batches plus DAN chunk propagation; expose/tune existing `prefill_chunk`. | Overlap stages for long prompts; chunk-size tradeoff is workload/network dependent. |
| Fewer/better stage hops | active placement; integration for refinements | Existing planner and measured route costs. Improve stage-edge evidence and local decisions, not a central scheduler. | WAN wait reduction; putting too much work on one slow GPU can lose the saving. |
| Direct return / persistent streams / NAT | active | Keep DAN ring plus go-libp2p discovery, hole punching and relay fallback. | Avoid client bounce and repeated connections. llama RPC is not a drop-in decentralized replacement. |
| QUIC/TCP selection | active transport support | Sidecar advertises QUIC and TCP using libp2p; inspect actual selected links before any change. | Do not claim QUIC is a missing feature or universally faster. |
| Socket buffering / coalescing / NODELAY | deferred pending evidence | Use OS/libp2p controls, not a new transport. Prior local probe found no material stall; C++ header/body are separate writes. | Small-message latency opportunity, not a demonstrated current bottleneck. |
| GPU/CPU boundary copies and synchronization | integration investigation | Worker calls synchronize, reads hidden rows and packs wire payloads. Reuse ggml buffer/copy capabilities where ownership allows; prove which synchronizations are redundant before removal. | Potential CPU/copy overhead reduction. Host network framing still needs valid data; upstream has no ready-made DAN zero-copy WAN path. |
| Compute/transfer overlap across requests | deferred with batching | Reuse backend async primitives; DAN still needs buffer lifetime/backpressure and route/session coordination. | Utilization; added in-flight state and queueing. |
| Prefill/decode disaggregation, KV migration, expert parallelism over WAN | deferred | Not a switch in this stage worker. Requires new placement/lifecycle and large state transfers even when using upstream compute. | No established benefit here; avoid speculative redesign. |

## Inspection result before selecting work

1. Existing knobs need no implementation: Q8 KV, activation precision, prefill
   chunks/microbatches, ordinary speculation, draft width, replicas and session count.
2. Smallest *new exposure* candidates are CPU thread counts, FA override, independent
   supported K/V types and load mode. Their usefulness depends on the actual use case;
   a small diff alone is not a reason to enable them.
3. N-gram and ordinary draft reuse require upstream helper integration. RAM prefix
   reuse and persistent HTTP clients address repeated-turn work but require lifecycle
   changes. Review them together before choosing one in isolation.
4. Batching, GPU sampling and local tensor splitting have unresolved latency tradeoffs.
   Trained speculation and newer architectures have model prerequisites. These are
   candidates to discuss, not defaults to switch on.
5. No performance ranking or calendar estimate is justified by this inventory alone.
   The owner handles WAN performance evaluation; future implementation checks should
   target correctness, cancellation, rollback and session isolation.

## Evidence

Local source inspected: `engine/stage_worker.cpp` (Stage initialization, sessions,
draft rounds and hidden output), `engine/provider_launcher.cpp` (config propagation),
`engine/planner.cpp` (F16 KV budgeting), `engine/dan_client.cpp` (API session lifetime),
`sidecar/cmd/dan-api-gateway/local.go` (queue, serialization and subprocess),
`sidecar/cmd/dan-api-gateway/tools.go`, `sidecar/network.go`, `CMakeLists.txt`.

Pinned upstream at `D:/Desktop/DAN/build/provider-owned-v0/llama.cpp`,
`95ef7fc16054e63b427a3ef00188e055ef7586d8`: `include/llama.h`,
`src/llama-context.cpp`, `common/speculative.{h,cpp}`, `common/ngram-map.{h,cpp}`,
`common/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt`,
`tools/server/README.md` and `README-dev.md`.

Online primary sources checked 2026-10-06:
- [Upstream server controls](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Speculation implementations](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [Recent releases](https://github.com/ggml-org/llama.cpp/releases): new extended batch
  API plus KV restore and speculative correctness fixes; these justify a focused
  compatibility/fix review, not a blanket update or claimed RTX 2070 speedup.
- [Adaptive MTP proposal #27210](https://github.com/ggml-org/llama.cpp/pull/27210):
  open when checked, not a merged general-purpose WAN controller. Its acceptance-driven
  depth policy is not equivalent to DAN's elapsed-round policy.

The existing [reuse tracker](OPTIMIZATION_RESEARCH.md) and
[WAN review](WAN_OPTIMIZATION_REVIEW.md) retain prior tests and measurements.
