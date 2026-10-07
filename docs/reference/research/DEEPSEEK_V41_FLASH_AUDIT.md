# DeepSeek-V4.1-Flash: dedicated DAN beta feasibility audit

Date: 2026-09-20. Status: **design/source audit, not implemented or model-tested in DAN**.

## 1. Executive decision

**DeepSeek-V4.1-Flash is a real released model, but it is not a small extension of DAN's OLMoE support.** It is a plausible research/porting target, not a model that can presently be selected for a working DAN beta by changing the catalog.

The networking architecture does not need replacing. Contiguous layer ownership and local experts remain possible. However, this particular model requires changes to the execution boundary, runtime state, tensor ownership, capacity estimation and allowed layer cuts. The largest obstacles are **runtime maturity, cross-layer attention dependencies and memory placement**, not the router.

Recommended decision:

1. Keep V4.1 as the explicit desired beta target if that is the product decision.
2. First establish an independently validated, full-model, text-only runtime on suitable hardware, outside DAN's distributed path.
3. Then prove a constrained two-stage cut before building automatic placement for it.
4. Do not promise a friends-on-consumer-GPUs beta, a launch date, or WAN throughput until those gates pass.

The existing RTX 2070 + A4500 setup is **not a viable all-resident configuration for the published model**. Making that pair run it would require substantial CPU/disk offload and expert-streaming work, not merely layer splitting. No such implementation is proposed here.

This audit uses the minimal-change skill: reuse runtime implementations where validated; keep the existing decentralized control plane; reject generic runtime frameworks and distributed expert parallelism as prerequisites. Minimal does not mean pretending required model state is absent.

## 2. Evidence, scope and reproducibility

### Inspected snapshots

| Component | Snapshot |
|---|---|
| DAN HEAD | `fe1f16cf80207dd361d0d861c688405657b0f420` |
| DAN actual worktree | Also has pre-existing changes to `CMakeLists.txt`, `docs/PROJECT.md`, and untracked `engine/tests/sessions_client.cpp`; preserved |
| DAN pinned llama.cpp | `95ef7fc16054e63b427a3ef00188e055ef7586d8`, with DAN's local patch |
| Upstream llama.cpp observed HEAD | `f072b103714dfa1eee531f80b24512faf38e3dd2` |
| Official model/config/reference | `deepseek-ai/DeepSeek-V4.1-Flash`, revision `dba1be0a40aa45a94ad051997016db3960a90277` |
| Shard observed HEAD and local checkout | `fcf728096948c7686bcf0897e9acb75d1abda1d5` |
| Vcruz runtime branch | `vcruz305/llama.cpp`, `runtime/deepseek41`, `5210c7c5ed61dddaee6ed476623abf4b63093d16` |
| JigSaw runtime branch | `JigSawPT/llama.cpp`, `dsv41-porte`, `3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5` |
| DwarfStar | `antirez/ds4`, `8db1d1d155cb0400a86a86b9c62d0defb3a6148b` |
| DwarfStar GGUF repository | `antirez/deepseek-v4.1-flash-gguf`, `dd8a266f7145edc19e2334b46e19b6821f221dc7` |

Read DAN's project guide and relevant engine, patch, build and protocol code. Inspected official configuration, weight index, reference execution/Engram/kernel code, upstream conversion PR, selected experimental runtime code, and Shard's V4 stage implementation. External code was read, not executed.

**Not done:** downloading model weights; validating a GGUF header against those weights; building any external runtime; running inference, performance tests or CUDA compatibility tests; provisioning hardware. External test claims remain their authors' evidence, not DAN measurements. This is a focused feasibility audit, not a full correctness/security review of every external fork.

Source links in this document identify the exact code where practical. Dynamic PR status and repository listings were checked on the date above.

## 3. Exact model: do not confuse V4.1 with V4

The official card describes a 552B backbone plus a 196B conditional-memory component. Repository parameter badges vary with packed storage and auxiliary components; they are not a reliable memory estimator. The official checkpoint index reports **510,286,023,000 bytes** across 48 safetensors shards. That includes components beyond the plain text backbone. [Model card][card], [checkpoint index][index]

The executable configuration supplies these details:

| Property | Value |
|---|---:|
| HF architecture | `DeepseekV41ForCausalLM` / `deepseek_v41` |
| Backbone layers | 40 |
| Hidden width | 5,120 |
| Persistent residual streams | 4 |
| Attention heads / latent head width | 64 / 512 |
| Q low-rank width | 1,280 |
| Routed experts per backbone layer | 384 |
| Selected routed experts per token | 6 |
| Shared experts per layer | 1 |
| Expert intermediate width | 2,304 |
| Sliding attention window | 128 |
| Global KV source layers | 2, 8, 14, 20 |
| Index source layers | 2, 8, 14, 20, 24, 28, 32, 36 |
| Candidate-pool source | 20 |
| Index top-k / candidate blocks / block size | 512 / 2,048 / 8 |
| Engram layers | 1 and 14 |
| Engram table rows | 384,006,168 and 384,016,682 |
| Engram row width | 256 |
| Engram n-grams / heads | up to 4 / 8 |
| Vocabulary | 129,280 |
| DSpark blocks / target taps | 3 / 37, 38, 39 |
| DSpark proposal block size | 5 |
| Output embeddings | Untied |

These numbers come from [official config][config] and [runtime config][runconfig], not from V4-Flash documentation. Compression-ratio arrays contain 43 entries because the three auxiliary prediction blocks follow the 40 backbone blocks. They must not become 43 ordinary DAN transformer layers.

The model's advertised context and active-parameter efficiency are not DAN capacity or performance guarantees. Advertised agentic results also use different sampling, context and tool harnesses from DAN's greedy chat. [Model card][card]

## 4. Actual runtime availability

### DAN's pin: no V4.1 execution support

The pin contains `src/models/deepseek4.cpp` and `llama-kv-cache-dsv4.cpp`: **earlier V4 support is present**. It does not provide the V4.1-specific graph/Engram implementation found in the forks below. DAN independently rejects both architectures: `manifest.cpp::load_manifest`, `planner.cpp::compatible_stage_model`, and the patched stage-load gate admit only `qwen2` and `olmoe`.

Consequently neither changing the whitelist nor labeling a V4.1 file `deepseek4` is a valid implementation.

### Current upstream: conversion work is not runtime support

PR [ggml-org/llama.cpp #28696][convertpr] was open, draft and unmerged when inspected. Its head was `b12818a24407175d941e9299e7b5fb7874a654d9`; changed files are conversion registration, `conversion/deepseek.py`, and GGUF constants. It explicitly separates `deepseek41` from V4 and says the PR is conversion-only.

The inspected upstream HEAD's source tree contains `deepseek4.cpp`, not the separate V4.1 runtime. The mere existence of downloadable GGUF quantizations does not establish upstream executability.

### Official Python reference

`inference/model.py` implements the math; `inference/kernel.py` supplies TileLang kernels. Its README calls it a readable reference, not a production serving engine. Dependencies include PyTorch >=2.10 and TileLang 0.1.8. The example conversion/run uses eight tensor-parallel ranks. This is **not** DAN's contiguous-layer execution model. [Reference instructions][inferreadme]

With `world_size > 1`, reference attention and expert modules use distributed reductions; Engram tables are row-sharded. These mechanisms should not be transplanted into DAN's WAN ring. `world_size=1` is conceptually local execution but does not solve memory fit.

### Experimental llama.cpp forks

**Vcruz:** a real `src/models/deepseek41.cpp` implements carried hyperconnection coefficients, Engram, shared KV/index sources and a V4.1 graph. Its source explicitly omits the hierarchical candidate mask and describes a corresponding context cap. Below the candidate-selection threshold the mask selects every block; above it omission changes the model. Confirm the cap at the runtime entry point before using it. [Pinned implementation][vcruz]

**JigSaw:** extends the V4 implementation with V4.1 branches, Engram support, shared caches, candidate logic and DSpark-related changes. Its separate report publishes reference comparisons, an actual inherited-Q-normalization bug, expert-streaming behavior and rollback fixes. Useful evidence, but the report's wording that remaining differences close correctness is stronger than logit correlation alone proves. Same-token argmax, correlation and reference quantization error need a larger held-out quality evaluation. [Pinned graph][jiggraph], [author's report][jigreport]

Neither is a drop-in DAN pin update. Both touch runtime internals beyond the model graph. They must be pinned and audited together with their converter and GGUF schema.

### DwarfStar: a genuine alternative, not a free backend swap

The inspected main tree includes `ds4_deepseek41_cuda.cuh`, `metal/dsv41.metal`, `ds4_engram.c`, model-specific tests and conversion tooling. CUDA code includes quantization, candidate selection, carried state and Engram kernels; Engram code reads rows using `pread` and batched reads. This is executable implementation evidence, not merely a support-list entry. [CUDA source][dscuda], [Engram source][dsengram]

Its README distinguishes V4.1 Metal support and CUDA text inference; do not infer that every existing GPU path or multi-GPU feature works for V4.1 on RTX 2070/A4500. Integrating this runtime would be a new backend project. Use it as an independent numerical/operational reference, not the default DAN integration strategy.

**Backend recommendation:** prefer a validated llama.cpp-derived V4.1 implementation because DAN already integrates that API. Do not choose a fork purely by reported tokens/s. There is insufficient independent evidence in this audit to certify one as beta-ready.

## 5. Trace the real model execution

The official `Transformer.forward`:

1. Computes Engram hashes from token IDs and per-sequence preceding tokens.
2. Embeds input IDs; optionally inserts vision embeddings.
3. Expands each hidden vector into four streams.
4. Initializes the first mixing vector to select the first stream.
5. Runs all 40 blocks, adding Engram contributions at layers 1 and 14.
6. Carries both residual streams and a mixing vector between blocks.
7. Collapses with the final block's carried vector, then applies final norm and the untied output head.
8. Returns sampled IDs, logits and optional DSpark taps. [Reference `Transformer` and `Block`][model]

Within `Block.forward`, attention computes coefficients whose pre-mix is consumed by the FFN; the FFN computes the pre-mix consumed by the **next block**. It returns `(x, ffn_pre)`, not just `x`. Recomputing this vector from the final block output is not generally equivalent: it was computed from an earlier intermediate residual.

Router computation is FP32 scoring, square-root softplus, correction-bias-assisted top-k, then unbiased selected-score normalization and scaling. The shared expert is always evaluated. Copying OLMoE's softmax/top-8 behavior would be wrong. The runtime should own this math. [Reference `Gate`, `MoE`][model]

### CED claim versus reference behavior

The publisher describes a causal encoder/decoder division. But the inspected reference's forward loop still executes **every backbone layer during prompt processing**. Do not budget a first port on encoder-only prefill or skip decoder layers. Decoder sliding-window states still need correct initialization, and the reference offers no such optimized serving path. Start with the complete reference computation; optimized bounded replay is later work.

The reference's nonzero-position compressor and window-cache branches assume a single token. Ordinary follow-up prompts and chunked prefill therefore require explicit runtime support or correct sequential replay; a multirow call at position >0 is not automatically supported just because initial prefill works. [Reference `Compressor.forward`, `Attention._window_kv`][model]

## 6. Boundary contract: what must cross a stage

**Can DAN's existing hidden-state activation contract remain unchanged? No.**

Current `Stage` in [stage_worker.cpp](../engine/stage_worker.cpp) uses `llama_model_n_embd()` as `hidden_`, allocates embedded input with that width, serializes that many output values in `put_hidden`, and checks the incoming shape in `run_middle`/tail processing. It does not transport the extra residual streams or mixing vector.

At a dependency-closed text-only cut, V4.1 requires at least:

| Item | Why |
|---|---|
| Four residual streams, `[rows, 4, 5120]` | Future blocks mix all four; averaging loses information |
| Carried pre-mix, `[rows, 4]` | Next block consumes previous FFN's coefficients |
| Token IDs, or equivalent exact Engram hash inputs | Layer 14 may be on another worker; hidden states do not encode recoverable token IDs |
| Existing session/request/position metadata | Associate both caches and history with the right sequence |

For arbitrary cuts, also carry or replicate shared compressed KV updates, index-key updates, current-query top-k selections and candidate masks as appropriate. Do **not** transmit the full historical cache on every decode step.

The minimal first contract can be one explicit, versioned V4.1 payload, not a generic graph/tensor message system. Reusing the outer frame and libp2p stream is reasonable; silently reinterpreting the old matrix is not. The receiver needs exact layout/length validation, explicit model semantics and a compatible runtime ABI.

Retain token IDs separately from `llama_batch.embd`. One inspected fork treats embedded input as image input for routing bias and treats missing token IDs as padding for Engram hashing. DAN's middle stage supplies embeddings for **text**, so directly reusing those conventions would silently compute the wrong model. The stage wrapper must distinguish text hidden input from vision input. See [Vcruz graph lines 886–892 and Engram input lines 308–339][vcruz].

### Payload size and a concrete current limit

FP32 streams plus FP32 pre-mix cost `(4*5120 + 4)*4 = 81,936` bytes per token, excluding token IDs and frame overhead. That is roughly four times a dense 5,120-wide boundary and ten times OLMoE's FP32 boundary.

- 512 rows: approximately 40 MiB per edge.
- 2,048 rows: approximately 160 MiB per edge.
- Existing [protocol.hpp](../engine/include/provider_owned/protocol.hpp) caps payloads at 64 MiB.

Even without IDs, only 819 such rows fit below that cap. Thus frame-size-aware prompt chunking is a **correctness requirement**, not optional optimization, for larger prompts. Verify chunk semantics first; do not simply raise the limit to accommodate arbitrarily large allocations.

## 7. Cross-layer KV: the crucial placement restriction

Official `SharedAttentionRuntime` contains `compress_kv`, `index_k`, `topk_idxs`, and `candidates`. Source layers overwrite references; later layers consume them. `Attention._compress_kv` and `Indexer.forward` prove these are real dependencies. They are not ordinary independent per-layer K/V. [Reference code][model]

For this exact checkpoint, a conservative dependency-closed partition is:

```text
[0,2) | [2,8) | [8,14) | [14,20) | [20,40)
            source 2     source 8     source 14     source 20
```

Cuts at 2, 8, 14 and 20 begin a new complete global KV/index source group. Within `[20,40)`, reindex layers still consume layer 20's shared keys and candidate pool. A new query indexer at layer 24 is **not** an independent KV source.

This partition is a source-derived design inference, **not a tested DAN route**. Validate it with captured reference intermediates and reject every unproven cut. Engram token history and carried mixing coefficients still cross even these cuts.

### Smallest implementation option

Keep source/consumer groups on the same worker and constrain the planner to legal cuts. Then all attention caches stay stage-local. A first candidate is `[0,20) -> [20,40)`; a later three-stage example is `[0,8) -> [8,20) -> [20,40)`.

The problem is physical fit: the last group contains 20 enormous expert banks. A smaller whole-model aggregate byte count does not guarantee any valid partition.

### If those groups do not fit

Either:

- use larger-memory workers or a validated worker-local offload policy; or
- implement ordered forwarding/replication of shared KV/index state to make finer cuts legal.

The latter preserves a ring and is not distributed expert parallelism, but it **is significant new execution-state machinery**: update ownership, session isolation, per-step index metadata, prefill, rollback and cleanup must all be designed. It cannot be hidden inside a metadata whitelist change.

Do not introduce that machinery until a measured hardware inventory shows constrained cuts are insufficient for the intended beta.

## 8. Weight and memory feasibility

### All experts remain assigned locally

For each layer, routed expert parameters are:

```text
384 experts * 3 projections * 5120 * 2304 = 13,589,544,960 parameters
```

At packed FP4 plus one byte of scale per 32 weights, that alone is about **6.724 GiB per layer**. Twenty layers require **134.47 GiB of routed expert storage**, before shared experts, attention, buffers, caches, head or allocator overhead. These are arithmetic storage estimates, not measured VRAM allocations. [Reference `Linear`/`Expert`][model]

Aggressive alternative GGUF quantization changes these numbers and requires quality validation. Six selected experts never means retaining only 6/384 of the weights.

### Engram is a second, separate memory problem

The two tables contain:

```text
(384006168 + 384016682) * 256 = 196,613,849,600 values
```

At native FP8 plus eight scale bytes per row, they occupy **188.833 GiB combined**, roughly 94.4 GiB on each of their layer owners. The official `ParallelEngramEmbedding` allocates them as tensors; it does not implement DAN-friendly disk residency. [Reference code][model]

A token accesses 24 rows per table (three n-gram lengths times eight heads), so local disk-backed lookup is a useful model-specific solution. It is not remote expert dispatch, and it should not become one HTTP fetch per lookup. Download the owned table to verified local storage first; then use a validated runtime's local mapped/batched reads.

This means a worker may own far more disk bytes than GPU bytes. DAN currently conflates owned tensor bytes with its GPU fit estimate. That is conservative but unusable for a disk-backed Engram implementation. Simply subtracting Engram from the estimate without implementing its actual residency/lookup path is unsafe.

### Published GGUF examples are not interchangeable

DwarfStar's publisher lists a Q2 file of 340.60 GiB and Q4 of 482.98 GiB, both including 188.83 GiB of native Engram. Main-weight portions are 151.77 and 294.15 GiB respectively, excluding runtime/context buffers. Its Q4 download is raw binary parts joined into one GGUF, not independently loadable split-GGUF shards. [Pinned artifact card][dsgguf]

Other converters quantize or represent Engram differently. A smaller filename/quant label is not proof of compatible metadata, identical hash tables or preserved quality. Pin the converter, runtime, artifact revision and complete hashes as one compatibility unit.

### Required estimator split

For the first supported backend, distinguish:

1. Download/physical disk bytes for every owned tensor and mandatory global tensor.
2. Resident GPU bytes for expert/attention/shared weights.
3. Host resident/working-set bytes for offloaded components.
4. Disk-backed Engram bytes and staging/lookup buffers.
5. Actual stage-owned attention/compression/history state times session count.
6. Peak loading, repacking, compute graphs, prefill workspace and logits.

Existing `planner.cpp::kv_bytes` assumes `hidden/heads` is head width and ordinary F16 K plus V per layer. Here that would compute **80 instead of the explicit 512-wide latent**, while also ignoring shared sources and compression. It is structurally wrong, not fixed by increasing the reserve.

The advertised compact KV figure is not an allocator estimate: the reference quantizes/dequantizes values in place and keeps BF16-shaped buffers, plus per-layer windows, compressor accumulators and index state. Neither the 890-byte claim nor the existing `max(1 GiB,15%)` reserve can certify fit.

DAN also caps planned candidates at eight (`placement.cpp::max_planned_candidates`). If deployment requires more workers, that is another explicit limit to address—but increasing it cannot solve an oversized indivisible dependency group.

## 9. GGUF ownership and sparse downloads

Existing `range_model.cpp::owned_tensor` includes owned `blk.N.*`, the head embedding and tail output/norm, and rejects other globals.

Reusable: a layer-prefix rule should include that layer's expert banks, router, shared expert, attention, norms, mixing weights, compressor/indexer and Engram tensors **if the selected converter names them under that prefix**.

Not automatically covered:

- Global Engram token-map, prime, offset and multiplier tensors. JigSaw stores these as global host tensors; Vcruz uses metadata arrays for some corresponding data. Validate the selected schema, not a generic assumption.
- Metadata arrays for compression ratios, KV/index sources, candidate selection, explicit head dimensions and hyperconnections.
- Auxiliary DSpark blocks, vision components and their ownership/exclusion.
- Multi-file GGUF input and raw split download artifacts.

`RangeModelRequest` currently represents one URL, one full hash and one logical sparse file. Do not point it at the first of 11 model shards and expect complete inspection/loading. The smallest approach is an immutable, compatible monolithic GGUF when operationally feasible. Otherwise a model-file bundle is real required download/index work, separate from inference networking.

For text-only, DSpark/vision weights may be omitted **only when the chosen loader explicitly supports their absence**. Tokenizer metadata, text routing, Engram and the ordinary text head must remain complete.

Do not scan/hash hundreds of GiB of cache synchronously before every warm replica reuse without measuring it. In-place hashing removes extra disk writes, not the read cost itself.

## 10. KV, sessions and rollback

Current DAN session lifecycle is reusable as an external API, but the state it owns is not just ordinary KV:

- Raw sliding-window rings can overwrite old entries.
- Ratio-2 compressors retain partial-group values and scores.
- Global compressed KV and index keys are shared between layers.
- Query top-k/candidate views must never survive incorrectly into another session/step.
- Engram needs token history, including reset and image-boundary behavior if vision is enabled.
- A more optimized CED implementation may additionally require replay state.

`Stage::reset`, `destroy`, `rollback` and implicit rollback currently call `llama_memory_seq_rm`. Keeping that API is desirable **if** the selected backend implements all corresponding model state. A successful ordinary-KV removal is not enough.

Even the older pinned `llama_kv_cache_dsv4::seq_rm` refuses a real rollback when `n_rs_seq == 0` and has bounded restore semantics. DAN's `Stage` constructor does not establish a V4.1 rollback budget. Inspect/configure the selected runtime explicitly; never infer support from the function name.

The official Python reference has process-global `shared_attn` aliases and batch-indexed buffers, not DAN sequence-ID isolation. Translating that runtime directly would need additional ownership work. Favor llama.cpp's validated sequence-state implementation where possible, but retest it.

Existing OLMoE two-session results prove the Qwen/OLMoE path only. V4.1 needs new interleaved, reset, destroy/recreate and continuation tests, particularly through layer 14 and the decoder source at layer 20.

## 11. Speculation: disable for the first milestone

DAN's present small Qwen draft cannot be used: incompatible tokenizer/model family, and worker/placement explicitly restrict drafts to Qwen2.

V4.1 DSpark is a model-conditioned auxiliary system. It consumes target taps at layers 37–39, three special blocks, Markov/confidence heads and its own proposal strategy. It is not a separate ordinary autoregressive draft that can be dropped into `load_draft_model` at the head. [Reference `DSparkBlock`, `Transformer.forward_spec`][model]

The reference contains DSpark forward code, but its provided generation loop is plain autoregressive. That distinction matters: an available class does not prove integrated accept/reject/rollback serving.

First milestone:

- no DSpark, no Qwen draft, no n-gram speculation;
- validate ordinary continuation and cancellation/cleanup;
- deliberately test rollback before enabling any speculative mode.

Later, DSpark would naturally need tail-local access to all three taps, correct proposal transport and full state rollback. Keeping `[20,40)` together happens to preserve the tap locality, but does not implement the algorithm. Shard's earlier V4 DSpark path is design evidence, not V4.1 compatibility.

## 12. Numerical and hardware behavior

Reference `linear` dispatches by weight type. FP8 and packed FP4 weights consume quantized FP8 activations; TileLang GEMMs declare FP32 accumulators and normally return BF16. Router scoring, mixing calculations and ratio-2 compressor state use FP32. Engram rows dequantize to BF16. [Model code][model], [kernel code][kernel]

Important distinction: `act_quant(..., inplace=True)` and `fp4_act_quant(..., inplace=True)` simulate quantized values by dequantizing back into the tensor. A reference comment calling a cache FP4/FP8 does not mean that buffer occupies packed FP4/FP8 physical storage.

| Boundary | First-port policy |
|---|---|
| Weights and internal arithmetic | Preserve the selected validated runtime's semantics; record deviations from official reference |
| Stage residual + pre-mix | FP32 wire initially; preserve original stream layout |
| Token IDs / routing metadata | Integer/exact; never quantize |
| KV/index state | Backend-specific, explicitly measured |
| FP16/FP8 activation wire | Disabled until full-width FP32 is validated |
| Draft | None |

FP32 transport does not restore precision already lost inside a kernel. FP16/FP8 transport can change both selected experts and sparse-attention positions. Small logit errors can become discrete routing changes. Do not promise CPU/GPU, cross-fork or quantized/reference token identity.

Conversely, coherent output does not excuse a systematic graph error. Compare intermediate streams, carried mixes, Engram hashes, compressed latents, index selections and logits before treating differences as expected numerical drift.

Official kernels use very recent low-precision Torch/TileLang paths. Execution on the RTX 2070 or A4500 was not verified. A GGUF fork may use alternative kernels, but successful OLMoE CUDA execution does not certify those new operations or memory paths.

## 13. What Shard contributes—and what it does not

The inspected Shard tree has `deepseek_v4`, `kimi_k3` and `minimax_m25` engines; no V4.1/Engram implementation was found. Its V4 engine is for the earlier architecture.

`engines/deepseek_v4/v4_stage.py` demonstrates contiguous local-expert ranges, four-stream boundaries, token-ID forwarding, snapshot/restore of window and compressor state, and replay for multi-position verification. It rejects distributed world sizes other than one for its stage execution. `v4_pipe.py` and DSpark files implement the associated transport and draft coordination. [Pinned Shard stage][shard]

| Lesson | DAN relevance |
|---|---|
| Keep expert banks local to their layer | Reuse DAN's ownership principle |
| Carry all residual streams | Required model semantics |
| Carry token IDs | Required here for Engram, not V4's older token-hash router |
| Snapshot/restore more than KV | Required principle; prefer backend state APIs |
| Draft tap locality | Relevant later for DSpark |
| Custom grouped kernels and pipelined speculation | Performance work, not first-port prerequisites |
| V4 layer independence/state never on wire | **Cannot transfer unchanged** to V4.1's shared attention |

Do not copy the V4 collapse head: V4.1 uses the carried final mix instead of V4's learned output hyperconnection tensors. Do not copy V4's per-head Q normalization without checking V4.1's reference; a fork author documented exactly this mistake.

## 14. Performance expectations: measured facts versus guesses

No DAN V4.1 speed estimate is established here.

Likely significant costs are:

- Resident expert-weight bandwidth and local expert kernels.
- Engram random-read latency and host-to-device staging.
- Larger multi-stream prompt traffic across WAN.
- Serialized stage execution and loop latency, as in current DAN.
- Very large initial downloads and cache verification.
- Load-time conversion/repacking and graph/workspace peaks.
- If selected: expert streaming, whose cold/content-dependent misses are a separate execution model from DAN's resident ranges.

The JigSaw report distinguishes cold-content and repeated-content measurements, describes reference/port numerical differences, and reports mixed speculative results. These are useful experimental lessons, not predicted DAN throughput. Repeating an identical greedy prompt can warm the **same expert access sequence**, so the usual single warm-up benchmark can conceal offload misses. [Author's report][jigreport]

Do not carry the OLMoE 10 tok/s WAN result over to a 552B backbone with larger activations. Nor assume the advertised lower prefill active count is already implemented in a selected backend.

Minimum telemetry: per-stage compute and wait, edge bytes/time, Engram reads and latency, disk cache misses, GPU/host loading peaks, session-state bytes, prompt rows/chunks, and end-to-end TTFT/decode rate. Record distinct-content and repeated-content workloads separately.

## 15. Smallest defensible implementation path

### Phase 0 — Feasibility gate, before DAN code

Select and pin one executable llama.cpp fork + compatible artifact. Validate full-model **text-only**, short-context, no-speculation behavior against official reference evidence and your own test set. Inventory real available GPU memory, RAM, SSD space and disk performance.

Deliverables: load log, exact artifact/schema/hashes, logits/reference comparisons, cold/warm memory peaks and capability matrix. Reject a runtime if it silently omits features within the promised context or cannot reset/continue correctly.

**Stop gate:** if no suitable worker can hold/execute the constrained groups, decide explicitly between bigger workers, local offload, or cross-stage state replication. Do not quietly expand scope.

### Phase 1 — Minimum metadata and storage support

Add only this architecture's necessary traits: expanded boundary shape, explicit latent geometry, source arrays, legal cuts, Engram ownership/residency and supported context/features. Extend index cache with a version bump. Pin one GGUF schema; add global-tensor rules and multi-file support only if the chosen artifact requires it.

No generic virtual ModelRuntime framework.

### Phase 2 — Full-range DAN worker

Port/rebase the chosen backend while preserving Qwen2/OLMoE tests. Implement full-range V4.1 through DAN with correct text input, formatting, all state lifecycle and measured admission. Keep speculation and lossy wire off.

### Phase 3 — Constrained two-stage split

Use a proven source boundary, initially 20 if hardware permits. Transport four streams, carried mix and token IDs. Keep all cross-layer attention consumers local. Compare stage-boundary tensors and output against the same full-model backend/artifact.

### Phase 4 — Three stages, placement and storage

Add a second legal cut; prove middle-stage Engram when applicable. Reject illegal cuts before reservation. Validate sparse download completeness and repeated cache reuse. Add only the placement rules demonstrated necessary.

### Phase 5 — Persistent sessions and replicas

Two interleaved conversations, reset/destroy/recreate isolation, follow-up multirow prompts, context exhaustion, cancellation, replica warm-up/reuse and failure cleanup. Never advertise READY on a stage whose required component remains absent.

### Phase 6 — WAN and beta acceptance

Two real machines, direct and relay separately, exact path logging, repeated short/long-prompt tests, cold acquisition and warm attachment. Then a trusted-friend install using the exact binaries/catalog.

Later: DSpark, FP16/FP8 wire, optimized CED prefill, expert streaming tuning, vision. Distributed experts remain a separate project, not a prerequisite.

## 16. Tests that specifically catch V4.1 mistakes

| Test | Failure it catches |
|---|---|
| Architecture and converter-schema rejection | V4.1 mislabeled as V4 or unknown metadata accepted |
| Global and layer tensor inventory | Missing Engram tables/maps, shared expert or mixing tensors |
| Boundary reconstruction of `x` and `pre_mix` | Averaging streams or dropping the lagged coefficient |
| Text embedding input through a middle stage | Incorrect vision routing bias or padding-only Engram lookup |
| Every legal cut / representative illegal cuts | Cross-worker access to absent shared state |
| First 4 tokens, reset and resumed conversation | Engram padding/history/hash drift |
| Positions 127/128/129 | Sliding-window overwrite/wrap errors |
| Odd/even positions around ratio-2 groups | Partial compressor state loss |
| More than 512 decoder positions and 1,024 encoder positions | Sparse top-k actually selects rather than retaining all candidates |
| More than 16,384 decoder positions, only if supported | Hierarchical candidate pruning; otherwise enforce a lower context cap |
| Prompt chunks at nonzero positions | Reference single-token assumption leaked into serving |
| Frame-size boundary and malformed lengths | Over-64-MiB frame, overflow or auxiliary-payload mismatch |
| Two sessions with unrelated token histories | Shared aliases/history contamination |
| Reset/destroy after ring wrap and partial compression | Clearing only conventional KV |
| Rollback at wrap/compression boundaries | Unsupported removal or stale state restoration |
| Cold and repeated-content runs | Disk/expert-cache effects misreported as steady compute speed |
| Qwen2/OLMoE regression suite and real routes | Backend rebase broke already-working models |

For non-identical backends, use stage-level numerical diagnostics and task-quality evaluation alongside token IDs. Same-backend full versus split is the primary isolation test; repeat the unsplit baseline to characterize its own nondeterminism before setting tolerances.

## 17. Complexity, risk and likely files

| Work | Complexity/risk | Why |
|---|---|---|
| Independent runtime/artifact qualification | High uncertainty | Experimental implementations, numerical and hardware gaps |
| Metadata and legal-cut validation | Medium | Concrete arrays/rules; unit-testable |
| GGUF ownership and artifact delivery | Medium–high | Global tables and enormous/multipart files |
| Engram residency/admission | High | Disk/host/GPU resources differ; current planner knows mostly VRAM |
| Full-range DAN runtime integration | High | Backend rebase and model-specific state |
| Dependency-closed stage boundary | Medium–high | Explicit extra tensors and input semantics |
| Arbitrary cross-source cuts | High; defer | Distributed shared state/rollback, not just activations |
| Sessions/replica correctness | High validation burden | Multiple mutable state types and long-lived resources |
| Networking/discovery redesign | Not required | Existing authenticated streams and route ownership remain useful |
| DSpark/vision/optimized CED | Separate later phases | Not needed to prove text-only split inference |

These are relative estimates, not calendar commitments.

Likely DAN changes after approval:

- `scripts/build_provider_owned.ps1`: pin selection and reproducible backend preparation.
- `patches/llama-provider-owned.patch`: V4.1 loading/graph boundary and stage-local memory filtering; more than the OLMoE graph wrapper.
- `CMakeLists.txt`: integration and ABI review. Current ABI hashes the patch; a backend rebase must also reliably identify the underlying backend/schema, not just leave the same patch identifier over different runtime code.
- `engine/manifest.cpp`: exact architecture/artifact admission.
- `engine/include/provider_owned/range_model.hpp`, `engine/range_model.cpp`: required metadata, globals, residency categories, cache version, selected artifact layout.
- `engine/planner.cpp`, `planner.hpp`, `engine/placement.cpp`: legal cuts, resource accounting, avoid dense byte-based speed assumptions.
- `engine/stage_worker.cpp`: expanded stage I/O, token-ID side input, state lifecycle, conservative context/feature gates, accurate metrics.
- `engine/include/provider_owned/protocol.hpp`, `activations.hpp`: explicit payload contract and bounded sizes; not a generic graph protocol.
- `engine/dan_client.cpp`: correct DeepSeek prompt/output handling; Qwen markers or raw unformatted prompts are not adequate beta chat.
- `engine/replica_owner.cpp`: likely mechanical boundary/feature integration and validation; retain ownership/leases/discovery behavior.
- Existing tests plus V4.1-specific reference/boundary fixtures; installer catalog only after runtime gates pass.

The official release supplies a Python prompt encoder, not a Jinja template. Use its test cases as the formatting oracle; avoid importing its complete API/tool ecosystem merely to support text chat. Reasoning/output conventions and sampling must be explicit in the beta's supported feature set. [Encoding reference][encoding]

## 18. Beta recommendation and acceptance criteria

**Conditional go for a dedicated V4.1 port; no-go for treating it as a near-finished beta configuration.**

If V4.1 is the non-negotiable target, the smallest scope is:

```text
Text only
One pinned model/converter/runtime combination
Short, explicitly validated context
FP32 boundary transport
Plain decoding, no DSpark
All experts local to their layer owner
Local disk-backed Engram where validated
Dependency-closed contiguous layer groups
Existing DAN ring, discovery, leases and replica ownership
```

Before calling that a beta, require:

1. Full-model runtime correctness and actual hardware fit demonstrated.
2. Two/three-stage results localized against the full-model baseline.
3. Two persistent sessions and replica lifecycle passing with all V4.1 state.
4. Direct/relay WAN measurements on the target deployment, including long prompts within the supported cap.
5. Useful answer quality and acceptable measured latency on representative tasks—not merely coherent output.
6. Repeatable installation, complete local artifact acquisition and clear resource/context limits.

No consensus, centralized scheduler, remote experts, payments, generic DAG runtime or new transport is required. But **a model-specific execution contract and much larger hardware/storage requirements are unavoidable**. The immediate next task is backend/artifact/hardware qualification, not coding the network or adding the architecture to the whitelist.

## Sources

[card]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md
[config]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json
[index]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/model.safetensors.index.json
[runconfig]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/config.json
[model]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
[kernel]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py
[inferreadme]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/README.md
[encoding]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py
[convertpr]: https://github.com/ggml-org/llama.cpp/pull/28696
[vcruz]: https://github.com/vcruz305/llama.cpp/blob/5210c7c5ed61dddaee6ed476623abf4b63093d16/src/models/deepseek41.cpp
[jiggraph]: https://github.com/JigSawPT/llama.cpp/blob/3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5/src/models/deepseek4.cpp
[jigreport]: https://github.com/JigSawPT/deepseek-v41-flash-on-5090
[dscuda]: https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4_deepseek41_cuda.cuh
[dsengram]: https://github.com/antirez/ds4/blob/8db1d1d155cb0400a86a86b9c62d0defb3a6148b/ds4_engram.c
[dsgguf]: https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/blob/dd8a266f7145edc19e2334b46e19b6821f221dc7/README.md
[shard]: https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_stage.py
