# DAN: minimal MoE support design

Reviewed September 20, 2026. Saved from the read-only design analysis delivered in chat.

> **Implementation status (2026-09-20).** Phases 1-4 of §17 are implemented and validated on
> one PC: architecture admission (`qwen2` + `olmoe`), expert-aware metadata and validation,
> the patched OLMoE range loader and graph, padded-KV admission, and single/two/three-stage
> routes whose token ids match the full-model reference on CPU and on one RTX 2070. See
> `docs/PROJECT.md` §9.9 for what was measured and what was not. Phases 5-7 (sessions and
> replicas, speculation, WAN) are not started. Two findings from implementation: the KV
> padding rule also applies to the existing Qwen2 path (the planner now models it for both),
> and the design's §15.5 static-harness command needs a manifest carrying `layers`/
> `hidden_size`, because a fixed `--provider` route never reads the GGUF header.

## Executive recommendation

**Add OLMoE-1B-7B-0924-Instruct as the first MoE target, using its Q4_K_M GGUF. Keep every layer’s experts on the worker that owns that layer.**

The existing activation contract can remain unchanged:

```text
A → B → C

Each boundary carries:
rows × hidden_size
+ existing position/session/request metadata
```

DAN should continue owning partitioning, sparse weight acquisition, placement, networking and sessions. The pinned llama.cpp should own routing, top-k selection, expert computation and combination.

The necessary runtime change is **range-aware loading and graph construction for one additional architecture**, not a new inference backend.

The investigation was read-only: no implementation files were changed, models executed, tests run, or resources provisioned. The selected GGUF’s header was read through a bounded HTTP range request; that verifies its metadata and tensor inventory, not successful inference. This document was subsequently saved at the owner’s request.

### Source snapshots

| Component | Inspected snapshot |
|---|---|
| DAN | `39c4b0f56ed150671c479ee64bc4062766c8dfc3` |
| Pinned llama.cpp | `95ef7fc16054e63b427a3ef00188e055ef7586d8`, with DAN’s local patch |
| Shard | `fcf728096948c7686bcf0897e9acb75d1abda1d5`; remote HEAD checked and unchanged |

The inspected DAN checkout includes the recent range-hashing, middle-stage copy, speculation-boundary, vocabulary and draft-memory changes. The older technical report is not authoritative for those paths. Recommendations below are proposals, not implemented support.

## 1. Current DAN dense-model assumptions

Classification below distinguishes genuinely reusable behavior from restrictions that must be changed or explicitly retained.

| Current assumption | Classification | Implementation and implication |
|---|---|---|
| Manifest architecture is absent or `qwen2` | **Qwen2-specific** | [manifest.cpp](../engine/manifest.cpp), `load_manifest`. Explicit `olmoe` currently fails. |
| Shape metadata uses `qwen2.*` keys | **Qwen2-specific, GGUF-specific** | [range_model.cpp](../engine/range_model.cpp), `parse_index`. Reads only Qwen2 layer/hidden/head geometry. |
| Inspected/cached model architecture must be `qwen2` | **Qwen2-specific** | `inspect_range_model`, `load_model_index` in the same file. Both need updating; changing only the live parser is insufficient. |
| Tensor directory entries need only name and byte span | **GGUF-specific** | `parse_index` discards dimensions and tensor type. Adequate for copying bytes, insufficient for early expert-shape validation. |
| Every layer has some `blk.N.*` tensor | **Dense/Qwen2 support gate, insufficient structural validation** | [planner.cpp](../engine/planner.cpp), `compatible_dense_qwen2`. It does not verify a complete layer’s required tensors. |
| Layer ownership is `blk.N.*` | **Generic and reusable within supported GGUF conventions** | `block_layer`, `owned_tensor`, `required_ranges`. Already includes packed expert tensors. |
| Head owns embeddings; tail owns final norm/output | **Generic for the selected architectures** | `owned_tensor` and patched architecture loader. |
| Missing output weight means tied embeddings | **Architecture-dependent, llama.cpp-specific** | Existing Qwen2 loader duplicates `token_embd.weight` on the tail. Valid for Qwen2; OLMoE’s loader requires an explicit output tensor. |
| Stage-enabled llama.cpp permits only `LLM_ARCH_QWEN2` | **Qwen2-specific, llama.cpp-specific** | [llama-provider-owned.patch](../patches/llama-provider-owned.patch), change to `llama_model_load`. |
| Only Qwen2 tensor loading and graph loops respect stage boundaries | **Qwen2-specific, llama.cpp-specific** | Patched `llama_model_qwen2::load_arch_tensors` and `graph::graph`. |
| Boundary state is one FP32 hidden matrix | **Generic for ordinary residual-stream transformers** | `build_inp_embd_direct`, `Stage::put_hidden`, `get_hidden`. Reusable for OLMoE, not universally for all MoE architectures. |
| KV is ordinary position-indexed attention KV | **Architecture-specific, llama.cpp-specific** | Stage-local filter in `llama_model::create_memory`; `llama_memory_seq_rm` for rollback/reset. Reusable for OLMoE. |
| KV head width equals `hidden / heads`, with equal K/V width | **Model-geometry assumption** | `kv_bytes` and worker metrics. Correct for selected OLMoE; not a general formula for MLA or arbitrary architectures. |
| K and V use FP16 | **llama.cpp-specific configuration assumption** | `Stage` inherits default `type_k/type_v = F16`. |
| Fit uses all selected tensor bytes, KV and reserve | **Generic and reusable** | `stage_model_bytes`, `stage_fits`. Already structurally suitable for resident MoE weights. |
| Decode cost is proportional to resident weight bytes | **Dense-oriented performance assumption** | `decode_bytes`, `estimate_token_ms`, `update_speed`. Not a reliable cross-model MoE cost model. |
| Draft consumes the target’s token IDs | **Generic speculation requirement** | `draft_proposals`, `vocabularies_match`; compatibility must be established, not inferred from family names. |
| Chat prompt uses Qwen’s `im_start`/`im_end` markers | **Qwen-specific application behavior** | [dan_client.cpp](../engine/dan_client.cpp), `chat_prompt`. Wrong for OLMoE’s supplied chat template. |
| Leases, ring routes and replica discovery identify a model by SHA | **Generic and reusable** | `lease.hpp`, `placement.cpp`, `replica_owner.cpp`, sidecar discovery. No expert-specific fields needed. |
| Workers must agree on runtime ABI | **Generic safeguard, current implementation llama.cpp-specific** | [CMakeLists.txt](../CMakeLists.txt) hashes the normalized patch text into `DAN_RUNTIME_ABI`. |

### Current execution path

`Stage::Stage` sets:

```text
dan_stage_start/end = assigned range
n_ctx              = context × sessions
n_batch             = context
n_ubatch            = context
n_seq_max           = sessions
embeddings          = !last_stage
pooling              = NONE
```

The patched Qwen2 graph:

1. Loads embeddings only at the head.
2. Loads only assigned transformer layers.
3. Uses direct hidden input on non-head stages.
4. Returns the unnormalized residual stream on non-tail stages.
5. Applies final normalization and output projection only at the tail.
6. Allocates KV only for owned layers.

The worker then handles prompts, activations and sessions independently of the internal FFN implementation. Relevant methods are `run_first`, `run_middle`, `run_last`, `execute`, `create`, `reset`, `destroy` and `rollback` in [stage_worker.cpp](../engine/stage_worker.cpp).

### Current documentation discrepancies relevant here

`PROJECT.md §9.2` correctly describes the strengthened vocabulary checks and context clamp. Some later limitations still say draft memory is unaccounted for, although `stage_with_draft_fits` now checks it at load time.

Likewise, the replica description still contains an older statement that draft correctness depends on serialized requests; current draft guess state is per session.

Neither discrepancy changes the MoE recommendation, but implementation—not those older sentences—is the basis here.

## 2. llama.cpp MoE capabilities in DAN’s pinned version

### 2.1 This pin already contains substantial MoE support

The pinned source includes executable model implementations for these relevant families:

| Family/group | Examples in the pinned source | Suitability for this milestone |
|---|---|---|
| Conventional attention plus MoE | OLMoE, Qwen2MoE, Qwen3MoE, Mixtral through the Llama implementation, DBRX, Arctic, Granite MoE, PhiMoE | Potentially compatible with layer-range execution; each still needs its own loader/graph audit |
| Larger conventional/model-specific MoE | MiniMax-M2, DeepSeek/DeepSeek2, GLM4-MoE, Hunyuan-MoE, Grok, OpenAI-MoE, ERNIE MoE | More geometry, attention or runtime qualifications |
| Hybrid/recurrent MoE | Qwen3Next, Qwen3.5MoE, Jamba, Nemotron-H-MoE, Granite hybrid, Kimi variants | Not covered by ordinary KV rollback assumptions |
| Expanded-state/special-attention models | DeepSeek4, Kimi-K3 and other specialized implementations | Not suitable for the first unchanged-contract milestone |
| Multimodal/encoder variants | Qwen3VLMoE, Nomic-BERT-MoE and others | Outside this text-decoder milestone |

This is source support, **not a claim that every architecture/quantization/backend combination has been tested in DAN**. Architecture registration and implementations are visible in the pinned [architecture registry](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/src/llama-arch.cpp) and [model sources](https://github.com/ggml-org/llama.cpp/tree/95ef7fc16054e63b427a3ef00188e055ef7586d8/src/models).

DAN’s own stage gate currently prevents all of these MoE implementations from being used.

### 2.2 Router, expert selection and combination already exist

`llm_graph_context::build_moe_ffn` implements:

```text
hidden
→ router projection
→ gating scores
→ top-k expert selection
→ selected expert up/gate projections
→ activation
→ selected expert down projections
→ routing-weight multiplication
→ expert reduction
```

It supports architecture-dependent choices including:

- Softmax, sigmoid and other gating functions.
- Selection biases.
- Group-limited selection.
- Optional selected-weight renormalization.
- Routed scaling.
- Separate or merged gate/up expert tensors.
- Expert biases and scales.
- Several FFN activations.

Those choices must remain controlled by the architecture’s llama.cpp graph. DAN should not recreate or standardize them. See [build_moe_ffn](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/src/llama-graph.cpp).

### 2.3 OLMoE’s exact implementation

The pinned `llama_model_olmoe` uses:

- Ordinary causal attention and RoPE.
- Q/K normalization.
- Ordinary per-layer attention KV.
- A softmax router.
- Top-k expert selection.
- SwiGLU experts.
- **No renormalization of the selected top-k probabilities** in its `build_moe_ffn` call.
- No shared expert.
- A conventional residual addition.
- Final RMS normalization and output projection.

That “no selected-weight renormalization” detail matters: copying Granite’s or another model’s router settings would silently change the model. The official checkpoint configuration also specifies `norm_topk_prob: false`. Sources: [pinned OLMoE implementation](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/src/models/olmoe.cpp), [checkpoint configuration](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct/blob/main/config.json).

### 2.4 Quantized experts and backend execution

Packed expert tensors use GGML’s indexed matrix multiplication, `GGML_OP_MUL_MAT_ID`.

The CUDA implementation selects among quantized matrix-vector, quantized matrix-matrix and floating-point paths according to tensor type, shape and hardware. A fallback path copies expert IDs to the host, synchronizes, groups tokens by expert and dispatches the corresponding computations.

Therefore:

- Quantized MoE execution already exists.
- It is not necessarily fully GPU-asynchronous.
- CUDA graph compatibility depends on which execution path is selected.
- No custom DAN expert kernel is required.

The pin also includes CUDA top-k/MoE and weighted-reduction fusion machinery. These are backend capabilities to reuse, not reasons to add DAN-specific kernel code. Source: [pinned CUDA backend](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/ggml/src/ggml-cuda/ggml-cuda.cu).

### 2.5 GPU/CPU offload

Upstream common CLI options include:

```text
--cpu-moe
--n-cpu-moe N
--override-tensor ...
```

They implement tensor-buffer placement overrides. DAN’s `Stage` uses the library API and currently sets `n_gpu_layers`; it does **not** automatically inherit those CLI options. Source: [pinned common argument handling](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/common/arg.cpp).

For the first milestone:

- Validate CPU execution separately.
- Validate fully resident GPU stages separately.
- Do not introduce expert-specific CPU offload configuration.
- Do not equate “all experts local” with “all experts necessarily on a GPU”; locality is a worker boundary, while residency is a separate runtime policy.

### Can llama.cpp’s MoE execution remain unchanged?

**Yes, for OLMoE’s router, experts and combination.**

**No, for its surrounding loader/graph entry and exit:** those currently execute the whole model and must become range-aware.

That distinction is the core of the proposed extension.

## 3. Shard’s actual MoE implementations

### 3.1 MiniMax-M2.5

Relevant executable files:

- `engines/minimax_m25/m25_stage.py`
- `engines/minimax_m25/m25_pipe.py`
- `shard/plan.py`
- `shard/probe.py`

#### Partitioning and state

`m25_pipe._load` constructs `Layer(i)` for `range(lo, hi)`. Each worker therefore owns contiguous transformer layers.

`_build_moe` loads **all experts for each owned layer**. Its vLLM setup uses world size one. There is no remote expert request in that stage’s inference path.

Ordinary layer boundaries carry hidden states. Optional EAGLE speculation adds auxiliary hidden-state taps, including the configured layers `1,30,58`.

KV stays on the stage. The eager path crops KV at `start_pos`; static KV paths overwrite positions and bound reads by the logical sequence end. Sources: [M2.5 stage](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/minimax_m25/m25_stage.py), [M2.5 pipeline](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/minimax_m25/m25_pipe.py).

#### MoE execution

The implementation uses vLLM `FusedMoE` with:

- 256 experts and top-8 routing for the inspected model.
- Sigmoid scoring.
- Expert selection correction bias.
- Configured renormalization and routed scaling.
- NVFP4 expert weights.
- No shared expert.
- Backend-dependent weight processing after loading.

The backend can change with GPU architecture. Consequently, checkpoint bytes alone are not a universal resident-memory estimate.

The per-stream and batched paths also explicitly distinguish numerical behavior: changing token counts changes expert grouping and potentially kernel scheduling.

#### Memory and speculation

Shard has model/backend-calibrated profiles rather than a universally exact allocator model. The M2.5 profile includes:

- Measured resident layer footprint.
- KV allowance.
- General reserve.
- Separate head/draft and tail/output reserves.
- A conservative layer cap.

The source documents an earlier estimate that was too small and packed stages close to OOM. Its probe distinguishes resident allocation from loading peak.

The transferable lesson is **measure resident and peak memory separately**, not “copy Shard’s MiB constants.” Sources: [placement profiles](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/plan.py), [probe](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/shard/probe.py).

### 3.2 DeepSeek-V4 / V4-Flash

Relevant files:

- `engines/deepseek_v4/v4_stage.py`
- `v4_pipe.py`
- `v4_dspark_draft.py`
- `v4_moe_decode.py`
- `v4_moe_grouped.py`
- `v4_moe_multi.py`
- `vendor/deepseek_v4_ref/inference/model.py`

#### Partitioning

`Stage(lo, hi)` still constructs a contiguous layer interval.

Importantly, `_set_globals` **rejects world size other than one**. The vendored reference contains distributed expert/all-reduce code, but Shard’s stage explicitly excludes that execution mode.

Thus the reference’s distributed primitives are not evidence that Shard’s active pipeline distributes a layer’s experts remotely. Source: [V4 Stage](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_stage.py).

#### Boundary and persistent state

V4’s boundary is not DAN’s current boundary:

```text
hidden: [batch, positions, 4, hidden_size]
token IDs
```

Four hyperconnection streams survive between layers. Token IDs are needed by hash-routed layers.

Stage-local attention state includes:

- Window KV.
- Compressed KV regions.
- Indexer cache.
- Compressor `kv_state`.
- Compressor `score_state`.

Rejected speculation cannot be repaired by cropping an ordinary KV sequence alone. Shard snapshots selected state, restores it and may replay an accepted prefix.

These are **V4 architecture requirements**, not generic MoE requirements.

#### Routing and experts

The inspected V4-Flash configuration has 256 routed experts, six selected experts and one shared expert. Its routing includes:

- Token-ID-based expert selection in initial hash layers.
- Score-based selection elsewhere.
- Selection bias distinct from combination weights.
- Architecture-specific score transformation and scaling.

The optimized paths use FP4 expert banks, FP8-related projection paths and grouped/custom kernels. Bank layout can avoid duplicate expert storage, but allocation ordering affects loading peak. Sources: [V4 reference](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/vendor/deepseek_v4_ref/inference/model.py), [grouped expert implementation](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_moe_grouped.py).

#### Speculation and placement constraints

DSpark uses target-layer taps `40,41,42` in the inspected configuration.

`Stage.tail_main_hidden()` refuses a stage that does not own every required tap. With that speculation implementation enabled, the tail must own those layers together and hold the extra draft state/weights.

This is a real placement constraint—but it comes from the selected model/drafter contract, not from MoE generally. Source: [V4 stage/tap checks](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/deepseek_v4/v4_stage.py).

There is no justification for adding equivalent constraints to OLMoE.

### 3.3 Kimi-K3

Shard also has executable Kimi-K3 stage and pipeline code.

Its boundary carries both:

```text
hidden/prefix stream
block_residual snapshot stack
```

Its stage-local state combines MLA KV and KDA recurrent/convolution state.

The stage’s `_seek` explicitly refuses rollback when KDA layers are present. Current executable behavior is therefore narrower than “generic speculative MoE support.”

Its packed expert backend uses local MXFP4 experts with architecture-specific `situ` activation handling; substituting a familiar activation would be incorrect. Sources: [K3 stage](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/kimi_k3/k3_stage.py), [K3 expert backend](https://github.com/leyten/shard/blob/fcf728096948c7686bcf0897e9acb75d1abda1d5/engines/kimi_k3/k3_moe_mxfp4.py).

K3’s planning profile separately considers resident weights, mixed state types and a large repacking transient. Those measured constants are specific to that checkpoint/backend, not transferable to GGUF OLMoE.

### 3.4 What not to infer from Shard

- Multiple engines do not establish a working generic runtime abstraction.
- Vendored distributed primitives do not establish active remote expert execution.
- Model-specific rollback machinery does not imply OLMoE needs snapshots.
- A resident expert bank is not an on-demand expert cache.
- Shard’s README claims of single-machine identity are broader than its cited V4 matrix, which describes equality against a greedy baseline on the **same ring**.
- Historical GLM/gpt-oss runs are not proof that the three current engine modules provide interchangeable support for those models.

## 4. Shard → DAN mapping

| Shard concept | Classification | DAN implication |
|---|---|---|
| Local router and top-k | **ALREADY PROVIDED BY LLAMA.CPP** | Leave inside the selected model graph |
| Local expert execution/combine | **ALREADY PROVIDED BY LLAMA.CPP** | No DAN expert executor |
| Contiguous layer assignment | **DIRECTLY RELEVANT TO DAN** | Already DAN’s execution model |
| Loading every expert of an owned layer | **DIRECTLY RELEVANT TO DAN** | Existing GGUF prefix ownership captures packed expert tensors |
| Shared experts | **ALREADY PROVIDED BY LLAMA.CPP; MODEL-SPECIFIC** | Not present in chosen OLMoE; audit separately for a later model |
| Full resident-weight accounting | **DIRECTLY RELEVANT TO DAN** | Never replace full expert bytes with active-parameter bytes |
| Backend/load-peak calibration | **RELEVANT BUT NEEDS DIFFERENT IMPLEMENTATION** | Measure llama.cpp allocations and driver VRAM, not PyTorch/vLLM constants |
| V4 hyperconnection boundary | **MODEL-SPECIFIC** | Not needed for OLMoE |
| V4 token-ID routing input | **MODEL-SPECIFIC** | OLMoE routing depends on hidden state, not raw IDs |
| V4 compressor snapshot/replay | **MODEL-SPECIFIC** | Ordinary OLMoE KV truncation remains sufficient |
| K3 residual snapshot stack | **MODEL-SPECIFIC** | Outside this milestone |
| DSpark tail/tap co-residency | **MODEL-SPECIFIC** | Do not add generic stage constraints |
| EAGLE auxiliary activations | **NOT NEEDED FOR FIRST MOE SUPPORT** | DAN’s existing independent draft does not consume these |
| Custom grouped/Triton/TileLang/vLLM kernels | **PERFORMANCE-ONLY** for this milestone | Use existing GGML backends |
| Expert-bank repacking/cache mechanisms | **NOT NEEDED FOR FIRST MOE SUPPORT** | GGUF already stores packed expert tensors |
| Cross-worker expert dispatch/all-reduce | **ARCHITECTURALLY INCOMPATIBLE** with the requested milestone | Explicitly deferred |
| Cross-session expert batching | **PERFORMANCE-ONLY** | Do not introduce a new scheduler |

## 5. Candidate first MoE models

| Candidate | Advantages | Why not necessarily first |
|---|---|---|
| **OLMoE-1B-7B-0924-Instruct** | Dedicated compact llama.cpp implementation; ordinary KV; single residual stream; official GGUF; approximately 4.21 GB Q4_K_M | Different tokenizer/chat template from Qwen; no existing compatible small DAN draft |
| Granite-3.1-1B-A400M | Considerably smaller; ordinary attention; pinned support | Additional embedding/residual/logit scaling and tied-head behavior; graph shared with other Granite variants creates a broader regression surface |
| Qwen1.5-MoE-A2.7B | Familiar Qwen lineage; conventional hidden boundary; exercises shared experts | **14.3B stored parameters**, despite 2.7B active; larger test footprint; shared-expert branch adds another correctness case |
| Qwen3-30B-A3B | Conventional MoE graph and official GGUF availability | Much larger resident footprint than needed to prove layer-range MoE |
| MiniMax-M2.5 | Relevant to Shard comparison | Far too large for the smallest correctness milestone |
| DeepSeek-V4-Flash | Rich MoE functionality | Expanded residual state, hash routing, compressed attention state and specialized speculation |
| Kimi-K3 / hybrid MoE families | Future relevance | Recurrent state and/or expanded boundaries complicate rollback and testing |

The size distinction for Qwen1.5 is explicitly documented by its publisher; official OLMoE and Qwen3 GGUF artifacts are available. Sources: [Qwen1.5 model card](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B), [OLMoE GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF), [Qwen3 GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF).

**Granite is the smaller download. OLMoE is the recommendation for the cleaner isolated runtime patch.** Granite remains a sensible alternative if testing hardware cannot accommodate OLMoE.

## 6. Recommended first target

### Exact proposed artifact

```text
Repository:
allenai/OLMoE-1B-7B-0924-Instruct-GGUF

File:
olmoe-1b-7b-0924-instruct-q4_k_m.gguf

Revision:
02ab6ea6894a8418eb14f6d8ee1bfb08bd298080

SHA-256:
8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f

File bytes:
4,213,512,672
```

The revision and hash came from the repository’s artifact metadata. The actual file header was inspected at that revision, without downloading the whole file. [Pinned artifact repository](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/tree/02ab6ea6894a8418eb14f6d8ee1bfb08bd298080).

### Verified GGUF geometry

| Property | Value |
|---|---:|
| Architecture | `olmoe` |
| Transformer layers | 16 |
| Hidden size | 2048 |
| Attention heads | 16 |
| KV heads | 16 |
| Expert intermediate width | 1024 |
| Experts per layer | 64 |
| Selected experts per token | 8 |
| Training context | 4096 |
| Vocabulary width | 50,304 |
| Tensor count | 195 |
| Shared experts | None |
| Output projection | Explicit, untied |

Start DAN validation at **context 512, one session, FP32 wire, speculation disabled**.

An 8 GB GPU is a plausible first single-worker target for this quantization and short context, but that is a capacity expectation—not a measured guarantee. CPU validation avoids making GPU availability a prerequisite for initial correctness.

## 7. Required DAN changes

### 7.1 Admit one additional architecture

Update manifest, inspection and cached-index validation to recognize exactly:

```text
qwen2
olmoe
```

Do not replace the whitelist with “anything llama.cpp recognizes.”

Parse the corresponding architecture-prefixed geometry. GGUF metadata ordering must not matter: do not assume `general.architecture` appears before every architecture-specific key.

For OLMoE, also retain:

```text
feed_forward_length
expert_count
expert_used_count
```

Retain tensor dimensions for structural validation. Tensor type is useful for diagnostics and supported-quantization checks; actual backend support remains llama.cpp’s responsibility.

### 7.2 Patch OLMoE’s range loader

Extend `llama_model_olmoe` to:

- Validate `[begin,end)`.
- Load embeddings only when `begin == 0`.
- Load final norm and output only when `end == layers`.
- Create layer tensors only for `begin <= layer < end`.
- Preserve global layer indices.

Do not reduce the expert dimension from 64 to eight. Eight is the selected count, not the resident count.

### 7.3 Patch OLMoE’s graph boundary

Mirror the existing Qwen2 stage wrapper:

```text
head:
    token embedding

non-head:
    build_inp_embd_direct()

all:
    execute owned layers using existing OLMoE operations

non-tail:
    return residual hidden states before final norm

tail:
    final norm → output projection → logits
```

Crucially, output-row selection must remain **tail-only**. A middle stage must return every prompt/speculation row, not just the final row.

The internal attention, Q/K normalization, router and FFN code should remain unchanged.

### 7.4 Extend the runtime gate and rebuild consistently

Allow `LLM_ARCH_OLMOE` in the stage-enabled loader gate.

Changing the patch automatically changes the existing runtime ABI hash. Rebuild the client and workers together; old and new workers should not form one route.

No new ABI field or wire version is necessary.

### 7.5 Keep chat correctness explicit

Current `chat_prompt` is Qwen-specific.

For the initial inference milestone, use raw prompts or externally rendered OLMoE prompts. Do not silently advertise Qwen-formatted `--chat` as supported.

Before shipping interactive OLMoE chat, either:

- Add a narrowly scoped formatter for the pinned model’s template; or
- Reuse a suitable existing llama.cpp chat-template facility where practical.

This is application formatting, not a network architecture change.

## 8. Changes that are not required

For this target, no changes are required to:

- DHT model discovery.
- PeerID authentication.
- NAT traversal or relay behavior.
- Worker lease semantics.
- Replica ownership.
- Ring construction.
- Session identifiers.
- Activation frame layout.
- Candidate-token suffix format.
- Tail-to-head token feedback.
- Owner forwarding.
- Result framing.
- Contiguous-layer planner search.
- Expert routing protocol—because none should be introduced.
- Distributed expert ownership.
- Custom CUDA/Triton kernels.
- A generic `ModelRuntime` interface.

The smallest extension is deliberately asymmetric: **support two audited architectures, not every architecture that llama.cpp can load.**

## 9. GGUF tensor ownership rules

### Exact OLMoE layer inventory

For each owned layer `i`, retain:

| Tensor | Purpose | GGML dimensions for selected artifact |
|---|---|---|
| `blk.i.attn_norm.weight` | Attention RMS norm | `[2048]` |
| `blk.i.attn_q.weight` | Query projection | `[2048,2048]` |
| `blk.i.attn_k.weight` | Key projection | `[2048,2048]` |
| `blk.i.attn_v.weight` | Value projection | `[2048,2048]` |
| `blk.i.attn_output.weight` | Attention output | `[2048,2048]` |
| `blk.i.attn_q_norm.weight` | Q normalization | `[2048]` |
| `blk.i.attn_k_norm.weight` | K normalization | `[2048]` |
| `blk.i.ffn_norm.weight` | Pre-MoE RMS norm | `[2048]` |
| `blk.i.ffn_gate_inp.weight` | Router | `[2048,64]` |
| `blk.i.ffn_gate_exps.weight` | Packed expert gate projections | `[2048,1024,64]` |
| `blk.i.ffn_up_exps.weight` | Packed expert up projections | `[2048,1024,64]` |
| `blk.i.ffn_down_exps.weight` | Packed expert down projections | `[1024,2048,64]` |

There are no shared-expert tensors or expert biases in this selected artifact.

The pinned converter explicitly stacks individual experts into three-dimensional tensors. Source: [OLMoE conversion](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/conversion/olmo.py).

### Range rule

For `[first_layer,last_layer)`:

```text
include all blk.i.* where first_layer <= i < last_layer

if first_layer == 0:
    include token_embd.weight

if last_layer == model.layers:
    include output_norm.weight
    include output.weight

always:
    retain the complete GGUF metadata/tensor directory
```

**The existing `blk.<layer>.*` ownership rule already selects every OLMoE layer tensor.**

No special expert selector is needed.

### Important safeguards

- Keep the complete header, tokenizer metadata and original offsets.
- Keep original layer numbers; do not renumber a stage’s first layer to zero.
- Keep the entire expert bank tensor.
- Reject missing router/up/gate/down tensors before assignment where possible.
- Reject expert dimensions inconsistent with `expert_count`.
- Preserve the original quantized bytes.
- Reject unsupported global tensors rather than silently assuming the current head/tail rules cover future architectures.

For OLMoE, require `output.weight`; do not inherit Qwen2’s tied-head fallback simply because `owned_tensor` supports it generically.

## 10. Memory and planner changes

### 10.1 Stored and active parameters are different quantities

The stage must hold:

```text
all attention weights
all norms
router
all 64 experts of every owned layer
owned embedding/output tensors
```

Only eight experts execute per token, but **all 64 must be available locally**.

Therefore:

```text
resident expert bytes != expert bytes × 8/64
```

Applying that fraction to admission would be a serious underestimation.

### 10.2 Existing tensor-byte accounting mostly already works

`stage_model_bytes` sums byte spans of every owned tensor. Because the expert bank is one complete owned tensor, it already counts every expert.

It also counts:

- Head/tail tensors.
- Header bytes.
- Padding included in tensor spans.

That is conservative for file storage, but not an exact GPU allocation measurement. Backend alignment, repacking and buffers can differ.

No new “active parameter” field belongs in the fit calculation.

### 10.3 KV formula remains applicable

For OLMoE:

```text
head_dim = 2048 / 16 = 128
kv_heads = 16

KV bytes =
allocated_positions_per_session
× sessions
× owned_layers
× 128
× 16
× (2 bytes K + 2 bytes V)
```

Examples:

| Configuration | Logical full-model KV |
|---|---:|
| 512 positions, one session | 64 MiB |
| 512 positions, two sessions | 128 MiB |
| 4096 positions, one session | 512 MiB |

These exclude allocator overhead.

**Existing issue to account for:** llama.cpp rounds context allocation. In this pin, the context code pads total context and per-sequence context to 256-position boundaries. DAN’s planner currently multiplies the requested context directly. The estimator should mirror that allocation rule, or conservatively round upward with checked arithmetic. Source: [pinned context allocation](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/src/llama-context.cpp).

### 10.4 Temporary expert workspace

Workspace is influenced by:

- Prompt/verification row count.
- Expert intermediate width.
- Selected experts per row.
- Backend grouping/sorting strategy.
- Quantized matrix multiplication buffers.
- Attention implementation.
- Logits/output buffers.
- Graph allocation/reuse.

Expert intermediates can scale with `rows × selected_experts × intermediate_width`, but that is **not a complete allocator formula**.

Do not introduce an unmeasured universal multiplier and call it exact.

### 10.5 Conservative first admission policy

Use:

```text
selected tensor bytes
+ padded ordinary KV
+ existing general reserve
+ any additional backend margin demonstrated necessary by measurement
```

Initial restrictions should be explicit:

- One pinned quantization.
- Context 512.
- One session, then two after validation.
- No draft initially.
- Validated CPU and GPU backend configurations.
- No expert-specific CPU offload.
- No claim of arbitrary context/batch/backend fit.

Retain the current `max(1 GiB,15%)` reserve as a starting allowance, **not proof that MoE peaks fit**. Measure loading, first decode, longest permitted prefill and multi-session peaks before widening support.

### 10.6 Draft memory

Current `stage_with_draft_fits` already adds:

```text
head-stage weights + head KV
+ complete draft weights + draft KV
```

That structure is reusable. It needs the same padded-KV and measured-workspace qualifications as target admission.

No draft should be selected automatically for OLMoE during the first phases.

### 10.7 Planner search

`plan_stages` already:

- Searches contiguous ranges.
- Calls actual range-fit checks.
- Tries alternate cuts through backtracking.
- Limits stages to nonempty layer intervals.

Its proportional cut is an ordering heuristic, not the final memory test.

For OLMoE’s repeated layer structure, that is acceptable initially. Quantization may still make layers differ in bytes, but `stage_model_bytes` accounts for those differences.

No special co-residency constraints are required. One complete layer remains indivisible; if no provider can hold it, the plan must fail.

### 10.8 Performance heuristic caveat

`estimate_token_ms` prices all resident decode weights using a scalar speed. `update_speed` persists that scalar across workloads.

MoE makes that calibration model-dependent:

- Only selected experts execute.
- Expert selection varies.
- Batch shape affects execution.
- Dense and MoE measurements are not interchangeable.

For first self-forming MoE replicas, either bypass the speed-based “another owner is faster” deferral for this architecture or keep its calibration explicitly model-scoped. Do not redesign placement or introduce expert-activity scheduling.

A dedicated OLMoE test catalog also avoids confusing “largest GGUF wins” with a meaningful quality ranking across dense and MoE models.

## 11. KV, sessions and rollback

OLMoE retains ordinary causal attention KV semantics.

| Operation | Existing DAN mechanism | MoE-specific change |
|---|---|---|
| Create | Allocate a sequence ID in the shared context | None |
| Decode | Write owned-layer KV for that sequence and position | None |
| Multiple sessions | Separate sequence IDs, frame-by-frame execution under worker synchronization | None |
| Explicit rollback | `llama_memory_seq_rm(sequence, position, -1)` | None |
| Implicit speculative rollback | Lower-position input truncates future KV | None |
| Reset | Remove all sequence KV, reset sampler/position | None |
| Destroy | Remove KV and release sequence slot | None |
| Disconnect | Clear resident sessions | None |

The MoE router has no persistent inference state. Its expert indices and intermediate activations belong to the current forward computation.

No router history, expert-cache state or load-balancing state needs rollback.

The existing llama.cpp stage KV filter is in the ordinary-attention memory branch used by this architecture. It must nevertheless be tested with an OLMoE middle stage; support should not be inferred solely from the filter’s presence.

## 12. Speculation implications

### Initial policy

**Disable speculation until plain single-worker and split OLMoE inference are validated.**

### Can the current small Qwen draft be reused?

**Not as a compatible draft for this target.** OLMoE uses a different vocabulary/tokenizer from the currently cataloged Qwen models.

Current `vocabularies_match` checks the full token table, attributes, special-token behavior, tokenizer metadata and a tokenization corpus. It should reject this pairing.

The corpus check does not mathematically prove equality of every possible tokenizer merge behavior. A future approved draft pair should have a known matching tokenizer artifact as well as passing runtime checks. Source: [vocab_compat.hpp](../engine/include/provider_owned/vocab_compat.hpp).

### What remains reusable?

Once a genuinely compatible draft is available:

```text
head drafts candidate IDs
→ head verifies current token + candidates
→ middle forwards hidden rows and candidate suffix
→ tail greedily checks acceptance
→ next input position truncates rejected KV
```

No MoE-specific message is needed.

### Does batching speculative positions change routing semantics?

Mathematically, each position’s router depends on that position’s hidden state. There is no cross-token expert-capacity/drop policy in this OLMoE graph.

Thus verification batching is semantically valid.

Numerically, batching can select different kernels or accumulation orders. A router near-tie can then choose a different expert set. That is a reason to test carefully—not to redesign speculation.

### How to validate without finding a useful draft first

Use test-only deterministic proposals or a compatible but non-beneficial draft setup to force:

- All accepted.
- First candidate rejected.
- Partial acceptance.
- Repeated rollback.
- Context-boundary shortening.
- Final-token commit.
- Two-session isolation.

Do not mistake such a correctness harness for a throughput optimization.

## 13. Wire-protocol impact and numerical behavior

### Explicit answer

**Yes: the current A → B → C activation contract can remain unchanged for OLMoE.**

At a boundary after a complete transformer layer, the required inter-stage value is:

```text
one residual hidden vector per token position
```

The router, expert selection, expert outputs and combine all finish inside that layer.

The next layer does not need:

- Previous expert IDs.
- Router probabilities.
- Individual expert outputs.
- Shared-expert state.
- Token IDs for routing.
- Additional recurrent state.

Existing position and sequence metadata still drive stage-local attention.

### Example payload sizes

For hidden width 2048, excluding existing frame/compute overhead:

| Wire format | One row | 512 rows |
|---|---:|---:|
| FP32 | 8192 bytes | 4 MiB |
| FP16 | 4096 bytes | 2 MiB |
| FP8 with per-row FP32 scale | 2052 bytes | 1,050,624 bytes |

Existing speculative candidate-ID suffixes remain unchanged.

### Actual selected weight path

The inspected artifact is mixed precision, not uniformly four-bit.

For its first layer:

- Router and norms: FP32.
- Expert up/gate: Q4_K.
- Expert down: Q6_K.
- Attention projections include Q4_K and Q6_K.
- Embedding: Q4_K.
- Output projection: Q6_K.

Other layers must be read from the tensor directory rather than assuming the same recipe everywhere.

### Boundary and state dtypes

| Component | Expected path |
|---|---|
| Weights | Stored GGUF mixed quantization/types |
| Internal computation | GGML/backend-selected mixed kernels |
| Router scores/top-k graph | Floating-point graph values; exact arithmetic depends on selected backend |
| Expert indexed-matmul graph output | FP32 in the inspected CUDA path |
| KV | FP16 K and V by current context defaults |
| Stage output exposed to DAN | FP32 |
| FP32 wire | FP32 bytes |
| FP16 wire | FP32 → FP16 → next-stage FP32 |
| FP8 wire | FP32 → scaled E4M3 → next-stage FP32 |
| Tail logits exposed to sampler | FP32 |

Do not describe all internal compute or accumulation as FP32 merely because the boundary is FP32.

### Numerical expectations

| Path | Expected behavior |
|---|---|
| FP32 wire, same hardware/backend/batching | No additional boundary quantization; strongest parity target, but still must be measured |
| FP32 wire across heterogeneous backends | Can drift |
| FP16 wire | Lossy boundary; can change router selection and later tokens |
| FP8 wire | Larger perturbation risk; can change expert selection and speculation acceptance |
| Plain versus speculative batching | Mathematically compatible, not guaranteed bit-identical |
| Same quantized checkpoint versus original BF16 model | Not expected to be identical |

There is a concrete tie-handling concern in this pin: the CUDA fused top-k implementation contains lower-expert-ID tie-breaking, whereas the CPU argsort comparator compares values without an explicit index tie-break. Do not promise universal cross-backend expert identity on exact ties. Sources: [CUDA top-k](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/ggml/src/ggml-cuda/topk-moe.cu), [CPU operations](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/ggml/src/ggml-cpu/ops.cpp).

This is numerical sensitivity, not necessarily random nondeterminism on repeated identical runs.

## 14. Minimal compatibility abstraction

Do **not** introduce a virtual runtime interface.

The minimum useful change is a renamed/shared validation function, for example:

```text
compatible_stage_model(index, reason)
```

with two explicit branches:

```text
qwen2 → existing supported dense contract
olmoe → audited OLMoE contract
other → reject
```

Keep `ModelIndex` as a data structure.

Add only metadata actually needed to validate or estimate the two supported paths:

- Expert count.
- Selected expert count.
- Expert FFN width.
- Tensor dimensions.
- Tensor type if used for validation/diagnostics.
- Explicit handling/rejection of attention geometry overrides incompatible with the current KV formula.

No ownership strategy class is needed because both supported models use the same `blk.N.*` selection.

No rollback interface is needed because both use ordinary KV truncation.

No stage-constraint solver is needed because OLMoE has no cross-layer constraint.

If new fields are serialized, bump the local model-index cache version and regenerate incompatible cached indexes. This is a **disk-cache schema change**, not a network protocol change.

## 15. Test plan

None of the following MoE validation was executed during this analysis.

### 15.1 Unit tests

Extend existing test targets rather than creating a separate testing framework.

| Test | Required assertion |
|---|---|
| Architecture parsing | `qwen2` and `olmoe` accepted; unrelated architectures rejected |
| Metadata order | Architecture-specific keys work before or after `general.architecture` |
| Expert metadata | Zero experts, zero selected experts, selected > total rejected |
| Tensor ownership | Every expert/router/norm tensor of owned layers included |
| Exclusion | No tensors from unowned layers selected |
| Shape consistency | Packed expert dimension matches metadata |
| Output ownership | OLMoE tail requires explicit output projection |
| Tied Qwen regression | Tail embedding fallback remains correct |
| Memory | All expert bytes counted, never multiplied by active fraction |
| KV | Multiple sessions, context padding and overflow checked |
| Planning | Uneven layer sizes, no-fitting-layer case, two/three-stage coverage |
| Cache | New fields round-trip; old incompatible index rejected/regenerated |
| ABI | Mixed patch ABIs refused |
| Negative routing support | `qwen2moe`, V4 and other unimplemented stage architectures remain rejected |

Use synthetic GGUF fixtures containing small tensors with the real naming/dimension conventions. Tiny fixtures validate selection and metadata—not real expert computation.

### 15.2 Single-worker real-model baseline

First run the selected artifact through the pinned ordinary llama.cpp path, without stage parameters.

Then run DAN with `[0,16)`.

Match:

- Exact artifact.
- Prompt token IDs.
- BOS/EOS handling.
- Greedy sampling.
- Context.
- Backend.
- Batch/ubatch settings.

Compare token IDs and, where instrumented, logits and residuals. Text equality alone is insufficient to diagnose numerical differences.

A difference should be localized, not automatically excused as “MoE drift.”

### 15.3 Two-stage split

Use a split such as:

```text
[0,8) → [8,16)
```

Validate:

- Only owned weights loaded.
- Head output is before final norm.
- Tail does not re-embed.
- KV exists only for owned layers.
- Token output matches the controlled reference where expected.
- Prompt lengths include multirow prefill.

### 15.4 Three-stage split

Use:

```text
[0,5) → [5,10) → [10,16)
```

This exercises a genuine middle stage.

Test multiple cut points, including one-layer stages, because a single successful split can miss boundary bugs.

### 15.5 Existing command entry points

After implementation, a pinned manifest and a full local model are available, the existing static harness can establish a baseline:

```powershell
.\scripts\Test-DAN-Client-Static.ps1 `
  -BuildDir build-client `
  -Model <full-local-olmoe.gguf> `
  -Manifest <olmoe-manifest.json> `
  -OutDir build-client/results/olmoe-single `
  -Transport hub `
  -Boundaries @(0,16)
```

Then exercise sparse range acquisition and placement:

```powershell
.\scripts\Test-DAN-Placement.ps1 `
  -BuildDir build-client `
  -Manifest <olmoe-manifest.json> `
  -OutDir build-client/results/olmoe-two `
  -BaselineDir build-client/results/olmoe-single `
  -Transport direct `
  -OfferedMib @(4096,4096) `
  -MinStages 2
```

```powershell
.\scripts\Test-DAN-Placement.ps1 `
  -BuildDir build-client `
  -Manifest <olmoe-manifest.json> `
  -OutDir build-client/results/olmoe-three `
  -BaselineDir build-client/results/olmoe-single `
  -Transport direct `
  -OfferedMib @(3072,3072,3072) `
  -MinStages 3
```

These are proposed **CPU rehearsal** commands using artificial admission capacities. They do not prove GPU fit. Replace angle-bracket placeholders with actual quoted paths before execution.

Important harness limitation: the current scripts compare rendered output strings. Add token-ID comparison and the required numerical diagnostics before treating them as sufficient MoE parity gates.

### 15.6 Sessions

Test at least:

- Two resident sessions.
- Interleaved requests.
- Reset one without affecting the other.
- Destroy/recreate a sequence slot.
- Follow-up prompt after generation/final commit.
- Context exhaustion in one session.
- Route reuse without model reload.
- Replica formation and teardown.

The existing replica harness has Qwen-oriented model/capacity assumptions; parameterize only what the OLMoE test requires.

### 15.7 GPU, speculation and WAN ordering

1. CPU single/split correctness.
2. GPU single/split correctness.
3. Loading and runtime memory peaks.
4. Two sessions.
5. Speculation with compatible proposals.
6. FP16/FP8 as separate numerical experiments.
7. Two-machine LAN/direct tests.
8. WAN direct tests.
9. WAN relay tests.

Three processes on one GPU are not three-GPU validation. Local forced relay is not physical WAN validation.

## 16. Failure modes

| Failure | Earliest useful detection |
|---|---|
| Unsupported architecture silently admitted | Model inspection and compatibility validation |
| Missing router tensor | Inspection |
| Missing expert gate/up/down tensor | Inspection |
| Expert dimension/count mismatch | Inspection; independently rechecked at load |
| Incorrect top-k or renormalization | Architecture metadata/graph review and reference tests |
| Only selected experts downloaded | Tensor ownership unit test |
| Wrong global layer numbering | Range tests and split-reference comparison |
| Final normalization applied at every stage | Boundary activation comparison |
| Middle stage emits only final prompt row | Multirow split test |
| Q/K normalization omitted | Loader inventory and reference comparison |
| OLMoE missing output treated as tied | Compatibility validation |
| Memory priced using active parameters | Planner unit test |
| Stage loads but expert execution OOMs | Worst-shape warm-up/real-model validation |
| Load-time repacking OOM | Loading peak measurement and conservative admission |
| Unowned sparse-file holes read as weights | Ownership/load tests and corrupted-range negative tests |
| Old cached index omits new metadata | Cache version validation |
| Wrong chat template | Prompt/tokenization tests |
| Draft tokenizer mismatch | Draft load compatibility check |
| Unsupported recurrent model uses KV-only rollback | Architecture rejection |
| Lossy wire changes routing/acceptance | Numerical validation |
| Dense speed calibration distorts MoE formation | Formation tests; bypass unvalidated cross-model heuristic |
| Failed warm-up still yields advertised replica | Replica readiness test |

Prefer failures before replica **READY** advertisement.

Current `stage_ready` means loading/context creation succeeded; it is not proof of every runtime shape. Replica warm-up is stronger, but an eight-token warm-up does not establish worst-case prefill memory safety.

Use the existing readiness mechanism. No new control-plane state is needed to make the validation stricter.

## 17. Implementation phases

### Phase 1 — Model detection and tensor ownership

- Add exact OLMoE metadata recognition.
- Strengthen OLMoE tensor/shape validation.
- Extend manifest and cached-index handling.
- Keep existing ownership rule.
- Add negative tests for unsupported architectures.

### Phase 2 — Range loading and memory planning

- Validate sparse extraction of complete expert tensors.
- Retain all-expert byte accounting.
- Correctly account for context padding.
- Verify head/tail costs and uneven layer sizes.
- Keep speculation disabled.

### Phase 3 — Single-worker MoE through DAN

- Extend the llama.cpp stage whitelist.
- Add OLMoE range-aware loading.
- Add direct hidden input and pre-final-norm stage output.
- Preserve upstream MoE math.
- Compare `[0,16)` against the pinned full-model reference.

### Phase 4 — Two/three-stage local correctness

- Exercise true middle stages.
- Validate FP32 boundaries.
- Confirm sparse loading and stage-local KV.
- Repeat on CPU and supported GPU configurations.
- Measure admission margins.

### Phase 5 — Sessions and persistent replicas

- Two resident sessions.
- Reset/destroy/reuse.
- Replica warm-up and cleanup.
- Avoid unvalidated dense-to-MoE speed calibration.

### Phase 6 — Speculation

- Establish a compatible proposal source.
- Validate acceptance and rollback.
- Test context-end and final commit.
- Measure actual draft use.
- Do not promise a useful small draft until one is demonstrated.

### Phase 7 — WAN validation

- Direct first, relay separately.
- Record exact artifact/backend/ABI.
- Validate failures and persistent-session behavior.

### Later

- Expert kernel performance.
- Quantization comparisons.
- Prefill/verification batching.
- Graph behavior.
- CPU expert offload.
- Expert caches only if a measured need appears.
- Distributed expert parallelism as a separate execution-model project.

## 18. Estimated complexity and risk

These are relative engineering estimates, not elapsed-time commitments.

| Phase | Complexity | Main risk | Completion evidence |
|---|---|---|---|
| Detection/ownership | Low–medium | Accepting malformed or unsupported metadata | Synthetic GGUF and rejection tests |
| Range/planner | Low–medium | Underestimating resident/peak memory | Byte-accounting tests plus real allocation measurements |
| OLMoE graph patch | Medium | Wrong boundary, output-row selection or global layer indexing | Full-model versus full-range reference |
| Two/three-stage validation | Medium | Numerics and sparse-loader interaction | Real-model token/logit/boundary comparisons |
| Sessions/replicas | Low–medium implementation; substantial validation | Cross-session KV contamination or stale state | Interleaved lifecycle tests |
| Speculation | Medium, potentially blocked on draft availability | Tokenizer compatibility and numerical acceptance differences | Forced accept/reject/rollback cases |
| WAN | Low model-code complexity; operational validation | Confusing transport failures with inference defects | Separate direct/relay receipts |

The largest correctness risk is not the router. It is **the range wrapper around the existing model graph**.

The largest capacity risk is not forgetting active parameters. It is incorrectly assuming file bytes plus a fixed reserve fully describe backend loading and execution peaks.

## 19. Exact source files/functions likely to change

| File | Function/area | Expected change |
|---|---|---|
| [patches/llama-provider-owned.patch](../patches/llama-provider-owned.patch) | `llama_model_load` gate; new OLMoE loader/graph hunks | Admit OLMoE and implement its stage boundaries |
| Pinned `src/models/olmoe.cpp` through the patch | `load_arch_hparams`, `load_arch_tensors`, `graph::graph` | Range validation, selective loading, direct input, non-tail residual output |
| [engine/range_model.cpp](../engine/range_model.cpp) | `parse_index`, `inspect_range_model`, `save_model_index`, `load_model_index` | OLMoE metadata and structural index support |
| Same file | `owned_tensor`, `required_ranges`, `stage_model_bytes` | Tests expected; ownership algorithm should remain unchanged |
| [range_model.hpp](../engine/include/provider_owned/range_model.hpp) | `ModelIndex`, `ModelTensor` | Minimal expert/shape metadata |
| [engine/manifest.cpp](../engine/manifest.cpp) | `load_manifest` | Exact architecture whitelist and error text |
| [engine/planner.cpp](../engine/planner.cpp) | `compatible_dense_qwen2` replacement; `kv_bytes` | Two-architecture validation and allocation-aware KV estimate |
| [planner.hpp](../engine/include/provider_owned/planner.hpp) | Compatibility declaration | Match renamed validator |
| [engine/stage_worker.cpp](../engine/stage_worker.cpp) | Catalog validation; possibly warm-up/admission diagnostics | Use updated validation; verify—not rewrite—existing Stage execution |
| Same file | `update_speed` | Avoid contaminating generic speed calibration with unqualified MoE measurements |
| [engine/placement.cpp](../engine/placement.cpp) | Speed-based alternative-plan comparison | Minimal guard if needed for initial MoE formation |
| [engine/dan_client.cpp](../engine/dan_client.cpp) | `read_models`, `chat_prompt`/`run_chat` | Metadata consistency; correct or explicitly unsupported chat formatting |
| [engine/coordinator.cpp](../engine/coordinator.cpp) | Existing compatibility caller | Mechanical update so older build/test path continues compiling |
| `engine/tests/range_model_test.cpp` | Metadata/cache/range tests | OLMoE fixtures and expert ownership |
| `engine/tests/formation_test.cpp` | Compatibility tests | Accept chosen model; retain other rejection cases |
| `engine/tests/placement_test.cpp` | Fit and split tests | Full expert costs, KV padding, nonuniform sizes |
| `scripts/Test-DAN-*` | Existing model/capacity/report assumptions | Smallest harness parameterization and token-ID comparison |
| New model manifest | Pinned OLMoE artifact | Explicit catalog entry, initially context 512 |
| Packaging/docs | Catalog inclusion and support statement | Only after validation gates pass |

`CMakeLists.txt` already derives the runtime ABI from the patch; no ABI mechanism redesign is required.

The sidecar, lease protocol, activation serializer and ring routing should not need model-specific edits.

## 20. Smallest implementation path

**Implement OLMoE as a second explicitly supported llama.cpp stage architecture, not as the start of a general runtime framework.**

The design boundary should remain:

```text
DAN:
    contiguous layer ownership
    complete expert-weight residency
    range files
    placement
    networking
    sessions

llama.cpp:
    attention
    router
    top-k
    expert kernels
    combine
    ordinary KV
```

This recommendation follows the minimal-design skill: reuse the existing execution contract, change only the architecture gates and range wrapper, and require measurements before introducing performance machinery.

### Explicitly deferred: distributed expert parallelism

Remote experts would require a different contract: token dispatch, expert ownership/discovery, fan-out/fan-in, backpressure, partial failures and additional synchronization.

None is necessary here.

To avoid creating unnecessary future obstacles, retain accurate architecture/expert metadata and document that a layer is the current placement unit. Do not add remote-expert protocol fields “for later.”

### Final implementation sequence

```text
Phase 1
OLMoE detection + tensor/ownership validation

Phase 2
range loading + conservative memory planning

Phase 3
single-worker OLMoE through DAN's patched runtime

Phase 4
2/3-stage local distributed correctness

Phase 5
persistent sessions + replica lifecycle

Phase 6
compatible speculation + rollback validation

Phase 7
WAN direct and relay validation

Later
performance tuning
distributed expert parallelism
```

**Bottom line:** OLMoE requires a new audited model graph wrapper, not a new network architecture. The existing `blk.N.*` extraction, contiguous planner, hidden-state wire contract and ordinary session/KV machinery are the foundation to keep.
