# Optimization research tracker

Last checked: 2026-10-06. Owner's target: research each area below and keep its
upstream status current, one area at a time. Prefer adopting upstream work.

Owner's clarified rule: prefer the llama.cpp implementation even when DAN has
an equivalent, unless the custom implementation has a required capability or
demonstrated advantage. Network latency is the first priority. Do not substitute
throughput, draft acceptance or smaller local KV buffers for first-token and
inter-token latency. Performance testing belongs to the owner; run bug checks
only during implementation unless asked otherwise.
Latency takes precedence over upstream reuse: reject response-latency regressions.
If research and inspection leave an optimization's latency tradeoff unclear, ask
the owner before adopting it. The inventory below describes availability and
integration work, not proof of a speed advantage or authorization for every switch.

2026-10-06 clarification: latency means network transfer latency. KV caches are
local GPU memory and do not reduce DAN activation traffic. Prioritize existing
FP8 versus F16 activation wire formats; do not move to item 2 yet. For hidden
3584, one row is 7168 bytes F16 versus 3588 bytes FP8 (including its 4-byte scale),
before protocol headers. Future owner testing should distinguish packing/unpacking,
transfer time, RTT and output effects on a split route. Single-worker chat cannot
prove WAN gains.

Local cache experiment completed: F16/512 short response ~2.06 s, F16/1024
~1.96 s; 2000-word first text ~1.30 -> 1.22 s. Local config keeps F16/1024.
Q8/512 saves 420 MiB KV but is slower at long-context generation; it remains
opt-in. Near-full-context speed gains from 1024 were inconsistent. Backend
placement/formation/client checks pass; arithmetic and native tools passed on Q8.

For each investigation record: source revision/date, merged/released status,
supported hardware/models, DAN integration gap, measured results, and decision.
Recheck sources before implementing; a feature on upstream master may be absent
from DAN's pinned runtime. This document is a manual tracker, not a background monitor.

## 1. KV cache and runtime tuning — existing upstream controls

Upstream supports separate K/V cache types, including F16, Q8_0 and Q4_0,
Flash Attention, and physical microbatch controls.
[Server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

DAN already uses llama.cpp's kernels, Flash Attention and CUDA graphs. Worker
settings expose symmetric F16/Q8_0 K/V and `prefill_batch` physical microbatches
of 256/512/1024. Local config selects F16/1024. Current 7B/16K cache is 896 MiB
F16 versus 476 MiB Q8_0. KV metrics reflect the selected type; placement retains
a conservative F16 estimate. These are upstream implementations with DAN settings,
not separate DAN cache/attention algorithms. KV changes do not shrink network frames.

Watch: an upstream report describes CPU fallback for certain quantized cache
combinations with Flash Attention. It is a reported issue, not evidence that
Q8_0 on this RTX 2070 is affected.
[Issue #28455](https://github.com/ggml-org/llama.cpp/issues/28455).

## 2. API compatibility and sampling

[Upstream server API](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
documents sampling parameters, stop strings, usage/timing information and tools.
Use it as an implementation reference; verify external API contract details
against the relevant official specification during implementation.

DAN: SSE, history, auto/none/required tools, bounded queue and cancellation implemented.
The local API now sends sampling controls to upstream common samplers on the tail;
JSON schemas/tool grammars and model templates also use upstream common. Nonstream
usage includes prompt/completion/cached tokens. Seeded constrained output and cache
reuse passed real CPU/GPU whole/split checks. Stop strings and streaming usage remain
unsupported; the output tool adapter remains Qwen2-specific.

## 3. Speculative decoding

[Upstream speculative guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
documents draft-model, EAGLE-3, DFlash, DSpark and n-gram methods. Hidden-state
drafts are target-specific; these are not interchangeable draft checkpoints.

DAN: standalone draft speculation exists on single and split routes, opt-in.
Current local API/provider do not enable it. The 0.5B model is already cached.
2026-10-06: adaptive linear width is implemented; the first stage uses each
session's complete round cost per committed token, including network transit.
It can return to ordinary decode and periodically probe again. Worker/provider
settings are `adaptive_draft=true` and `draft_width` as the cap (default 4,
maximum 32); disabling adaptation retains a fixed width. Deferred draft input
history stays in session RAM, with catch-up before probing or lifecycle controls.
See [WAN_OPTIMIZATION_REVIEW.md](WAN_OPTIMIZATION_REVIEW.md) for the algorithm,
replay costs, correctness evidence and the existing batching drift limitation.
The owner requested bug tests only and will evaluate latency across two PCs;
no further performance benchmarking is run. The local 7B-only catalog still
does not enable a draft. Do not extrapolate earlier 14B/remote results to it.

N-gram drafting needs no extra model weights. All named implementations above,
including MTP and the n-gram variants, already exist at DAN's exact pinned revision.
Ordinary draft and simple n-gram proposals now call their upstream implementations;
the other methods still need integration and compatible models. Session history,
verification and rollback remain integrated with DAN's ring.
Keep the WAN elapsed-cost controller around that code: upstream proposal policies
are not a replacement for measuring a distributed round. Trained hidden-state
drafts still need compatible models and feature transport between owned stages.

## 4. RAM-only prefix reuse

Upstream cache_prompt reuses the common prompt prefix so only a changed suffix
is evaluated. Its documentation notes possible numerical differences with
different batching.
[Cache reference](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

DAN: the local API now retains one RAM-only client/session for five idle minutes,
scoped by user/cache key/first user text. Exact prefix comparison and upstream KV
trimming run on every stage; only the suffix is processed. Scope changes, errors,
cancellation and shutdown release the cache. Distinct `prompt_cache_key` values
provide explicit chat isolation. Correctness checks cover reuse and expiry; no
latency benchmark was run. See PROJECT.md's latest integration note for limitations.

## 5. Upstream runtime upgrades

[Release v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0),
published 2026-10-05, is listed latest at this check. It adds `llama_batch_ext` /
`llama_process`, models, backend improvements and speculative correctness fixes.
DAN build script pins 95ef7fc16054e63b427a3ef00188e055ef7586d8 and applies its
provider-owned stage patch. A release DLL swap is not a valid upgrade strategy.
The existing n-gram/EAGLE-3/DFlash/DSpark/MTP algorithms do not themselves require
this upgrade. New upstream helper files may depend on the new API; start with the
pinned helpers or deliberately port the patch and relevant fixes together.

Investigate in an isolated build: patch/API diff, supported RTX 2070 CUDA kernels,
ABI identity, whole-model versus split reference parity, sessions and tools,
then leave real two-machine performance evaluation to the owner. Keep the working
runtime until the necessary correctness checks pass.

## 6. Pinned llama.cpp reuse inventory

Source checked locally: `D:/Desktop/DAN/build/provider-owned-v0/llama.cpp`, exact
HEAD `95ef7fc16054e63b427a3ef00188e055ef7586d8`, dated 2026-09-03. The inspected
speculative and n-gram implementation files are unmodified at that revision.
DAN links `llama` and now `llama-common` (`LLAMA_BUILD_COMMON=ON`). Only the
explicitly integrated helpers are active; llama-server flags do not apply to DAN.

| Area from the optimization list | Classification | DAN state / upstream reuse decision |
|---|---|---|
| CUDA kernels, quantized weights, Flash Attention, CUDA graphs | already active | Already upstream; active on the local RTX 2070. No competing DAN algorithm to replace. |
| KV precision, context, compute microbatches | already active | Upstream context controls. F16/1024 selected locally; Q8_0 opt-in. Other types need settings/admission work. KV precision does not reduce WAN frames. |
| Bounded final-stage output reservation | already active (updated worker) | Reuses pinned `llama_context_params.n_outputs_max`, sized for at most 32 verified outputs and the session-pool minimum. Intermediate stages still expose every hidden row. No new inference algorithm. |
| Ordinary draft-model proposals | already active, opt-in | Uses upstream `common/speculative.cpp`; DAN retains distributed verification/rollback. Prepared API chats fall back to plain decode. |
| Adaptive speculative width | already active | Session-local complete-round cost adaptation, opt-in with drafting. Preserve WAN behavior around upstream proposals; no measured superiority over upstream is claimed. |
| Model-free n-gram drafting | already active simple helper, opt-in | `ngram_draft=true` uses pinned `common_ngram_simple_draft` with session-local history. Other n-gram methods remain integration work. No blanket latency gain claimed. |
| EAGLE-3 | model-dependent | Pinned helper and layer-input capture present. Compatible trained draft, selected target features and stage routing required. Not a switch for the current 7B model. |
| DFlash / DSpark | model-dependent | Pinned parallel draft helpers present. Compatible target-specific checkpoints and hidden-feature transport required; extra WAN traffic remains an evaluation cost. |
| MTP | model-dependent | Pinned `load_mtp`/`draft-mtp` present. Needs trained prediction heads plus ownership/hidden-state integration, not an ordinary draft checkpoint. |
| GPU sampling | deferred / not worth changing | Pinned `llama_set_sampler` can skip raw-logit readback, but also allocates vocabulary-sized backend buffers and defaults to one output per sequence. Needs careful speculative/lifecycle integration; latency/memory tradeoff is unresolved. Do not enable merely for upstream reuse. |
| RAM-only prefix reuse | already active | One scoped idle-expiring API client; exact token prefix and upstream KV trimming across stages. No disk prompt/cache files. |
| Prefill chunks / pipeline | already active | Upstream batched compute plus DAN chunk propagation. Provider exposes `prefill_chunk`, locally 512. Only split routes use the pipeline. |
| Pipelined drafting / tree verification | deferred / not worth changing | Core batching/KV primitives reusable. Asynchronous WAN scheduling, epochs, cancellation and rejected-branch cleanup require substantial integration. No measured justification for a redesign. |
| Continuous batching | already active, opt-in | Owner approved default-off ready-frame ring decode batching through upstream `llama_decode`. Not the full llama-server scheduler; serialized local API and whole-model loops remain unbatched. |
| Sampling, grammar, constrained JSON | already active | Local API carries per-request settings to upstream common samplers/schema grammars. Grammar compute applies only to constrained requests. |
| Native tools and token streaming | already active | Gateway handles tool framing and SSE. Additional upstream chat/parser reuse must fit the Go gateway and current API behavior; no wholesale server replacement. |
| Activation transport / F16 / FP8 compression | already active | Core F16 conversion reused; DAN ring framing and FP8 row encoding remain required. F16/FP8 opt-in; F32 default. Preserve wire/quality behavior. |
| Placement, direct return, NAT, relay, discovery, persistent routes, weight cache | already active | Preserve DAN's decentralized integration. Upstream host-controlled RPC and local device splitting do not supply the same worker-owned execution or route lifecycle. |
| More architectures | model-dependent | Reuse upstream model graphs/backends with compatible provider-owned stage support. Qwen2 and OLMoE work; other upstream models are not automatically split-compatible. |
| Additional hardware backends | integration needed | Reuse upstream Vulkan/Metal/SYCL etc. Requires hardware-specific builds and stage correctness checks; no reason to change the working RTX 2070 CUDA backend. |
| Whole upstream runtime upgrade | deferred / not worth changing | Existing algorithms are already pinned. Upgrade/port only for a needed fix/API/model/backend benefit; no DLL swap or upgrade solely to collect features. |

Primary references: [speculative methods](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md),
[common implementation](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp),
[core API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h),
[server controls and cache](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[RPC architecture](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md).
Current master documentation can describe fixes/options newer than the pinned code;
the presence claims in the table come from the local pin, not master documentation.

Upstream `ngram-mod` shares its hash pool across server slots, while `ngram-cache`
can load/save statistics. For DAN's data policy, isolate dynamic state per session
in RAM and omit persistent statistics. Reuse does not require inheriting upstream
server defaults for cross-conversation or disk caches.

## 7. Implementation order under the latency priority

1. Integrate upstream `ngram-simple` into the current verifier: no draft weights,
   minimal additional compute, and possible multiple committed tokens per WAN round
   on repeated text. Fall back through the existing elapsed-cost controller.
2. Reuse upstream draft proposal/confidence and supported backend sampling logic;
   delete equivalent custom code only after distributed correctness is preserved.
3. Reduce repeated-turn prefill with bounded RAM-only prefix reuse when its session
   lifetime is in scope. This avoids recalculating/transmitting unchanged history.
4. Integrate compatible trained drafts if model/hardware support is chosen; use
   upstream implementations and add only the necessary distributed feature path.
5. Continue WAN overlap/topology work where upstream lacks the required ring behavior.

This is a candidate integration order, not a measured latency ranking. The initial
2026-10-06 reuse audit changed documentation only. The subsequent implementation
review selected bounded final-stage output reservation for a concrete allocation
benefit; other candidates remain inactive pending their benefit/scope decision.

## 8. Small implementation selected after pipeline review

The final stage requests one output for ordinary prefill/decode and at most 32 for
speculative verification. It previously left llama.cpp's `n_outputs_max` at zero,
which defaults to the logical prefill batch size. Upstream reserves a worst-case
graph with vocabulary outputs for each physical prefill row even though DAN does
not need those prompt logits.

The worker now sets the upstream output bound to the larger of the session-pool
minimum and `min(context, 32)` on final stages only. The first/middle stages retain
all hidden-state outputs. The existing width limit is shared with the controller
and CLI; oversized speculative token/activation frames return a protocol error
before decode instead of reaching an upstream output-buffer assertion. Ordinary
long prompts, prompt chunks, commits and distributed routing are unchanged.

Regression check: `engine/tests/output_rows_integration.py HOST:PORT [--hidden N]`
against a static whole-model or tail worker (`--ctx 128 --max-sessions 2`). It checks
long prompt handling, two sequences, reset/reuse, width 32, and rejection/recovery
at width 33 without collecting timings or saving prompt/response content.

Validation completed 2026-10-06:
- Worker rebuilt; protocol, client, formation and placement unit checks all pass.
- Real cached Qwen2.5-0.5B checks pass on RTX 2070 whole-model and tail-only stages,
  and on CPU: long prompt with one sampled output, two sessions, width 32,
  oversized-width rejection, reset/reuse and clean shutdown. An unbound malformed
  control connection closes and clears sessions by the existing contract; recovery
  is checked through a new connection.
- Same-model/same-context allocation check (`ctx=128`, two sessions): CUDA compute
  buffer drops from **74.62 MiB to 18.66 MiB** (55.96 MiB, about 75%). Shutdown
  logs confirm it remains at the reserved size after width-32 verification. This
  is a measured allocation benefit, not a throughput or WAN-latency benchmark.
- Three CPU stages (0..10, 10..18, 18..24) produce the same token IDs as the old
  whole-model worker for two short prompts and a 48-word prefill, with reset/reuse.
  Cached weights were used directly: the catalog-based placement rehearsal could
  not reach Hugging Face in this environment. No public-network/WAN claim follows.
- The tested worker was installed in `build/ui-launch/gpu-runtime`, retaining its
  matching CUDA DLLs; the existing provider was restarted with the same configuration.
  Its Qwen2.5-7B / 16K / F16 / 1024-microbatch CUDA compute allocation drops from
  **608.00 MiB to 296.02 MiB** (311.98 MiB). All 29 layers stay offloaded, Flash
  Attention resolves enabled, and the persistent replica reforms and warms up.
  A single local API correctness request returns `4` for `2 + 2`; no timings are
  collected. The installed executable hash matches `build-runtime/dan-stage-worker.exe`.
- Distributed stage transport/placement, model selection, wire precision, retained
  prompt lifetime and gateway/API settings are preserved. No UI verification or
  performance benchmark was run. This establishes memory savings and correctness
  for the checked cases; WAN latency and other supported models are not measured.

## Baseline and decisions

- Local persistent replica enabled, 2026-10-05. Repeated short-request first
  text: about 2.14 s -> 0.067 s; 2000 background words: 3.56 s -> 1.33 s.
- Reproducible backend probe: scripts/Measure-DAN-API.ps1. Output has timing and
  content-event counts only; events are not guaranteed token counts.
- Adaptive speculation and provider prefill chunking are implemented with local
  correctness checks; other experiments remain researched candidates.
- Current scope stays on network latency; two-PC performance testing belongs
  to the owner. API item 2 has not started.
