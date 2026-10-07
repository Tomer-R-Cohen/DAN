# Response: frontier-WAN feasibility

Date: 2026-09-20. Replies to `docs/AGENT_HANDOFF_FRONTIER_FEASIBILITY.md`. Analysis only: no
engine/network code changed, nothing committed, no weights downloaded, no nodes stopped, no
benchmarks started on live infrastructure for this assessment. The measurements quoted as
"measured today" were taken earlier in this session under the owner's direction, before this
handoff was read.

Worktree preserved: `CMakeLists.txt`, `docs/PROJECT.md`, `engine/tests/sessions_client.cpp`.

## 0. Status update (appended 2026-09-20, after implementation)

**The verdict below stands. Several of its numbers and open items do not** — the owner acted on
the §4 recommendation and had Qwen3-MoE implemented in the same session, which closed Gate 1,
partly closed Gate 3, and replaced estimates with measurements. Read this section first; the rest
is unchanged from the original reply.

| Item below | Superseded by |
|---|---|
| §4: "≈1.36 GiB/layer, ≈133 GiB" (estimate) | **1.499 GiB/layer, 141.70 GiB**, measured from the real artifact's per-layer byte spans |
| §4 table: 8×24 GB "fits (~149 GiB usable)" | Confirmed by DAN's own planner: **8 stages of 11–12 layers**, 18.3 GiB on the head |
| §4 table: 6×32, 4×48 "fits" | Confirmed: **6 stages** (15–16 layers) and **4 stages** (23–24 layers) |
| §4 table: 8×16 GB "short unless Q3/Q2" | Confirmed: **no plan** (98 GiB usable vs 141.7 GiB) |
| §6 Gate 1 "not yet run, ~30 min" | **Done.** `provider_owned_plan_check` runs the real `plan_stages` against an inventory |
| §6 Gate 3 "~1 day port" | **Implemented and unit-tested**; real-weights token-id comparison still outstanding |
| §7 Q6 "the pod is still running" | **Gone** (connection refused). Gate 2 needs new hardware |

### New findings that post-date the original reply

1. **The attention head width is declared, not derived.** Qwen3-30B-A3B and Qwen3-235B-A22B both
   set `attention.key_length = 128` while `hidden / heads` is 64. DAN's planner KV estimate and
   the worker's reported `kv_bytes_per_session` both used the quotient, so both **understated
   Qwen3 KV by half**. Fixed; Qwen2/OLMoE are unaffected because their declared widths equal the
   quotient. Nothing in either earlier report anticipated this.
2. **The planner hung on deep models.** A 94-layer model against 8×24 GB did not finish in
   **90 seconds**: `plan_stages` proved each too-small stage count impossible by searching every
   ordering and cut point. It now skips a count whose roomiest workers cannot hold the model's
   bytes plus KV — a necessary condition, so no reachable plan is lost. Same answers, ~100 ms.
   This is a second hard placement limit alongside §2.2's eight-candidate cap.
3. **The 235B artifact is a split GGUF and DAN cannot load one.** The official Q4_K_M is five
   parts; `RangeModelRequest` is one URL, one hash, one sparse file. Pointed at part 1 the worker
   correctly refuses it ("layer 18 is missing ffn_up_exps.weight"), so this fails closed — but the
   **user-facing model needs multi-file GGUF support or a single-file artifact**. That is download
   and indexing work, separate from inference, and it was not on any roadmap here.
4. **Editing the llama.cpp patch did not change the advertised runtime ABI.** CMake hashed the
   patch at configure time without treating it as an input, so a rebuilt worker kept the old ABI
   string and could still form a route with a mismatched peer. Fixed with
   `CMAKE_CONFIGURE_DEPENDS`.
5. **Qwen3-MoE keeps llama.cpp's tied-output fallback**, unlike OLMoE which requires an explicit
   `output.weight`. Validation allows both, per architecture.

### What is still true and still blocking

- §2.2's eight-candidate cap, §5's latency projections and §7's six questions are unchanged. The
  projections remain **projections from one measured two-node point**; Gate 2 was never run.
- No Qwen3 inference has been executed. The implementation is unit-tested and the real artifact's
  header validates, but no reference-versus-split token-id comparison exists, because no available
  hardware holds the 30B control model (pod gone; local is an 8 GB GPU and 16 GB RAM against an
  18.56 GB Q4_K_M / 11.26 GB Q2_K artifact).

Implementation detail is in `docs/PROJECT.md` §9.10.

## 1. Verdict

Three different answers, because the handoff's three scenarios are three different products:

| Target | Verdict |
|---|---|
| DeepSeek-V4.1-Flash on ordinary contributors | **No-go**, and not primarily for latency reasons |
| A frontier-adjacent MoE (Qwen3-235B-A22B) on **few large** contributors (≥24 GB × 8) | **Constrained experiment with a credible path to beta** |
| Any frontier model on **mostly 8–12 GB** contributors | **No-go today**, and the first blocker is DAN's planner, not the network |

The handoff's pessimism about long WAN pipelines is structurally correct — my own measurement
supports the model's shape — but two of its inputs are wrong in ways that move the answer in
opposite directions, and one hard code limit that decides the question is missing from both
reports.

The honest summary for the owner: **frontier quality, ordinary residential contributors and
interactive speed are not simultaneously reachable today. Any two are.** The lever that changes
the arithmetic most is not a better scheduler — it is fewer, larger contributors.

## 2. Corrections

### 2.1 The contributor memory proxy is wrong (this one helps you)

The handoff states Steam's August 2026 **Windows-only** survey has 8 GB most common at 34.03%,
12 GB 14.57%, 16 GB 11.23%, 24 GB 1.66%.

The August 2026 survey I can retrieve reports, for dedicated video memory:

| 8 GB | 12 GB | 16 GB | 20 GB | 22 GB | 24 GB | 32 GB |
|---:|---:|---:|---:|---:|---:|---:|
| 25.74% | 12.99% | **26.92%** | 1.37% | 1.12% | 5.41% | 1.34% |

16 GB is now the most common tier, having passed 8 GB in July 2026. That gives roughly **36% of
surveyed systems at ≥16 GB and ~8% at ≥20 GB** — against the handoff's implied ~13% and ~1.7%.
The 24 GB figure differs by more than 3×.

The retrieved page states these are combined Windows/Mac/Linux figures, so a genuine Windows-only
view could differ slightly — but not enough to restore a 34% 8 GB plurality, and the July
crossover is independently reported. I believe the handoff's numbers are from an older survey.
Recheck before relying on either set.

Caveats that survive the correction: this is a voluntary gaming-hardware sample, not willing DAN
contributors; card VRAM is not free VRAM (display and desktop compositing take a real slice); and
the tier distribution says nothing about uplink, uptime or willingness.

### 2.2 The missing blocker: DAN cannot plan more than 8 stages

`engine/placement.cpp:17` sets `max_planned_candidates = 8`, applied at line 397
(`pool.resize(max_planned_candidates)`). Every route DAN plans today has **at most eight stages**.

Neither report treats this as a constraint, and it decides the "many small contributors"
question before any latency argument does:

| Contributor tier | Usable per worker after DAN's reserves | Max model weights in 8 stages |
|---|---:|---:|
| 8 GB | ~5.6 GiB | ~45 GiB |
| 12 GB | ~8.9 GiB | ~71 GiB |
| 16 GB | ~12.3 GiB | ~98 GiB |
| 24 GB | ~18.7 GiB | ~149 GiB |
| 32 GB | ~26 GiB | ~208 GiB |
| 48 GB | ~39 GiB | ~312 GiB |

Usable = card VRAM − the launcher's `reserve_vram_mib` (1536 default) − the planner's
`max(1 GiB, 15%)` reserve, before KV and workspace. So the 25 × 8 GB thought experiment is not
merely slow — **the current planner refuses to place it at all.** Raising the cap is a small
change, but it is a change, and it would then expose the latency wall the handoff describes.

### 2.3 The matched direct/relay comparison now exists

The handoff says it was not yet available. It was measured today, same layer cut, same hardware,
same prompt, 10 repetitions each, warm-up discarded, loading excluded:

| Ring link | TTFT median (range) | decode tok/s median (range) |
|---|---|---|
| relay, 88 ms | 269.7 ms (195.3–321.3) | 9.99 (9.81–10.48) |
| direct, 77 ms | 164.4 ms (160.6–201.9) | 11.25 (10.13–12.15) |

Direct gave +12.6% decode and −39% TTFT. Part of the decode gain is the shorter measured path
(77 vs 88 ms), not relaying itself, so treat +13% as an upper bound for "relay overhead" on this
pair. The full baseline, including same-GPU splits, is in `docs/PROJECT.md` §9.9.

### 2.4 The latency model's shape is validated; only its constants are open

At N=2 with one WAN edge each way: RTT 77 ms, measured 89 ms per committed token (11.25 tok/s).
Stage compute for 16 OLMoE layers is a few ms. So per-token time ≈ ring propagation + ~12 ms of
compute, serialization and queueing. The handoff's `T_token ≈ Σ(edges) + compute` is therefore
the right form, and its instruction to count directed traversals rather than an RTT per edge is
correct.

Calibrated from that point, for N workers in a ring with mean one-way delay `d`:

```text
T_token ≈ N·d + c,   c ≈ 10–25 ms for a small MoE, larger for a frontier model
tok/s   ≈ 1000 / T_token         (plain decode, no speculation)
```

Two consequences worth stating plainly:

- At a typical residential `d ≈ 25 ms`, **10 tok/s needs the whole ring under ~100 ms**, i.e. about
  **four WAN hops**. Not twelve, not twenty-five.
- Adding contributors *always* costs latency for a single conversation. More contributors help
  capacity, concurrency and model size — never single-user speed.

### 2.5 V4-Flash-0731 was a defensible instinct but not a cheap one

The recommendation was not baseless: DAN's pinned llama.cpp does contain
`src/models/deepseek4.cpp` (V4, 1,502 lines) plus `llama-kv-cache-dsv4.cpp`, so a runtime exists,
which is more than V4.1 has in the pin. V4's layers are also independent, unlike V4.1's shared
KV/index source groups — that matters more for DAN than parameter count, because it permits
arbitrary cuts.

But it is still not a whitelist change. Per Shard's V4 stage (and the audit's own table), V4
carries four residual streams and token IDs across a boundary. DAN's wire contract assumes one
`hidden`-wide FP32 matrix (`stage_worker.cpp`, `put_hidden`/`get_hidden`, `run_middle` shape
checks). So V4 needs a new versioned boundary payload too — cheaper than V4.1, far dearer than
OLMoE was.

### 2.6 The newest Qwen flagship is the wrong shape for DAN

Qwen3.5-397B-A17B is reported as a hybrid Gated-DeltaNet plus full-attention stack. I verified
the consequence locally rather than trusting the description: the pin's `src/models/qwen35moe.cpp`
(738 lines) calls `build_inp_mem_hybrid()` and `build_rs(...)` for recurrent state.

DAN's rollback, reset and implicit speculative truncation all assume `llama_memory_seq_rm` over
ordinary attention KV. Recurrent state is a different branch with different semantics — the same
class of problem the MoE design doc flagged for Jamba/Qwen3Next/Kimi. Newer and better on
benchmarks does not mean closer for DAN.

### 2.7 What the same-GPU split numbers do and don't say

The handoff is right that they are not scaling measurements. Quantified: on one A4500 with **no
network at all**, 1→2 stages costs 18.2% of decode rate and 1→3 costs 26.2% (TTFT flat at ~50 ms).
That is process hops, per-frame serialization and three processes contending for one GPU, and
these runs do not separate those three causes. Do not reuse those percentages as a per-hop
penalty for distinct machines; on distinct GPUs the compute overlaps differently.

## 3. Current DAN capabilities that matter here

Verified in source this session, not inferred from docs:

| Capability | State | Relevance |
|---|---|---|
| Contiguous layer ownership, whole expert bank per owned layer | Working (OLMoE) | The MoE ownership question is already answered |
| Second architecture added via a 5-hunk patch + validator branch | Working | Sets the real cost of adding another *ordinary* MoE |
| Loop mode (client out of the token loop) | Default | Already removes one WAN round trip per token |
| Speculation | Qwen2 only, by design | 44–49% acceptance, ~2.3 tokens/round measured on Qwen |
| Two sessions interleaved, reset, destroy/recreate | Verified on OLMoE, both GPUs | Concurrency, not single-user speed |
| Persistent replicas, cache-aware and latency-aware placement | Working | Removes setup cost, not per-token cost |
| FP16/FP8 activation wire | Opt-in, lossy | Halves/quarters boundary bytes; changes routing decisions |
| Chunked prefill | Implemented in the worker, **not exposed** in `provider.conf` | Becomes a correctness requirement for wide boundaries |
| Payload cap | 64 MiB (`protocol.hpp:43`) | Caps rows per prefill frame |
| Planner candidates | **8** (`placement.cpp:17`) | Caps total stages per route |
| Async/pipelined speculation, cross-session batching, multi-GPU per worker, expert offload | Not implemented | The things that would change the latency arithmetic |

## 4. Model recommendation and contributor requirements

**Flagship: Qwen3-235B-A22B-Instruct-2507.** Reasoning, in DAN's terms rather than benchmark terms:

- The pin's `src/models/qwen3moe.cpp` is **179 lines and structurally identical to
  `olmoe.cpp` (199 lines) at exactly the five points the OLMoE patch touches**: `build_inp_embd`,
  `build_inp_out_ids`, the `for (il = 0; il < n_layer; ++il)` loop, the `il == n_layer - 1`
  output-row selection, and the final `cur = inpL; build_norm; lm_head` tail. It uses ordinary
  `build_attn_inp_kv()`.
- So the runtime work is the OLMoE patch again, plus a `compatible_stage_model` branch and a
  manifest entry. **The wire contract does not change.** No new payload, no token IDs, no shared
  KV groups, no Engram.
- 94 layers gives fine partition granularity; 128 experts × 8 active is an ordinary
  `build_moe_ffn` configuration.

Estimated fit (arithmetic, **not** measured — see §6 experiment 1): ~2.42 G expert parameters per
layer, ≈1.36 GiB/layer at Q4_K_M, ≈133 GiB for the artifact. Therefore:

| Contributor mix | Verdict under today's 8-stage cap |
|---|---|
| 8 × 24 GB | **Fits** (~149 GiB usable) — the minimum credible beta |
| 6 × 32 GB | Fits comfortably |
| 4 × 48 GB | Fits, and is the only mix that also reaches interactive speed |
| 8 × 16 GB | **Short** (~98 GiB usable vs ~133 GiB) unless Q3/Q2, or the planner cap is raised |
| Any number of 8–12 GB | Not placeable |

Controls: OLMoE-1B-7B (already validated end to end) and Qwen3-30B-A3B.

**If the owner insists on V4.1-Flash**, it is a research track: new versioned boundary payload
(4 streams + pre-mix + token IDs, ~82 KB/token/row), an indivisible 20-layer dependency group of
~134 GiB, 188 GiB of Engram tables needing a disk-residency path DAN does not have, and no
audited runtime in the pin. I agree with the V4.1 audit's conclusion and found nothing in it that
was too pessimistic.

**Smaller contributors are not useless** — they are just not useful *for a frontier replica*.
They should run smaller models and additional replicas, which is exactly what DAN's replica
mechanism already supports. Say that to contributors honestly rather than implying their 8 GB
card helps run the 235B.

## 5. Performance scenarios

Using the §2.4 calibrated model. `d` = mean one-way delay between consecutive ring members.
**These are projections from one measured two-node point, not measurements.**

| Scenario | Workers | `d` | Plain decode | With speculation at 2.3 tok/round |
|---|---:|---:|---:|---:|
| 4 × 48 GB, same region | 4 | 15 ms | ~11 tok/s | ~25 tok/s |
| 4 × 48 GB, continental | 4 | 25 ms | ~8 tok/s | ~18 tok/s |
| 6 × 32 GB, continental | 6 | 25 ms | ~5.7 tok/s | ~13 tok/s |
| 8 × 24 GB, continental | 8 | 25 ms | ~4.3 tok/s | ~10 tok/s |
| 8 × 24 GB, intercontinental | 8 | 60 ms | ~2.0 tok/s | ~4.6 tok/s |
| 12 × 16 GB (needs planner change) | 12 | 25 ms | ~3.0 tok/s | ~7 tok/s |
| 27 × 8 GB (needs planner change) | 27 | 25 ms | ~1.4 tok/s | ~3.3 tok/s |

Uncertainties I cannot close without measurement:

- `c` (compute + serialization) is taken from a 1B-active model. Qwen3-235B-A22B reads roughly
  22 B active parameters per token; at Q4 that is ~8 GB of weight traffic spread across the ring,
  perhaps 15–30 ms total on consumer GPUs. For the shortest rings that is not negligible.
- The 2.3 tokens/round figure is **Qwen2.5-14B's measured acceptance with a 0.5B draft**. There is
  no compatible draft for Qwen3-235B today, and no measured acceptance. Treat the speculation
  column as "what the target would need", not a forecast.
- Residential uplink is ignored above. A 4096-wide FP32 boundary is 16 KB/token/row — fine for
  decode, but a 2K-token prefill is ~32 MiB per edge, which at 20 Mbit/s is ~13 s of serialization
  before any compute. TTFT on long prompts, not decode rate, is where ordinary uplinks hurt.
- Nothing here models availability. One member leaving dissolves the route.

## 6. Smallest decisive experiment

Three gates, cheapest first. Each can return "stop".

**Gate 1 — placement feasibility, no weights downloaded (~30 min).**
Read only the GGUF *header* of the chosen Qwen3-235B artifact over HTTP range requests
(`inspect_range_model`, the same few MB DAN already reads for any manifest), then run the real
`plan_stages` against synthetic inventories of 8×16 GB, 8×24 GB, 6×32 GB, 4×48 GB.
*Go:* a legal plan exists for at least one mix the owner can actually recruit.
*Stop:* no mix places → the flagship choice is wrong, before any porting.
This replaces my §4 estimates with DAN's own accounting, including real per-layer bytes.

**Gate 2 — latency ladder, existing OLMoE, existing hardware (~1 h).**
On the PC + pod pair already running, measure per-token time at 2, 3 and 4 WAN boundaries
(placing extra stages on the pod's second node), forced direct and forced relay, 10 reps × 128
tokens each, and fit `T = N·d + c`.
*Go:* the fit predicts the measured points within ~15%, so §5's projections can be trusted for
planning.
*Stop:* per-hop cost is materially worse than `d` (queueing, Nagle, sidecar overhead) → fix that
before choosing any large model, because it dominates everything else.

**Gate 3 — port Qwen3-MoE and measure single-machine first (~1 day).**
The five-hunk patch, validator branch and manifest, validated exactly as OLMoE was: full-model
reference vs `[0,N)` vs two- and three-stage token-id equality, on the A4500 with a small Qwen3
MoE (30B-A3B) as the control.
*Go:* token-id identity, and single-machine decode at a rate that leaves room for the ring.
*Stop:* any correctness gap → the architecture family is not as close as the graph suggests.

Only after all three: rent the multi-GPU hardware for one timed 235B session. Do not buy or rent
anything before Gate 1 returns.

Proposed go/no-go for the beta itself, **for the owner to set, not me**: a single conversation at
≥10 committed tok/s with TTFT ≤2 s on a 512-token prompt, on a mix the owner can actually
recruit. My projection says that needs ~4 workers at ≤25 ms, or speculation working on the target
model. If the threshold is 15–20 tok/s as previously suggested, the honest answer today is that
only the 4 × 48 GB same-region mix can reach it, and only with working speculation.

## 7. Questions only the owner can answer

1. **What tok/s and TTFT make it "usable" to you?** Everything above changes with this number, and
   it has never been agreed.
2. **Which contributor tier can you actually recruit?** The whole feasibility question collapses to
   this. Eight friends with 24 GB cards is a different project from fifty strangers with 8 GB.
3. **Is Qwen3-235B-A22B "frontier enough"?** If yes, there is a credible path. If only V4.1-class
   quality counts, the answer is a research track with no beta date.
4. **May the 8-candidate planner cap be raised?** It is a small change, but it only converts an
   impossible placement into a slow one.
5. **Regional or global contributors?** `d` is the single most sensitive input in §5.
6. **Budget for the pod?** It is still running. Gate 2 needs it; Gates 1 and 3 mostly do not.

## Sources

- [Steam Hardware & Software Survey](https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam)
- [Steam Hardware Survey August 2026 summary](https://tech-insider.org/steam-hardware-survey-august-2026/)
- [16 GB GPUs pass 8 GB, July 2026](https://wccftech.com/steam-hardware-survey-july-2026-16-gb-gpus/)
- [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
- [Qwen3.5-397B-A17B release coverage](https://www.marktechpost.com/2026/02/16/alibaba-qwen-team-releases-qwen3-5-397b-moe-model-with-17b-active-parameters-and-1m-token-context-for-ai-agents/)
- DAN source, this worktree: `engine/placement.cpp:17`, `engine/include/provider_owned/protocol.hpp:43`,
  `engine/planner.cpp`, `engine/stage_worker.cpp`; pinned llama.cpp `src/models/{qwen3moe,olmoe,qwen35moe,deepseek4}.cpp`
- DAN measurements: `docs/PROJECT.md` §9.9
