# Beta model and GPU selection — missions and acceptance checklist

Date: 2026-10-07. **Implementation approved 2026-10-07.** Progress: M0 done; M1 device
binding, live memory and backend-agnostic detection done; M2 replica context check done.

This plan covers selection, formation and the prerequisites for a usable public,
heterogeneous-hardware beta. Checked items mean decisions/research completed, not
that proposed features exist. PROJECT.md remains the standing rules source.

## Decisions and boundaries

- [x] Prioritize the best supported model that meets interactive usability constraints.
- [x] Target **20 accepted output tokens/s per active chat**, delivered to its client.
  Speculative guesses, SSE events and aggregate replica throughput do not count.
- [x] If no plan qualifies, offer an explicit slower fallback; never silently relax
  the selected speed, model capabilities or context requirement.
- [x] Warm first response: **2 s target, 5 s allowed maximum** for an ordinary chat turn.
- [x] Public providers and broader hardware are day-one requirements.
- [x] Reuse llama.cpp, existing DAN code and suitable upstream algorithms first.
- [x] No implementation, UI inspection, runtime changes or benchmarks in this task.
- [x] Day-one OS/GPU matrix (owner): NVIDIA (CUDA), AMD (ROCm) and Apple Silicon
  (Metal). Not yet packaged or validated for AMD/Metal (M7).
- [x] Default context (owner, revised): **32K**; a request may ask for up to 256K and
  then gets a plan that holds it or an explicit "no plan". Context is reserved per
  session up front (7B: ~1.8 GiB at 32K, ~14 GiB at 256K), so a 256K default would
  exclude most GPUs and every current model. Later (M2): each replica serves its
  model's trained maximum, capped by its GPUs' memory.
- [x] Cold-start policy (default chosen; owner gave no answer): answer on an eligible
  ready model and offer a better cold model separately (`ColdStart::ready_first`);
  `ColdStart::wait` is the alternative mode.
- [x] Configurable cap of **five authenticated provider nodes** per replica (upper
  bound). Local GPU count is separate. PeerID does not prove distinct machines.
- [x] Ordinary turn: at most 2048 uncached input tokens. Stall: p95 gap between
  accepted tokens at most 1 s. Margin: predictions must beat each limit by 20%.

**Open issue (2026-10-07):** the Qwen2.5-7B manifest and the local API still use 16K.
On one 8 GB GPU, 7B Q4 at 32K does not fit DAN's memory rule (weights + KV + reserve),
so raising them would stop the owner's single-GPU local replica from forming.

## What the audit established

- [x] `dan_client.cpp::read_models` orders by GGUF logical tensor storage bytes,
  not an explicit quality rank. Quantization can therefore distort model preference.
- [x] `plan_stages` finds the first feasible split with the fewest stages; layer
  allocation is initially proportional to offered memory. Latency is estimated later.
- [x] Placement truncates to eight candidates although discovery can return more.
  Cache reuse can replace a split without comparing its sustained latency.
- [x] Ready replicas are tried before new placement; idle small replicas can keep
  providers reserved and prevent a larger useful combination from forming.
- [x] Launcher GPU selection advertises a chosen NVIDIA device but does not explicitly
  bind that selection through llama.cpp's device parameters. This must precede
  reliable multi-GPU planning. Current advertised memory derives from total VRAM
  minus a reserve, rather than fresh free memory at admission.
- [x] Replica owner startup does not propagate the API's requested context policy;
  a manifest's 512-token replica can coexist with a 16K API requirement.
- [x] Existing reusable pieces include `stage_fits`, padded KV accounting, model
  indexes, catalog hashes, edge probes, speed estimates, leases and fake-worker tests.

Key locations: `engine/{dan_client,planner,placement,replica_owner,provider_launcher,
stage_worker}.cpp`, `engine/include/provider_owned/{placement,formation,lease}.hpp`,
`sidecar/{discovery,replica}.go`, and the local API's `chat_process.go`.

## Research and reuse decisions

| Existing source | Reuse candidate | Decision and limits |
|---|---|---|
| [Pinned llama.cpp](https://github.com/ggml-org/llama.cpp/tree/95ef7fc16054e63b427a3ef00188e055ef7586d8) | `ggml_backend_dev_*`, model `devices`/split parameters, `common_get_device_memory_data`, performance counters | Direct C++ reuse for inventory, affinity, local splitting and memory estimates. These APIs were found in the local pin. Validate their behavior with DAN's stage patch and each backend; no blanket upstream upgrade. |
| [Shard topology code](https://github.com/leyten/shard/blob/master/shard/topology.py) | Pure group selection, ordering, layer assignment and cost helpers | Closest small implementation to adapt into DAN's C++ planner. Reuse its algorithm and focused tests with attribution. Replace its memory approximations and coordinator-path costs with DAN's existing range accounting and ring path. |
| [Petals routing](https://github.com/bigscience-workshop/petals/blob/main/src/petals/client/routing/sequence_manager.py) | Layer-progress graph with compute/link costs and shortest-path selection | Useful for routing through hosted spans. It does not by itself solve free layer assignment, device reuse constraints and participant caps. Use as a reference/alternative, not a second simultaneous production planner. |
| [Petals block selection](https://github.com/bigscience-workshop/petals/blob/main/src/petals/server/block_selection.py) | Decentralized rebalancing thresholds and coverage checks | Adapt stability rules for idle upgrades. Its throughput objective must be replaced by DAN's per-chat usability policy. |
| [exo placement](https://github.com/exo-explore/exo/blob/main/src/exo/master/placement.py) and [utilities](https://github.com/exo-explore/exo/blob/main/src/exo/master/placement_utils.py) | Topology/backend filtering and download-aware decisions | Reference for heterogeneous resources. Its current memory-oriented placement is not a ready-made solution to DAN's latency objective; importing its runtime would expand the task. |
| [distributed-llama](https://github.com/b4rtaz/distributed-llama) | Existing local-cluster implementation | Reviewed as a comparison. Its Ethernet-oriented tensor-parallel execution is not a replacement for DAN's WAN stage transport. |
| [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) | Host-controlled remote backends | Keep outside the public DAN scheduling/transport path; retain provider-owned catalog/admission and authenticated libp2p links. |
| [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver) | Existing C++ constraint solver | Consider only if the bounded adapted planner cannot express required constraints cleanly. It adds a build/package dependency; no runtime solver dependency is proposed initially. Small exhaustive fixtures can check optimality without it. |

License references: [llama.cpp MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE),
[Shard Apache-2.0](https://github.com/leyten/shard/blob/master/LICENSE),
[Petals MIT](https://github.com/bigscience-workshop/petals/blob/main/LICENSE),
[exo Apache-2.0](https://github.com/exo-explore/exo/blob/main/LICENSE).
Before copying, record the exact commit and preserve applicable license/notice text.
Branch links above identify inspected sources; they are not reproducible dependency pins.

Relevant papers: [Petals](https://arxiv.org/html/2312.08361v1) for decentralized routing
and rebalancing; [Parallax](https://arxiv.org/abs/2509.26182) for separating allocation
and request routing. [Parallax code](https://github.com/GradientHQ/parallax) uses a
different runtime stack; no isolated drop-in module was verified in this review.

Do not copy Shard's subnet-separation rule, centralized orchestrator, fixed speculative
acceptance assumption or single total-time objective. DAN needs separate startup,
prefill and decode constraints. No inspected project supplies this entire policy
as a drop-in library.

Shard's greedy layer assignment is optimal only within its linear, uniform per-layer
decode-cost model. DAN must compare alternate feasible splits when endpoint memory,
nonuniform costs or first-response constraints invalidate that assumption.

## Target selection contract

Choose the highest curated quality tier among feasible plans. Within a tier, prefer
the plan with better confidence-adjusted user latency; use cache warmth and lower
resource cost as further tie-breakers. Preserve active conversation model/configuration.

A plan specifies model artifact/quantization, required capabilities/context, provider
groups, explicit local devices, contiguous stage ranges, cache precision, transport,
session capacity and whether it is ready or requires loading.

Feasibility includes per-device and shared-host memory, supported backend/model/ABI,
available capacity, provider limits, the participant cap, decode target and the
applicable warm/cold first-response policy. A free session slot alone does not prove
that another chat can sustain 20 tokens/s.

For ordinary decoding, estimate the sequential sum of stage compute and forward/
feedback transfer costs, with queuing and client delivery accounted for in observed
latency. Include tail-to-head feedback and tail-to-client return according to the
actual critical path; do not double-count concurrent transfers. RTT/2 is an estimate,
not measured one-way latency. For prefill, account for uncached prompt rows and actual
chunk overlap. For speculation, use actual accepted-token yield and enabled mode;
do not credit chat API paths with unavailable draft-model acceleration. Batching is
opt-in and its observed per-chat contention belongs in the model.

Keep model-load/download time separate from warm inference. Bound the planner's own
execution work and network discovery so optimization cannot consume the first-token
budget. Unknown, stale or self-reported performance is not proof of eligibility.

## Missions — execute only after implementation approval

### M0 — Freeze the beta contract

Dependencies: remaining owner decisions above. Keep thresholds configurable in one
shared selection policy used by clients and replica owners.

- [x] Resolve hardware matrix, context, cold start and participant-cap decisions.
- [x] Define useful output, TTFT, sustained per-chat rate, stalls and load conditions
  (`engine/include/provider_owned/selection.hpp`, `SelectionPolicy`).
- [x] Define explicit below-target fallback (`Selection::fallback`, offered only when
  nothing qualifies; context is never relaxed). Manual model choice: M5.
- [x] Curated preference: `quality_tier` in each manifest (32B 50, 14B 40, 7B 30,
  OLMoE 20, 1.5B 15, 0.5B 10); file size only breaks ties.

Done: `selection_test.cpp` holds 13 worked examples (which plan wins and why,
including no qualifying plan).

**Done when:** deterministic examples specify which plan should win and why, including
no qualifying plan. Speed is an estimated service target, not a guarantee about
uncontrolled public hardware.

### M1 — Bind the hardware DAN actually advertises

Reuse upstream ggml inventory and llama model device parameters. Start in provider
launcher/worker capability handling; keep device configuration provider-local.

- [x] Enumerate backend/device capabilities (`dan-stage-worker --list-devices`, any
  ggml backend); explicitly bind the selected device (`--device PCI|NAME` →
  `llama_model_params.devices`). The launcher passes nvidia-smi's PCI bus ID.
- [x] Use current free memory plus owner quotas: greetings and reservations offer
  min(quota, the bound GPU's free memory) while no stage is loaded.
- [ ] Distinguish device, provider node and shared host memory; avoid double counting.
- [ ] Prefer one provider-owned local GPU group as one WAN stage, using upstream
  local splitting. Validate before relying on combined capacity. Multiple worker
  processes/identities per card are a fallback, not the first architecture change.
- [x] Remove NVIDIA-only launch gating: without nvidia-smi the launcher uses the
  worker's device list and accepts only CUDA, ROCm and MTL backends.

Checked on the RTX 2070: listing, PCI ID match, bound 0.5B load (all layers on CUDA0),
unknown device refused, launcher fallback with nvidia-smi missing. Not checked: a
second GPU (no hardware), AMD/Metal (no builds), UMA host-memory limits.

**Done when:** selecting GPU 1 loads GPU 1; shared/UMA memory is not added twice;
unsupported backends fail clearly; memory changes cause a clean refusal. Each promised
backend gets a correctness check; the existence of a llama backend is insufficient.

### M2 — Make model/context/memory admission consistent

Dependencies: M0–M1. Reuse `Manifest`, `ModelIndex`, `stage_fits`, KV padding and
upstream no-allocation memory breakdown. Avoid model downloads merely to score plans.

- [ ] Add catalog preference and capability metadata; keep immutable artifact hashes.
- [ ] Propagate context/session requirements through launcher, owner, client and API.
  Done so far: a client never uses a READY replica with less context than the chat
  needs (the request's, else the model manifest's).
- [ ] Include weights, endpoint tensors, KV, compute buffers, backend overhead, draft
  state when enabled, and local host-memory limits. Retain conservative estimates
  until upstream stage-specific reporting is validated; cache reports by configuration.
- [ ] Keep requested context and offload policy fixed when using upstream fit helpers.
- [ ] Filter unsupported templates/tools/architectures and precision/backend combinations.

**Done when:** a larger file cannot override quality preference; a 512-token replica
cannot satisfy 16K; boundary layers and concurrent-session KV fit on every device;
no automatic reduction of context or silent CPU fallback is used to pass admission.

### M3 — Reuse measurements and model their uncertainty

Dependencies: M1–M2. Extend existing worker timing, owner rate and sidecar edge probes;
no telemetry service or automated benchmark suite.

- [ ] Separate prefill and decode observations by relevant model, backend, precision,
  context bucket, active sessions and execution mode; bound the record count.
- [ ] Carry timestamps/sample count/confidence; age out stale optimistic claims.
- [ ] Bootstrap unseen hardware/configurations with conservative provisional estimates
  and existing bounded formation warm-up. Offer an explicitly unverified trial through
  the slower-fallback choice; do not claim it meets 20 tokens/s before observations.
  New providers must have a path to collect ordinary-traffic evidence.
- [ ] Observe transfer size/time and path changes during normal traffic. Reuse bounded
  candidate-edge probes; avoid an all-pairs network survey or bandwidth stress tests.
- [ ] Treat peer-reported speeds/memory as hints. Prefer locally observed completion
  and transfer behavior; retain worker-side admission and bounded retries.
- [ ] Keep content out of telemetry; preserve existing permitted diagnostics.

**Done when:** unknown latency never means zero; dense/MoE/configuration measurements
are not blindly mixed; aggregate throughput and speculative guesses cannot satisfy
the per-chat target; a new provider can enter an explicitly accepted trial; protocol
fixtures reject invalid numeric/capability values.

### M4 — Adapt an existing bounded placement algorithm

Dependencies: M0–M3. Prefer a narrow C++ adaptation of Shard's pure topology helpers
inside DAN's existing planner. Reuse DAN memory checks and tests. Do not build a
Python scheduling service or replace libp2p.

- [ ] Pin/review the reused source and attribute copied/translated code.
- [ ] Keep diverse candidate groups from the discovered pool: capacity, compute,
  connectivity and cache state. Separate candidate-search budget from provider cap.
- [ ] Compare eligible model/group/order/split combinations; use the existing feasible
  result as an initial candidate while bounded search improves it.
- [ ] Evaluate alternate feasible layer splits through existing planner machinery
  where Shard's uniform-layer greedy allocation does not satisfy DAN's cost/TTFT model.
- [ ] Enforce contiguous complete layer coverage and nonduplicated resource use.
- [ ] Apply separate TTFT/decode constraints and DAN's actual feedback/return paths.
- [ ] Score cached and uncached splits consistently; prune dominated alternatives.
- [ ] Bound work deterministically, cache plans against capability versions, and
  return a feasible best-found result or an explicit no-plan reason. Do not call
  bounded heuristic output globally optimal.

**Done when:** synthetic fixtures handle slow large GPUs, asymmetric links, a useful
ninth provider, cache-versus-speed tradeoffs, participant limits and heterogeneous
memory. Tiny cases match an exhaustive reference; large discovery sets terminate
within the configured work budget. No performance benchmark is required for these tests.

### M5 — Unify ready-replica and new-placement choice

Dependencies: M4. Reuse `open_replica`, `PlacementRequest`, existing leases and the
local API's retained session.

- [ ] Apply one quality/context/load/speed policy to ready and prospective routes.
- [ ] Honor the selected cold-start mode; expose the model and reason for fallback.
- [ ] Include expected queueing and per-chat capacity; keep admission worker-owned.
- [ ] Keep active conversation routing stable; do not discard cached context because
  a marginally better score appears. New chats can select a different model.
- [ ] Return the selected artifact/configuration to the client and carry its explicit
  choice on later turns, including after the five-minute process-cache expiry or a
  switch between chats. Reuse model request/response metadata; do not retain KV or
  server-side conversation content indefinitely. An unavailable pinned model requires
  an explicit new choice, not a silent model switch.
- [ ] Preserve one active request per local API initially; multiple users' local APIs
  already connect independently. A shared public web gateway is separate scope.

**Done when:** an under-target large ready replica cannot bypass the policy; warm
prefix reuse affects startup cost appropriately; the model stays pinned across cache
expiry; reservation races release partial leases and select another valid option
without centrally assigning providers.

### M6 — Allow idle replicas to improve without oscillation

Dependencies: M5. Adapt Petals' stability ideas around DAN's owner loop/status/leases.

- [ ] Advertise bounded idle/reclaimable state without treating busy GPUs as free.
- [ ] Let each owner reconsider its own idle replica after meaningful changes.
- [ ] Require an improvement margin and cooldown, with existing deterministic owner
  ordering/backoff to reduce competing proposals. No global network optimizer.
- [ ] Define how cached idle sessions expire before yielding; no active or retained
  conversation may be silently destroyed for an upgrade. New incoming work cancels
  a pending idle-yield decision.
- [ ] Recheck leases/capacity and restore usable formation if an upgrade cannot complete.

**Done when:** idle smaller replicas can release resources for an eligible better
arrangement; active chats survive; competing owners do not leak leases or repeatedly
dissolve/reform the same layout. This does not add mid-request failover.

### M7 — Meet public, heterogeneous beta boundaries

Dependencies: M1–M6. Reuse existing authenticated connections, parsers, quotas and
libp2p controls rather than inventing a new identity or transport system.

- [ ] Validate and bound advertised device counts, model lists, rates, sizes and
  timestamps; reject malformed frames and incompatible runtime/feature versions.
- [ ] Check resource limits/timeouts for public connection and reservation churn.
  Cap per-peer discovery/probe work; stale advertisements cannot hold capacity forever.
- [ ] Preserve worker-owned SHA-to-catalog resolution; no client download URLs.
- [ ] Build/package the agreed CUDA/AMD/Metal/etc. variants with matching dependencies.
  Verify stage extraction, hidden states, sampling, KV and device affinity per backend.
- [ ] Keep algorithm predictions distinct from guarantees about untrusted provider
  behavior; document the deferred correctness/privacy mechanisms accurately.

**Done when:** all promised OS/backend combinations have supported packages and
correctness evidence; hostile capability fixtures fail safely; unsupported hardware
gets an actionable response. Public participation does not imply trusted results.

### M8 — Complete the beta acceptance checklist

Dependencies: M0–M7. Extend existing `formation_test`, `placement_test`, discovery/
replica tests and real-model chat lifecycle checks; do not add another framework.

- [ ] Check selection policy, memory admission, device affinity and context consistency.
- [ ] Check stale claims, impossible targets, bounded search, concurrency and lease races.
- [ ] Check malformed tool/schema requests, cancellation and subsequent session reuse.
- [ ] Check provider arrival/exit and idle upgrades; an interrupted answer fails clearly
  and the user can retry. No automatic failover promise.
- [ ] Rebuild matching binaries/installers and update PROJECT.md with actual support.
- [ ] Owner checks the UI and a real multi-PC large-model/tool conversation.
- [ ] Owner evaluates actual TTFT, streaming rate and stalls at requested context/load.
  Correctness tests alone cannot establish the 20 tokens/s target.

**Release gate:** all required checklist items pass on the agreed hardware matrix;
remaining unsupported capabilities are explicit. Speculation and broader batching
can improve the attainable plans later, but are not substitutes for correct selection.

## Keep the implementation short

1. Share one policy/estimator/planner between clients and replica owners.
2. Reuse llama.cpp inventory/memory/local execution, DAN admission/leases/transport,
   one adapted placement core, and existing test harnesses.
3. Use a curated catalog before attempting automatic quality evaluation.
4. Collect normal-traffic measurements; avoid a new metrics service or calibration fleet.
5. Treat local GPU groups as one WAN stage when upstream splitting passes correctness.
6. Keep candidate exploration bounded; do not attempt globally optimal network packing.
7. Require a concrete missing capability before adding OR-Tools, Boost Graph, a Python
   daemon, Hivemind, or another runtime dependency.
8. Land missions as small reviewable changes after approval. Pass each acceptance gate
   before moving on; keep batching/speculation flags independent of selection policy.

Shortest dependency order: **M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8**.
Backend packaging and public-input fixtures can progress alongside M3–M6 once the
hardware and wire contracts are fixed. No implementation or tests were run for this plan.
