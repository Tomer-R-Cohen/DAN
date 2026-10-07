# Beta model and GPU selection — missions and acceptance checklist

Date: 2026-10-07. **Implementation approved 2026-10-07.** Status: M0–M8 implemented and
checked on this PC (CPU and RTX 2070). Open: link bandwidth (M3), first-token estimate
inside the search (M4), ROCm and Apple Silicon builds (M7), and the owner's UI check and
real two-PC speed evaluation (M8).

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
  needs (the request's, else the model manifest's). A replica owner without
  `--context` forms at the largest context that needs no extra stage, from the
  manifest's context (floor) up to the GGUF `context_length` (cap 256K); the replica
  advertises the context it got (`PlacedRoute::context`). Model index format 3 stores
  `context_length`; older index files are re-read once.
- [x] Split-route context no longer limited by one activation frame: only the first
  stage checks the wire limit, against its largest frame (prompt chunks of at most
  `prefill_chunk` rows). Before, any split route refused contexts above 64 MiB /
  (hidden x 4): ~4.6K positions for 7B, ~3.2K for 14B/32B. Checked: three CPU stages
  of 0.5B at 32K context, a ~12K-token prompt sent as 512-row chunks, answer returned.
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

- [x] Separate prefill and decode observations by model, context bucket (powers of two
  from 512) and active-session bucket (1, 2, 4 … 64); at most 64 records per worker,
  oldest dropped (`engine/include/provider_owned/measurement.hpp`). Speculative verify
  batches (9–63 rows) are not recorded. Backend and KV precision are per worker, not
  in the key: a worker restarted with another backend or `kv_cache` on the same cache
  folder keeps its old entries until they age out (known gap).
- [x] Each record carries a sample count and age; entries older than 7 days are
  ignored; under 32 samples an estimate is "not measured". Greetings send ages, not
  timestamps, so machine clocks need not agree. Kept in `<cache>/speeds.txt`.
- [ ] Bootstrap unseen configurations. Done: placement uses an exact or heavier
  (larger context, more sessions) measurement of the same model, else the generic
  speed, and reports `PlacedRoute::estimate_measured`. Remaining (M5): route an
  unmeasured plan through the explicit trial/fallback choice.
- [x] Observe link times during normal traffic (latency; bandwidth not yet): each
  sidecar pings, at most every 30 s, the ≤ 16 peers it has DAN streams with, and keeps
  the replica owners' edge-probe results; its network status lists ≤ 32 measured links
  (`links`: peer, rtt, direct/relay; libp2p's average for other connected peers). The
  worker passes them on (`link=` greeting lines, ≤ 32, rtt ≤ 60 s). Planners use a
  measured worker-to-worker round trip (plus the relay penalty) instead of the
  via-planner guess; the search's pruning bound no longer assumes that guess. Not
  measured: bandwidth, i.e. how long a large prompt chunk takes on a link.
  Checked: sidecar `TestNetStatusLinks`, greeting bounds, 600 brute-force cases (half
  with links), a fixture where two far-but-close workers win once their link is known,
  and the replica rehearsal (nodes listed their ring peers, direct or relayed).
- [x] Peer-reported speeds are hints: they only rank plans; workers still admit or
  refuse every reservation themselves, and the owner measures the formed replica.
- [x] No content in measurements: model hash, bucket numbers, speed, sample count, age.

Checked 2026-10-07: three CPU stages of 0.5B at 32K context; after one ~12K-token
request every stage recorded 25 prefill chunks and its decode steps. Greeting parser
rejects malformed, implausible (0, > 1e9 µs/GiB, non-bucket, too old) or more than 64
measurements (`provider_owned_measurement_test`).

**Done when:** unknown latency never means zero; dense/MoE/configuration measurements
are not blindly mixed; aggregate throughput and speculative guesses cannot satisfy
the per-chat target; a new provider can enter an explicitly accepted trial; protocol
fixtures reject invalid numeric/capability values.

### M4 — Adapt an existing bounded placement algorithm

Dependencies: M0–M3. Prefer a narrow C++ adaptation of Shard's pure topology helpers
inside DAN's existing planner. Reuse DAN memory checks and tests. Do not build a
Python scheduling service or replace libp2p.

- [x] Reused source: none copied. For a fixed group order the best split is an exact
  dynamic program over layer boundaries with DAN's own `stage_fits` rule, which
  Shard's uniform-layer greedy cannot match for endpoint tensors or uneven speeds; so
  no Shard code was translated and no attribution is needed
  (`engine/include/provider_owned/search.hpp`, `engine/search.cpp`).
- [x] Diverse pool from all discovered workers (default 12): the head, then round-robin
  from closest link, most memory, fastest measured speed and cached layers. The work
  budget (4000 split evaluations) is separate from the participant cap (5).
- [x] Groups of 1..5, every order for up to four non-head members (three heuristic
  orders beyond), exact split per order. The old first-fit planner's result is scored
  too and kept when better (it never is when the search was exhaustive).
- [x] Contiguous full coverage, one stage per worker, required head first (tests).
- [ ] Separate TTFT/decode constraints. Done: decode time per token with DAN's ring
  links. Not done: prefill/first-token estimate in the search (M5 uses the policy).
- [x] Cached and uncached plans scored with the same cost; a plan from layers already
  on disk wins when at most 10% slower. Groups are pruned by total memory and by a
  lower bound on time. Even stage times break exact ties.
- [x] Bounded and deterministic; `complete` only when every group and order was
  evaluated; otherwise "best found". No-plan returns a reason (cap, memory, budget).
  Plans are not cached across calls (each search takes well under a second).

Measurements and splits: only a confident (32+ samples) measurement of the model shapes
a split; otherwise all workers share a neutral speed. A worker's generic speed comes
from its last stage, and per-step overhead inflates it on small stages: trusting it
gave a worker that ran one layer one layer again (seen on CPU). Known gap: confident
measurements still carry this small-stage bias; fitting a fixed-plus-per-GiB cost once
a worker has run two stage sizes would remove it.

Checked: 600 random small cases (half with a tied output head) equal brute force;
fixtures for slow large GPUs, slow links, asymmetric/relayed links, a useful ninth
provider, the five-provider cap, uneven memory, cache versus speed, required head, even
split; 200 candidates stay within budget (`provider_owned_search_test`). Real CPU route
(three 0.5B workers): exhaustive search in 0.5 ms, cached split chosen, answer returned.

**Done when:** synthetic fixtures handle slow large GPUs, asymmetric links, a useful
ninth provider, cache-versus-speed tradeoffs, participant limits and heterogeneous
memory. Tiny cases match an exhaustive reference; large discovery sets terminate
within the configured work budget. No performance benchmark is required for these tests.

### M5 — Unify ready-replica and new-placement choice

Dependencies: M4. Reuse `open_replica`, `PlacementRequest`, existing leases and the
local API's retained session.

- [x] One policy for both: `dan-client` scores ready replicas (owner-measured ms per
  token and first token, plus the client's link) and placement previews (search
  estimate, prefill measurements, links) with `select_plan`. Placement can preview
  every model without reserving (`PlacementRequest::previews`); only the selected model
  is then placed. `--policy target` uses only plans that meet the target and otherwise
  fails with `below_target` and the reason; `--policy any` (CLI default) also accepts
  slower or unmeasured plans, still preferring a ready replica in ready-first mode.
- [x] Cold start: `--cold-start ready|wait`. Ready-first answers on a qualifying ready
  replica without looking at new placements (keeps warm turns fast); a better plan that
  must load is reported only when previews were made. The API's first event names the
  model, its SHA-256 and whether it met the target.
- [x] Load: a ready replica's time per token is the owner's measurement times the
  sessions sharing it (conservative; batching would do better). Workers still admit or
  refuse every reservation.
- [x] Stable routing: selection runs when a client process starts; turns of a live
  conversation stay on its route and retained KV.
- [x] Pinning: the local API offers `dan-auto` (policy target) and `dan-any` (accept
  slower/unmeasured). It remembers each conversation's model SHA-256 by its scope hash
  (RAM only, at most 256 entries, 24 h; no content) and restarts the client with
  `--pin-model` after the five-minute expiry. An unavailable pinned model returns an
  explicit error asking for a new chat; there is no silent switch.
- [x] One active request per local API (unchanged).

Races: a ready replica that fails to open is dropped and selection runs again; placement
keeps its existing lease release and retries. Replica owners now advertise
`first_token_ms` (status JSON and a 14th field on REPLICA lines; older lines parse).
Known gaps: the first-token estimate of a ready replica comes from its real requests and
warm-up, not from a fixed ordinary-turn size; a new placement needs confident prefill
measurements for every stage, so a fresh network is "unmeasured" until used (`dan-any`).

**Done when:** an under-target large ready replica cannot bypass the policy; warm
prefix reuse affects startup cost appropriately; the model stays pinned across cache
expiry; reservation races release partial leases and select another valid option
without centrally assigning providers.

### M6 — Allow idle replicas to improve without oscillation

Dependencies: M5. Adapt Petals' stability ideas around DAN's owner loop/status/leases.

- [x] Idle state: a worker holding a stage adds `idle_s` (seconds since its stage last
  computed) and `open_sessions` to its greeting. The sidecar lists busy workers only
  for `DAN-CANDIDATES/1 +busy` (an 8th `available|busy` field); placement previews
  with `reclaim_idle_s` may count a busy worker with no session that has been quiet
  that long. Real placement never reserves a busy worker.
- [x] Each owner checks only its own replica (`--upgrades`, launcher
  `replica_upgrades=true`, off by default), every 60 s (jittered) once the replica has
  had no request and no open session for 2 min. It previews every catalog model with
  itself as head over its members, free peers and idle busy peers.
- [x] Gives way only for a higher curated tier, or the same model at least 25% faster
  by a measured estimate, and only if the better plan uses a GPU outside its replica
  (`engine/include/provider_owned/upgrade.hpp`). After giving way it leaves its GPU
  free for 2–3 check intervals, and until a deadline (idle time + four check
  intervals, ~6 min) forms only a plan at least as good as the one it gave way for,
  with at least as many GPUs. So when two owners give way at different times, the
  later one still finds the earlier one's GPU free; the existing rank delay and leases
  settle who leads. An upgrade that did not happen backs off 10 min, doubling to 4 h;
  success (or another replica taking this GPU) resets it.
- [x] Retained conversations count as open sessions (the API keeps its client session
  for five idle minutes), so they are never dissolved; any queued or new request, or a
  session opened since the check, cancels the decision at the last moment.
- [x] Giving way is an ordinary dissolve: leases are released and members keep their
  layers loaded, so whatever forms next (the better replica, or the old one again)
  reuses them.

Checked 2026-10-07 (two CPU nodes, fake GPUs sized so each holds 0.5B but only both
together hold 1.5B, upgrade idle 15 s / checks 5 s): A alone formed 0.5B; B then formed
its own 0.5B (the stuck state from PROJECT.md §14 item 5); each owner gave way once; the
1.5B replica formed across both from cached layers and stayed for the next minute. The
first attempt, without the formation deadline, showed the failure it fixes: the owners
gave way at different times and each re-formed its small replica alone. Not checked
end to end: an open conversation blocking an upgrade (unit-tested decision only), and
the real network.

**Done when:** idle smaller replicas can release resources for an eligible better
arrangement; active chats survive; competing owners do not leak leases or repeatedly
dissolve/reform the same layout. This does not add mid-request failover.

### M7 — Meet public, heterogeneous beta boundaries

Dependencies: M1–M6. Reuse existing authenticated connections, parsers, quotas and
libp2p controls rather than inventing a new identity or transport system.

- [x] Bounds, refusing the whole answer instead of clamping:
  - Sidecar (`sidecar/limits.go`): a remote capability needs protocol version 1, names
    ≤ 128 bytes, ≤ 16 TiB memory, context ≤ 1M, ≤ 256 sessions, ≤ 64 models with valid
    hashes, ≤ 4096 layers, ≤ 64 cached ranges inside the model, ≤ 8 data protocols. A
    remote replica status needs valid IDs, 1–256 sessions (in use ≤ max), context ≤ 1M,
    1–64 members with valid ranges, latencies ≤ 10⁷ ms. Messages stay ≤ 64 KiB.
  - Worker greeting (`formation.hpp`): ≤ 64 KiB, the same name/memory/context/session
    bounds, ≤ 64 models, ≤ 256 cached ranges ending ≤ layer 4096, speeds ≤ 10⁹ µs/GiB,
    ≤ 64 measurements (M3 rules). Unknown keys are still skipped (forward compatible).
  - Client: ≤ 512 candidate or replica lines from the local sidecar, plausible numbers.
  - Runtime ABI and feature versions were already exact matches (`runtime_abi`).
- [x] Churn limits (checked, one added): leases expire (≤ 60 s), unreserved control
  connections close after 60 s idle, serving ones after 10 min, a worker takes ≤ 16
  control connections and now ≤ 4 per PeerID; probes answer `ERR busy`; discovery
  queries ≤ 64 candidates with deadlines; worker statuses and replica answers older
  than their freshness limits count as offline.
- [x] Catalog: clients still name models only by SHA-256 and workers resolve them
  in their own catalog; nothing in M0–M7 added a URL path.
- [ ] Packages per backend. Done: `scripts/build_provider_owned.ps1 -Backend
  cpu|cuda|rocm|metal`; the launcher detects ROCm/Metal GPUs through the worker (M1).
  Not done (needs that hardware): a ROCm build and an Apple Silicon build, each passing
  `provider_owned_stage_reference`, the placement and replica rehearsals, and a device
  binding check. CUDA (RTX 2070) and CPU are the only verified backends.
- [x] Predictions versus guarantees: every speed is a prediction from peer-reported
  and locally observed numbers; `dan-auto` says "predicted to meet", and nothing here
  verifies results, protects prompt privacy from providers, or resists Sybil peers
  (PROJECT.md §3, §11).

Checked: `provider_owned_measurement_test` (hostile greetings), sidecar
`TestValidCapability` / `TestValidReplicaStatus`, replica rehearsal with the per-peer cap.

**Done when:** all promised OS/backend combinations have supported packages and
correctness evidence; hostile capability fixtures fail safely; unsupported hardware
gets an actionable response. Public participation does not imply trusted results.

### M8 — Complete the beta acceptance checklist

Dependencies: M0–M7. Extend existing `formation_test`, `placement_test`, discovery/
replica tests and real-model chat lifecycle checks; do not add another framework.

Results 2026-10-07 (this PC: CPU workers with fake GPUs, plus the RTX 2070 where noted):

- [x] Selection policy, memory admission, device affinity, context: 16 C++ unit tests
  (selection, measurement, search incl. brute force, upgrade, placement, formation,
  range model, protocol, client, route, lease, UI, platform) and the Go tests pass;
  RTX 2070 device binding (M1); split route at 32K context (M2).
- [x] Stale claims, impossible targets, bounded search, concurrency, lease races:
  stale and hostile claims refused (M3/M7 tests); `--policy target` refuses an
  unmeasured route with `below_target` and names the slower option (M5); 200
  candidates within budget (M4); placement rehearsal over libp2p with a two-client
  race: one winner, every worker free afterwards, output identical to the baseline;
  replica rehearsal: four owners racing for three GPUs formed exactly one replica
  (two lost races released cleanly); two concurrent chats interleaved on one replica,
  both byte-identical to the baseline.
- [x] Schema requests, cancellation, session reuse: `api_lifecycle_integration.py`
  against a local stack (CPU 0.5B replica + gateway) passed three runs in a row:
  JSON schema output, prefix reuse, a cancelled stream followed by a new chat.
  Malformed tool/schema requests are covered by the gateway's Go tests.
- [x] Provider arrival/exit and idle upgrades: replica rehearsal (a member killed
  mid-answer gives the client an error, the replica dissolves, re-forms from loaded
  layers when the member returns; owner death releases everyone); two-node upgrade test
  (M6). No automatic failover is promised.

- [x] Matching binaries and installer: portable CUDA build (sm 75/86/89/120) of this
  branch; replica rehearsal on the RTX 2070 (three CUDA workers on one GPU) passed every
  functional check, ~11 ms per token. Its text differs from the CPU baseline after ~20
  tokens, as CPU and GPU arithmetic do; on the GPU the replica, a three-stage route and
  a single whole-model worker all produced identical text. `DAN-Setup-1.1.0.exe`
  (484 MB, public network node as bootstrap) built; govulncheck clean apart from the
  documented GO-2024-3218; the staged launcher's `--check` and the packaged worker's
  `--list-devices` work on this PC. `build_installer.ps1` now also ships
  `llama-common.dll`, which the chat runtime needs. Not installed or run as a node.

Found and fixed while running this checklist: a client that found every worker taken
reported `below_target` (now "no placement fits"); a new chat right after a cancelled
one could find the replica's only session still closing (the client now waits up to
~16 s for a fitting replica that is only full); the lifecycle test used `dan-auto`, so a
busy machine (42 ms per token while a CUDA build ran) correctly made it refuse; it now
uses `dan-any`, since it checks correctness, not speed.
- [x] Rebuild matching binaries/installers and update PROJECT.md with actual support
  (see results above; AMD and Apple remain unverified, M7).
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
