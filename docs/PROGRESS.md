# DAN Progress

The canonical answer to three questions — what works, what was proven, what
happens next — plus the private-beta launch gate. Design details are in
[ARCHITECTURE.md](ARCHITECTURE.md), build/run/operate commands in
[OPERATIONS.md](OPERATIONS.md), and physical-test receipts in
[TESTS_AND_STATS.md](TESTS_AND_STATS.md) and `reference/design/`.

## Working rules

- Build the smallest working distributed primitive, understand it, document it,
  and only then add the next layer.
- Use C++23, raw POSIX TCP sockets, CMake, and the standard library.
- Preserve the current poll-based coordinator, persistent providers, and framed text protocol.
- Do not add AI, cryptocurrency, blockchain, marketplaces, distributed training,
  or scalability infrastructure until explicitly requested.
- Before changing code, inspect the repository and read this `docs/` directory.
- After a meaningful change, build, test, update the relevant documents, and
  record exact progress in this file.
- Keep architecture documentation aligned with what actually works.
- Treat model weights and model choice as registry configuration, never as
  family-specific networking or scheduling logic.

## Current focus

DAN's main distributed-inference direction is persistent provider-owned
execution with runtime replica formation and range-backed sparse GGUF storage.
The C++23 coordinator selects a Qwen2 model from configuration, reads metadata
only, plans contiguous stages from live provider VRAM, and routes activations
through the resulting ordered replica without owning model weights.
Models that fit one provider use a single local stage; larger models are split
across as many providers as the planner needs.

On 2026-09-08, Qwen2.5 32B Q5_K_M completed a physical 20-token inference run
across a Windows RTX 2070 and Linux RTX A5000. The coordinator formed layers
0-6 and 7-63 automatically. Decode reached 4.279 tok/s; the relayed Tailscale
path consumed 184.791 ms/token and dominated performance. See the
[complete physical result](reference/tests/aggregate-vram-test.md).

The current productization milestone is Windows Release v1.0.1. It includes a
separate interactive coordinator package with automatic replica formation and
persistent chat, plus a contributor package that puts the provider-owned CUDA stage worker behind the existing one-click
NVIDIA/Tailscale launcher and adds a live terminal dashboard, automatic
reconnect, bounded resumable range downloads, disk preflight, cache status,
layer assignment, VRAM, request, and token telemetry. The old managed RPC
worker is not used by this contributor package.

Local Windows acceptance passed 100/100 sequential 20-token requests without a
worker PID change or model reload. Persistent follow-up, independent resident
sessions, reset, and graceful shutdown also passed. See
[the v1 report](reference/design/runtime-v1.md). The v1 physical Windows/Linux rerun
was skipped at the user's request; the earlier physical execution proof remains
in [the v0 report](reference/tests/provider-owned-execution-v0.md).

On 2026-09-12, local CUDA recovery was validated by killing the active provider,
reforming the replica with a replacement, and resuming inference. Restarting the
coordinator with the original provider still alive also reused the resident stage.
Automatic two-provider CUDA ring formation was then validated from provider-
advertised endpoints, including queue/backpressure acceptance, idle failure
detection, replacement reformation, and inference after recovery.
Token-by-token streaming was validated through both the binary protocol and the
DAN SSE gateway on that CUDA ring.
Active cancellation was then validated on the same ring: a running 4,096-token
request stopped at a token boundary, rolled back cleanly, and released the
replica for subsequent work.

On 2026-09-12, the rebuilt coordinator and contributor archives were also run
together from their extracted package directories. Encrypted libp2p onboarding,
CUDA loading, readiness, non-streaming and SSE generation passed. Stopping the
provider changed readiness to `503`; restarting the unchanged package reused its
cache, restored readiness, returned `recovered`, and reported
`replica_reformations=1`. This proves same-machine package integration, not the
still-open clean second-machine acceptance or 24-hour soak gates.
The packaged soak checker then completed 123 stateless CUDA requests in five
seconds with no resident-session/KV residue, and a second run waited through a
forced provider outage and continued after cached rejoin.
The packaged libp2p path now carries activations directly between authenticated
provider peers and returns the tail directly to the coordinator. A packaged
interactive run formed balanced 12/12 stages on two providers and completed real
CUDA generation. This remains same-machine evidence; clean cross-machine/NAT
acceptance is still open.
An opt-in served run with the same direct encrypted ring, CUDA graphs, a local
draft model, and pipeline depth 4 completed buffered and SSE requests. Metrics
reported 8 speculative rounds and 27 accepted of 32 proposed draft tokens with
zero residual sessions/KV.

On 2026-09-16, `Start-DAN-Service.ps1` was changed to run
`Select-DAN-Model.ps1`, a new interactive picker, on every launch: it lists
every bundled model config, lets the operator pick the target model (served
over the WAN) and an optional same-family draft model for speculative
decoding, remembers the choice (`-NoPicker` reuses it for unattended
restarts), and downloads+verifies the chosen draft model's GGUF once
(`scripts/DanModel.psm1`, generalized to resolve either an explicit
`artifact_url` or an `hf_repo`/`gguf_filename`/`artifact_revision` triple the
same way `engine/coordinator.cpp`'s manifest loader does). `Show-DAN-Dashboard.ps1`
now runs in the foreground by default instead of hiding the coordinator/gateway
behind log files, and shows a compute-vs-network per-token time breakdown using
new `dan_queue_wait_ms`/`dan_time_to_first_token_ms`/`dan_*_compute_ms_per_step`/
`dan_network_ms_per_step` series added to the gateway's `/metrics`.
`Install-DAN-Provider.ps1` was added as a one-line friend-side installer
(download-verify-extract-launch against GitHub Releases).

The same day, `dan-api-gateway` stopped treating every chat request as
stateless: it now recognizes when a request's message history is a prior
turn's history plus new messages and continues that turn's coordinator session
(only the new text is sent; the coordinator's resident KV cache is resumed)
instead of resending the full conversation and reprocessing it from scratch
every message, with a tested fallback to a fresh session when continuation
isn't recognized or fails. Covered by a new Go test
(`TestChatCompletionReusesSessionOnContinuation`) plus the existing suite, all
passing including under `-race`; this required no coordinator/protocol change
since session support already existed server-side, just an unused capability.

None of the PowerShell-side changes (picker, dashboard stats, installer) have a
physical run behind them yet — the Go gateway changes are unit-tested, but
nothing here has been exercised against a real coordinator/provider/GPU. This
needs a real build and a real friend run before any of it can be marked Done
below.

## What works

- Persistent C++23 provider-owned stage workers and metadata-only coordinator.
- Session create/reset/destroy, stateless requests, persistent follow-ups,
  multiple resident KV contexts, strict v2 framing, metrics, and graceful stop.
- Persistent coordinator/provider connections with framed messages, request IDs,
  concurrent providers, FIFO queuing, disconnect handling, and per-provider timing.
- Automatic served-mode replica reformation after provider loss. Active and
  queued work fails cleanly; replacement providers restore service without a
  coordinator restart, idle heartbeats detect failures without client traffic,
  and unchanged reconnecting stages reuse loaded weights.
- A provider can host the metadata-only coordinator beside its own GPU stage;
  the coordinator remains a replaceable role rather than a dedicated machine.
- Automatic formation can wire selected providers into a direct activation ring;
  the coordinator is removed from intermediate hot-path hops and reforms the ring
  after provider loss.
- A dependency-free DAN gateway exposes authenticated model listing
  and streaming or buffered chat completions over the bounded binary service.
  Repeating `-coordinator` round-robins requests across independent replica
  groups and fails over from unreachable, reforming, failed, or saturated
  coordinators before any response content is emitted.
- Gateway metrics expose whether speculation is configured, pipeline depth,
  speculative rounds, and proposed/accepted draft-token totals.
- Model-aware routing across whole-model providers and llama.cpp RPC groups.
- A replaceable model registry with test, main, and distributed roles.
- Managed `dan-main` control plane with arbitrary provider count, deterministic
  assignment, shard states, cache identity, heartbeats, offline recovery, and
  automatic eligible-spare replacement.
- Managed providers that download and SHA-256 verify artifacts, retain cache,
  supervise one RPC worker, and recover from worker exit.
- One persistent coordinator-owned `llama-server` that starts only after every
  assigned provider is ready and serves repeated requests without per-request
  process startup.
- Windows NVIDIA provider onboarding, CUDA packaging, persistent provider identity,
  Tailscale address configuration, automatic reconnect, and browser dashboard.
- Operator telemetry for VRAM offered to DAN and VRAM currently used by DAN.

CPU-only regression coverage exercises control-plane assignment, cache validation,
worker lifecycle, persistent serving, request correlation, provider replacement,
and failure recovery. The strict CMake build and existing regression tests passed
after the 2026-09-07 VRAM-gate change.

## Verified GPU results

### Single NVIDIA A40

Qwen3-30B-A3B Q4_K_M passed 10/10 persistent-provider requests at 32,768
context. Mean DAN latency was 3,844.7 ms and observed peak VRAM was 21,227 MiB.
The processes exited cleanly. Raw evidence is committed in
[`dan-qwen3-30b-a3b-results.tar.gz`](../dan-qwen3-30b-a3b-results.tar.gz).

### RTX 3090 + RTX A4500 smoke test

On 2026-09-05, Qwen2.5-1.5B-Instruct Q4_K_M ran through standalone llama.cpp
RPC and DAN's distributed-group path using llama.cpp
`95ef7fc16054e63b427a3ef00188e055ef7586d8`, TCP, 4,096 context, `RPC0,RPC1`,
and tensor split `6,5`.

| Path | Total latency | Model load | Generation |
|---|---:|---:|---:|
| Standalone | 55,279 ms | 51,984.54 ms | 17.67 tok/s |
| DAN group | 57,223 ms | 51,139.01 ms | 17.98 tok/s |

Pod A peaked at 890 MiB and 5% utilization; Pod B peaked at 1,093 MiB and 3%.
Both GPUs participated, but the roughly 940 MiB model fit on either GPU, so this
did not prove aggregate-VRAM necessity. Pod B's generated hostname did not
resolve; its numeric private IPv4 endpoint worked.

### RTX 3090 + RTX A4500 Qwen3 test

Qwen3-30B-A3B Q4_K_M (18,556,685,824 bytes, SHA-256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`)
generated responses through both the standalone and DAN group paths at 32,768
context.

| Path | Total latency | Load/distribution | Generation |
|---|---:|---:|---:|
| Standalone | 806,286 ms | 781,425.76 ms | 13.73 tok/s |
| DAN group | 804,709.8 ms | 783,180.33 ms | 16.94 tok/s |

The RTX 3090 peaked at 11,958 MiB/34% and the RTX A4500 at 9,839 MiB/43%; peaks
were 219 ms apart. This proved simultaneous participation, but not aggregate-VRAM
necessity because the model may fit on one worker. These two-pod summaries were
operator supplied; their raw pod evidence is not in this checkout.

## Windows friend test: 2026-09-07

### Setup and outcome

- Coordinator/provider A: RTX 2070, 6656 MiB offered.
- Provider B: RTX 2050, 2560 MiB offered.
- Combined offer: 9216 MiB over Tailscale.
- Qwen2.5 7B Q5_K_M (5,444,831,936 bytes) ran successfully on the RTX 2070.
  Its one-shard manifest correctly left the RTX 2050 spare.
- Qwen2.5 14B Q3_K_M used a two-shard manifest and both PCs received distinct
  assignments. The RTX 2070 reached `READY`; the RTX 2050 remained downloading
  until its internet connection failed.
- Actual two-GPU 14B startup and inference remain unverified.

The official Qwen2.5 14B Q3_K_M files are:

| File | Bytes | SHA-256 |
|---|---:|---|
| 0 | 4,000,429,472 | `3f38798df3987883b1b37c923e6a98fbfebc88cff2d94743e9935baccf1d19ad` |
| 1 | 3,338,774,688 | `fee42fd8be1b06f48c2e2313bfa09116446f56efded331543303a4642ba875ad` |

Source: [Qwen/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/tree/main).

## Legacy RPC architecture truth discovered by the friend test

Managed manifest artifacts and llama.cpp tensor placement are separate layers:

1. DAN assigns and downloads one artifact per provider as a cache/readiness step.
2. Each provider starts a model-unaware llama.cpp `rpc-server`.
3. The coordinator starts `llama-server` with the complete GGUF locally.
4. llama.cpp sends tensors to workers using `--tensor-split`, currently derived
   from offered VRAM (for the friend test, `6656,2560`).

The provider's cached GGUF is not passed to `rpc-server`. The current backend can
prove pooled GPU compute and aggregate VRAM, but it cannot provide provider-owned
weight storage where the coordinator has no complete model. Provider GGUF
downloads are redundant traffic under the current llama.cpp RPC design.

That fork is resolved for DAN's main path: the C++ provider-owned engine loads
verified weight ranges on providers and keeps the coordinator metadata-only.
The old llama.cpp RPC path remains legacy and still has coordinator-owned weights;
its provider artifact downloads do not prove provider-owned storage.

## Priority fixes

### Implemented after the friend test

- Downloads now use one stable partial path per artifact hash, HTTP range resume,
  low-speed reconnects, and ten retries. Existing PID-specific partial files are
  migrated by selecting the largest valid candidate. Final size and SHA-256
  verification remain mandatory.
- Protocol v2 reports downloaded bytes, total bytes, and current speed once per
  second. Both provider and coordinator show `Downloading required files` with a
  meter; artifact names remain hidden.
- VRAM wording now distinguishes `used now` from `offered to DAN` and explicitly
  says when the model is not loaded. A roughly 0.1 GB idle RPC worker before
  distributed model startup is expected, not the assigned tensor allocation.

### Existing heartbeat behavior

The provider already sends heartbeats from a separate thread during artifact
preparation. The observed friend timeout followed a whole-machine internet and
Tailscale outage; it was not evidence that the download blocked heartbeats.

### P1: correct resource semantics

Disk artifact bytes are not current RPC-worker VRAM use. The 2026-09-07 change
makes the shared free-VRAM gate honor manifest `min_vram_mib` instead of artifact
size plus 512 MiB. Calibrate those minimums against real allocation and overhead;
the RTX 2050's 2560 MiB offer is tight.

The coordinator UI should show only VRAM offered to DAN and VRAM currently used
by DAN. Physical VRAM is intentionally omitted. User-selectable contribution
limits remain future work; currently the provider offers detected VRAM minus its
fixed reserve.

### P1: smoother networking and access

- Validate the new low-speed reconnect and resume behavior against a real
  interrupted Hugging Face transfer. No HTTP 429 was observed; the session's
  20-30 MB/s to 0.1 MB/s variation was consistent with CDN/routing instability,
  not a proven formal rate limit.
- Keep chat loopback-only by default. Tailscale Serve is the simplest private way
  to let a friend use the coordinator's browser UI, but a tailnet administrator
  must enable it first.
- Treat Tailscale `NoState` as a Tailscale startup problem: restart the Windows
  service, run `tailscale up`, then verify `tailscale status` before debugging DAN.
- A tailnet invitation does not select a local profile. Use `tailscale switch
  --list` and switch by the listed profile name.

### P2: packaging clarity

- Distribute the complete provider release directory. A lone `dan-provider.exe`
  fails with `Could not find the bundled managed provider`.
- Put large CUDA packages in GitHub Releases rather than normal repository uploads.
- Keep the localhost dashboard (`http://127.0.0.1:9090`) as the reliable operator
  view when an automated coordinator has no visible console.
- NVIDIA/CUDA is the current automatic path. Intel Iris needs a supported backend
  and should receive an explicit compatibility message.

## Known limitations

- Provider-owned control and activation traffic is authenticated and encrypted by
  libp2p; capacity claims and returned computation are not independently verified.
- Legacy llama.cpp RPC traffic is unencrypted and must remain on a trusted private
  network.
- Each coordinator serves one model and one replica; the gateway can route across
  multiple coordinators, but one coordinator does not yet own multiple replicas.
- Legacy distributed groups start a process per request; managed serving does not.
- Request queues and coordinator state are not persisted across coordinator restarts.
- Model quality and memory estimates depend on the selected GGUF and manifest.
- Conversation state is not shared across round-robin whole-model providers.
- Provider hardware/capacity claims are self-reported.

## Private beta launch gate

This is the release gate for the provider-owned service. A checked item needs a
repeatable test or physical receipt; a feature existing in source is not enough.

| Gate | State | Evidence or remaining work |
|---|---|---|
| Provider-owned split inference | Done | Qwen2.5 32B ran across RTX 2070 + RTX A5000; local automatic two-stage CUDA ring also passed. |
| Persistent workers and range-backed weights | Done | Workers reuse loaded stages and verified sparse range caches across coordinator reconnects. The pinned default artifact returned its declared 491,400,032-byte length, honored a one-byte range request, and matched the cached full-file SHA-256. |
| Automatic formation and recovery | Done | Idle failure detection, replacement formation, and inference after recovery passed on CUDA. |
| Bounded concurrent API | Done | Fair queue, backpressure, timeouts, cancellation, buffered replies, and token SSE pass acceptance. |
| API readiness and horizontal routing | Done | Health checks replica state; repeated coordinator endpoints round-robin independent replicas and fail over from unreachable, reforming, failed, or saturated replicas before output begins. The runbook keeps coordinator binary ports private and scales independent gateway/replica groups behind a standard TLS load balancer. |
| One-command coordinator service | Done | Extracted Windows coordinator and contributor packages completed encrypted onboarding, CUDA generation, SSE, provider loss, cached rejoin, and post-reformation generation. The launcher holds a native Windows system-required execution state while running. |
| Encrypted automatic activation ring | Done | The extracted libp2p packages formed a direct authenticated provider-to-provider ring with a checked tail return, balanced 12/12 stages, real CUDA generation, and recovery after provider loss. Tailscale mode wires the same ring explicitly. |
| NAT relay operation | Done | The sidecar has an optional bounded circuit-v2 relay-service mode. Its test proves an allowlisted reservation carries an authenticated circuit stream and an unlisted reservation is rejected. Provider packages now reserve and advertise relay addresses for provider-to-provider ring fallback; Windows/Linux cross-build and real provider preflight pass. Cross-NAT physical acceptance remains separate. |
| CUDA graph and speculative acceleration | Partial | CUDA graph reuse is compiled in by default for every release build and active in provider workers (confirmed 179 reuses in a 2026-09-13 cross-machine run). Speculative decoding + pipelining measured a real Windows-to-RunPod WAN run on Qwen2.5-14B: decode went from 4.70 to 12.11 tok/s (2.58x), 46.7% draft-token acceptance (see [`reference/tests/remote-gpu-test-method.md`](reference/tests/remote-gpu-test-method.md)). `Start-DAN-Service.ps1` now runs an interactive picker (`Select-DAN-Model.ps1`) that lets the operator choose a draft model and fetches/verifies it, falling back cleanly to non-speculative if unavailable, but this packaging path has not yet had its own physical confirmation run — the WAN number above is from the underlying binary flags run manually, not from the new picker/launcher. |
| Stranger-machine acceptance | Done | Packaged archives ran successfully across multiple distinct friend PCs on different networks/NATs. |
| Production soak and churn | Done | An operator-run soak held for roughly 8 hours without failure. |
| Distribution license and notices | Done | DAN is Apache-2.0 licensed and both packages include its license/notice. Builders also include pinned llama.cpp/sidecar notices, the exact pinned Qwen2.5 Apache-2.0 license, the installed CUDA license, and a tested 109-file Go dependency bundle (including reciprocal source). This machine has a complete, non-preview Visual Studio Community 2026 installation; [Microsoft's Community terms](https://visualstudio.microsoft.com/vs/community/) permit individual developers to build free or paid apps, and its [current redistribution list](https://learn.microsoft.com/visualstudio/releases/2026/redistribution) permits validly licensed users to redistribute files under `VC\Redist` unmodified. Organization eligibility remains the release builder's responsibility. |
| Reachable dependency vulnerabilities | Done | `govulncheck` v1.8.0 found three reachable Go advisories; DTLS, WebTransport, and QUIC were upgraded to fixed versions and the repeat scan reported zero reachable vulnerabilities. Package creation now reruns this gate. |
| Operations | Done | Authenticated per-replica Prometheus metrics, packaged alert rules, backup, upgrade, rollback, and incident actions are documented in [OPERATIONS.md](OPERATIONS.md). |

### Shard parity and c0mpute integration

These are not private-beta blockers. Shard is the inference engine; worker-account
policy and payments come from its surrounding c0mpute platform and are listed
separately instead of being misrepresented as engine features.

| Capability | DAN today | Gap to Shard |
|---|---|---|
| Execution evidence | Authenticated peer identity or trusted Tailscale; capacity and completed work are self-reported | Signed per-stage activation-chain receipts with complete layer coverage, plus randomized comparison against a trusted copy of the assigned block |
| Weight distribution | Verified HTTPS range fetch and local reuse | Content-addressed peer seeding and mirror-independent recovery |
| Failure semantics | Ring reforms and the next request succeeds | Preserve committed tokens and resume the interrupted request on a warm spare |
| Per-replica throughput | Fair bounded queue; one request executes on a replica at a time; opt-in pipelined speculative chunks | Continuous/batched verification across concurrent requests and production-tuned speculation |
| Privacy | Providers see their boundary activations | Trusted boundary placement and sensitive-job routing policy |
| Model/runtime breadth | Dense Qwen2 GGUF and greedy decoding | Multiple tuned model engines, tool semantics, lossless sampling, and proven long context |
| Swarm management | One replica per coordinator; a stateless gateway balances independent replica groups | One control plane managing multiple swarms with topology-aware placement and live rebalancing |
| Coordinator trust | One operator-run, single trusted coordinator (metadata-only, but it still routes every activation and picks stage placement) | Permissionless coordinator role with no single trusted operator |

c0mpute-only integration still absent from DAN: worker-account admission,
reputation/ejection policy, job accounting, pricing, refunds, and payouts. Those
are required for a paid permissionless network, but not for honest Shard engine parity.

**Decentralized coordinator selection (roadmap, not started).** Bitcoin has no
persistent coordinator role at all — leaderless, with miners winning the right
to propose each block via proof-of-work and the network converging on the
longest valid chain. DAN's coordinator is a different kind of role: it is
metadata-only, but it still routes every activation and decides stage
placement, so replacing "one trusted operator" with a permissionless,
Sybil-resistant election (plus the reputation/slashing needed once that role
can be adversarial) is real distributed-systems design work, not a config
change. Excluded for now by the blockchain/marketplace working rule at the top
of this document; kept here as a future direction, not a private-beta gate.

### Naming and data boundary

The public process is `dan-api-gateway`. It implements DAN's local JSON/SSE API
and never contacts an external inference vendor. Prompts and generated tokens
flow only through the configured DAN gateway, coordinator, and providers.
