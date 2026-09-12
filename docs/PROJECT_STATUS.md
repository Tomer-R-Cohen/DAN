# DAN Project Status

This is the canonical answer to three questions: what works, what was proven,
and what should happen next. Operational commands for the current
provider-owned path belong in
[PROVIDER_OWNED_SETUP.md](PROVIDER_OWNED_SETUP.md); design details in
[ARCHITECTURE.md](ARCHITECTURE.md); the legacy managed/whole-model/RPC path's
commands and wire format are both in [LEGACY_PATH.md](LEGACY_PATH.md).
The remaining private-beta, Shard-engine parity, and separate c0mpute-platform gates are tracked in
[DAN_LAUNCH_READINESS.md](DAN_LAUNCH_READINESS.md).
Production metrics, alerts, backup, upgrade, rollback, and incident steps are in
[DAN_OPERATIONS.md](DAN_OPERATIONS.md).

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
[complete physical result](AGGREGATE_VRAM_TEST.md).

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
[the v1 report](PROVIDER_OWNED_RUNTIME_V1.md). The v1 physical Windows/Linux rerun
was skipped at the user's request; the earlier physical execution proof remains
in [the v0 report](PROVIDER_OWNED_EXECUTION_V0.md).

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
zero residual sessions/KV. The standard release remains non-speculative unless
an operator supplies a compatible draft model.

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
