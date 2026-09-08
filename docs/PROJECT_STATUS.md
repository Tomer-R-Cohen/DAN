# DAN Project Status

This is the canonical answer to three questions: what works, what was proven,
and what should happen next. Operational commands belong in [SETUP.md](SETUP.md),
design details in [ARCHITECTURE.md](ARCHITECTURE.md), and wire formats in
[PROTOCOL.md](PROTOCOL.md).

## Current focus

DAN's main distributed-inference direction is persistent provider-owned
execution with runtime replica formation and range-backed sparse GGUF storage.
The C++23 coordinator selects a Qwen2 model from configuration, reads metadata
only, plans contiguous stages from live provider VRAM, and routes activations
through the resulting ordered replica without owning model weights.

On 2026-09-08, Qwen2.5 32B Q5_K_M completed a physical 20-token inference run
across a Windows RTX 2070 and Linux RTX A5000. The coordinator formed layers
0-6 and 7-63 automatically. Decode reached 4.279 tok/s; the relayed Tailscale
path consumed 184.791 ms/token and dominated performance. See the
[complete physical result](AGGREGATE_VRAM_32B_RESULTS.md).

Local Windows acceptance passed 100/100 sequential 20-token requests without a
worker PID change or model reload. Persistent follow-up, independent resident
sessions, reset, and graceful shutdown also passed. See
[the v1 report](PROVIDER_OWNED_RUNTIME_V1.md). The v1 physical Windows/Linux rerun
was skipped at the user's request; the earlier physical execution proof remains
in [the v0 report](PROVIDER_OWNED_EXECUTION_V0.md).

## What works

- Persistent C++23 provider-owned stage workers and metadata-only coordinator.
- Session create/reset/destroy, stateless requests, persistent follow-ups,
  multiple resident KV contexts, strict v2 framing, metrics, and graceful stop.
- Persistent coordinator/provider connections with framed messages, request IDs,
  concurrent providers, FIFO queuing, disconnect handling, and per-provider timing.
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

## Architecture truth discovered by the friend test

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

Before production, choose one honest design:

- Keep llama.cpp RPC: the coordinator owns the GGUF and providers only run RPC
  workers; remove redundant provider model downloads.
- Build true provider-owned storage/loading: replace or substantially extend the
  RPC backend so each provider loads its own weights.

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

## Next acceptance run

Do not call aggregate VRAM proven until one run satisfies all of these:

1. The chosen model cannot start within either provider's offered limit alone.
2. Both assigned providers reach `READY` without heartbeat timeouts.
3. `llama-server` starts with both RPC endpoints and tensor split `6656,2560`.
4. Both GPUs show nonzero DAN VRAM during the same request.
5. A response completes and both providers receive participation credit.
6. An interrupted provider download demonstrably resumes rather than restarting
   from zero.

Use [NEXT_GPU_EXPERIMENT.md](NEXT_GPU_EXPERIMENT.md) for the detailed rental
runbook and [GPU_VALIDATION.md](GPU_VALIDATION.md) for evidence requirements.

## Known limitations

- Protocol traffic and capability claims are unauthenticated and unencrypted.
- llama.cpp RPC is proof-of-concept and must remain on a trusted private network.
- One managed model and one replica are supported; assignments do not rebalance.
- Legacy distributed groups start a process per request; managed serving does not.
- Request queues and coordinator state are memory-only.
- Cancellation and response streaming are absent.
- Model quality and memory estimates depend on the selected GGUF and manifest.
- Conversation state is not shared across round-robin whole-model providers.
- Provider hardware/capacity claims are self-reported.

## Later roadmap

After the aggregate-VRAM acceptance run:

1. Fix the P0 provider experience issues above.
2. Choose coordinator-owned versus provider-owned model storage.
3. Add authentication before widening access beyond trusted friends.
4. Validate quality, latency, bandwidth cost, and failure recovery on more GPUs.
5. Add contribution controls, bandwidth-aware placement, provider scoring, leases,
   accounting, and pricing only when measurements justify them.
