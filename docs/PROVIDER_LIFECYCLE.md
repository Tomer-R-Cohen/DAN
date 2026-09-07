# Provider lifecycle and weight reuse

## Current evidence and limits

DAN has persistent whole-model providers, model-compatible scheduling, and
manually configured distributed groups. The A40 whole-model run completed ten
requests with a loaded runtime. The reported RTX 3090 + RTX A4500 Qwen2.5 and
Qwen3 tests produced responses through both distributed entry points. See
[PROJECT_STATUS.md](PROJECT_STATUS.md). The two-pod summaries were supplied by
the operators; their raw pod artifacts have not been imported into this checkout.
They establish reported participation, not aggregate-VRAM necessity.

The Qwen3 distributed runs reported about 781–783 seconds loading/distributing
weights within about 805 seconds total. These timings identify model preparation
as the dominant cost; they do not independently isolate network, disk, hashing,
and allocation time. Each DAN group request currently launches a new runtime.

**Repeated full-model streaming during user requests is NOT the intended DAN
production architecture.** Model preparation must be amortized across many
requests. Otherwise download/transfer cost dominates latency, wastes bandwidth,
increases rental expense, and makes ordinary interactive use impractical.

## Implemented lifecycle and boundary

For a Linux gaming PC, `dan-provider` now performs the steps before `Join`: load
minimal local config, detect NVIDIA GPU/index/VRAM with `nvidia-smi`, subtract the
configured reserve, load or create a persistent random identity, select a private
RPC endpoint, and enter the existing managed-provider lifecycle. Coordinator or
network loss reconnects with capped backoff and re-advertises verified cache.

```text
provider joins
  -> capabilities measured
  -> model/shard assignment
  -> shard downloaded ahead of inference
  -> hash verified
  -> cached on disk
  -> optionally loaded into VRAM
  -> provider advertises cached/loaded state
  -> scheduler forms execution target
  -> many inference requests reuse weights
```

1. Join: provider identity, capabilities and cached inventory now register on the
   existing connection. Identity remains self-reported; sessions/leases and
   authorized connectivity are not implemented.
2. Measure: report available GPU VRAM, RAM, disk, backend revision, and network
   reachability. Distinguish total capacity from capacity reserved by other jobs.
3. Assign: the coordinator loads one versioned `dan-main` manifest and assigns
   its arbitrary shard collection one-per-provider. The manifest includes
   quantization, compatible runtime, context, tensor mapping, size,
   and cryptographic checksums. Placement is a control-plane operation.
4. Prepare: `managed_provider` copies local/`file://` bytes or downloads HTTP(S)
   bytes before user work, verifies optional length and SHA-256, and atomically
   publishes the completed cache entry. Failed verification becomes `ERROR`.
5. Cache: verified bytes remain under model/version/shard/hash identity. Startup
   scans and re-hashes inventory. HTTP downloads retain a stable partial file,
   resume it across restarts, and report size/speed progress. Eviction is not implemented.
6. Load: `LOAD_SHARD` starts one owned llama.cpp RPC worker and polls its TCP
   endpoint. It reports `READY` only after the child is alive and reachable.
7. Advertise: v1 tracks `UNASSIGNED`, `ASSIGNED`, `DOWNLOADING`, `CACHED`,
   `LOADING`, `READY`, and `ERROR`, plus online/offline heartbeat state. Cached
   is not loaded. Busy and draining control states remain future work.
8. Form target: the replica is `READY` only when every required shard has an
   online `READY` provider. The coordinator then starts one persistent server and
   separately health-gates runtime `READY`.
9. Serve: `/model dan-main` sends sequential HTTP completion requests to the same
   server PID. Request traffic does not restart workers, reacquire artifacts, or
   intentionally redistribute the model.
   Session/KV-cache ownership is separate from weight residency.
10. Recover: withdraw readiness on a lost worker, stop the runtime, move only the
    missing shard to an eligible spare, run normal preparation, and automatically
    recreate the runtime after every required provider is ready.

Automatic Provider Replacement / Reassignment v1 implements step 10 for the same
one replica. Provider loss stops the runtime and rejects requests while recovery
selects and prepares a spare. Runtime startup resumes automatically after replica
readiness. Leases, in-flight replay, proactive rebalancing, and multiple replicas
remain unimplemented. Manual groups remain a separate process-per-request path.

Assignments are a collection, not named A/B slots. Online unassigned providers
are spares. For each missing shard the coordinator prefers an exact
model/version/shard/hash cache match, then highest sufficient VRAM, then provider
ID. Offline ownership is released while sticky history remains. A reconnect
reclaims an unowned shard through the normal selection path, but never evicts a
healthy replacement. A provider reporting `ERROR` is excluded for that shard
until reconnect; another candidate is tried if available.

## Bridge experiments versus production design

RPC `--cache` stores large tensor payloads on each worker and can satisfy later
loads from disk. It still needs initial population, a client-side GGUF and tensor
metadata, hashing, allocation, and disk-to-VRAM loading after runtime restart.
It is not a portable, ahead-of-time shard distribution protocol. The pinned
implementation uses a 64-bit FNV cache key, not a cryptographic shard manifest;
do not describe it as production integrity verification. The experiment verifies
the source GGUF's SHA256 and records local cache file hashes for diagnostics.

A persistent `llama-server` now supplies DAN's managed serving path. Local tests
prove process reuse and automatic replacement behavior; real CUDA tensor-transfer
reuse remains a hardware question.

Gamer Provider Testnet v1 is implemented locally. Next run the first real friends
test using [FRIENDS_TESTNET.md](FRIENDS_TESTNET.md) and select follow-up work from
measured usability, GPU, network, and recovery evidence. The
[staged rental runbook](NEXT_GPU_EXPERIMENT.md) remains the detailed real-runtime
measurement reference. Aggregate-VRAM proof remains a separate question.
