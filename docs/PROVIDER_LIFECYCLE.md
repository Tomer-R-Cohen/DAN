# Provider lifecycle and weight reuse

## Current evidence and limits

DAN has persistent whole-model providers, model-compatible scheduling, and
manually configured distributed groups. The A40 whole-model run completed ten
requests with a loaded runtime. The reported RTX 3090 + RTX A4500 Qwen2.5 and
Qwen3 tests produced responses through both distributed entry points. See
[STATE.md](STATE.md), [small-model report](TWO_GPU_SMOKE_REPORT.md), and
[Qwen3 report](QWEN3_TWO_GPU_REPORT.md). The two-pod summaries were supplied by
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

## Intended lifecycle (not yet implemented for distributed providers)

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

1. Join: assign provider identity and session/lease; establish authorized
   connectivity. Current DAN metadata is self-reported, not measured attestation.
2. Measure: report available GPU VRAM, RAM, disk, backend revision, and network
   reachability. Distinguish total capacity from capacity reserved by other jobs.
3. Assign: coordinator chooses a versioned model manifest and shard layout,
   including quantization, compatible runtime, context, tensor mapping, size,
   and cryptographic checksums. Placement is a control-plane operation.
4. Prepare: download assigned bytes before dispatching user work. Use resumable
   temporary files, verify length and SHA256 against the manifest, and atomically
   publish the completed cache entry. Failed verification never becomes ready.
5. Cache: keep verified bytes on durable disk, indexed by model/shard identity
   and checksum. Define retention/eviction; a pod's writable layer is not a
   durability guarantee. A replacement pod must revalidate its inventory.
6. Load: optionally reserve VRAM and load the assigned tensors into a persistent
   runtime. Readiness includes successful worker connectivity and warmup.
7. Advertise: distinguish assigned, downloading, verified-on-disk, loading,
   loaded/ready, busy, failed, and draining states. Cached is not loaded.
8. Form target: scheduler verifies all required shards and compatible runtimes,
   reserves resources, and admits requests only after the whole group is ready.
9. Serve: keep weights loaded across many requests. Request traffic contains
   inputs, activations, and results; routine requests do not redistribute weights.
   Session/KV-cache ownership is separate from weight residency.
10. Recover/drain: withdraw readiness on lost workers, stop new dispatch, handle
    in-flight failures explicitly, retain verified disk state, and reload before
    re-admission. Resource leases prevent double allocation across groups.

These states, manifests, prefetch, measured capabilities, leases, and cache-aware
scheduling are design targets. No new protocol fields or implementation are
claimed by this document. Current groups do not automatically form or advertise
cached/loaded shard state; current registry memory/context fields are metadata.

## Bridge experiments versus production design

RPC `--cache` stores large tensor payloads on each worker and can satisfy later
loads from disk. It still needs initial population, a client-side GGUF and tensor
metadata, hashing, allocation, and disk-to-VRAM loading after runtime restart.
It is not a portable, ahead-of-time shard distribution protocol. The pinned
implementation uses a 64-bit FNV cache key, not a cryptographic shard manifest;
do not describe it as production integrity verification. The experiment verifies
the source GGUF's SHA256 and records local cache file hashes for diagnostics.

A persistent `llama-server` can separately test runtime reuse through RPC without
editing DAN. Success proves that backend configuration, not persistent DAN group
integration. Integrating that runtime with DAN scheduling is a later source task.

Next execute [the staged rental runbook](NEXT_GPU_EXPERIMENT.md): uncached versus
cold/warm disk cache, worker restart with retained cache, current DAN group
relaunch, then ten requests through one persistent server. Aggregate-VRAM proof
remains a separate subsequent milestone; neither caching nor persistence proves it.
