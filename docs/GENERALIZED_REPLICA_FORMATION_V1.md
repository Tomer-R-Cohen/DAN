# Generalized Replica Formation v1

Status: implemented and locally validated on Windows CPU and CUDA.

## Architecture

`dan-stage-worker` can now start without a model or layer range. It connects to
the provider-owned coordinator, registers its stable ID, GPU name, and offered
VRAM, and waits in `AVAILABLE`. The coordinator range-reads the selected GGUF's
metadata only, calculates the exact tensor bytes for every contiguous Qwen2
stage, and chooses the smallest provider count that fits.

Planning subtracts per-stage FP16 KV memory plus conservative runtime headroom:
the larger of 1 GiB or 15% of offered VRAM. Candidate provider orders and layer
boundaries are searched for the first valid plan, beginning with two providers.
The coordinator sends every assignment before waiting so provider range
downloads happen concurrently.

Each selected provider uses the existing verified sparse cache, loads only its
assigned stage, and reports `stage_ready`. Inference then routes through a
runtime-sized ordered stage list:

```text
tokens -> first stage -> activation -> zero or more middle stages
       -> activation -> final stage -> sampled token
```

Session lifecycle, KV commits, and request completion are sent to every stage.
A provider failure makes the replica unavailable and fails pending work; v1
does not attempt KV rollback or stage replacement.

## Automatic run

Providers may start before the coordinator; they wait and register when its
control listener becomes available. Start the coordinator with:

```powershell
dan-provider-owned-coordinator.exe `
  --manifest config/provider-owned-qwen2.5-0.5b-q4km.json `
  --provider-listen 0.0.0.0:50200 `
  --metadata-cache build/model-index.tmp `
  --prompt "The capital of France is" --tokens 20 --shutdown-workers
```

Every Windows provider starts with no model or layer arguments:

```powershell
dan-stage-worker.exe `
  --coordinator COORDINATOR_IP:50200 `
  --provider-id PROVIDER_ID --cache-dir build/provider-cache
```

The Linux command uses the same arguments and executable name. CUDA workers use
the CUDA build; GPU identity and currently free VRAM are detected through the
llama.cpp backend. `--gpu` and `--vram-mib` can override detection for tests or
a deliberately smaller contribution. `--gpu-layers 999` remains the default.

Change only `--manifest` to select another supported pinned single-file dense
Qwen2 GGUF. DAN derives its architecture, layer count, hidden size, attention
layout, tensor sizes, and stage split from the remote GGUF metadata. A minimal
selection contains:

```json
{
  "model_id": "qwen2.5-1.5b-instruct-q4-k-m",
  "context_size": 512,
  "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
  "gguf_filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
  "artifact_sha256": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
  "artifact_revision": "91cad51170dc346986eccefdc2dd33a9da36ead9"
}
```

An explicit HTTPS `artifact_url` remains supported instead of `hf_repo` plus
`gguf_filename`. Optional `architecture`, `layers`, and `hidden_size` fields
act only as assertions against discovered metadata. Existing sessions cannot
cross a model change.

## Validation

- Windows CPU and CUDA builds completed.
- Protocol, range-model, and formation tests: 3/3 passed in both builds.
- Existing explicit two-stage inference still generated 20 tokens.
- Automatic local three-stage CPU inference generated the same 20 token IDs as
  the explicit two-stage CPU run.
- Automatic local three-stage CUDA inference exactly matched the frozen CUDA
  result for `The capital of France is`: ` Paris. It is the largest city in
  Europe and the second largest in the world. It is also`.
- CUDA planner assignment was runtime-generated as layers `0`, `1..14`, and
  `15..23`; activation traffic correctly reported 7,168 bytes per decode token
  across two stage boundaries.
- The first-stage embedding was confirmed in the CUDA model buffer rather than
  silently consuming host RAM.
- Configuration-only model switching was exercised with Qwen2.5 1.5B Q4_K_M:
  metadata discovery found 28 layers and hidden size 1,536, two generic CUDA
  providers were assigned layers `0` and `1..27`, and 20 tokens completed at
  93.355 decode tok/s. The coordinator retained metadata only.
- The 1.5B run reported 6,144 activation bytes per decoded token, 1.246 ms of
  first-stage compute, 0.744 ms of local transport, and 8.722 ms of final-stage
  compute per token. Both workers loaded once and shut down cleanly.
- The 1.5B configuration was reduced to model source, immutable identity/hash,
  and context size; it no longer supplies architecture dimensions or a split.
  A cached CUDA run using that reduced configuration reproduced the same
  20-token output while automatically discovering 28 layers and width 1,536.

## Remaining assumptions

- Dense Qwen2 decoder models with standard GGUF tensor naming only. Other
  architectures and Qwen2 MoE/multimodal variants are rejected before planning.
- One active single-file GGUF and one replica.
- At least two and at most eight providers; contiguous stages only.
- Context and provider session pool are currently configured as 512 and eight
  by the model/formation path.
- Greedy sampling occurs on the final provider.
- FP32 little-endian activations and a 64 MiB frame limit.
- Model switching requires draining current work and reconnecting/reforming the
  replica; there is no zero-downtime cutover.
- Provider registration is now native to the provider-owned coordinator. The
  former RPC registry is not used by this path.
- Multipart GGUF, cache reuse across repartitioning, provider replacement,
  network-aware placement, arbitrary architectures, and authentication remain
  deferred.

Fully automatic formation still needs GPU discovery wired into the packaged
provider launcher, durable coordinator model selection, reconnect/reformation
after provider loss, and a user-facing model-selection surface.
