# Provider-Owned Execution Prototype v0

Status: feasibility inspection complete; implementation and measurements in progress.

This experiment is isolated from DAN's existing llama.cpp RPC runtime. The RPC
path remains the supported testnet baseline.

## Pinned source inspected

- llama.cpp commit: `95ef7fc16054e63b427a3ef00188e055ef7586d8`
- DAN runtime entry points: `src/coordinator.cpp`, `src/managed_provider.cpp`,
  `src/distributed_runtime*.cpp`, and `src/persistent_runtime*.cpp`
- llama.cpp model loader: `src/llama.cpp`, `src/llama-model-loader.cpp`, and
  `src/llama-model.cpp`
- Qwen2 graph: `src/models/qwen2.cpp`
- graph inputs/results: `src/llama-graph.h` and `src/llama-graph.cpp`
- KV cache: `src/llama-kv-cache.h` and `src/llama-kv-cache.cpp`
- GGUF reader: `ggml/src/gguf.cpp`
- RPC backend/server: `ggml/src/ggml-rpc/ggml-rpc.cpp`

Line numbers below refer to that pinned llama.cpp commit.

## Feasibility findings

1. **Where tensors load.** `llama_model_load_from_file_impl` in
   `src/llama.cpp:379` creates `llama_model_loader`. Its constructor calls
   `gguf_init_from_file` and indexes every tensor's metadata in
   `src/llama-model-loader.cpp:563-594`. Architecture-specific
   `load_arch_tensors` creates the selected runtime tensors, and
   `llama_model_loader::load_all_data` at `src/llama-model-loader.cpp:1486`
   maps or copies their bytes into backend buffers.

2. **Layer representation.** `llama_layer` in `src/llama-model.h:251` holds
   the attention, feed-forward, normalization, and architecture-specific tensor
   pointers for one transformer block. `llama_model::layers` is a vector at
   `src/llama-model.h:688`. Qwen2 creates one entry per block in
   `src/models/qwen2.cpp:29-42` and executes them in order at lines 65-124.

3. **Partial model construction.** The public API cannot request a contiguous
   layer subset. Stock Qwen2 unconditionally creates the embedding, head, and
   every layer tensor. Internally, partial construction is practical:
   `llama_model_loader::done_getting_tensors(bool partial)` already permits
   intentionally unused file tensors at `src/llama-model-loader.cpp:1385`, and
   `load_all_data` loads only tensors created in the runtime GGML contexts.
   A small architecture-specific load filter is therefore sufficient.

4. **External hidden-state input.** `llama_batch` already accepts FP32
   embeddings and `build_inp_embd` creates an input tensor for them at
   `src/llama-graph.cpp:2303`. Stock graphs still construct the token-embedding
   branch and begin at layer zero. A stage-two graph must select the existing
   embedding input directly and start its loop at the assigned layer.

5. **Stopping at a layer.** Stock Qwen2 always runs through final norm and the
   output head. `llm_graph_result` can expose an embedding tensor and marks it as
   an output in `src/llama-graph.cpp:1353-1370`; context decode copies that output
   to host memory at `src/llama-context.cpp:1860-1900`. A stage-one graph can set
   its last hidden state as `t_embd` and make that tensor the graph root.

6. **KV layout.** `llama_kv_cache` creates one K and V tensor per retained
   layer in `src/llama-kv-cache.cpp:164-240`. Its `layers` vector records the
   original model layer ID and `map_layer_ids` resolves that ID during attention.

7. **Provider-local KV.** The KV constructor already accepts a `layer_filter_cb`
   and skips rejected layers at `src/llama-kv-cache.cpp:169-176`. The normal
   model memory path passes such filters at `src/llama-model.cpp:2568-2663`.
   Applying the experimental stage range as that filter lets each independent
   context allocate and update only its own layers. No coordinator KV is needed.

8. **Required llama.cpp changes.** The proof needs four focused changes:
   experimental stage-range model parameters; Qwen2 tensor creation restricted
   to that range plus the embedding or head appropriate to the stage; a Qwen2
   graph that starts from tokens or an external hidden state and stops at the
   requested boundary; and a stage-range KV filter. A small stage-worker program
   then uses ordinary llama.cpp decode, tokenizer, output, and backend APIs.

9. **Wrapper or fork.** This cannot be a public-API-only DAN wrapper because
   model construction and architecture graph boundaries are internal. It does
   not require extracting transformer math or a new framework. The credible
   path is a small, pinned Qwen2-only llama.cpp patch plus an isolated DAN worker.
   The patch should remain reviewable and rebased with the pinned revision; it is
   not a general upstream-quality layer-partition API.

## Decision

Proceed with a Qwen2-family model, exactly two providers, one fixed contiguous
split, one request at a time, and greedy decoding. Provider A owns the token
embedding and early blocks. Provider B owns the remaining blocks, final norm,
output head, and sampling. Activations are transferred as validated FP32 arrays.

The first execution proof may keep a duplicate original GGUF on provider disks
while each process loads only its assigned tensors. If used, the result will be
reported exactly as:

```text
provider-owned execution: PROVEN/NOT PROVEN
provider-owned storage: NOT YET PROVEN
```

True per-stage GGUF artifacts are a follow-up only after correct execution.

## Current RPC comparison

The existing `ggml-rpc-server` is model-unaware. The coordinator-side model
loader creates tensors and `ggml_backend_rpc_buffer_set_tensor` sends their byte
contents with `RPC_CMD_SET_TENSOR` (`ggml/src/ggml-rpc/ggml-rpc.cpp:699-723`).
The server reconstructs submitted GGML graph nodes in `rpc_server::create_node`
and executes them in `rpc_server::graph_compute` at lines 1625-1749. Its optional
hash cache avoids repeat uploads but does not make providers owners of model
semantics or GGUF loading. This is why the new worker uses a separate protocol.

## Acceptance record

### Local two-process result (2026-09-07)

```text
Model: Qwen2.5 0.5B Instruct
Quantization: Q4_K_M
Layers: 24
Split: 12 / 12

Provider A device: CPU (CUDA Toolkit unavailable on this machine)
Provider A layers: embedding + 0..11
Provider A KV: 3.00 MiB, exactly 12 layers
Provider A tensors created: 145 / 291

Provider B device: CPU (CUDA Toolkit unavailable on this machine)
Provider B layers: 12..23 + final norm + tied output head
Provider B KV: 3.00 MiB, exactly 12 layers
Provider B tensors created: 146 / 291

Coordinator model storage/API dependency: 0 / metadata only
Prompt: The capital of France is
Prompt tokens: 5
Generated tokens: 20

Distributed output:
 Paris. It is the largest city in Europe and the second largest in the world. It is located

Ordinary llama.cpp output (same prompt, model, context, greedy sampling):
 Paris. It is the largest city in Europe and the second largest in the world. It is located

Rendered greedy output comparison: exact for all 20 generated token positions
Distributed prefill: 136.892 ms
Local full-model prefill: 172.870 ms
Distributed decode: 43.659 tokens/s
Local full-model decode: 51.160 tokens/s
Distributed slowdown: 14.66%

Stage A decode compute: 8.570 ms/token mean
A-to-B activation: 3,584 bytes/token (896 FP32 values)
Coordinator-routed activation transfer: 0.272 ms/token mean on loopback
Stage B decode and greedy sample: 13.787 ms/token mean
Total distributed decode: 22.905 ms/token mean

Prefill activation: 17,920 bytes (5 x 896 x 4)
Decode activation total after prefill: 68,096 bytes (19 x 3,584)
```

The coordinator process accepted only the JSON manifest, provider endpoints,
and prompt. It had no model-path option and never opened a GGUF. Activations
were received from provider A and forwarded as opaque framed bytes to provider
B. Both provider contexts remained alive for the full generation.

The first proof used two independent processes and separate provider cache
paths on one Windows machine. It did not use two physical machines, CUDA, or a
WAN/LAN link because the second Tailscale machine was offline and this machine
has an NVIDIA driver but no CUDA Toolkit/compiler. Therefore its timing is an
implementation baseline, not the requested network/GPU benchmark.

### Storage result

```text
provider-owned execution: PROVEN locally (two processes)
provider-owned storage: NOT YET PROVEN
physical two-provider acceptance: NOT YET RUN
```

Both providers currently store the complete 491,400,032-byte GGUF. The patched
loader creates only its stage tensors; it does not create or execute the other
stage's block tensors. Stage B also needs `token_embd.weight` because this model
ties that tensor to the output head. True storage sharding needs a GGUF writer
that copies shared metadata and only these selected tensors into two valid
artifacts, with the tied embedding intentionally present in both artifacts.

### Protocol and failure behavior

Frames have a fixed 40-byte network-order header containing magic, version,
message type, request ID, sequence position, rows, columns, dtype, and payload
length. Activations are FP32 little-endian `[tokens, 896]` with an eight-byte
compute-duration prefix. Payloads are capped at 64 MiB before allocation; shape,
dtype, exact byte count, request ID, and sequence position are validated. A
disconnect or invalid response ends the one-shot coordinator with
`experimental runtime unavailable: ...`; it does not affect the DAN testnet.

### Regression and portability status

- The unmodified/default llama.cpp path built and generated the exact local
  baseline after the patch (stage fields default to disabled).
- DAN's Windows Release build succeeded and all 6 CTest tests passed.
- The separate protocol/manifest test has 4 passing cases.
- The Linux-oriented Python integration suite was not a valid Windows run: it
  expects executables at Unix `build/coordinator` paths. Those resulting 18
  missing-file errors and one unavailable-Python child error are environmental,
  not product regressions.
- Existing DAN RPC source and packaging were not changed.
- The experimental worker compiled on Windows/CPU. Its source has Windows and
  POSIX sockets and the CMake path enables CUDA, but Linux and CUDA builds remain
  unverified.

## Critical questions answered

1. **Can providers execute contiguous stages?** Yes, locally: A executed 0..11
   and B executed 12..23 plus the head, with an exact 20-token match.
2. **Can the coordinator omit the full GGUF?** Yes in this runtime; it uses only
   a small JSON manifest and routes frames.
3. **Does each provider own only local KV?** Yes: logs show 12 filtered-in layers
   and 3.00 MiB of KV on each process.
4. **Weights or activations during inference?** Only activations cross this
   protocol. Each provider loads weights from its local file before listening.
5. **Bytes per decoded token?** 3,584 payload bytes, plus an 8-byte timing field
   and a 40-byte frame header between each hop.
6. **WAN/LAN penalty?** Unknown; the measured loopback routing component was
   0.272 ms/token.
7. **Compute, bandwidth, or latency bound?** Locally compute-bound: stage compute
   was 22.357 ms/token versus 0.272 ms routed transfer. WAN classification awaits
   the physical test.
8. **Architecturally viable?** The two-stage execution mechanism is viable.
   Network viability is not established until the physical CUDA run.
9. **What prevents arbitrary N?** Model graph stage generalization, direct
   provider-to-provider routing, per-stage artifact generation, topology-aware
   scheduling, and request lifecycle/recovery are intentionally absent.
10. **What is needed for physical shards?** Copy GGUF metadata and only each
    stage's named tensor records/data, preserve alignment/types, duplicate tied
    tensors where semantically required, then hash and assign each artifact.
11. **llama.cpp modifications?** Two experimental model parameters, partial-load
    acknowledgement, a KV range filter, a direct hidden-state graph input, and a
    Qwen2 layer-range tensor/graph path.
12. **Maintainable patch?** Yes for this pinned Qwen2 experiment: a small patch
    across seven llama.cpp files. It is not yet a general
    upstream API.
13. **Next experiment?** Build the same worker with CUDA on the RTX 2070 and the
    second provider, run the exact 20-token case over Tailscale, record GPU/VRAM,
    RTT and stage timings, then decide whether to build physical GGUF shards.

Recommended commit message: `Prototype provider-owned two-stage Qwen inference`
