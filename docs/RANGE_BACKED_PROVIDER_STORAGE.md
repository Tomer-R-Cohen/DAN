# Range-Backed Provider Model Storage

## Result

DAN providers can now build a stage-specific sparse GGUF directly from the
pinned Hugging Face file. The GGUF metadata and owned tensor byte ranges remain
at their original offsets; unowned regions are sparse holes. The coordinator
does not download or open model weights.

The source is pinned to revision
`9217f5db79a29953eb74d5343926648285ec7e67` and SHA-256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
HTTP range responses are validated against their `Content-Range`, repository
commit, linked object hash, and total size. Each cached range has a local
SHA-256 in the sidecar; missing, truncated, stale, or corrupt owned ranges are
rejected and rebuilt. DAN never silently falls back to a complete download.

## Windows acceptance

The Qwen2.5 0.5B Q4_K_M full file is 491,400,032 bytes with 291 tensors.

| | Provider A | Provider B |
|---|---:|---:|
| Stage | embedding + layers 0–11 | layers 12–23 + output |
| Logical sparse size | 491,400,032 B | 491,400,032 B |
| Physical disk usage | 225,705,984 B | 272,105,472 B |
| HTTP bytes downloaded | 231,771,488 B | 278,159,200 B |
| Full-file percentage downloaded | 47.17% | 56.61% |
| Tensors present/loaded | 145 | 146 |
| Shared tensor bytes | 0 | 0 |
| Restart cache | reused | reused |

The download count includes the bounded metadata/index probes. This model has a
separate output weight, so no tied embedding tensor is duplicated. `fsutil
sparse queryrange` confirmed four allocated regions per provider with large
unassigned holes remaining sparse before and after inference.

The stage-aware llama.cpp loader required one additional scoped change:
whole-file virtual-memory prefetch is disabled only while DAN stage selection
is active. Without it, Windows materialized the sparse holes even though llama
loaded only owned tensors. Normal llama.cpp and DAN's legacy RPC path retain
their original prefetch behavior.

## Correctness and recovery

- Provider A loaded 145/291 tensors and Provider B loaded 146/291.
- Both workers retained only their 12 layers of KV state.
- The deterministic 20-token output exactly matched the established full-model
  baseline: ` Paris. It is the largest city in Europe and the second largest in
  the world. It is located`
- A deliberately corrupted Provider A range and a truncated Provider B file
  were both detected and rebuilt.
- A clean restart reused both caches without downloading.
- Final local run: 35.66 decode tok/s, 10.33 ms A compute/token, 0.41 ms local
  transport/token, 17.31 ms B compute/token, and 3,584 activation bytes/token.
- Windows CTest: 8/8 passed, including provider-owned protocol/range tests and
  existing provider/coordinator regressions.

Linux uses `ftruncate` and allocated-block accounting, but was not rerun for
this milestone because the physical Linux provider was unavailable. Physical
CUDA/Tailscale validation remains the next required cross-platform check.

This proves provider-specific storage for the fixed two-stage Qwen model. It
does not yet add arbitrary-N stages, batching, pipeline overlap, pre-generated
shards, scheduling, provider replacement, or KV migration.
