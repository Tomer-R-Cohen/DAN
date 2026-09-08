# Persistent Provider-Owned Runtime v1

Status: implemented and locally accepted on Windows. The v1 physical
Windows/Linux CUDA rerun was intentionally skipped on 2026-09-08 at the user's
request; the earlier v0 physical proof remains documented separately.

## Architecture

The C++23 `dan-provider-owned-coordinator` connects once to both persistent
workers and routes validated frames. It receives Provider A's FP32 hidden state
and forwards it to Provider B. It has no model argument, does not link llama.cpp,
and never opens a GGUF.

Each `dan-stage-worker` loads its partial Qwen2 stage once. A session creates an
independent llama context, sampler where needed, position counter, and KV cache.
Up to `--max-sessions` contexts remain resident, while a worker-wide active
session guard permits only one executing request. Stateless requests create and
destroy a context around one request. Persistent requests keep the context and
commit the final sampled token to both providers before the next follow-up.

The original pinned llama.cpp patch and Qwen2.5 0.5B Q4_K_M 12/12 split are
unchanged. Both providers still store the complete GGUF but create/load only
their assigned runtime tensors. The existing llama.cpp RPC runtime was not
modified.

Maintained sources now live under `src/provider_owned`,
`include/provider_owned`, and `tests/provider_owned`; the patch is under
`patches`, the manifest under `config`, and the build helper under `scripts`.
The superseded Python v0 implementation was removed and remains recoverable
from its frozen Git commit.

## Protocol and lifecycle

Protocol v2 uses a fixed 48-byte network-order header:

```text
magic, version, type, session_id, request_id, position,
rows, columns, dtype, reserved, payload_bytes
```

It supports create, reset, destroy, prompt, token, activation, result,
end-request, metrics, final-token commit, and graceful shutdown frames. Payloads
are capped at 64 MiB before allocation. Frame type, reserved bits, session and
request IDs, monotonic request ordering, sequence position, tensor shape, FP32
dtype, and exact payload length are checked. A malformed frame closes that
coordinator connection. Any coordinator disconnect discards its sessions, so
partially advanced KV and orphaned inactive contexts cannot be reused.

## Local acceptance results

Environment: Windows, CPU execution, Qwen2.5 0.5B Instruct Q4_K_M, context 512.
This validates persistence and session behavior without replacing the earlier
two-machine CUDA performance record.

### 100 sequential stateless requests

```text
Requests: 100 / 100 passed
Generated tokens: 2,000 (20 each)
Deterministic output matches: 100 / 100
Worker A PID: 17472 before and after
Worker B PID: 14192 before and after
Model reload count: 1 per worker

Worker A cold model startup: 547.753 ms
Worker B cold model startup: 560.114 ms
Warm request latency p50: 594.348 ms
Warm request latency p95: 645.071 ms
Decode: 37.076 tokens/s
Provider A decode compute: 10.009 ms/token
Loopback network/routing: 0.645 ms/token
Provider B decode compute/sample: 16.318 ms/token
Activation: 3,584 bytes/token
```

All requests produced:

```text
 Paris. It is the largest city in Europe and the second largest in the world. It is located
```

The workers had already served six lifecycle/correctness requests before the
100-request run, so their final metrics reported 106 cumulative requests while
the acceptance report itself contains exactly 100 requests and 2,000 tokens.

### Session checks

- Two simultaneously resident sessions independently produced the exact
  20-token baseline; neither inherited the other's KV state.
- Resetting and reusing one resident session produced the initial deterministic
  output again.
- A persistent two-request session generated ` Paris. It is the largest city in`
  and then, after appending `. The capital of Germany is`, generated
  ` Berlin. It is the second largest city`.
- Ordinary full-model llama.cpp given the equivalent combined context generated
  the exact same second output.
- Graceful shutdown returned acknowledgements from both workers, then both
  listening PIDs exited cleanly.

### Windows CUDA smoke test

The final C++23 targets were also rebuilt with CUDA 13.3 for compute capability
7.5 and run on the RTX 2070. Two simultaneously resident sessions each matched
the frozen all-CUDA 20-token output. The first request included CUDA graph
warmup; the second completed in 131.345 ms. Across both requests, decode was
98.704 tokens/s with 4.021 ms/token on A, 0.585 ms/token loopback routing, and
5.525 ms/token on B. Both workers again reported one model load and shut down
cleanly. This is a local smoke test, not a replacement for the skipped physical
Windows/Linux v1 benchmark.

## Tests and preservation

- C++ protocol test: passed. It covers v2 field round-trip, session/request IDs,
  oversized payload rejection, unknown type rejection, and control payload
  validation.
- Provider-owned C++23 worker and coordinator: Release build passed with the
  pinned patched llama.cpp dependency.
- Existing Windows DAN/RPC Release build: passed.
- Root DAN CTest suite: 7/7 passed (the new protocol test plus the existing six
  RPC/application regressions).
- Linux source path remains POSIX-compatible, but v1 was not rebuilt or rerun on
  Linux after the user chose to skip that test.

## Remaining limits

- Exactly two fixed Qwen2 stages and one active request.
- No batching, simultaneous execution, arbitrary-N stages, scheduler v2,
  replacement, KV migration, authentication, rewards, or crypto.
- Session contexts consume their own compute/KV buffers; the configured resident
  session cap must fit provider memory.
- Both providers retain the full GGUF storage artifact.
- Physical v1 p50/p95, CUDA VRAM, Tailscale RTT/path, and WAN decode measurements
  are not claimed because the physical rerun was skipped.

Recommended next milestone: multiple concurrent resident sessions with a small
bounded request queue, then measure whether batching or pipeline overlap reduces
per-token latency before generalizing beyond two stages.
