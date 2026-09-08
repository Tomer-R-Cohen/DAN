# Concurrent Multi-Session Serving v2

## Result

The C++23 provider-owned runtime now accepts concurrent clients through a
bounded, fair request queue while executing exactly one distributed request at
a time. Provider stages stay loaded, session KV remains provider-owned, and the
coordinator remains model-free. The llama.cpp RPC runtime is unchanged.

The physical Windows RTX 2070 to Linux RTX 2000 Ada rerun was skipped because
the Linux provider was unavailable. The completed acceptance run was local
Windows CPU execution of the same Qwen2.5 0.5B Q4_K_M 12/12 stage split.

## Architecture

- Client TCP connections are persistent and may carry multiple framed requests.
- A fixed client-handler pool submits work to a bounded queue.
- The queue stores a FIFO per session and round-robins ready sessions. Stateless
  requests use their request ID as the fairness key.
- One executor owns both provider connections, so stage execution remains
  serialized and provider frame ordering cannot race.
- Cancellation and timeout remove queued work. Active inference is not
  interrupted in this milestone.
- A provider disconnect marks the replica unavailable and fails queued work.

The stage workers reuse llama.cpp's native multi-sequence model: one
`llama_context` per worker, `n_seq_max` slots, per-token `llama_seq_id` in each
`llama_batch`, and `llama_memory_seq_rm` for reset/destruction. DAN does not
reimplement KV storage or batch tensor construction. llama.cpp server-style
continuous batching was studied but deliberately not enabled because v2 allows
only one active request. llama.cpp backend pipeline parallelism is an
in-process multi-device scheduler and does not replace DAN's cross-machine
activation link.

## Acceptance result

Configuration:

- 20 concurrent persistent client connections
- 50 resident sessions
- 1,000 mixed stateless/persistent requests
- one generated token per load-test request; separate probes generated 20 tokens
- queue capacity 16
- per-session context 64 tokens, 64 native llama sequence slots

Result:

```text
completed=1000
correctness failures=0
queue max=16
queue-full probe=passed (15 explicit rejections)
queued cancellation=passed
queued timeout=passed
resident sessions=50
KV memory estimate=157286400 bytes
model reload count=1 per worker
reference first token=" Paris"
p50 request latency=1529.96 ms
p95 request latency=1809.35 ms
p99 request latency=3704.05 ms
mean queue wait=1273.44 ms
mean time to first token=1356.83 ms
aggregate generated tokens/s=11.058
```

The workers remained alive for all 1,019 successful inference requests
(acceptance workload plus probes), loaded their model stages once, retained
independent sequence KV, and shut down gracefully. Reset/fresh persistent
sessions and stateless requests produced the same deterministic first token.
Backpressure retries are counted as failed submissions by coordinator metrics;
they are expected explicit `queue_full` responses, not lost accepted work.

Provider A compute, network, Provider B compute, and activation bytes per
generated token are exported in coordinator metrics. A 97-request CPU smoke
sample measured 9.65 ms Provider A compute, 0.38 ms local transport, and 15.35
ms Provider B compute per stage step. Its aggregate traffic was 8,128 bytes per
generated token because that figure includes prompt-prefill activations. A
single-token stage transfer remains 896 FP32 values, or 3,584 bytes.

## Verification

- Provider-owned protocol/fairness test: passed.
- Full Windows CTest suite: 7/7 passed, including normal provider checks and
  coordinator regressions.
- Persistent TCP load exposed and fixed short-lived connection port exhaustion.
- Interrupting Provider B during load returned explicit failures, drained the
  queue, marked the replica unavailable, and left the coordinator alive.
- Physical CUDA/Tailscale v2 test: skipped; Linux provider unavailable.

## Remaining limits

There is still one active request, two fixed stages, no GPU batching or pipeline
overlap, no active-request cancellation, no provider replacement/KV migration,
and each provider still stores the full GGUF while loading only its stage.
Authentication remains the trusted-Tailscale boundary.

The runtime is ready for the next experiment: reuse llama.cpp continuous-batch
batch/sequence construction across several active sessions, then overlap the
two DAN stages. Measure first; do not add arbitrary-N scheduling yet.
