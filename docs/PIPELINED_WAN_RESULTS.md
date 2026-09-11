# Pipelined Speculative Decoding: First Real-WAN Result

Date: 2026-09-11

Status: **pipelining over a real WAN link passed and shows a real speedup**.
This is the first time [PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md)'s
design has been measured over an actual network between two different
machines rather than CPU loopback or one local GPU. A real bug in the
pipelined path's own metrics was found and fixed in the course of this test
(see below) — the numbers here are from the fixed binary, not the one the
test started with.

## System under test

```text
Windows RTX 2070 (6656-8191 MiB card, offered capped to 2000 MiB for this test)
  |
  | coordinator + draft model, connects to Linux over SSH port forward
  | (dan-sidecar/libp2p was set up but abandoned for this run -- the rented
  |   GPU had no port exposed for it; plain SSH -L was simpler and worked)
  v
RunPod rented Linux box, NVIDIA RTX 4000 Ada (20019 MiB card, offered capped
  to 2000 MiB for this test)
```

Both providers registered via the coordinator's auto-registration path
(`--coordinator`/`--provider-listen`), not the fixed-address `--provider`
flow. Offered VRAM was deliberately capped on both sides (`--vram-mib 2000`)
so the planner was forced into a genuine 2-provider split instead of fitting
the small model on one GPU alone.

## Model

- Qwen2.5-1.5B-Instruct Q4_K_M (`config/provider-owned-qwen2.5-1.5b-q4km.json`)
- 28 layers, hidden size 1536, context 1024
- Split: `linux-local` layers 0-2 (0.21 GiB), `windows-remote` layers 3-27
  (0.84 GiB)
- Draft model: the same weights, loaded locally by the coordinator
  (`--draft-model`) — not a separate smaller model. A dedicated draft model
  would likely show a different, probably better, real acceptance rate;
  that's out of scope here.

## A real bug found and fixed during this test

The first pipelined run reported `decode_tok_s=261.033` with
`A_ms/token=0.000` and `network_ms/token=0.000`. That was wrong, not fast:
`generate_pipelined()`'s decode loop (`src/provider_owned/coordinator.cpp`)
never recorded `a_compute_ms` or `network_ms` — only the very first prefill
step did — so `decode_tok_s` (computed as `1000 / mean(a_compute +
middle_compute + b_compute + network)`) was dividing by data that was
almost entirely missing. `latency_ms`/`tokens` (wall-clock, unaffected by
the gap) gave the honest number: `60 / 1.254 s ≈ 47.85 tok/s` — still a
large speedup, just not the one the broken field claimed.

Fixed by threading per-hop send timestamps and `(compute_ns, network_ns)`
pairs through the pipeline's sender/relay/receiver threads (new FIFOs on
`PipelineState`, matched to each chunk's produced-token count the same way
the existing `b_compute_ms` accounting already worked). Verified first on
CPU/GPU loopback (real non-zero `A_ms`/`network_ms`/`B_ms` that
`decode_tok_s` is now arithmetically consistent with), then confirmed again
over the real WAN link below. Commit `0a8a8fe`.

## Results

### Baseline: no draft model, auto-registration, real WAN

```text
tokens:              20
latency:              3,490.959 ms (one request)
decode:               6.640 tok/s
Linux (stage A) compute:   2.346 ms/token
network:                 138.451 ms/token
Windows (stage B) compute:  9.809 ms/token
activation:            6,144 bytes/token
speculative_rounds:         0
```

Network dominates: 138.451 ms of roughly 150.6 ms per token, ~92%. This is
the expected shape for a real internet link with no mitigation — one round
trip paid in full per generated token.

### Pipelined: `--draft-tokens 8 --pipeline-depth 6`, same link, same split

Before the metrics fix (broken, not a real measurement):

```text
decode_tok_s: 261.033   (A_ms/token=0.000, network_ms/token=0.000 -- bug)
```

After the fix, same configuration, rerun:

```text
tokens:              60
latency:              1,054.662 ms (one request)
decode:               42.099 tok/s
Linux (stage A) compute:   0.704 ms/token
network:                  19.180 ms/token
Windows (stage B) compute:  3.870 ms/token
activation:            6,144 bytes/token
speculative_rounds:          8
draft_accept:             0.906
```

`0.704 + 19.180 + 3.870 = 23.754 ms/token`; `1000 / 23.754 ≈ 42.10` —
matches the printed `decode_tok_s`, confirming the fixed metrics are
internally consistent this time.

Sample output (model quality, not infrastructure — Qwen2.5 1.5B, unrelated
to the network/pipelining result):

```text
Paris became the capital of France in 1360 when King Charles V moved his
court from the city of Tours to Paris. This decision was made due to
several historical, political, and geographical reasons.
```

### Comparison

| | Baseline (no pipelining) | Pipelined |
|---|---:|---:|
| decode_tok_s | 6.640 | **42.099** |
| network_ms/token | 138.451 | 19.180 |
| stage A compute ms/token | 2.346 | 0.704 |
| stage B compute ms/token | 9.809 | 3.870 |
| draft_accept | — | 0.906 |

**~6.3x throughput over the same real WAN link**, and the effective
per-token network cost dropped from 138 ms to 19 ms by batching 8 draft
tokens into each verification round trip at 90.6% acceptance and keeping
several rounds in flight (`--pipeline-depth 6`) instead of paying one full
round trip per token. This is the first real-network confirmation of the
core claim in [PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md).

## What this does and doesn't prove

- **Proves**: pipelined speculative decoding measurably hides WAN round-trip
  latency on a real link between two different machines, not just in
  projection or on loopback.
- **Doesn't prove**: ring topology or chunked prefill over real WAN (not
  exercised in this run — see
  [the ring physical test doc](PIPELINED_RING_PHYSICAL_TEST.md) for that,
  still pending); a production-realistic draft model's acceptance rate
  (same weights were reused for both roles here); behavior at larger models
  or higher `--pipeline-depth`; behavior under sustained/concurrent request
  load rather than one single request.
- **Setup note**: this run used plain SSH port forwarding, not
  `dan-sidecar`, because the rented GPU box only exposed an SSH port. The
  encrypted P2P transport path itself remains untested against this exact
  auto-registration flow — see
  [the ring physical test doc](PIPELINED_RING_PHYSICAL_TEST.md#why-dan-sidecar-and-whats-untested-about-it).
