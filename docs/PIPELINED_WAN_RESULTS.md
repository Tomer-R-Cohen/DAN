# Pipelined Speculative Decoding: First Real-WAN Result

Date: 2026-09-11

Status: **pipelining over a real WAN link passed and shows a real speedup, at
two model sizes**. This is the first time
[PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md)'s design has been
measured over an actual network between two different machines rather than
CPU loopback or one local GPU. A real bug in the pipelined path's own
metrics was found and fixed in the course of the first test (see below) —
every number in this doc is from the fixed binary, not the one the test
started with.

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

## Link characteristics

The rented box (RunPod, `213.173.108.5`) geolocates to Arad, Romania.
Measured ICMP round trip from Israel: 82-84 ms average. This explains why
`network_ms/token` sits around 138-140 ms in the no-pipelining runs above —
DAN's own protocol round trip (TCP + framing + processing) adds on top of
the raw ~83 ms link latency, and in the pipelined runs, dividing that
~83-100 ms round trip across several tokens per batch (rather than the
model itself being any faster) is exactly what the lower `network_ms/token`
numbers below reflect.

## Second model: Qwen2.5-14B-Instruct Q8_0, same link

Same setup, same link, forced 2-way split (`--vram-mib 13000` on Linux so
it can't hold the 14B model alone): `linux-local` got layers 14-47 (10.04
GiB), `windows-remote` got layers 0-13 (4.59 GiB) — provider ordering
depends on registration order, so which machine gets which end of the stage
list is not fixed across runs.

### Baseline (no draft model)

```text
tokens:              20
latency:              4,185.966 ms
decode:               5.006 tok/s
stage A compute:          29.296 ms/token
network:                 139.775 ms/token
stage B compute:          30.684 ms/token
activation:            20,480 bytes/token
```

Network's share of total time: 139.775 / (29.296+139.775+30.684) = **70%**,
down from the 1.5B baseline's 92%. Same link, same ~83 ms raw latency —
the only thing that changed is a bigger model needs more compute, so the
fixed network cost is a smaller fraction of a larger total. This is the
predicted mechanism, not a new one: pipelining's payoff is proportional to
how much of total time was network to begin with, and that share shrinks
as compute grows.

### Pipelined, with a *real* small draft model (1.5B drafting for 14B)

Unlike the 1.5B test above (which reused identical weights for both draft
and target — unrealistic), this run used the actual smaller Qwen2.5-1.5B
model as `--draft-model` against the 14B target, matching a real deployment
shape.

```text
tokens:              60
latency:              6,550.539 ms
decode:               15.313 tok/s
stage A compute:           7.560 ms/token
network:                  43.590 ms/token
stage B compute:          14.153 ms/token
activation:            20,480 bytes/token
speculative_rounds:          17
draft_accept:             0.324
```

`draft_accept=0.324` — far lower than the 1.5B/1.5B same-model test's
0.906, because a genuinely different, weaker draft model disagrees with
the target much more often. This is the realistic number; the earlier 0.906
was an artifact of the draft and target being identical weights.

### Comparison: both models, real-vs-artificial draft

| | 1.5B target, 1.5B draft (identical) | 14B target, 1.5B draft (real) |
|---|---:|---:|
| baseline decode_tok_s | 6.640 | 5.006 |
| pipelined decode_tok_s | 42.099 | 15.313 |
| speedup | ~6.3x | ~3.06x |
| network's share of baseline | 92% | 70% |
| draft_accept | 0.906 (artificial) | 0.324 (realistic) |

The smaller speedup on the 14B run is the expected result of two
compounding, independently-predicted effects: less network to hide (compute
is a bigger share of total time), and a lower, more realistic acceptance
rate (more rejected rounds means less speculative work survives). Both
factors point the same direction, and both were predicted before this run,
not fit to it afterward.

## What this does and doesn't prove

- **Proves**: pipelined speculative decoding measurably hides WAN round-trip
  latency on a real link between two different machines, at two different
  model sizes, with both an artificial (identical-weights) and a realistic
  (genuinely smaller) draft model; that the speedup shrinks as compute
  grows relative to network cost, as predicted before either run.
- **Doesn't prove**: ring topology or chunked prefill over real WAN (not
  exercised in either run — see
  [the ring physical test doc](PIPELINED_RING_PHYSICAL_TEST.md) for that,
  still pending); behavior at 32B+ or with more than 2 providers; behavior
  under sustained/concurrent request load rather than one single request
  per run; a draft model actually chosen/tuned for the target (1.5B
  drafting for 14B is a reasonable real shape, but arbitrary — a purpose
  -built draft would likely do better).
- **Setup note**: both runs used plain SSH port forwarding, not
  `dan-sidecar`, because the rented GPU box only exposed an SSH port. The
  encrypted P2P transport path itself remains untested against this exact
  auto-registration flow — see
  [the ring physical test doc](PIPELINED_RING_PHYSICAL_TEST.md#why-dan-sidecar-and-whats-untested-about-it).
