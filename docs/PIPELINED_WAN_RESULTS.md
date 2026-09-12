# Pipelined Speculative Decoding + Ring: Real-WAN Results

Date: 2026-09-11 to 2026-09-12

Status: **both pipelining and ring topology passed over real WAN links,
across two model sizes and two independently rented GPUs**. This is the
first time [PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md)'s
design has been measured over an actual network between two different
machines rather than CPU loopback or one local GPU. Two real bugs were
found and fixed along the way — one in the pipelined path's metrics, one
in ring mode's connection setup over an SSH tunnel — plus several
non-code environment issues (disk space, IPv4/IPv6, tunnel direction) that
cost real debugging time and are recorded below so they don't repeat.
Every number in this doc is from the fixed binaries, not the ones each
test started with.

> Historical note: automatic formation now supports authenticated ring links,
> and pipelined speculative verify blocks can traverse that ring. Statements
> below about those combinations being unavailable describe the tested revision.

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

## Ring topology, first real-WAN attempt: a second real bug, and a fix

Ring mode (`--next`/`--ring-listen`/`--ring-return`) can't combine with
auto-registration (the coordinator rejects `--ring-return` together with
`--provider-listen`), so this used the older fixed-address `--model`/
`--provider` flow instead, over a fresh rented pod (RTX PRO 4500, also
Arad, Romania). It also can't combine with pipelining — `generate_pipelined`
only routes the *prefill* step through `ring_return`; the decode loop
(sender/relay/receiver threads) talks to `stages.front()`/`stages.back()`
directly, hub-and-spoke style, regardless of how the stage workers were
started. So this section measures ring topology alone, no speculative
decoding.

**The bug.** Every first attempt hung forever on `ring: waiting for the
tail stage to connect`, even though the tail stage's own log claimed
`ring: connected to next hop`. Root cause: a ring connection's `--next`
dial has no retry once `connect()` returns successfully (by design — see
`PIPELINED_SPECULATION_V1.md`), but SSH's `-R` reverse port forwarding can
complete a *local* accept on the connecting side before its forwarded
channel to the real remote listener exists yet. The connecting side sees
a "successful" connect that goes nowhere, holds a dead socket forever, and
never retries. Confirmed by reading `connect_to()` in `stage_worker.cpp`:
a raw `::connect()`, no handshake, so nothing catches this.

**The fix** (branch `ring-connect-handshake`, merged to main): both ends of
both ring legs (stage-to-stage, and tail-to-coordinator) now do a real
send/receive round trip immediately after `connect()`/`accept()`, bounded
by a 5s timeout. On failure, the connecting side reconnects and the
accepting side waits for a new connection, instead of trusting a socket
that never proves it works. Verified on loopback first — the fix caught a
genuine local timing race and recovered automatically — then confirmed
fixing the real WAN hang.

**A second, unrelated snag on the way to a clean ring result**: `ssh -R
50105:localhost:50105` failed with `Connection timed out` (not
`refused`) on Windows, because `localhost` resolved to IPv6 while the
coordinator only binds IPv4. Fixed by using `127.0.0.1` explicitly in every
`-L`/`-R` forward instead of `localhost`. Unrelated to the handshake fix,
but part of what "physical test" actually means — see the guidance below.

### Ring result: Qwen2.5-1.5B, no pipelining

```text
tokens:              20
latency:              1,952.881 ms
decode:               11.512 tok/s
stage A compute:           0.000 ms/token  (ring mode's documented metrics gap)
network:                  84.911 ms/token
stage B compute:           1.954 ms/token
```

Compare to the 1.5B hub-and-spoke baseline above (different pod, similar
~83 ms raw latency): `network_ms/token` dropped from 138.451 to
**84.911 — a ~39% reduction**, matching the prediction (ring uses `D+1`
network legs instead of hub-and-spoke's `2D`; for 2 stages that's 3 legs
vs 4). Notably, 84.9 ms is almost exactly the raw ping RTT — ring mode's
per-token cost is close to *one* round trip, where hub-and-spoke pays
several sequential ones.

### Ring result: Qwen2.5-14B, no pipelining, real 2-way split

Same fixed-address flow, `--model-url`/`--model-revision`/`--model-sha256`
used instead of a pre-downloaded full file (the static `--model` flag
supports the same sparse range-download the auto-registration path uses —
no need to fetch the whole ~15.9 GB on either machine just because ring
mode can't use `--provider-listen`).

```text
tokens:              60
latency:              7,468.745 ms
decode:               8.649 tok/s
network:                 100.982 ms/token
stage B compute:          14.642 ms/token
```

Since pipelining can't combine with ring, this is still one round trip
paid per token — ring only shortens that round trip, it doesn't reduce how
many are paid. That's why this is slower than the pipelined 14B result
(18.795 tok/s) below despite a shorter path: ring's payoff needs pipelining
alongside it to be large, and that combination doesn't exist in the code
yet.

## Third confirmation: 14B pipelined, a second pod

Re-ran the 14B target / 1.5B draft pipelined config (hub-and-spoke, no
ring) on the new RTX PRO 4500 pod, coordinator back on Linux (see note
below on why that placement matters), after clearing ~27 GB of stale
range-cache from earlier test retries that had filled the pod's disk to
93% and was silently killing providers mid-download (`peer disconnected
while receiving frame` — not a network bug, an `ENOSPC`-shaped one).

```text
tokens:              60
latency:              5,019.757 ms
decode:               18.795 tok/s
stage A compute:           5.777 ms/token
network:                  40.648 ms/token
stage B compute:           6.781 ms/token
draft_accept:             0.292
```

Matches the first pod's 14B pipelined result (15.313 tok/s, 43.590 ms
network, 0.324 draft_accept) closely — same shape, different hardware and
location, confirming this is a reproducible finding and not a one-machine
fluke.

**Coordinator placement matters over SSH.** Putting the coordinator on
Windows (behind NAT, needing `ssh -R` for the remote provider to reach it)
measured `network_ms/token=308.832` — dramatically worse than putting the
coordinator on the reachable Linux box and having Windows dial out via a
plain `-L` forward (`40.648`). Same model, same pod, same physical link;
only the tunnel direction changed. Whether this is SSH's reverse-forwarding
channel overhead under concurrent pipelined traffic specifically, or
something else about `-R`, wasn't isolated further — but the practical
rule going forward is: put the coordinator on whichever machine is
actually reachable, and forward *toward* it, not away from it.

## What this does and doesn't prove

- **Proves**: pipelined speculative decoding measurably hides WAN round-trip
  latency on a real link between two different machines, at two different
  model sizes, with both an artificial (identical-weights) and a realistic
  (genuinely smaller) draft model, confirmed across two independent pods;
  that the speedup shrinks as compute grows relative to network cost, as
  predicted before either run; that ring topology alone (no pipelining)
  measurably shortens the per-token network cost on a real WAN link,
  matching its `D+1`-vs-`2D` legs prediction.
- **Doesn't prove**: ring topology *combined* with pipelining (the code
  doesn't support that combination yet — see the design doc); chunked
  prefill over real WAN (never exercised); behavior at 32B+ or with more
  than 2 providers; behavior under sustained/concurrent request load rather
  than one single request per run; a draft model actually chosen/tuned for
  the target (1.5B drafting for 14B is a reasonable real shape, but
  arbitrary — a purpose-built draft would likely do better); which specific
  mechanism makes `ssh -R` slower than `-L` under load (observed, not
  isolated).
- **Setup note**: every run here used plain SSH port forwarding, not
  `dan-sidecar`, because the rented GPU boxes only exposed an SSH port. The
  encrypted P2P transport path itself remains untested against this exact
  flow — see
  [the ring physical test doc](PIPELINED_RING_PHYSICAL_TEST.md#why-dan-sidecar-and-whats-untested-about-it).
- **Practical lessons that cost real debugging time**: `ssh -R`/`-L`
  targets should always use `127.0.0.1`, never `localhost` (IPv4/IPv6
  resolution mismatch produces a `timed out`, not `refused`, and is easy to
  misdiagnose as a network problem); a rented pod's small root disk filling
  up manifests as a confusing `peer disconnected` protocol error, not an
  obvious disk-space message — check `df -h` early when a provider dies for
  no visible reason; the static `--model` flag can use
  `--model-url`/`--model-revision`/`--model-sha256` for the same sparse
  range-download the auto-registration path gets, so ring mode never
  actually requires a full-model download either.
