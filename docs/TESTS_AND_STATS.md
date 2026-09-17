# Tests and Stats

> **Start with [PROJECT.md](PROJECT.md)** — the current, canonical guide (decentralized path,
> state, rules, roadmap). This document holds detail and history; most results here are from the coordinator-based path.

What was actually run, on what hardware, with what result. Full receipts —
exact commands, byte counts, hashes, raw logs — live in `reference/tests/`;
this page just has the summary table and the numbers worth remembering.
Design rationale is in [ARCHITECTURE.md](ARCHITECTURE.md) and
`reference/design/`; current status and the beta launch gate are in
[PROGRESS.md](PROGRESS.md).

## Headline results

| Date | Test | Hardware | Result |
|---|---|---:|---|
| 2026-09-08 | [Aggregate-VRAM 32B](reference/tests/aggregate-vram-test.md) | Windows RTX 2070 (6,656 MiB) + Linux RTX A5000 (24,000 MiB), Tailscale | First proof neither GPU alone could hold the model. 20 tokens, 4.279 tok/s, 184.791 ms/token network-dominated (relayed Tailscale path). |
| 2026-09-11/12 | [Pipelined speculation, real WAN](reference/tests/pipelined-wan-results.md) | Two independently rented pods, SSH tunnels | Qwen2.5-1.5B: 6.640 → 42.099 tok/s pipelined (6.3x). Qwen2.5-14B: 5.006 → 15.313 tok/s with a real draft model (3.1x). |
| 2026-09-11/12 | [Ring topology, real WAN (SSH tunnel, fixed addressing)](reference/tests/pipelined-wan-results.md#ring-topology-first-real-wan-attempt-a-second-real-bug-and-a-fix) | Same rented pods | Qwen2.5-1.5B: 11.512 tok/s. Qwen2.5-14B: 8.649 tok/s (no pipelining — ring and pipelining didn't combine at the time). A real hung-forever bug found and fixed (`ring-connect-handshake`, merged). |
| 2026-09-13 | [Provider-owned v2, Windows+Linux](reference/tests/windows-linux-v2-test.md) | Windows RTX 2070 + rented Linux GPU | Range-backed sparse GGUF caching validated cross-machine: partial download, `cache=reused` on restart, correct 20-token output. |
| 2026-09-13/14 | [Cross-GPU 14B, RunPod, libp2p+relay](reference/tests/remote-gpu-test-method.md) | Windows RTX 2070 (8,191 MiB) + rented RunPod RTX A4000 (16,376 MiB), no VPN | Genuine forced split (real VRAM limits, no caps) on a 14.63 GiB model. Baseline 4.70 tok/s → 12.11 tok/s with speculative decoding + pipelining (2.58x), 46.7% draft acceptance. CUDA graphs confirmed (179 reuses in logs). |

## What's proven vs. not, per capability

| Capability | Proven | Not yet proven |
|---|---|---|
| Forced cross-GPU split from real VRAM limits | Yes — 32B (2026-09-08), 14B (2026-09-13/14) | — |
| Range-backed sparse weight caching, cross-machine | Yes — reuse survives coordinator restart | — |
| CUDA graph reuse | Yes — confirmed in logs, not just assumed | — |
| Speculative decoding + pipelining | Yes — 2.58x-6.3x speedups across three separate runs, two different network transports | Production-default speculation (standard package ships no draft model) |
| Ring topology (coordinator removed from hot path) | Yes, once — SSH tunnel, fixed `--provider` addressing, no pipelining, no auto-registration, no peer-auth (2026-09-11/12) | Combined with libp2p relay circuits + auto-registration + `--provider-peer-auth` (attempted 2026-09-13/14, see below) |
| NAT relay (circuit-v2) | Yes — a NAT'd peer reserved a slot and got a dialable circuit address | Full ring data flow through a relay circuit (see below) |
| Stranger-machine acceptance | No | Open beta gate — needs a genuinely clean host, packaged release, different NAT |
| 24-hour soak | Attempted, result lost (logging gap, now identified) | Needs a rerun with `Test-DAN-Soak.ps1` writing to a file |

## The ring-mode gap, precisely

Don't conflate these — they're two different ring attempts, months apart, over
two different transports.

**2026-09-11/12, SSH tunnel, fixed addressing.** This one worked, but only
after chasing down a real hung-forever bug: a dead `connect()` that never
retried once an SSH `-R` reverse forward had quietly gone nowhere. The
headline table's 11.512/8.649 tok/s numbers come from here. It predates
auto-registration and `--provider-peer-auth`, and it doesn't combine with
pipelining — the decode loop bypasses `ring_return` and talks to
`stages.front()`/`stages.back()` directly regardless.

**2026-09-13/14, libp2p relay circuit, auto-registration, peer-auth.** This
combination never got that far. The coordinator's `--ring-return` is one
string doing two jobs — bind address and externally-advertised address — which
only works when they happen to be the same thing, like on Tailscale. Fixed
that by exposing the coordinator's already-existing `ring_target` field as a
`--ring-target` flag (one line in `src/provider_owned/coordinator.cpp`, builds
and tests clean). Registration and the two-stage plan both started working
after that, but the tail's first data frame fails a magic-number check on the
coordinator side — `bad frame header` — even though the peer-ID header and
handshake code match byte-for-byte on both ends. That's as far as it's been
traced; going further needs a packet capture or temporary logging, not more
code reading. Full trace in
[`reference/tests/remote-gpu-test-method.md`](reference/tests/remote-gpu-test-method.md).

## Real bugs physical testing found that code review wouldn't have

None of these showed up until something actually ran over a real network.
Worth remembering before trusting a design doc's claims at face value:

- A ring `--next` dial's lack of retry, combined with SSH `-R`'s ability to
  complete a local accept before the forwarded channel exists, caused a
  silent dead-socket hang. Fixed with a real send/receive handshake round
  trip after every `connect()`/`accept()`.
- `localhost` resolving to IPv6 while the coordinator only binds IPv4 caused
  an SSH forward to time out instead of failing fast. Use `127.0.0.1`
  explicitly.
- `git -C <dir> apply <relative-patch>` resolves the patch path relative to
  `<dir>`, not the caller's cwd — silently wrong unless the path is absolute.
- `pkill -f <pattern>` matches its own invoking shell's command line when
  that exact text appears in the remote command sent over SSH — kills the
  SSH session instead of the target. Kill by PID instead.
- Under `--provider-peer-auth`, `--provider-id` must exactly equal the
  worker's libp2p PeerID, with no error explaining the mismatch otherwise.
- The VRAM planner's reserve floor (`max(1024 MiB, 15%)`) makes any
  `--vram-mib` below 1024 silently unusable for any layer — the provider
  just sits registered forever.
- A one-shot coordinator (`--requests 1`) races ahead as soon as *any*
  sufficient replica forms, which can silently defeat a test meant to force
  a multi-provider split if one provider alone already has enough VRAM.

## Historical / superseded

- [`reference/tests/pipelined-ring-physical-test.md`](reference/tests/pipelined-ring-physical-test.md) —
  the original design validation for pipelining, ring topology, and chunked
  prefill, done single-machine (CPU loopback / one local GPU). Superseded by
  the real-WAN results above; still worth keeping for the design detail.
- [`reference/tests/provider-owned-execution-v0.md`](reference/tests/provider-owned-execution-v0.md) —
  the earliest local-plus-physical two-provider proof, from before
  range-backed storage existed.

## How the remote tests were actually driven

[`reference/tests/remote-gpu-test-method.md`](reference/tests/remote-gpu-test-method.md)
covers the SSH-orchestration setup that made the 2026-09-13/14 cross-GPU
iteration fast: one machine driving both sides non-interactively, `nohup`
plus `disown` for detached remote processes, kill-by-PID, a background poll
loop for the multi-minute waits. It also has its own punch list of what's
still untested — bounded concurrent API over WAN, recovery under real churn,
`dan-api-gateway` routing over this link, the 32B model on this same link,
multiple concurrent resident sessions.
