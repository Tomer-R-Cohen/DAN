# Pipelined Speculative Decoding v1

## Status

All six phases in this document's original plan are done. Phases 1-5 are
**built and verified** (phases 4 and 5 scoped down from their original
descriptions, below); phase 6 needed no new code — see its own entry in
"Phasing" for why. Phases 1-5, in order: implicit rollback, the pipelined coordinator itself
(`generate_pipelined` in `src/provider_owned/coordinator.cpp`, opt-in via
`--pipeline-depth N`, 0 the unchanged serial default), speculative/pipelined
decoding on the served (`--listen`) path (not just the `--prompt` CLI
benchmark), direct stage-to-stage ring return for `route_step` — prefill and
non-speculative decode only, not the speculative/pipelined path, which turned
out to need its own design (see phase 4 in "Phasing") — and, on top of that
ring, chunked pipelined prefill (a new wire type, `prompt_chunk`; opt-in via
`--prefill-chunk N` on `dan-stage-worker`). Verified on CPU with the real
Qwen2.5 1.5B Q4_K_M two-stage split: output is byte-identical to the serial
K-chunk speculative baseline at `--pipeline-depth` 1, 4, and 8, for a
`--tokens 60` generation with natural partial rejections (`draft_accept`
0.70-0.75); four sequential stateless requests show no cross-request state
leakage; a real client against the served path over the actual
TCP/queue/scheduler machinery gets correct deterministic output with
backpressure, cancellation, and timeout all still working; over an actual
stage-to-stage ring with direct return to the coordinator, both a single
stateless request and a 4-request persistent session produce output
byte-identical to the same requests over hub-and-spoke; and, with
`--prefill-chunk 32` against an ~80-token prompt (confirmed to actually chunk
via the stage logs), output stays byte-identical to the same prompt with
chunking off, both stateless and persistent. See "Rollback is implicit", the
end of "Coordinator concurrency", and phases 3-5 in "Phasing" for what was
actually built, which differs from the original sketch in ways caught during
implementation (documented in place at each point): the epoch stayed
in-process rather than going on the wire, the termination signaling needed a
per-stage-boundary sent count rather than the single FIFO counter first
sketched to avoid a real deadlock, `Replica` needed to own an optional
`DraftModel` rather than the served path staying draft-free, ring mode needed
every listener a stage owns bound before any blocking outbound connect or the
whole topology deadlocks at startup, and chunked prefill needed one narrow
drop rule in the ring thread so a chunk's ack at the last stage doesn't get
misread as the coordinator's actual result. Four implementation bugs were
found and fixed across all of this verification: two in phase 2 (draft model
KV never rolled back on a correction, and a fixed draft width drafting past
`token_limit` — see "Bugs found during verification" after "Coordinator
concurrency") and two in phase 4 (a startup-ordering deadlock and commit
frames misrouted into the ring — see phase 4's own entry in "Phasing"). Phase
3 additionally surfaced a test-harness timeout that was not a code bug (also
in its own entry); phase 5's ack-misrouting issue (previous paragraph) was
caught by design review before it was ever run, not by a failing test.

Throughput figures in "Projection" remain projections, not measurements — the
CPU verification above exercised correctness, not the WAN throughput case this
document is written for. The numbers that would confirm or refute the
projection are named in "Instrumentation" and have not been collected.

Verifying implicit rollback also surfaced a correctness fact that changes this
document's acceptance criteria: batched multi-token verification is not
bit-identical to serial single-token decode on this runtime, independent of
rollback or pipelining. See "A correctness fact, found while verifying
rollback" below. The acceptance criteria section has been corrected to name
the right baseline, and the CPU verification above was run against that
corrected baseline.

## Problem

The provider-owned coordinator sends one speculative chunk and blocks until it
has traversed every stage. `route_speculative` (`src/provider_owned/coordinator.cpp:782`)
exchanges with each stage in turn, and `generate` (`:917`) runs the next draft
round only after the previous round has committed. Exactly one frame is ever in
flight, so every stage but one is idle, and each committed token costs a full
loop traversal.

The measured 32B WAN run shows what that costs. From
[the Qwen2.5 32B results](AGGREGATE_VRAM_TEST.md):

```text
RTX 2070 stage compute:  14.156 ms/token
network:                184.791 ms/token
RTX A5000 stage compute: 34.762 ms/token
loop total:             233.709 ms/token
decode:                   4.279 tok/s
activation:              20,480 bytes/token
```

Network was 79.1% of the loop. Compute on the binding stage was 34.762 ms, so
the same hardware could clear 28.8 frames per second if the loop were kept
full. It is not: the round trip, not the GPU, sets the rate.

Two further limits apply today. A partially rejected round pays an extra full
traversal, because `rollback_all` (`:818`) broadcasts a rollback to every stage
and waits for each acknowledgement before the next round starts. And
speculative decoding is benchmark-only: `--draft-model` is rejected unless
`--prompt` is set and `--listen`, `--persistent`, and `--interactive` are all
off (`:486`), and `Replica::run` (`:1187`) calls `generate` with no draft, so
the served path has never run it.

## Design

Keep several speculative chunks in flight at once. The draft model proposes
past positions the replica has not yet verified, the coordinator streams those
chunks back to back, and a rejection discards the chunks built on it. Committed
output is unchanged: the accept rule is the one already in `generate`
(`:941`), applied per reply instead of per round. Pipelining changes only when
chunks are sent, not the per-chunk computation, so the pipelined path must
produce token-identical output to today's serial K-chunk speculative path — not
to plain serial decode. See "A correctness fact, found while verifying
rollback" for why that distinction matters.

### Chunk granularity is kept

A chunk stays `[cur] + K-1 drafts` as one `token` frame with `rows = K`,
exactly as sent today. Splitting a block into K single-token frames was
rejected. That split exists in the reference implementation to avoid a
per-position replay inside its stage; llama.cpp has no such penalty, because a
K-token `llama_batch` is one forward pass and decode is bound by weight
streaming rather than by token count. Splitting would cost roughly K times the
per-stage compute to buy the same pipeline fill.

### Frame ordering and reply matching

Total frame order is established at stage 0 and preserved by first-in-first-out
forwarding: only stage 0 receives injected frames, every other stage receives
only what its predecessor produced, in the order it produced it, and each stage
worker is a single `recv`/`handle`/`send` loop
(`src/provider_owned/stage_worker.cpp:892`). Replies therefore return in send
order.

Given that, matching replies to requests needs a queue and a counter, not
identifiers:

- Per active request the coordinator keeps a FIFO of in-flight chunks, each
  holding its base position and the tokens sent, plus an epoch tag.
- One reply pops one FIFO entry. Each reply is validated structurally against
  that entry (session, request, `position == base + rows`, `rows`), so a
  desynchronised stream fails loudly instead of committing wrong tokens.
- On a rejection, the epoch is bumped; a popped entry whose tag doesn't match
  the current epoch is known-stale and its reply is discarded rather than
  applied. This replaced an earlier "count of stale replies still owed"
  design: since it's strict FIFO, everything after a rejection is stale until
  the epoch bump, so a per-entry tag makes that automatic instead of counted.

**Built as in-process state only, not on the wire — a deliberate change from
the original plan below, made during implementation.** The original design
called for a `uint16` epoch on the wire (bytes 38-39 of the frame header,
currently reserved and validated as zero —
`include/provider_owned/protocol.hpp:196` — with `version` moving from 2 to
3), echoed by every stage in its reply, "as an assertion rather than the
mechanism." Implementing it, that assertion turned out to have nothing to
verify: this topology has exactly one writer per stage socket (the sender
thread owns stage 0's socket; each relay thread owns exactly one downstream
socket) and v1 has no redials or retries, so a stage's replies are
*definitionally* in the order the single writer sent them — there is no
concurrent producer whose frames could arrive interleaved for the wire-level
epoch to catch. The in-process epoch (`PipelineState::epoch`, compared per
popped FIFO entry) does the actual job. If a later phase adds direct
stage-to-stage links with reconnect/redial, or more than one writer per
socket, that assumption stops holding and the wire-level version should be
revisited.

### Rollback is implicit

**Built and verified.** `rollback_all` is removed. The next chunk's position becomes the rewind
command. The stage worker's position check in `execute`
(`src/provider_owned/stage_worker.cpp:350`, `:359`, `:484`, `:579`) changes
from "position must equal `session.position`" to:

- greater than `session.position` is still an error, because it is a gap;
- less than `session.position` truncates with
  `llama_memory_seq_rm(sequence, position, -1)`, sets `session.position`, then
  decodes normally;
- equal keeps today's path.

That body already exists in `rollback` (`:267`); the change folds it into the
execute path. Stale frames sent before a correction are processed first by
every stage, advancing them onto a rejected future, and the correction that
follows truncates it away. The wasted stage time is the pipeline bubble; the
state cannot be corrupted, because truncation by absolute position is exact and
every position at or after the correction is rewritten in order.

Nothing else in a session needs to be rolled back. Stage KV is plain llama.cpp
KV — the DAN patch adds only a layer filter to `create_memory`
(`patches/llama-provider-owned.patch`) — Qwen2 is dense with no recurrent or
windowed state, and the per-session sampler is
`llama_sampler_init_greedy()` (`src/provider_owned/stage_worker.cpp:68`), which
is stateless. Session rollback state is therefore exactly the KV cache and
`session.position`. The draft model already exposes `rollback`
(`src/provider_owned/coordinator.cpp:342` region).

Verified on a local CPU run: Qwen2.5 1.5B Q4_K_M, 28 layers split 0-13/14-27
across two `dan-stage-worker` processes, `--draft-tokens 8`, 60-token
generation, partial rejections occurring (`draft_accept=0.747` over 10 rounds).
Output and every reported metric were byte-identical between the original
`rollback_all` code and this change. Confirmed on unmodified code by testing
both versions against the same running stages via `git stash`.

This change is worth taking on its own. It removes a full traversal from every
partially rejected round on the existing serial path, before any pipelining
exists.

### In-flight depth

Depth is a tug of war: below the bandwidth-delay product the loop runs at
`depth / RTT` rather than at `1 / tau_max`, while every added chunk is a deeper
speculation that is both least likely to be accepted and destroyed by the next
rejection. The reference implementation's measured rule is one below the
bandwidth-delay product:

```text
W = clamp(round(RTT / tau_max) - 1, 2, cap)
```

On the measured 32B configuration, `RTT` is 233.709 ms and `tau_max` is
34.762 ms, giving a bandwidth-delay product of 6.7 and `W = 6` at `K = 4`.
Both `K` and `W` are runtime flags: `--draft-tokens` (existing) and
`--pipeline-depth` (**built**; 0 selects the serial path, matching
`generate()` exactly). **Not built:** the adaptive-`K` rule
(`K = clamp(round(EMA(accepted)) + 2, k_min, k_max)`) — v1 uses a fixed
`--draft-tokens` every round. One consequence of a fixed `K`: the last round
of a request can draft past `token_limit`. `generate()` avoids this by
shrinking `count` for the final round; the pipelined sender can't, because it
doesn't know the confirmed token count in advance (that's the whole point of
pipelining). The receiver truncates instead — see "Bugs found during
verification" below for why that's output-equivalent, not just a workaround.

Two limits bound `W` below the bandwidth-delay product:

- Draft budget. At `W = 6`, `K = 4` the coordinator drafts about 24 tokens per
  cycle for roughly 3.3 committed, near 7 drafted per committed token. Holding
  30 tok/s would require about 215 draft tokens per second, a budget under
  4.7 ms per draft token. Once the round trip is hidden, the draft model is
  expected to become the binding stage.
- Context headroom. Stages reject `rows` beyond `context_size - position`
  (`src/provider_owned/stage_worker.cpp:374`, `:495`), so `W` must be clamped
  as a session approaches its context limit.

### Coordinator concurrency

**Built as designed**, with one addition the original sketch didn't have (see
below): sender, relay ×(N-1), receiver — `pipeline_sender`, `pipeline_relay`,
`pipeline_receiver` in `src/provider_owned/coordinator.cpp`. The receiver runs
on the calling thread rather than a spawned one; sender and each relay are
`std::thread`s, joined before `generate_pipelined` returns.

```text
sender      draft K tokens, send chunk to stage 0, push to in-flight FIFO;
            blocks while the FIFO holds W chunks
relay i     forward stage i's replies to stage i+1 untouched  (i = 0..N-2)
receiver    pop the FIFO per tail reply; commit, discard, or trigger a correction
```

`Connection` already separates `send` and `receive`
(`src/provider_owned/coordinator.cpp:118`). Concurrent send and receive on one
socket is permitted on both Winsock and POSIX, but two concurrent senders would
interleave frames, so each stage socket has a single sending owner. Backpressure
is inherent: a slow stage fills its socket buffer, blocking the upstream reader
and so its predecessor, and `W` bounds the whole chain.

The draft model is touched only by the sender thread. The receiver publishes a
correction (position and token) and bumps the epoch under one lock; the sender
observes it, calls `draft.rollback()`, and re-drafts from the corrected prefix
(this call was missing in the first implementation — see "Bugs found during
verification").

**Addition not in the original sketch: per-stage-boundary sent counts, not a
single "sender finished" flag.** The first implementation had each relay
thread decide when to stop by comparing its own processed count against a
single shared "total chunks sent" value, set once the sender was done. That
is provably too early for any relay before the last stage: a relay can only
know it forwarded the *last* chunk once the *tail's reply* for that chunk has
propagated all the way back to the receiver — which happens strictly *after*
the relay already forwarded it, not before. So a relay waiting to be told "no
more are coming" can end up blocked in a `receive()` call for a frame that
will genuinely never arrive, with no cross-thread way to interrupt a blocking
socket read. Caught by tracing the causal order before running anything, not
by hitting it — though the first pipelined smoke test would have hung on
exactly this (2 stages is exactly the "one relay between sender and tail"
case where it bites).

The fix, and what's actually built: `PipelineState` keeps `sent[i]` (frames
placed on stage `i`'s socket so far) and `stage_finished[i]` (no more ever
will be) per stage, not one pair of counters for the whole pipeline. Relay `i`
reads `sent[i]`/`stage_finished[i]` — set by whoever writes to stage `i`
(the sender, for `i = 0`; relay `i-1`, for `i > 0`) — and only calls
`receive()` once `sent[i]` proves a corresponding frame already reached that
socket; it writes `sent[i+1]`/`stage_finished[i+1]` for whoever reads stage
`i+1` next. This forms a cascade: sender finishes and finalizes `sent[0]`,
which lets relay 0 finish and finalize `sent[1]`, and so on to the receiver —
each stage's "done" signal only fires once it's *certain*, never guessed.

The stage worker needs no concurrency change, and none was made. Overlap is
across stages, not within one, so its single-threaded loop remains correct
as-is; the only stage-worker change for the whole design was the implicit
rollback relaxation, already covered above. There is no wire-level epoch to
echo (see "Frame ordering and reply matching").

### Bugs found during verification

Two bugs were found and fixed while verifying `generate_pipelined` against the
real two-stage CPU setup, beyond the termination-signaling design issue caught
before testing (previous section). Both are now fixed in the code; recorded
here because both were real, not hypothetical, and both trace back to the same
root cause: writing the sender's per-round logic by adapting `generate()`'s
loop body without re-deriving which of its side effects a *free-running*
sender still needs.

**The draft model's own KV was never rolled back on a correction.** The
sender updates its local `base_position`/`carry` from the receiver's
correction, but the first version never called `draft.rollback()` — unlike
`generate()`, which always does (`draft->rollback(keep_position)`). Since the
sender's draft model free-runs forward every round regardless of what the
receiver later decides, skipping the rollback means it keeps decoding a
continuation nobody committed to, on top of state that should have been
discarded. Over enough rejections its KV runs past `context_size` and
`llama_decode` fails outright ("failed to find a memory slot") — which is
how this was caught, on the very first pipelined run. Fixed: the sender now
calls `draft.rollback(base_position)` whenever it picks up a pending
correction, before drafting the next round.

**A fixed `K` can draft past `token_limit`, and the fix is to truncate the
commit, not shrink the draft.** Covered under "In-flight depth" above:
`generate()` shrinks `count` for the last round so it lands exactly on
`token_limit`; the pipelined sender can't know the confirmed count in time to
do that. First symptom: `--tokens 60` produced 65. The fix adds a check to the
receiver's accept loop, structurally identical to its existing EOG check —
stop applying tokens the instant `token_ids.size()` reaches `token_limit`,
mid-round if needed, exactly as it already stops on EOG. This is not an
approximation of `generate()`'s behavior, it's equivalent to it: verification
is causal, so `checked[i]` depends only on `chunk.guesses[0..i]`, never on
what the block contains after position `i` — truncating what gets *applied*
after the fact produces the identical committed sequence as never having
drafted the rest, just with a few wasted (already-computed, already-verified,
discarded) tokens at the very end of a request. The extra KV positions this
leaves resident on the stages are cleaned up by the *same* implicit-rollback
mechanism from phase 0: `commit_final_token` sends the truncated
`output.position`, which is lower than what the stage actually holds, so the
stage truncates back down to it automatically.

Verified after both fixes: output byte-identical to the serial K-chunk
baseline at `--pipeline-depth` 1, 4, and 8 (same prompt, `--draft-tokens 8`,
`--tokens 60`, natural partial rejections), and token count exactly 60 in
every case.

### Control frames and request drain

**Built more simply than the original sketch, because thread-join already
gives a stronger guarantee than the "drain fence" it proposed.** `control_all`
(`src/provider_owned/coordinator.cpp:684`) broadcasts session control directly
to every stage and waits for each acknowledgement; that bypasses the pipeline
order and would be unsafe if any pipeline thread were still touching a stage
socket. The original plan was to gate control frames on "the in-flight FIFO is
empty." What's actually built doesn't need that check: `generate_pipelined`
calls `sender.join()` and `relay.join()` for every pipeline thread, and
`pipeline_receiver` runs to completion on the calling thread, all *before*
`commit_final_token`/`control_all(end_request)` are called. By the time
control frames are sent, no thread exists that could still be mid-send or
mid-receive on any stage socket — a strictly stronger guarantee than "the FIFO
is empty," and one that falls out of normal thread lifetime instead of needing
its own bookkeeping. In-band control frames that traverse the stages remain
the right answer if a later phase allows more than one request active at a
time (join-before-control only works because v1 keeps exactly one pipelined
request in flight per replica, matching
[runtime v2](PROVIDER_OWNED_RUNTIME_V2.md)).

Draining mid-pipeline replies at end-of-generation needed no special handling
either, for the same reason: `pipeline_receiver`'s own loop keeps consuming
tail replies (discarding the stale ones once `state.finished` is set) until
its own termination condition — `stage_finished[read_index]` cascaded all the
way from the sender, `sent[read_index]` caught up — is met, which by
construction can't happen before every chunk anyone sent has been accounted
for. There's no separate "drain" step; finishing normally already is one.

Reader-thread timeouts remain a real, documented limitation, unchanged from
the original plan: `Connection::set_timeout` (`:87`) sets 30 s on send and
receive. A relay or the receiver blocked in `receive()` at the moment of a
genuine connection failure elsewhere in the pipeline won't notice until that
call times out — bounded, but not instant. Not exercised by the CPU
verification above (no failures were injected).

## A correctness fact, found while verifying rollback

Verifying implicit rollback against plain serial decode (`--draft-model`
omitted) surfaced something this document originally got wrong: **speculative
output is not bit-identical to plain serial greedy output on this runtime**,
independent of rollback and independent of pipelining. Same prompt, same
model, same two-stage split, `--draft-tokens 8`: serial decode gives "...in
1256 when King Louis IX..."; the existing single-chunk speculative path gives
"...in 1368, when King Charles V...". Both are fluent, plausible continuations
— the divergence is a different token chosen partway through, not corruption.

Isolated with `--draft-tokens 1`: at K=1 the "batched verify" is a single-row
`llama_batch`, structurally identical in shape to serial's own decode call, and
output was byte-identical to serial (`draft_accept=1.000`). At K=8 it diverges.
The only variable between those two runs is batch width, so the cause is
floating-point non-associativity in multi-row batched matrix multiplication —
a quantized (Q4_K_M) GEMM/attention kernel does not guarantee identical
rounding for a batch of K query rows versus K sequential single-row calls.
Under greedy sampling, a small rounding difference near a close logit gap can
flip which token wins argmax, and that single flip cascades into a different
continuation. This is a general property of batched versus sequential
floating-point compute, not specific to DAN's implementation, and it does not
indicate corrupted state or a bug in the accept rule, position handling, or
causal masking — K=1 exercises that same code at rows=1 and matches serial
exactly.

This does not implicate implicit rollback: the K=1 and K=8 runs above were
both run against the change in this document, and separately the K=8
speculative run was confirmed byte-identical between `rollback_all` and
implicit rollback (previous section). The divergence is entirely a property of
K>1 batched verification, present before this document's changes and before
any pipelining exists.

**Consequence for this design:** the correctness bar for phase 1 cannot be
"pipelined output matches plain serial decode," because that bar is already
failed by the unpipelined K>1 path today. The correct bar is "pipelined output
matches today's serial K-chunk speculative output" — both do the identical
batched-verify computation in the identical shape per chunk; pipelining only
changes when chunks are sent, not what each one computes. The acceptance
criteria below have been written against that baseline, not against plain
serial decode.

## Projection

Not measured. Throughput is modelled as

```text
tok/s = (committed tokens / computed frames) * min(1 / tau_max, W / RTT)
```

with committed tokens per cycle `R + 1`, `R = a / (1 - a)` for per-draft
acceptance `a`, and computed frames per cycle `ceil((R + 1) / K) + (W - 1)`.
Applied to the measured 32B configuration at `K = 4`, `W = 6`, hub routing
unchanged:

```text
  a     R+1   tok/frame   tok/s   vs measured 4.279
0.60    2.50       0.42    10.7              2.5x
0.70    3.33       0.55    14.3              3.3x
0.80    5.00       0.71    18.4              4.3x
```

Acceptance `a` for DAN's draft and target pair has never been measured, and it
is the largest single input to the table. The frame ceiling on this hardware is
28.8 frames per second; the projections sit below it because discarded
speculation consumes the binding stage.

Direct stage-to-stage return is a separate follow-on. Hub routing costs `2D`
network legs per traversal, a ring costs `D + 1`: at the measured two stages
that is 4 legs against 3, an estimated 25% of the wire term, giving about
17.7 tok/s at `a = 0.70`. The reduction grows with stage count (12 legs against
7 at six stages). The hub topology is retained for v1, and it has one property
a ring does not: the coordinator holds a direct socket to every stage, so an
out-of-band cancel that lets stages skip stale frames rather than compute them
is reachable later.

## Phasing

1. Implicit rollback. Stage position relaxation, `rollback_all` deleted.
   Improves the existing serial path and is a prerequisite for the rest.
   **Built and verified** — see "Rollback is implicit" above.
2. Pipelined coordinator, opt-in by `--pipeline-depth`, serial path untouched.
   **Built and verified** on CPU — see "Status" above and "Bugs found during
   verification". Not yet measured on a WAN/multi-machine configuration; the
   throughput projection below is still unconfirmed.
3. Speculative decoding on the served path: wire a draft into `Replica::run`
   and lift the `--prompt`-only restriction. **Built and verified.** `Replica`
   now optionally owns a `DraftModel` (constructed from `--draft-model` when
   given); `generate_request` branches to `generate_pipelined` when
   `--pipeline-depth > 0`, `generate` with the draft otherwise, matching the
   `--prompt` CLI path's own branch. `--draft-model` and `--pipeline-depth`
   are now valid with `--listen` (previously rejected outright); `--listen`
   and `--prompt` remain mutually exclusive as before, unrelated to this
   change. Verified against the real two-stage CPU setup through the actual
   TCP/queue/scheduler path (`provider_owned_concurrency_client`, not the
   `--prompt` shortcut): a stateless generate returns the correct deterministic
   text (`" Paris"` for the fixed reference prompt) with pipelining active,
   and the existing backpressure/cancellation/timeout probes in that harness
   all pass unchanged (`queue_full`, `cancelled`, `queue_timeout` all fire
   correctly) — confirming pipelining doesn't interfere with the scheduler's
   queueing or the sender/relay/receiver threads' lifetime. One caveat, not a
   defect: the harness's default 30s per-request queue timeout is tuned for
   fast decode and is too short for a CPU dual-model (target + draft) run —
   the first attempt timed out 12 of 32 concurrent backpressure-probe
   requests; rerun with `--queue-timeout-ms 240000` and 0 failures. On real
   (GPU) hardware this shouldn't come up. `DraftModel` is unsynchronized
   inside `Replica`, safe only because the scheduler's single executor thread
   guarantees exactly one `generate`/`generate_pipelined` call is ever active
   — unaffected by, and not a fix for, the multi-session-concurrency question
   already listed under "Remaining questions".
4. Direct stage-to-stage return. **Built and verified for `route_step`
   (prefill and non-speculative decode) only; `route_speculative` and the
   pipelined sender/relay/receiver remain hub-and-spoke.** New opt-in flags:
   `--next HOST:PORT` and `--ring-listen HOST:PORT` on `dan-stage-worker`,
   `--ring-return HOST:PORT` on the coordinator. Every stage's connections
   become forward-only in ring mode: it reads hot-path input from one
   connection and writes the outcome — success or error alike — to `--next`,
   never back on the connection it arrived on. That one rule is what makes
   error propagation general over any stage count without per-stage
   special-casing: a stage can't usefully reply to whoever handed it a frame
   (the previous stage isn't equipped to interpret a reply), so both outcomes
   just ride forward, and the last stage's `--next` is the coordinator's
   return listener, so everything surfaces there. Control frames (session
   lifecycle, `commit_token`/`commit_activation`, metrics, shutdown) are
   unaffected — they always reply on the connection they arrived on, exactly
   as without `--next`.

   Verified on the real two-stage CPU setup: a single stateless request and a
   4-request persistent session both produce output byte-identical to the
   same requests over hub-and-spoke.

   Two real bugs surfaced during that verification, both fixed:
   - **Startup deadlock.** The first version connected to `--next` before
     opening this stage's own listeners (both the ring-input listener and the
     regular control listener). For any ring longer than one hop this
     deadlocks unconditionally: the last stage blocks dialing the
     coordinator's return listener before its own control listener is even
     open, and the coordinator won't open its return listener until every
     stage's control port has already answered — no startup ordering breaks
     that cycle. Fixed by binding every listener a stage owns before any
     blocking outbound connect; the OS accept queue holds an incoming
     connection until `accept()` is called, so binding early and accepting
     late is safe, and does not require accepting early too.
   - **Commit frames misrouted into the ring.** `commit_token`/
     `commit_activation` were included in the hot-path set that gets
     forwarded via `--next`, but `commit_final_token` (coordinator.cpp) was
     deliberately left on hub-and-spoke and sends those types over the direct
     per-stage connection, blocking on a reply there. With the bug, a stage
     forwarded the reply into the ring instead, so `commit_final_token`'s
     `exchange()` call never saw it. Symptom: persistent multi-request
     sessions failed on the second request (the first request's own
     end-of-request commit silently broke the connection state); a single
     stateless request happened not to exercise `commit_final_token` at all
     under the exact test run that first looked clean, which is why this
     wasn't caught by the single-request test alone. Fixed by excluding
     commit_token/commit_activation from the forwarded set — only
     prompt/token/activation/speculative_activation ride the ring.

   Prefill validation is weaker in ring mode than hub-and-spoke: the
   coordinator never sees the intermediate activation that would tell it the
   tokenized prompt length, so `require_result` falls back to its
   single-stage check (final position greater than the start) instead of the
   exact one hub-and-spoke gets for multi-stage prefill. Disclosed, not
   hidden — the stage's own KV/position bookkeeping remains the actual source
   of truth regardless; this only weakens a coordinator-side sanity check.

   Not measured: WAN latency reduction, the actual point of this phase. Not
   built: ring integration for `route_speculative`/pipelining, which needs
   its own design (per-hop metrics currently flow back through the
   coordinator's own relay threads; in ring mode they never pass through it
   at all) — the design doc originally scoped this as a likely-necessary
   follow-up, and verification confirmed it is a separate problem, not an
   incidental extension of the `route_step` work above.
5. Pipelined chunked prefill. Independent of the above and the larger
   time-to-first-token win on long prompts. **Built and verified, ring mode
   only.** New opt-in `--prefill-chunk N` on `dan-stage-worker` (0, the
   default, keeps today's single-batch prefill). One new wire type,
   `po::Type::prompt_chunk` (`include/provider_owned/protocol.hpp`): a
   non-final chunk of a prefill, shaped exactly like `activation` (extends KV,
   carries embeddings) but never triggers sampling, since more of the prompt
   is still coming.

   The design turned out simpler than expected once ring mode (phase 4)
   already existed: only the **first** stage needs new logic at all, because
   it is the only stage that ever turns one incoming frame into several —
   it's the one place raw prompt text gets tokenized, so it's the only place
   that can know where chunk boundaries fall. Splitting `tokens` into pieces
   and sending all-but-the-last immediately via `--next` (through a small
   `emit_chunk` callback threaded into `Stage::handle`/`execute`/`run_first`),
   then letting `tokens` fall through holding only the final chunk so the
   unchanged code below handles it exactly like an unchunked prompt would.
   Middle and last stages needed only one line each — `prompt_chunk` added to
   the small set of frame types each of `run_middle`/`run_last` already
   accepts — because ring mode's existing forward-only design means each
   chunk arrives as its own ordinary wire frame; a middle stage forwards it
   with the same generic `output = input` code path that already forwards
   `activation`, no new logic needed, and the coordinator's `route_step` is
   completely unaffected — it still does exactly one send and one receive per
   call, unaware that several chunks crossed the wire between them.

   One real design gap, caught while working through the design rather than
   at test time: the **last** stage's reply to a `prompt_chunk` is a plain KV
   extension with nothing useful to send anywhere — there is no next stage,
   and the coordinator's single `ring_return` read is waiting for the
   eventual real result, not a per-chunk acknowledgement. The existing ring
   thread forwards every outcome unconditionally; sending one of these into
   `ring_return` would have been misread as the actual result and failed
   validation. Fixed by adding a narrow, explicit drop rule to the ring
   thread — skip forwarding only when the input was `prompt_chunk`, the
   handling succeeded, and this is the last stage — instead of a more general
   "sometimes don't reply" mechanism (an `std::optional<Frame>` return, or
   similar) that would have widened `Stage::handle`'s contract for every
   caller to account for.

   Verified on the real two-stage CPU setup, `--prefill-chunk 32` against an
   ~80-token prompt (confirmed chunking fired: stage logs show two 32-token
   chunks plus a final partial one at both stages): output byte-identical to
   the same prompt with chunking off, for both a single stateless request and
   a 3-request persistent session. The persistent run's requests 2 and 3
   produced output identical to request 1 regardless of the actual follow-up
   prompt text — confirmed to be a raw-continuation model-behavior artifact
   (no chat template in this CLI path), not a bug: the identical
   pattern reproduces exactly on plain hub-and-spoke with no ring and no
   chunking involved at all.

   Not measured: the actual time-to-first-token win on long prompts, the
   stated purpose of this phase — everything verified is CPU, same-machine,
   an ~80-token prompt too short to show a meaningful difference either way.
6. Draft-model acceleration. Deliberately last: until the round trip is hidden,
   the draft is not the bottleneck. **Turned out to need no new code at all.**
   The premise going in (see the earlier "is it just translating it to C++?"
   discussion in this project's history) was that CUDA-graphing the draft
   model would be the hardest, most bespoke part of this whole plan, on the
   theory that it would need the same hand-rolled static-KV graph-capture
   machinery Shard built for their PyTorch draft model. That premise was
   wrong for this stack specifically: llama.cpp's own CUDA backend
   (`ggml-cuda.cu` in the vendored `llama.cpp` tree) already implements
   generic graph capture and replay for any repeated-shape decode
   sequence — capture on first use, replay on subsequent calls, automatic
   recapture on a shape change — gated only by a CMake option
   (`GGML_CUDA_GRAPHS`) and GPU architecture (disabled below Volta, `cc<70`).
   `DraftModel::propose`'s decode loop (`src/provider_owned/coordinator.cpp`)
   is exactly the shape this was built for: a single-token batch, same shape,
   called repeatedly. Nothing about it needed to change.

   What this phase actually consisted of: checking whether DAN's existing
   CUDA build already had the option on (`build-provider-owned-cuda`'s
   CMake cache: `GGML_CUDA_GRAPHS:BOOL=ON` — it did, already, before this
   session), rebuilding the coordinator and stage worker against that
   configuration to pick up every change from phases 1-5, and confirming on
   real hardware (RTX 2070) that capture actually engages for this specific
   workload rather than assuming it from the CMake flag alone. It does:
   `CUDA Graph id 27 reused` appears **122 times** in the coordinator's own
   log across one 120-token generation (`--draft-tokens 8`, GPU stages,
   GPU draft) — one captured graph, replayed continuously, no recapture. The
   target stages show the same pattern for their own (differently-shaped,
   batched) verify calls, 14 reuses across that run's 16 speculative rounds —
   graph capture benefits the whole GPU path here, not only the draft.

   Measured on that same run, not isolated from the surrounding GPU-vs-CPU
   difference (no ablation build exists to isolate graph capture alone — see
   below): `draft_ms/token` 5.664, `decode_tok_s` 122.665, `draft_accept`
   0.959, all GPU stages and GPU draft. For comparison, the CPU runs earlier
   in this document's own verification measured `draft_ms/token` in the
   60-130 range. That comparison conflates raw GPU-vs-CPU compute with the
   graph-capture win specifically, so it is reported as one honest data point
   from real hardware, not as an isolated measurement of what graphs alone
   are worth — producing that number would need a second CUDA build with
   `GGML_CUDA_GRAPHS=OFF`, not attempted here (a full CUDA backend rebuild,
   and the question it would answer is upstream llama.cpp behavior, not
   anything about DAN's own code).

## Instrumentation

The existing `speculative_rounds` and `draft_accept` counters
(`src/provider_owned/coordinator.cpp:1918`) change meaning once chunks overlap
and are redefined rather than reinterpreted. Per request, export:

- per-stage service time `tau_k`, separated from queueing delay
- loop `RTT`, and the derived `RTT / tau_max`
- time-weighted mean in-flight depth, and maximum in-flight depth
- computed frames per committed token, and discarded frames
- acceptance by draft index, `accept_by_depth[1..K]`
- draft milliseconds per drafted token

Two of these decide whether the design pays at all: `RTT / tau_max`, which is
the fill headroom and is worthless if it is near 1, and the decay of
`accept_by_depth`, which sets how much speculation survives.

## Acceptance criteria

- **Verified.** Pipelined output is token-identical to **today's serial
  K-chunk speculative path** (same `--draft-tokens`, `--pipeline-depth 0`),
  on CPU with the real Qwen2.5 1.5B Q4_K_M two-stage split (0-13/14-27), at
  `--pipeline-depth` 1, 4, and 8, `--draft-tokens 8`, `--tokens 60`, one
  prompt, with natural (not forced) partial rejections at every depth tested
  (`draft_accept` 0.70-0.75). Not against plain serial decode — see "A
  correctness fact, found while verifying rollback": batched K>1 verification
  already disagrees with plain serial decode today, for reasons unrelated to
  pipelining, so plain serial decode is the wrong baseline for this
  comparison. **Not yet done:** a fixed prompt *set* (only one prompt tried),
  forced (as opposed to naturally-occurring) rejections at every position in a
  chunk, and depths beyond 8.
- **Verified.** The token limit landing mid-chunk truncates cleanly to the
  exact requested count (`--tokens 60` produces exactly 60; see "Bugs found
  during verification" for the bug this replaced, where it produced 65).
  **Not yet done:** end-of-generation (EOG) landing mid-chunk was exercised
  incidentally (the verified prompt ends the same way in every run) but not
  deliberately forced at a chosen position.
- **Verified.** Four sequential stateless pipelined requests on one replica,
  alternating between two prompts, produce deterministic, repeating output
  (request 1 == request 3, request 2 == request 4) — no cross-request state
  leak.
- **Not yet done.** A stale reply arriving after a correction is dropped in
  every run so far (implied by the byte-identical output above, since a stale
  reply being wrongly applied would have changed it), but no test has
  disabled the epoch check specifically to confirm it — as opposed to some
  other part of the design — is what's catching it.
- **Verified.** Existing CTest suite (`provider_owned_protocol_test`,
  `provider_owned_formation_test`, `provider_owned_range_model_test`,
  `windows_coordinator_test`) passes unchanged, 4/4, with the pipelined code
  present but `--pipeline-depth 0` (none of these tests exercise
  `--draft-model` at all, pipelined or not — see the phase-0 verification
  note in "Status" history for that gap).
- **Not done.** A physical WAN rerun of the 32B configuration, reporting the
  instrumentation below alongside tok/s. Everything verified above is CPU,
  same-machine, small-model — it establishes correctness, not the throughput
  claim this document exists to make.
- **Verified.** Ring topology (`--next`/`--ring-listen`/`--ring-return`)
  produces output byte-identical to hub-and-spoke for a single stateless
  request and a 4-request persistent session, on the real two-stage CPU setup.
  **Not done:** ring mode with a draft model / `--pipeline-depth` active (out
  of scope — `route_speculative` and the pipelined threads stay hub-and-spoke
  regardless of `--ring-return`), three or more stages (only two-stage
  verified, though the design and code are not hardcoded to two), and any
  actual WAN latency measurement (the stated purpose of ring mode).
- **Verified.** Chunked pipelined prefill (`--prefill-chunk`) produces output
  byte-identical to unchunked ring-mode prefill, for both a single stateless
  request and a 3-request persistent session, with chunking confirmed to
  actually fire via the stage logs. **Not done:** the actual
  time-to-first-token win (the stated purpose of this phase — the verified
  prompt, ~80 tokens, is too short to show a meaningful difference either
  way), three or more stages, and chunk sizes other than 32.
- **Verified.** CUDA graph capture is active for the draft model on real GPU
  hardware (RTX 2070): `CUDA Graph id 27 reused` 122 times across one
  120-token GPU generation, confirming replay rather than a one-off capture.
  **Not done:** an isolated measurement of what graph capture alone is worth
  (would need a second CUDA build with `GGML_CUDA_GRAPHS=OFF`) — the reported
  `draft_ms/token` numbers reflect GPU-vs-CPU as a whole, not graphs in
  isolation.

## Remaining questions

- Acceptance `a` and the shape of `accept_by_depth` are unmeasured, so the
  projection is untested at its most sensitive input.
- `tau_max` is 34.762 ms against 14.156 ms on the other stage, a 2.4x spread on
  a two-stage split. Rebalancing the layer split by measured stage speed is
  untried and is a separate lever from pipelining.
- The design assumes one active request per replica, as in
  [runtime v2](PROVIDER_OWNED_RUNTIME_V2.md). Pipelining several sessions
  concurrently needs the join-before-control approach (see "Control frames and
  request drain") replaced by in-band control frames, since there would no
  longer be a moment where every pipeline thread for every session is
  guaranteed joined, and the stage worker's single-active-session guard
  (`src/provider_owned/stage_worker.cpp:330`) revisited.
- Greedy only. Lossless sampling at temperature requires speculative-sampling
  rejection at the last stage and is out of scope here.
- Stage execution is Qwen2-only by the loader's own check
  (`patches/llama-provider-owned.patch`), so nothing here has been considered
  against a windowed or recurrent architecture, where truncation by position
  would not be exact.
- Batched verification's divergence from plain serial decode (previous
  section) is unquantified beyond one prompt at K=8. Whether it grows with K,
  whether it is rare enough to ignore, and whether users should be told that
  `--draft-model` output can differ from a plain run at the same prompt are
  all open. None of this blocks pipelining — pipelining's own correctness bar
  is against the K-chunk speculative baseline, which already carries this
  property — but it is a real gap in what DAN currently tells users to expect.
- Ring mode's error-propagation rule (every stage's ring connections are
  forward-only; both success and error outcomes ride `--next`) was verified
  only via the happy path — no test forced a mid-ring error to confirm it
  actually surfaces at the coordinator's return listener rather than hanging
  or silently dropping. The rule is load-bearing for anything beyond two
  stages and untested beyond reasoning about it.
- Ring mode's graceful shutdown is incomplete: `ring_thread.join()` in
  `dan-stage-worker` can block forever if that thread is parked in a blocking
  `accept()` or `recv_frame()` call when shutdown is requested, since neither
  is interruptible by the `jthread` stop token used elsewhere in this file.
  Not exercised by verification, which stopped processes directly rather than
  through the graceful `Type::shutdown` path in ring mode specifically.
