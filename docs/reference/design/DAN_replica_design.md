# DAN: Self-Forming Persistent Replicas — Design Report

*2026-09-18. Starting point: `DAN_local_replica_formation.md`. Scope: one model, one ABI.*

> **Implemented (v1, 2026-09-18)**; see `docs/PROJECT.md` §9.8 for what was built and how
> it was tested. Three decisions changed from the first version of this report and are
> corrected below: (1) every planned ring link is measured from its sending side before
> anything is reserved; the proposer's own RTT is only a first filter (internet paths are
> not a metric, so no triangle bound). (2) The owner's relay is outside the ring but not
> free: it is asynchronous, with a bounded queue per client, and a client that cannot keep
> up is dropped. (3) `sessions_max` is a configured value (default 1), not chosen by the
> planner.

## 0. Summary

1. **Replicas can form with almost no new machinery.** Today a chat client already does the
   whole formation job (`place_route`: discover → plan → reserve → assign → link). Run that same
   code inside a provider node, keep the connections open instead of closing them at the end
   of a chat, and you have a persistent replica.
2. **The members need no new protocol.** A worker already serves a route for as long as the
   control connection that holds its lease stays open. It already keeps a separate KV
   cache per session (llama sequence ids), and it already keeps the route bound across
   sessions. "Replica member" is just "serving, with the lease held by the owner's connection".
3. **The coverage rule in the `.md` doesn't fit DAN.** In DAN any provider can take *any*
   contiguous range at formation time: `plan_stages` picks the split from memory. Which
   ranges exist does not limit how many replicas can form; **free memory reachable over good
   links** does. Cached ranges only change *how long* formation takes. So v1 needs no coverage
   balancing. "Complete first" becomes: *can `plan_stages` find a plan over me plus my
   viable free peers?* Coverage survives only as a tie-breaker when choosing among plans
   (and as an optional disk prefetch later).
4. **Data path: the owner handles the client edge, the ring stays direct.** The owner never
   enters the token loop: the tail feeds each token straight to the head, as today. The
   owner only relays the client's prompt in and the streamed text out, asynchronously and
   with a bounded queue per client, so it adds no hop to the token loop. That saves the most complex parts of a direct client path: per-session return
   sockets at the tail, per-client authorization on every member, and a NAT dial from the
   tail to every client. The owner should normally be the head, so the prompt path is the
   same as a direct one.
5. **Failures:** any member or owner loss dissolves the replica. Survivors keep their weights
   loaded, so re-forming with the same ranges takes the existing "stage already loaded"
   path, which takes seconds. There is no repair, election or consensus.

---

## 1. Current-state map

| Step | What happens today | Where |
|---|---|---|
| Provider join | `dan-provider` reads `provider.conf`, checks the GPU, starts `dan-sidecar` (`-dht client`, relay, `-inbound`, `-ring-inbound`, `-ring-proxy`, `-status-file`) and the serve-mode worker (`--control-listen`, `--ring-listen`, `--catalog`). The worker reads each catalog manifest plus its GGUF index and loads **nothing**. A status thread writes `worker-status.json`. | `engine/provider_launcher.cpp`, `engine/stage_worker.cpp` (serve main, status thread) |
| Advertise | The sidecar re-reads the status file each second (fresh status only) and publishes `dan/model/1/<sha>` for each catalog model (5 min validity, refreshed every validity/3, bounded). This means only "may serve M". | `sidecar/discovery.go` `advertiseModels` |
| Discovery | The client's sidecar candidate API (`DAN-CANDIDATES/1`, up to 8 SHAs): DHT providers → `/dan/capabilities/1.0.0` query to each (RTT and direct/relay measured **from the querying node**) → a local control forward per peer → `CANDIDATE <peer> <control> <MiB> <abi> <rtt> <path>`. Dead peers in the DHT cost up to the 10 s query timeout. | `discovery.go` `startCandidateAPI`/`findCandidates`/`queryCapabilities`; `placement.cpp` `discover_candidates` |
| Placement | `place_route`: open all forwards in parallel and read `provider_available` (state, memory, ABI, limits, `cached=`). Try models largest first; filter with `unusable()`; sort by `link_cost` (rtt/25 + relayed), then memory; try `plan_from_cache` (**exact tiling** from cached ranges only) and then `plan_stages` (fewest stages, memory-proportional split, ≤8 candidates, first plan wins). `stage_fits` holds back max(1 GiB, 15%) and counts weights + KV(context × sessions). | `engine/placement.cpp`, `engine/planner.cpp`, `formation.hpp` |
| Reservation | `reserve` goes to every chosen worker in parallel. The worker checks its own catalog, limits and `stage_fits`, then `WorkerLease::reserve` (**first wins**, ≤60 s). If any refuses, the client releases the rest, waits 100–500 ms and replans (3 attempts). The lease is tied to the reserving TCP connection (`held` in `serve_connection`). | `stage_worker.cpp` ~2008–2105, `lease.hpp` |
| Range loading | `assign_stage` → `begin_loading` → `load_assigned_stage`: `prepare_range_model` downloads only the missing byte ranges into the sparse `<model>-<b>-<e>.gguf` + `.ranges`; if the same range/context/sessions is already loaded it is **reused**; otherwise a new `Stage` is built. The draft model loads with `max_sessions = 1`. Reply `stage_ready`, lease `serving`. | `stage_worker.cpp` `load_assigned_stage`, `load_draft_model` (~1853–1966); `range_model.cpp` |
| Ring formation | `create_ring_session` sends `create_session` with a route payload (`next`, `previous_peer`, `loop`) to all stages in parallel. `bind_route` dials `next` through the sidecar ring proxy (retrying each second within 20 s), sets `expected_previous`, and dials the loop link in the background. The tail's `next` is the client's return listener (`/p2p/<client>`, the client sidecar's single `-ring-inbound`). **The binding stays for the whole control connection** (`bound`); a later `create_session` with the same route returns at once. | `client.cpp` `create_ring_session`; `stage_worker.cpp` `bind_route`, `serve_control` |
| Sessions / KV | `Stage` has `n_seq_max = sessions` KV sequences; `create/reset/destroy_session` manage one sequence each. **Only one session executes at a time** (`active_session_`). The draft mirrors every session command. | `stage_worker.cpp` `Stage::create/reset/execute` |
| Inference | Loop mode: `stream_prompt` (token budget) → tail, prompt → head over control. Activations go A→B→C over the ring; the tail samples, streams `client_chunk` to `next`, and feeds the token to the head over `loop`. Speculation: `draft_round` at the head, `verify_round` at the tail, guesses appended after the activations. At the end the client sends `end_request`/`commit_token` to every stage. `LoopState` holds **one** active request. | `continue_loop`, `run_ring`, `draft_round`, `verify_round`; `client.cpp` `generate_loop` |
| Client disconnect | Each worker's control connection closes → `ring.disconnected()` (**all** sessions and draft sessions cleared) → the lease is released → `release_route` (next/loop closed, `expected_previous` cleared). **The loaded `Stage` stays in VRAM**; range files stay on disk. | `serve_control`, `RingState::disconnected`, `release_route` |
| Worker disconnect | Neighbours see their ring socket close → `ring.disconnected()` clears all sessions; the client's control socket to the dead worker fails and the request fails; the client exits and every lease is released. There is **no reconnect** at route level: a sidecar only re-resolves a peer when a new stream is dialed. | `run_ring`, `client.cpp` |
| Idle | The control socket's receive timeout is `serving_idle_ms` = 10 min after `stage_ready`; an unreserved connection times out after 60 s. | `stage_worker.cpp:1744`, `:2091` |

The existing code already provides three properties the redesign needs:
- The lease acts as a fence. Authority is an open TCP connection, not a belief, so two
  holders are impossible.
- Weights (on disk and resident) outlive routes and sessions.
- KV is per session.

---

## 2. Minimal decentralized architecture

```text
every provider node = worker + sidecar + formation loop (new, but it is today's place_route)

formation loop (on a FREE node)            members (plain workers, unchanged protocol)
  discover free viable peers ─────────────▶ capabilities answers (existing)
  plan_stages / plan_from_cache (existing)
  reserve / assign_stage (existing) ──────▶ lease: reserved → loading → serving
  link ring + warm-up session (existing) ─▶ bind_route, loop link
  READY: advertise dan/replica/1/<sha>
  front door for clients ◀──────────────── client: create_session / prompt / chunks
```

**Roles**
- **Provider:** every node. Its formation loop runs only while its own worker is FREE.
- **Owner:** the provider whose loop won the reservations. It holds its members' leases over
  its control connections (exactly as a chat client does today), advertises the replica,
  admits sessions, and relays the client edge. Its authority covers only its own replica, and
  only while its connections live. The owner is always a member, normally the head.
- **Replica:** `route_id` + an ordered member list, READY only when fully linked and warmed up.
- **Session:** one KV sequence on every member, allocated by the owner.
- **Client:** talks only to the owner.

**Where the owner runs (v1):** `dan-client` in a new mode (`--form`), started by `dan-provider`
and using the node's existing sidecar (which gains `-candidate-api` and a front-door
forward). This reuses `place_route`, `InferenceClient`, speculation choice and the model index
unchanged. It can be folded into `dan-provider` later.

**What is not added:** no member-to-member control traffic, no replica membership protocol,
no consensus, no election, no new worker states, no global view.

---

## 3. Formation algorithm (provider side)

```text
formation_loop(self):                       # runs only while self's lease is AVAILABLE
  sleep(jitter(5..15 s) + rank_delay)       # rank_delay: see "races"
  if self.lease != AVAILABLE: continue      # someone else reserved me -> I'm not a proposer

  peers  = candidate_api(model_sha)         # existing discovery: state, MiB, ABI, cached=, rtt, path
  viable = [p for p in peers
            if p.state == available and p.abi == self.abi
            and p.rtt <= RTT_MAX                 # hard locality gate (v1: 60 ms)
            and (p.path == direct or ALLOW_RELAY)]
  viable = [self (rtt 0, direct)] + sort_by(link_cost, memory)(viable)[:7]

  # demand hook, v1 supply rule: don't duplicate an idle replica next door
  if exists READY replica R nearby (owner rtt <= RTT_MAX) with sessions_in_use == 0: continue

  plan = best_plan(viable, context=C, sessions=S)      # §4: reuse-first, then plan_stages
  if not plan or self not in plan: continue            # the proposer must be a member
  if not quality_ok(plan): continue                    # stages <= 4, est. ring time <= T_MAX

  route_id = random_route_id()
  placed = place_route(plan)          # existing: parallel reserve, first wins; on refusal
                                      # release all, back off 100-500 ms, replan (3 attempts)
  if not placed: continue             # partial reservations already released

  # members load in parallel (existing): reuse resident stage > sparse cache > download
  link ring: create_session(route payload) to all   # existing create_ring_session
  warm-up: one 1-token prompt, then destroy_session  # also removes first-request warm-up
  READY: write replica status -> sidecar advertises dan/replica/1/<sha>
  serve(front door) until any member connection or the ring breaks -> dissolve
```

**Network feasibility:** v1 uses what DAN already measures (the RTT and path of the
capabilities query) as a first filter, plus `link_cost` for ordering. That RTT is only
proposer↔peer; internet routing is not a metric (A↔B and A↔C can be 20 ms while B↔C is
150 ms). So after planning, and before reserving anything, the owner asks each sending side
of every planned link (each hop and the loop back to the head) to measure it
(`/dan/probe/1.0.0`: dial, the ring's 5 s hole-punching wait, best of 3 pings). A link over
the limit leaves its non-head end out and the planner runs again. The warm-up token is the
final end-to-end check.

**Why bandwidth can be ignored for now:** a decode step carries hidden × 4 B of activations
(20 KB for 14B), so decode is latency-bound. Prefill (context × hidden × 4 B, ~10 MB at 512
tokens) is where bandwidth matters. Add `S/B` later, once chunked prefill lands.

**Quality gate (v1):**
- valid plan,
- all edges pass the gate,
- stages ≤ 4,
- estimated ring time `Σ (rtt/2 over the m ring edges incl. tail→head) ≤ T_MAX`.

There is no GPU speed model yet. A plan that passes the gate is formed even if it isn't
globally optimal.

**Races:**
- Leases stay the only correctness mechanism: the first reservation wins, the loser releases
  and replans.
- **Rank delay** (to avoid livelock when many nodes start together): each node computes its
  rank among the viable free set, by memory (descending) then PeerID, and waits
  `rank × 3 s` more. The largest free GPU usually proposes first; the others then see its
  members as `reserved` and plan around them.
- This is only a heuristic. Two nodes with different views still just race, and the leases
  settle it.

**Why the proposer must be a member:**
- A node can't grab other GPUs without contributing.
- The number of proposers is bounded by the number of free nodes.
- The owner shares its fate with a member. Owner death then counts as one member failure, not
  an extra failure domain.
- The proposer sorts first (rtt 0), so `plan_stages` will usually make it the head.

**Formation commit:** the replica is READY only after every member has replied `stage_ready`,
every `bind_route` has succeeded, and a warm-up token has gone all the way around the ring.
Any failure before that releases everything; the leases already guarantee this cleanup.

---

## 4. Coverage representation: what "least represented weights" should mean

**Why the `.md` rule is wrong for DAN**
1. **Ranges aren't fixed resources in DAN.** The `.md` treats a provider as holding a
   segment S_j, and replicas as needing one copy of each segment, so max-min coverage limits
   the replica count. In DAN a FREE provider holds no committed range: the range comes from
   `plan_stages` at formation, sized to its memory, and any missing bytes are downloaded.
   Formation is limited by **how much free memory is reachable over viable edges, and how it
   splits into ≤ 8 stages that pass `stage_fits`**, not by which layers are cached.
   - *Counterexample:* three viable free providers all cached `[0,10)`, so coverage is
     `(3,0,0)`. `plan_stages` still forms A`[0,10)` B`[10,20)` C`[20,30)`; B and C download.
   - *The reverse:* coverage `(1,1,1)` with a 400 ms relayed edge must not form.
2. **Fixed buckets conflict with the proportional split.**
   - A 24 GB + 8 GB pair splits about 75/25. Equal pre-cached buckets never tile that, and
     `plan_from_cache` only accepts **exact** tilings from cached start points.
   - Pre-caching "scarce buckets" would therefore mostly waste downloads, and could
     push the planner toward worse, bucket-shaped splits.
3. **Caching in VRAM because a range is scarce would be wrong.** It holds volunteers' GPU
   memory for replicas that don't exist.

**What coverage should mean in DAN**
- **v1: feasibility first.** "Can I complete a replica?" becomes "does `plan_stages` succeed
  over me plus my viable free peers?" There are no partial-replica objects, only FREE
  providers, some holding warm or cached ranges.
- **Cost to ready as the tie-breaker (GENERALIZE `plan_from_cache`).** Today it is all or
  nothing. Replace "exact tiling first" with a score over the candidate plans:
  - `missing_bytes = Σ stage bytes not on that member's disk`
  - `reload = members whose resident stage ≠ assigned range`
  - Rank plans by (stages, reload, missing_bytes, link cost). Candidate plans are the exact
    cache tiling (if any), plus `plan_stages` with cut points snapped to cached boundaries
    whenever `stage_fits` still holds, plus the plain proportional plan.

  This keeps the existing 201 s → 14 s cache win and also captures partial reuse.
  `prepare_range_model` already downloads only missing bytes, so any overlap helps.
- **Later, if downloads become the bottleneck: byte-weighted per-layer coverage.** Define
  `c(ℓ) = Σ over viable peers holding layer ℓ on disk`, with layer bytes as weights. A FREE
  provider that couldn't form for a long time may **prefetch to disk only** the contiguous
  window, of its proportional-share size, that minimizes Σ c(ℓ).
  - Layers are the right unit, because `.ranges` and `cached=` are already per layer range.
    Buckets are not.
  - Disk only, never VRAM.
  - This is the max-min idea restricted to where it is valid: download time. It becomes
    capacity-weighted (speed, sessions) only when replicas are demand-limited.

**Verdict:** drop "otherwise load the scarcest range" from v1. The v1 rule is: **"form if a
viable plan exists (reuse-first among plans); otherwise stay FREE and discoverable."** Add
the disk prefetch only if measurements show downloads dominate formation.

---

## 5. Replica data structure

What the owner holds, and what a live query returns:

| Field | Why | Source |
|---|---|---|
| `replica_id` | identity; one per formation, never reused | = the existing random `route_id` used in every lease |
| `owner` | who to talk to | the authenticated PeerID of the answering stream (not a payload claim) |
| `model_sha256`, `draft_sha256` | what it runs; whether it speculates | the formation request |
| `members` | ordered `(peer, begin, end)`; head = first, tail = last | the plan |
| `context`, `sessions_max` | fixed at formation (the `Stage` is built with them) | `StageRequest` |
| `sessions_in_use` | capacity (live only, never in the DHT) | the owner's session table |
| `runtime_abi` | compatibility | the members' greetings |
| `state` | `forming` or `ready` (only ready is advertised) | the owner |

**Left out:**
- epoch/term: a new formation gets a new `route_id`, and the lease-connection is the fence,
- heartbeat counters,
- tok/s predictions: added later, when clients rank by them,
- a separate head/tail field: implied by the order.

---

## 6. Protocol changes

**Members:**
- **No new frames.**
- *Small:* the idle timeout must not dissolve an idle replica. Either the owner sends a
  periodic harmless frame (`metrics`, if it accepts no session; otherwise a no-op), or
  `serving_idle_ms` doesn't apply once the holder has linked a route.

**New or extended:**

| # | Change | Kind |
|---|---|---|
| 1 | DHT key `dan/replica/1/<sha>`, advertised by the **owner's** sidecar from a replica status file (the same fresh-status mechanism as `advertiseModels`) | GENERALIZE |
| 2 | Capabilities response gains an optional `replica` message (§5). The owner's sidecar already answers `/dan/capabilities/1.0.0` from a status file | GENERALIZE |
| 3 | Candidate API: `REPLICAS <sha>…` → `REPLICA <owner> <control-forward> <rtt> <path> <id> <free sessions> <context> <stages> <draft>` | GENERALIZE |
| 4 | Front door: the client opens a control stream to the owner (a new protocol id forwarded to the owner's local port, like `/dan/transport/1.0.0` to the worker). It speaks **existing frames**: `create_session`, `reset_session`, `destroy_session`, `stream_prompt`, `prompt`, `cancel_request`, `end_request`, `commit_token`. Replies (`client_chunk`, `result`, `ack`, errors) come back **on the same stream** | NEW (small) |
| 5 | Return path to the owner: the tail's `next` must reach the owner's return listener. The owner is usually the head, and a node's sidecar has one `-ring-inbound` (to the worker). So a return stream needs its own protocol id or target suffix (e.g. `/p2p/<peer>/return` → `-return-inbound`) | SMALL CHANGE |
| 6 | The owner maps each client connection's session ids to its own internal ids, so clients can't collide with or touch each other's sessions | owner-internal |

To the client, the owner looks like **a one-stage route in loop mode**, so
`InferenceClient::generate_loop` needs only one change: read chunks from the control stream
instead of a return listener.

---

## 7. KV / session architecture

- **Weights vs KV lifetime are already decoupled.** Weights live as long as the `Stage`
  (beyond routes). KV lives in session sequences, which are cleared by
  `destroy_session`/`reset_session`, or all at once by `disconnected()` when the owner's control
  connection or the ring breaks. A replica's weights therefore survive clients by
  construction, and all KV dies when the replica does. Both are correct.
- **Capacity:** `S` is configured (`replica_sessions`, default 1) and planned like any
  session count (`stage_fits` counts KV × S). Trading sessions against model fit is later work. `sessions_max = S` and free sessions =
  `S − in use`. `reserve` already refuses `sessions > max_sessions`.
- **Create:** a client's `create_session` → the owner allocates an internal id and sends
  `create_session` to all members (the existing parallel call; the route payload is identical,
  so `bind_route` returns at once). If any member fails, the owner destroys the others, as
  `create_ring_session` already does, and returns an error.
- **Reset** (`/new`) → `reset_session` to all. **Destroy** → `destroy_session` to all, and the
  slot is freed.
- **Client disconnect / idle:** the owner cancels any active request (`cancel_request` to the
  tail, waits for the final result, `end_request`), then destroys **only that client's
  sessions**. An idle-session timeout (e.g. 10 min) frees slots held by vanished clients.
- **Multiple sessions in v1:** KV slots already coexist, but `Stage::execute` allows one
  executing session and `LoopState` one streaming request. So the **owner queues requests
  FIFO** and runs them one at a time. Sessions persist between turns, so several chats share
  a replica with turn-level interleaving and no worker change.
  - Later: per-request `LoopState`, lifting the `active_session_` gate, and batching.
- **Replica failure:** all sessions fail. KV never migrates.

---

## 8. Speculation

- **Where it lives:** the draft sits on the head and is chosen by the owner at formation, as
  `place_route` does now; `draft_sha256` goes in the replica record. The draft
  stays resident with the stage (warm).
- **Ring path unchanged.** `draft_round`, `verify_round`, guesses appended after activations,
  and position-based implicit rollback all apply per session sequence. So persistent replicas
  keep today's 9.6 → 17.4 tok/s on split routes and 19.7 → 36.8 on one GPU.
- **Required fix:** `load_draft_model` builds the draft with `max_sessions = 1`. A replica with
  `S > 1` needs the draft built with `S` sequences, or the second `create_session` fails on
  the draft.
- **Per-session state:** `RingState.guessed/guessed_at` and `draft_round`'s catch-up are
  per-route values. They are correct while the owner serializes requests (v1). Make them
  per-session before requests run concurrently. (`draft_session` is declared but unused;
  remove it or use it then.)
- **Opt-in semantics:** a replica with a draft always speculates (`draft_round` runs whenever
  `ring.draft` is set). Because speculation can rarely flip a word, v1 lets a
  node's config choose it, and the client may prefer replicas with or without a draft. A
  per-request flag comes later.

---

## 9. Client path

```text
REPLICAS <sha>  →  live answers only (the DHT is a hint)  →  drop full / non-READY
→ rank by owner link_cost, then free sessions  →  open front door  →  create_session
→ stream_prompt + prompt  →  client_chunk… result  →  end_request / commit_token  →  … →  close
```

**Direct (Client→head…tail→Client) vs owner edge, measured against the real code:**

| | Direct client path | Owner edge (recommended) |
|---|---|---|
| Token loop (tok/s) | tail → head | tail → head (same, **no cost**) |
| Prompt in | client → head | client → owner, where owner = head normally: **same** |
| Text out | tail → client | tail → owner → client: +1 one-way hop on display only |
| Session setup | the tail dials each client over a ring stream (direct-wait up to 5 s, then relay for NATed clients) | one control stream to the owner (no direct-wait), ~1 RTT |
| Worker changes | tail: session → return socket map; head and all members: accept control frames from arbitrary clients, which requires an owner-issued admission per session on every member (a new message); `end_request`/`commit_token` fan-out from untrusted clients | none |
| Multi-session | per-session sockets and admission everywhere | owner demultiplexes by session |
| Relay load | a relayed stream per client per tail | text only (~20 B/token) via the owner |

**Recommendation:** the owner edge with a direct ring. The direct path buys only one display
hop and costs a new admission protocol on every member plus per-session NAT dials. Revisit
the direct path only if owner fan-in becomes a real bottleneck.

**No READY replica:**
- During migration: fall back to today's client placement (it exists and works; the leases
  keep it safe alongside formation).
- Final: wait up to ~30 s, polling (free providers form on their own), then fail cleanly with
  "no replica ready; N free providers seen".
- No "form for me" request in v1.

**Replica full:** try the next one. **Slow client:** the owner buffers a bounded amount per
client; on overflow it cancels that request and drops the client. It never blocks the tail's
return socket.

---

## 10. Failure table

| Failure | Minimal correct behavior | Mostly existing? |
|---|---|---|
| Member dies during inference | Neighbours' ring sockets close and `disconnected()` clears their sessions. The owner sees the control failure or keepalive timeout, fails active sessions with an error, closes all member connections (leases release, stages stay resident), withdraws the advertisement, and re-forms after jitter. | yes, plus owner logic |
| Member dies while idle | Detected by the owner keepalive (~30 s); same as above. | keepalive is new |
| Proposer dies while forming | Connections close, reservations release (or expire ≤60 s), members return to FREE. | yes |
| Owner process dies | Every member control connection closes → release. **The replica dissolves.** The data path could technically run on, but nobody could admit or clean up sessions, so there is no reason to keep it. | yes |
| Owner's node dies | Same as above (the owner is a member). | yes |
| Network partition | Each member obeys only the connection holding its lease. A cut connection releases the lease, so no second owner can exist for the same lease, and there is no split brain. A partitioned owner loses its members. Future failover would need fencing by (`route_id`, epoch) on the lease, not "I think the owner died". | yes |
| Client disappears | The owner cancels and destroys only that client's sessions. | owner logic |
| Partial session create | The owner destroys the sessions it created, then errors (as `create_ring_session` already does). | yes |
| Formation race | First-wins leases; the loser releases and replans; rank delay makes this rare. | yes |
| Stale DHT record | The client live-queries. Roadmap item 2 (don't wait for dead peers) applies to replica queries too. | partly |
| Relay limit (4 GiB / 2 h) or path change | Persistent control and ring streams over the relay **will** be cut. DAN has no route-level reconnect: a cut link dissolves the replica, and it re-forms from resident weights in seconds. The v1 gate prefers direct edges; raise the relay's duration cap for low-traffic control streams, or refresh routes periodically. | new concern |
| Member's GPU pressure (friend starts a game) | Out of scope for v1: loading/serving errors dissolve the replica. | — |

---

## 11. Optimization opportunities

| | Required for correctness | Required for viable performance | Later |
|---|---|---|---|
| Cache reuse | — | reuse-first plan scoring (GENERALIZE `plan_from_cache`) | disk prefetch of scarce layers |
| Warm GPU state | re-form reuses resident stages (existing) | warm-up token at formation | idle unload after N min FREE (check: today a released stage stays resident indefinitely) |
| Network cost | hard RTT gate; ring edges must bind | prefer direct; `link_cost` order | peer↔peer RTT from members, `S/B` for prefill |
| Heterogeneous speed | — | — | GPU speed class in the greeting and in the estimate |
| Stage count | ≤ 8 (planner) | ≤ 4 gate, fewest first | — |
| KV capacity | `S` fixed at formation; `stage_fits` counts it | pick the largest `S` that fits | KV-aware capacity, variable `S` |
| Multi-session scheduling | owner FIFO, one active request | — | concurrent requests, batching |
| Replica demand | supply rule: no duplicate idle replica nearby | — | "all full" → form; long idle → dissolve |
| Idle rebalancing | none (sticky by construction: serving members are invisible) | — | hysteresis rules |
| Speculation | draft with `S` sequences | kept as today | per-session guess state, per-request opt-in |
| Direct vs relay | — | gate + prefer direct | relay cap policy |

---

## 12. Reuse classification

| Subsystem | Verdict | Note |
|---|---|---|
| Provider launcher | SMALL CHANGE | also start `dan-client --form`; sidecar gains `-candidate-api` and the front-door and return forwards |
| Stage worker (serve) | KEEP | plus the draft sessions fix, keepalive/idle rule |
| `WorkerLease` | KEEP | its holder is now the owner; it is the fence |
| `plan_stages`, `stage_fits` | KEEP / MOVE RESPONSIBILITY | run by the provider's formation loop |
| `plan_from_cache` | GENERALIZE | reuse score instead of exact-tiling-only |
| Model index + cache | KEEP | |
| Range cache/download | KEEP | |
| Worker catalog | KEEP | workers still refuse unknown SHAs; no URLs from anyone |
| Candidate discovery | GENERALIZE | add `REPLICAS`; the formation loop reuses `CANDIDATE` |
| DHT advertisements | GENERALIZE | add `dan/replica/1/<sha>` from the owner |
| Capabilities protocol | GENERALIZE | optional `replica` message |
| PeerID identity | KEEP | owner = the authenticated PeerID |
| Path measurement (RTT, direct/relay) | KEEP | becomes the formation gate |
| Route/session creation (`create_ring_session`) | MOVE RESPONSIBILITY | client → owner |
| `RingState`, `bind_route`, loop | KEEP | route bound for the replica's lifetime |
| `InferenceClient` | KEEP (owner) / SMALL CHANGE (client) | the owner drives members with it; the client reads from the control stream |
| Prompt streaming (`stream_prompt`, `client_chunk`) | KEEP | relayed by the owner |
| Tail return path | SMALL CHANGE | return target distinct from the worker's ring inbound |
| KV/session state | KEEP | `S` sequences; owner FIFO in v1 |
| Speculative decoding | KEEP / SMALL CHANGE | draft `max_sessions = S` |
| Status file / dashboard | SMALL CHANGE | show `replica_id`, role (owner/member), sessions |
| WAN reconnection | KEEP / DEFER | dial-time re-resolution only; no route repair |
| Client-side placement | KEEP, then debug-only | fallback during migration |

---

## 13. Migration plan (DAN works after every step)

1. **Owner mode by hand.** `dan-client --form --manifest M` on a provider node: it runs
   `place_route` including its own worker, keeps the connections, links the ring, warms up,
   keeps members alive, and exposes the front door. Includes the fixes in §6 #4–#6, §8 and the
   keepalive. The normal chat path is unchanged.
2. **Replica discovery.** The owner writes a replica status file; its sidecar advertises
   `dan/replica/1/<sha>` and answers the `replica` capability; the candidate API gets `REPLICAS`.
3. **Client `--replica`.** Discover, rank, and chat through an owner; fall back to today's placement.
4. **Automatic formation.** `dan-provider` runs the formation loop (gate, rank delay, supply
   rule, reuse scoring). DAN Chat prefers replicas.
5. **Concurrency.** Per-request loop and speculation state; lift the `active_session_` gate.
6. **Client placement becomes debug-only.**

"Promote a client's route into a replica" is not worth it: the leases are tied to the chat
client's own connections, so handing them over would need a transfer protocol. Step 1
runs the same code in a long-lived process from the start.

---

## 14. Prototype

**Formation test** (one PC + one RunPod pod, as before; or the local libp2p rehearsal with 3 CPU workers):

1. Start providers A, B and C with formation enabled and nothing loaded. **No client runs.**
2. Within ~30 s, exactly one owner (the highest rank) logs
   `replica <id> READY: A[0,x) B[x,y) C[y,N)`. The other loops log `not free` or `lost race`.
   `dan/replica/1/<sha>` is advertised.
3. Client 1 (`dan-client --replica --chat`) finds it, gets output **byte-identical** to
   today's placement baseline on the same split, and disconnects. Worker logs show only
   `destroy_session`, with no `release_route` and no lease change.
4. Client 2 finds the **same `replica_id`**. Its worker logs show **no** reserve, assign,
   load or bind. Time to first token ≈ discovery + one owner round trip.
5. Both clients connected with `S ≥ 2`: both chats work, turns run one after another, and
   each conversation keeps its own context.
6. With `--speculate` on the replica, tok/s is within noise of today's split-route numbers.

**Failure test:**

7. Kill B mid-generation. The client gets a clean error within the keepalive interval; A and C
   log `route … ended` (FREE, stage still resident); the advertisement disappears.
8. Restart B. The replica re-forms with the same ranges via the reuse path (no downloads,
   no reload on A and C) and advertises a new `replica_id`.
9. Race: start 4 providers at once where only 3 are needed. One replica forms and one
   provider stays FREE, with no stuck reservations after 60 s.

---

## 15. Challenges to the proposal

1. **"Complete a partial replica" has no object to complete.** DAN has no partial
   replicas; it has FREE providers, some with warm or cached layers. Completing one is the
   same as planning over the free viable set, which the planner already does.
2. **Max-min coverage fits fixed-segment systems, not DAN.** Replica count is limited by
   reachable free memory and stage fit, not by which layers exist. Coverage matters only
   for download time, and there it should be per layer, byte-weighted, disk-only, and used
   as a tie-breaker (§4).
3. **"Load the scarce range" contradicts warm-state hygiene.** It either holds VRAM for
   nothing or guesses boundaries the planner won't choose. Disk prefetch is the most it
   should do, and only if needed.
4. **The owner should be in the client edge.** With loop mode, the owner edge is free in
   tok/s. The direct client path is the *more* complex option (§9).
5. **Most of the redesign already exists.** The lease is the membership and the fence; a
   bound route already survives across sessions; KV is already per session; weights
   already outlive routes; failed formation already cleans up. The real new pieces are:
   - the formation loop and its gate,
   - the owner's front door and session table,
   - replica advertisement and query,
   - a distinct return stream,
   - a keepalive,
   - the draft's session count.
6. **Relay limits are the new risk.** Persistent streams meet the relay's 2 h cap.
   Re-forming from resident weights makes that cheap, but the gate should prefer direct
   edges.

**Rough cost:**
- steps 1–3 plus the prototype: ~1 week,
- step 4 (auto-formation): 2–3 days,
- step 5 (concurrency): later.

**Deferred:**
- global anything, consensus, election, handoff, KV migration, repair,
- batching, dynamic rebalancing,
- payments, reputation, proof of compute, Sybil resistance,
- multi-model supply (the replica record and DHT key are already per SHA, so nothing here
  blocks it).

## 16. Invariants (as listed, plus these from the code)

- A provider belongs to ≤ 1 replica: one lease, one bound route, `next` refuses a second route.
- Replica membership = the set of open lease-holding connections of one owner. It ends the
  moment any of them closes.
- A `route_id` is never reused; a re-formed replica is a new replica.
- A READY replica has passed one warm-up token end to end.
- Only the lease holder's connection can bind a route or send session commands to a member.
  A ring predecessor must be the authenticated `previous_peer`.
- Clients never reach members' control ports; they only reach the owner.
- Session ids on members are allocated by the owner and never chosen by clients.
- `context` and `sessions_max` are fixed for a replica's lifetime (the `Stage` is built
  with them).
- The planner currently accepts only dense Qwen2 models (`compatible_dense_qwen2`); formation
  inherits that limit.
- The DHT holds no volatile state (sessions, load). Live queries are authoritative.
- A worker is FREE ⇔ its lease is available. Only FREE workers are candidates, so READY
  replicas are never stolen, and placement is sticky without extra rules.
