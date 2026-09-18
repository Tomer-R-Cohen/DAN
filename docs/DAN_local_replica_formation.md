# DAN: Local Replica Formation for a Single Model

## Goal

Assume DAN serves **one model**.

The network contains:

- provider nodes with GPUs,
- client nodes,
- network locality,
- heterogeneous VRAM and compute,
- persistent model ranges already cached or loaded on providers.

The goal is to let providers organize themselves into useful model replicas **without a global scheduler**.

The central rule is:

> **A new provider should first try to cheaply complete an existing local replica. If it cannot, it should load the locally least-represented useful model range.**

This turns replica formation into a local, decentralized coverage problem.

---

## 1. Model representation

Let the model contain layers

\[
L = \{0,1,\ldots,n-1\}.
\]

A provider \(v\) can host some contiguous range

\[
I_v = [a_v,b_v).
\]

For example, with a 30-layer model:

```text
A: [0,10)
B: [10,20)
C: [20,30)
```

together form one complete replica:

```text
A → B → C
```

A **replica** is a set of providers whose assigned ranges collectively cover the full model exactly once in execution order.

---

## 2. Locality

A provider should not reason about the entire DAN network.

For provider \(v\), define a local neighborhood:

\[
N(v)
\]

containing providers that are cheap enough to use in the same inference path.

The neighborhood may be defined using:

- measured RTT,
- direct vs relayed connectivity,
- bandwidth,
- recent observed activation-transfer time,
- a maximum network-cost threshold.

The important point is:

> Coverage is measured locally, not globally.

A globally balanced network can still be locally unusable.

---

## 3. Local coverage

Partition the model into useful candidate ranges:

\[
S_1,S_2,\ldots,S_k.
\]

For the simplest prototype, these can be the same coarse ranges the planner already tends to produce.

Define local coverage for range \(S_j\) around provider \(v\):

\[
c_v(S_j)
=
\#\{u \in N(v) : u \text{ currently holds or serves } S_j\}.
\]

Example:

```text
local neighborhood around P

[0,10)   × 3
[10,20)  × 2
[20,30)  × 1
```

The local coverage vector is:

\[
C_v=(3,2,1).
\]

The bottleneck is `[20,30)`.

If a new provider can host that range, loading `[20,30)` improves the vector to:

\[
(3,2,2).
\]

---

## 4. First priority: complete a replica

Before balancing coverage, a new provider should determine whether it can complete a useful local replica.

Example:

```text
A: [0,10)
B: [10,20)
P: new provider
```

If \(P\) can host `[20,30)`, then:

```text
A → B → P
```

becomes a complete replica.

This should normally be preferred over creating another copy of an already-covered range.

But not every possible completion is worthwhile.

A candidate completion should be evaluated using predicted inference cost.

---

## 5. Replica completion cost

For a candidate path

\[
R=(v_1,v_2,\ldots,v_m),
\]

with layer assignment

\[
I_1,I_2,\ldots,I_m,
\]

estimate token time as:

\[
T(R)
\approx
\sum_{i=1}^{m} C_{v_i}(I_i)
+
\sum_{i=1}^{m-1} N(v_i,v_{i+1}),
\]

where:

- \(C_{v_i}(I_i)\) is estimated compute time for node \(v_i\) to execute its assigned range,
- \(N(v_i,v_{i+1})\) is estimated activation-transfer cost between consecutive providers.

Then approximately:

\[
\text{tok/s}(R) \approx \frac{1}{T(R)}.
\]

The exact estimator can become more sophisticated later.

For v1, even a rough model using RTT/direct-vs-relay plus coarse GPU speed classes is enough.

---

## 6. Provider join rule

When a new provider \(p\) joins:

### Step 1 — Discover the local neighborhood

Find nearby compatible providers and collect:

- model range currently loaded/cached,
- VRAM,
- availability,
- runtime compatibility,
- link cost,
- current replica membership.

### Step 2 — Search for cheap replica completions

Find incomplete local formations that \(p\) could complete.

For every candidate, estimate:

\[
T(R).
\]

If at least one completion is below an acceptable cost threshold:

> Join the cheapest valid completion.

### Step 3 — Otherwise improve local coverage

If no useful replica can be completed, choose a range that improves the weakest part of the local coverage vector.

The simplest rule is:

\[
S^*
=
\arg\min_{S_j} c_p(S_j),
\]

subject to:

- the range fits in \(p\)'s memory,
- the provider supports the runtime/model,
- the range is useful for a valid future partition.

If several ranges tie, prefer:

1. a range already cached,
2. the range requiring the least download,
3. the range that produces the best predicted future path,
4. otherwise random/deterministic tie-breaking.

---

## 7. Why max-min balancing appears naturally

Suppose local coverage is:

\[
C=(5,5,1).
\]

At most one independent complete replica can be formed if every replica needs one copy of each segment.

The least-covered segment limits capacity.

A natural objective is therefore:

\[
\max \min_j c(S_j).
\]

Equivalent intuition:

> Increase the local bottleneck coverage.

Another useful imbalance metric is:

\[
\max_j c(S_j)-\min_j c(S_j).
\]

Lower is better, but maximizing the minimum is more directly tied to the number of disjoint replicas that can exist.

---

## 8. Example: replicas emerge without a global scheduler

Assume three equal-sized ranges:

```text
S1 = [0,10)
S2 = [10,20)
S3 = [20,30)
```

Providers arrive sequentially.

### Provider A

No coverage exists.

```text
A → S1
```

Coverage:

\[
(1,0,0)
\]

### Provider B

Least-covered ranges are `S2` and `S3`.

```text
B → S2
```

Coverage:

\[
(1,1,0)
\]

### Provider C

`S3` is the bottleneck.

```text
C → S3
```

Now:

```text
A → B → C
```

is a complete replica.

Coverage:

\[
(1,1,1)
\]

### Provider D

All ranges tie.

```text
D → S1
```

Coverage:

\[
(2,1,1)
\]

### Provider E

```text
E → S2
```

Coverage:

\[
(2,2,1)
\]

### Provider F

```text
F → S3
```

Coverage:

\[
(2,2,2)
\]

Now two replicas can emerge:

```text
Replica R1:
A → B → C

Replica R2:
D → E → F
```

No global machine assigned the six providers.

The replicas emerged from local coverage balancing plus local leases.

---

## 9. Locality matters more than global counts

Consider:

```text
Region X:
A [0,10)
B [10,20)

Region Y, 500 ms away:
C [20,30)
D [20,30)
```

Globally the model may look reasonably covered.

Locally, Region X has:

\[
(1,1,0).
\]

A new provider in Region X should strongly prefer `[20,30)` even though that range already exists elsewhere.

Therefore DAN should never use global range counts as the primary signal.

Use:

\[
c_v(S_j)
\]

rather than:

\[
c_{\text{global}}(S_j).
\]

---

## 10. "Complete first, balance second"

The provider policy can be summarized as:

```text
NEW PROVIDER P

        ↓

discover local providers

        ↓

Can P complete a sufficiently cheap local replica?

        ├── yes
        │    ↓
        │  join the cheapest useful completion
        │
        └── no
             ↓
        find locally least-covered useful range
             ↓
        load/cache that range
```

This is the core decentralized primitive.

---

## 11. Sticky assignments

Providers should not constantly move between ranges.

Downloading/loading weights and destroying good replicas has a real cost.

A provider assignment should therefore be **sticky**.

Reconsider the range only when conditions are strong enough, for example:

- provider has been idle for a long time,
- its range is heavily overrepresented locally,
- another range is a severe local bottleneck,
- its replica dissolved,
- operator explicitly requests rebalancing.

Do not rebalance simply because a marginally better arrangement appears.

---

## 12. Heterogeneous GPUs

Counting providers equally is only a first approximation.

Eventually local coverage should represent **serving capacity**, not just copies.

Instead of:

\[
c(S_j)=\#\text{providers serving }S_j,
\]

use:

\[
c(S_j)
=
\sum_{v:S_j\subseteq I_v} q_v,
\]

where \(q_v\) may incorporate:

- compute throughput,
- number of sessions it can support,
- VRAM,
- reliability,
- availability.

However, this should be deferred until the basic decentralized behavior works.

A simple provider-count model is easier to validate.

---

## 13. Client behavior is a separate problem

Replica formation and client routing should remain separate.

Providers solve:

> Which local providers and ranges should exist so useful replicas emerge?

Clients solve:

> Which existing READY replica gives me the best expected experience?

For client \(u\) and replica

\[
R=(v_1,\ldots,v_m),
\]

a simple client cost is:

\[
T_u(R)
=
N(u,v_1)
+
T(R)
+
N(v_m,u).
\]

The client chooses:

\[
R^*
=
\arg\min_R T_u(R)
\]

among READY replicas with available capacity.

Thus:

```text
PROVIDER FORMATION:
optimize local replica quality and model coverage

CLIENT SELECTION:
optimize client → replica → client execution cost
```

These should not be collapsed into one algorithm.

---

## 14. Replica membership and races

Every provider may be capable of proposing a replica.

But one provider should belong to only one active replica in v1.

Existing first-wins leases can resolve competing formations.

Example:

```text
A wants: A + C + D
B wants: B + C + D
```

Both try to reserve C.

```text
C accepts A first.
C rejects B as busy.
```

B releases partial reservations and retries/replans.

This avoids requiring a global scheduler.

---

## 15. What happens if no replica needs the provider?

Then the provider is still useful.

It becomes a **prepared partial replica resource**:

```text
P:
cached/loaded range = locally scarce range
state = available
```

Future providers can discover it and complete a replica around it.

The distinction is:

```text
complete model path
    = replica

provider with useful range
    = replica-building resource
```

A partial range by itself is not a replica.

---

## 16. Initial decentralized algorithm

For a first implementation:

```text
on_provider_join(P):

    neighbors = discover_local_compatible_providers(P)

    completions = find_replica_completions(P, neighbors)

    if completions:
        R = minimum_predicted_cost(completions)

        if cost(R) <= COMPLETION_THRESHOLD:
            attempt_to_reserve(R)

            if success:
                form_replica(R)
                return

    coverage = calculate_local_range_coverage(neighbors)

    range = least_covered_range_that_fits(P, coverage)

    load_or_cache(range)

    remain_available_for_future_formation()
```

Formation races remain protected by worker leases.

---

## 17. Important open questions

The architecture is simple, but several parameters need experimental answers:

### Neighborhood definition

What makes another node "local"?

Possible definitions:

- RTT below X ms,
- activation-transfer benchmark below X ms,
- direct connection preferred,
- weighted network-cost radius.

### Range granularity

Should coverage be measured per:

- individual layer,
- fixed layer blocks,
- currently cached ranges,
- planner-generated partitions?

Fixed coarse blocks are probably easiest initially.

### Completion threshold

At what predicted token cost should a provider prefer completing a replica over improving coverage elsewhere?

This should come from measured DAN throughput.

### GPU heterogeneity

When should provider-count coverage be replaced by capacity-weighted coverage?

### Idle replica policy

How long should an unused replica remain intact before its providers reconsider their assignments?

These are optimization questions, not reasons to change the underlying architecture.

---

## 18. Relation to Shard / c0mpute

As of the current public design material reviewed in September 2026:

- Shard/c0mpute separates inference swarms from placement/control logic.
- Its currently implemented/live formation architecture has relied on a placement/control-plane brain rather than this local self-balancing rule.
- Their newer design work proposes moving placement toward a protocol in which multiple actors can derive/verify formations rather than trusting one permanent global decider.
- Their design discusses cached/held ranges and incentives around useful/rare ranges.
- I did **not** find the exact DAN rule proposed here:
  - first complete a cheap local replica,
  - otherwise increase minimum local model-range coverage.

So this idea should be treated as DAN's own decentralized formation primitive unless later research finds an equivalent implementation elsewhere.

---

## 19. Core rule

For a single-model DAN network with locality:

\[
\boxed{
\text{Complete a cheap local replica if possible;}
\quad
\text{otherwise increase the minimum local model coverage.}
}
\]

This creates a useful emergent behavior:

```text
providers do not ask:
"what did a central scheduler assign me?"

they ask:
"what is locally missing?"
```

Replicas then emerge from local decisions rather than global placement.
