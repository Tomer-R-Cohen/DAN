# DAN Cryptography, Verification, and Useful-Work Security

**Status:** Technical design draft  
**Scope:** Provider identity, authenticated control, signed replica membership, client authorization, work receipts, probabilistic verification, dispute handling, and the future payment boundary.  
**Non-goal:** Designing a blockchain, consensus protocol, token, or full zkML system.

---

# 1. Security model

DAN should separate four different properties that are often incorrectly grouped together under "proof":

| Property | Question | Primary mechanism |
|---|---|---|
| Authentication | Who is this peer? | libp2p authenticated identity / PeerID |
| Authorization | Was this operation permitted? | Signed capability / job authorization |
| Integrity / attribution | Who claimed or emitted this record? | Digital signature + hash |
| Computational correctness | Was the inference actually computed correctly? | Independent verification / redundant execution |

A digital signature proves attribution:

\[
\operatorname{Verify}(pk_A, m, \sigma_A)=1
\]

means:

> the holder of the private key corresponding to \(pk_A\) signed message \(m\).

It does **not** prove:

\[
y = f(x)
\]

for a claimed inference computation.

Therefore DAN must never treat a signed activation hash, signed receipt, or signed work claim as a proof of correct inference.

---

# 2. Threat model

Initially assume any individual provider can behave Byzantine.

A malicious provider may:

- claim work it never performed;
- return arbitrary activations;
- replay an old authorization;
- alter request metadata;
- claim another provider's layer range;
- double-claim overlapping work;
- lie about VRAM, model availability, latency, or loaded state;
- disconnect after accepting work;
- selectively behave correctly only when it predicts verification;
- create multiple PeerIDs;
- collude with one or more other providers;
- send malformed protocol frames;
- attempt to impersonate a replica member;
- attempt to reuse a valid signature in a different protocol context.

Initially **out of scope**:

- an adversary controlling the majority of all available compute;
- global Byzantine consensus;
- censorship resistance against a nation-state adversary;
- perfect anonymous Sybil resistance;
- generic succinct proof of arbitrary neural-network execution;
- secure multiparty computation;
- privacy-preserving inference against all replica members.

---

# 3. Cryptographic primitives

## 3.1 Provider identity

DAN already uses libp2p authenticated connections and PeerIDs.

That identity should remain the provider's primary network identity.

Conceptually:

```text
provider private key
        |
        v
provider public key
        |
        v
PeerID
```

Application-level signatures should be produced either:

1. directly by the libp2p identity key if the implementation exposes a stable signing API; or
2. by a dedicated application signing key cryptographically bound to the PeerID.

If option 2 is used:

```text
PeerKeyBinding {
    peer_id
    application_public_key
    key_epoch
    valid_from
    valid_until
}
```

must itself be signed by the libp2p identity key.

Do not silently create an unrelated signing identity.

---

## 3.2 Hash function

Use one protocol-wide cryptographic hash primitive for commitments unless a subsystem has a specific reason not to.

Recommended class:

```text
SHA-256
```

or another well-supported 256-bit cryptographic hash already available in the codebase.

The exact primitive is less important than:

- collision resistance;
- stable cross-platform implementation;
- explicit domain separation;
- canonical byte serialization.

Define:

\[
H_D(x)=H(D \parallel x)
\]

where `D` is a protocol-specific domain string.

Examples:

```text
DAN/REPLICA-SNAPSHOT/v1
DAN/JOB-AUTH/v1
DAN/WORK-CLAIM/v1
DAN/FRAME-COMMITMENT/v1
DAN/WALLET-BINDING/v1
```

---

## 3.3 Canonical serialization

Never sign arbitrary JSON text.

Bad:

```json
{"a":1,"b":2}
```

versus:

```json
{ "b": 2, "a": 1 }
```

These are semantically equivalent but byte-different.

All signed protocol objects need deterministic encoding.

Suitable approaches include:

- deterministic Protobuf;
- canonical CBOR;
- another explicitly specified binary encoding.

The signed bytes must be uniquely determined by the logical object.

Pseudo-interface:

```cpp
Bytes canonical_encode(const SignedObject& obj);
Hash digest = sha256(domain || canonical_encode(obj));
Signature sig = sign(private_key, digest);
```

---

# 4. Protocol versioning and domain separation

Every signed object should contain:

```text
protocol_version
network_id
object_type
```

and also be hashed under a type-specific domain.

This provides two independent protections:

```text
internal type field
+
cryptographic domain separator
```

For example, a valid `WorkClaim` signature must not be reusable as a `JobAuthorization`.

---

# 5. Replay protection

Every authorization or economically relevant claim must be bound to a unique context.

Recommended fields:

```text
network_id
replica_id
replica_generation
session_id
request_id
nonce
created_at
expires_at
```

Not every object needs every field, but all signed operations need sufficient freshness and context binding.

A receiver should maintain replay state at the appropriate scope.

Example:

```text
(client_peer_id, nonce)
```

or:

```text
(client_peer_id, job_id)
```

must not be accepted twice when the operation is defined as single-use.

Expiration is not enough by itself. A message valid for ten minutes can still be replayed repeatedly during those ten minutes unless reuse is tracked.

---

# 6. Replica identity

A persistent replica needs a cryptographically stable identity separate from any individual session.

Recommended:

```text
replica_id        = random 128/256-bit identifier
replica_generation = monotonically increasing generation or random incarnation ID
```

`replica_id` identifies the logical replica.

`replica_generation` distinguishes reincarnations.

Example:

```text
Replica R generation 17:
A -> B -> C

B dies

Replica R generation 18:
A -> D -> C
```

No receipt from generation 17 should be valid as evidence for generation 18.

If the current implementation creates a completely new replica ID after dissolution, generation is less important, but preserving an incarnation field is still useful for future repair/reconfiguration.

---

# 7. Signed replica snapshot

Before economically meaningful inference can be attributed, there must be a committed execution plan.

Example structure:

```text
ReplicaSnapshot {
    protocol_version
    network_id

    replica_id
    replica_generation

    model_hash
    tokenizer_hash
    runtime_abi_hash
    execution_profile_hash

    owner_peer_id

    members[] {
        peer_id
        stage_index
        layer_begin
        layer_end
        role
    }

    context_length
    sessions_max
    activation_dtype
    speculation_profile

    created_at
    expires_at
}
```

Define:

\[
R = H_{\text{REPLICA}}(\operatorname{encode}(ReplicaSnapshot))
\]

Every member signs `R`.

For member \(i\):

\[
\sigma_i = \operatorname{Sign}(sk_i, R)
\]

Replica membership evidence becomes:

```text
ReplicaCertificate {
    snapshot
    member_signatures[]
}
```

A replica is economically valid only if all required members have signed the exact same snapshot hash.

This prevents the owner from later claiming:

> provider B participated in this replica

when B never accepted that plan.

---

# 8. Why the owner is not a trust root

The owner may:

- discover providers;
- construct the plan;
- acquire leases;
- create the ring;
- advertise READY state;
- accept client sessions;
- assemble receipts.

But it should not be able to invent economically binding statements for members.

Trust model:

```text
owner proposes
members independently authenticate proposal
members sign their participation
```

The owner is therefore:

```text
orchestrator of one replica
```

not:

```text
cryptographic authority for the replica
```

---

# 9. Model and execution commitments

"Same model" must mean more than the marketing/model name.

A verification-relevant execution commitment should include hashes or normalized identifiers for:

```text
model weights
tokenizer
runtime ABI
backend version
quantization configuration
activation dtype
context configuration
sampling configuration
speculation configuration
architecture-specific flags
```

Conceptually:

```text
ExecutionProfile {
    model_hash
    tokenizer_hash
    runtime_abi_hash

    backend
    quantization
    activation_dtype

    deterministic_mode
    speculation_mode
    sampling_mode
}
```

Then:

\[
E = H_{\text{EXEC}}(\operatorname{encode}(ExecutionProfile))
\]

This is necessary because two peers using the same model weights but different numerical execution paths may legitimately produce different bytes.

---

# 10. Client job authorization

Before billable work begins, the client authorizes a bounded job.

Example:

```text
JobAuthorization {
    protocol_version
    network_id

    job_id
    client_peer_id

    replica_id
    replica_generation
    replica_snapshot_hash

    model_hash
    execution_profile_hash

    prompt_commitment

    max_output_tokens
    max_compute_units
    max_payment

    created_at
    expires_at
    nonce
}
```

The client signs:

\[
J = H_{\text{JOB}}(\operatorname{encode}(JobAuthorization))
\]

\[
\sigma_C=\operatorname{Sign}(sk_C,J)
\]

The economic rule is:

\[
\text{TotalSettlement}(J) \le \text{max\_payment}(J)
\]

Provider claims **cannot increase this bound**.

This is one of the most important invariants in the design.

---

# 11. Prompt privacy in accounting records

Do not persist raw prompt text merely to prove job identity.

Use:

\[
P = H_{\text{PROMPT}}(\text{prompt bytes})
\]

The encrypted/authenticated live inference stream still carries the prompt.

The durable accounting layer stores `prompt_commitment = P`.

This gives later evidence that two receipts refer to the same prompt without unnecessarily duplicating user content into settlement records.

---

# 12. Request identity

Inside a client session:

```text
session_id
request_id
request_sequence
```

should be unique.

Recommended hierarchy:

```text
replica_id
  └─ replica_generation
      └─ session_id
          └─ request_id
              └─ request_sequence
```

For speculation:

```text
speculation_round
candidate_index
token_position
```

can extend that namespace.

These fields matter for both replay resistance and dispute localization.

---

# 13. Work units

DAN must explicitly define what can be claimed and paid.

For current pipeline parallelism, the natural work unit is approximately:

```text
(request_id, stage_index, token/position range)
```

or, depending on accounting granularity:

```text
(request_id, layer_range, sequence interval)
```

A work unit must be deterministic enough that two claims can be tested for:

```text
same
disjoint
overlapping
missing
```

Do not define accounting only as:

```text
provider says: "I did 27% of the job"
```

The system needs a structural claim.

---

# 14. Work claim

Example:

```text
WorkClaim {
    protocol_version
    network_id

    job_id
    replica_snapshot_hash

    provider_peer_id

    request_id
    stage_index
    layer_begin
    layer_end

    sequence_begin
    sequence_end

    input_commitment
    output_commitment

    compute_units

    created_at
    nonce
}
```

Provider \(i\) signs:

\[
C_i = H_{\text{CLAIM}}(\operatorname{encode}(WorkClaim_i))
\]

\[
\sigma_i=\operatorname{Sign}(sk_i,C_i)
\]

Important:

```text
input_commitment
output_commitment
```

are commitments to bytes or normalized numerical objects.

They are **not correctness proofs**.

---

# 15. Frame commitments

For an activation frame:

```text
FrameHeader {
    replica_id
    replica_generation
    session_id
    request_id
    stage_index
    sequence
    dtype
    shape
    byte_length
}
```

Define:

\[
F = H_{\text{FRAME}}(
    \operatorname{encode}(FrameHeader)
    \parallel
    \text{raw payload bytes}
)
\]

A sender can record `F_out`.

The next stage can record `F_in`.

If both independently attest to the same hash:

```text
sender says: F
receiver says: F
```

then later we know exactly which byte sequence crossed that edge.

This establishes **chain of custody**, not mathematical correctness.

---

# 16. Optional chained transcript

A request can maintain an append-only transcript hash:

\[
T_0 = H(\text{request metadata})
\]

\[
T_{k+1} = H(
    D_{\text{TRANSCRIPT}}
    \parallel T_k
    \parallel \text{event}_k
)
\]

Events could include:

```text
stage input hash
stage output hash
token output
error
session reset
verification result
```

This creates a tamper-evident compact transcript.

It does not replace signatures, because a single malicious party could construct an internally consistent fake transcript.

It is useful when combined with independently signed checkpoints.

---

# 17. Structural accounting validation

Before doing any expensive verification, perform deterministic structural checks.

For a linear replica with model layer interval:

\[
[0,L)
\]

valid stage claims should satisfy:

### Coverage

\[
\bigcup_i R_i = [0,L)
\]

### No unauthorized overlap

For normal work:

\[
R_i \cap R_j = \emptyset
\quad \forall i \ne j
\]

except explicitly designated verification overlap.

### Membership

Each claim's provider must appear in the signed replica snapshot.

### Assignment

Claimed range must match the provider's assigned range.

### Request binding

All claims must bind to the same:

```text
job_id
request_id
replica_snapshot_hash
execution_profile_hash
```

### Payment ceiling

\[
\sum_i p_i \le P_{\max}
\]

These checks are cheap and should happen before probabilistic verification.

---

# 18. Overclaiming

Example:

```text
snapshot:
A = [0,12)
B = [12,27)

claim:
A = [0,20)
```

A's claim is invalid immediately because:

\[
[0,20) \not\subseteq [0,12)
\]

No recomputation is necessary to reject that structural overclaim.

This is why signed replica membership substantially simplifies accounting.

---

# 19. Cross-provider overlap

If two providers legitimately have ambiguous or disputed claims:

\[
R_A \cap R_B = O
\]

then the conflict should be localized to:

\[
O
\]

rather than invalidating unrelated work.

Policy principle:

> No two providers are paid for the same ordinary work unit.

Initially:

```text
payment(O) = withheld
```

until dispute resolution.

Uncontested work outside \(O\) can still settle.

---

# 20. Verification is probabilistic redundant execution

The practical first correctness mechanism should be sampled recomputation.

Suppose provider A claims:

\[
y = f_s(x)
\]

for stage \(s\).

A verifier independently evaluates:

\[
y' = f_s(x)
\]

Then under a byte-deterministic execution profile:

\[
y \stackrel{?}{=} y'
\]

The verification event should itself be recorded:

```text
VerificationRecord {
    claim_hash
    verifier_peer_id
    verifier_execution_profile_hash

    expected_output_commitment
    observed_output_commitment

    result

    created_at
}
```

and signed by the verifier.

---

# 21. Detection probability

Let:

- \(p\) = probability a malicious work unit is independently checked;
- \(n\) = number of fraudulent work units submitted.

Assuming independent sampling, probability of avoiding detection is:

\[
P(\text{undetected after }n)=(1-p)^n
\]

Therefore:

\[
P(\text{detected})=1-(1-p)^n
\]

Example:

If:

\[
p=0.05
\]

and a provider cheats 50 times:

\[
P(\text{detected})
=1-0.95^{50}
\approx 92.3\%
\]

For 100 fraudulent units:

\[
1-0.95^{100}
\approx 99.4\%
\]

This illustrates why low sampling rates can still make sustained cheating unattractive.

It does **not** imply that 5% is the correct production rate.

That must be measured economically.

---

# 22. Adaptive verification rate

Verification rate should be provider-dependent.

Define:

\[
p_i = f(
    \text{age}_i,
    \text{verified\_successes}_i,
    \text{mismatches}_i,
    \text{disconnects}_i,
    \text{claim anomalies}_i
)
\]

Possible policy shape:

```text
new provider:
    high p

established provider with clean history:
    lower p

provider after mismatch:
    very high p

provider under active dispute:
    possibly p = 1 for bounded probation
```

Avoid treating reputation as one opaque scalar if the underlying evidence categories matter.

---

# 23. Unpredictable verification selection

A provider should not know far in advance which jobs will be checked.

Otherwise:

```text
if checked:
    compute honestly
else:
    cheat
```

Verification selection therefore needs unpredictability.

Possible design:

1. commit to request/job;
2. derive verification decision from randomness unavailable before commitment;
3. dispatch verifier.

For example, later:

\[
r =
H(
\text{job\_id}
\parallel
\text{claim\_hash}
\parallel
\text{epoch randomness}
)
\]

Verify if:

\[
r \bmod M < K
\]

The randomness source must not be controllable by the provider being checked.

For an early prototype, centralized test harness randomness is acceptable.

For a fully decentralized economic protocol, randomness generation requires more careful design.

---

# 24. Verification overlap versus fraudulent overlap

These must be distinct protocol states.

## Ordinary work

```text
A owns work unit W
```

Expected payable count:

```text
1
```

## Deliberate verification

```text
A owns W
V independently recomputes W
```

This is valid overlap tagged:

```text
purpose = verification
```

## Conflicting claims

```text
A claims W
B also claims W
```

without a verification assignment.

This is:

```text
purpose = conflict
```

Do not infer intent only from identical ranges.

The protocol metadata should make the purpose explicit.

---

# 25. Deterministic verification classes

Exact hashes work only when exact output equality is a valid expectation.

Define execution profiles by verification class.

Example:

```text
EXACT
NUMERIC_TOLERANCE
NONDETERMINISTIC
```

## EXACT

Verifier must match output bytes:

\[
H(y)=H(y')
\]

Use only when all relevant numerical behavior is expected to be deterministic.

Potential requirements:

```text
same model hash
same quantization
same runtime ABI
same backend class
same activation dtype
same speculation configuration
same deterministic kernel path
same sampling configuration
```

Even identical GPUs may not guarantee determinism if kernels use nondeterministic reductions.

This must be empirically validated.

## NUMERIC_TOLERANCE

For floating-point paths such as lossy FP16 boundary compression:

\[
\|y-y'\| \le \epsilon
\]

Possible metrics:

\[
\|y-y'\|_\infty
\]

\[
\frac{\|y-y'\|_2}{\|y'\|_2+\delta}
\]

or task-specific logit/error bounds.

This is not a cryptographic equality proof.

The exact tolerance must be model/runtime validated.

## NONDETERMINISTIC

If the execution path cannot support meaningful deterministic comparison, hash equality should not be used as correctness evidence.

Such paths may need:

- a canonical verification rerun;
- a higher-level output test;
- or no economic correctness claim until a verifier is designed.

---

# 26. FP16 activation implications

If DAN's baseline sends FP32 boundary activations and:

```text
--f16-activations
```

performs:

\[
FP32 \rightarrow FP16 \rightarrow FP32
\]

then the transmitted activation has irreversible rounding error.

Therefore:

```text
FP32 claim output hash
```

and:

```text
FP16-wire claim output hash
```

must not be expected to match.

The activation dtype / codec is part of the signed execution profile.

A verifier checking an FP16-wire claim should reproduce the same quantization boundary if byte equality is required.

---

# 27. Speculative decoding implications

Speculation can change the execution schedule and, depending on implementation, floating-point order.

Verification metadata should bind:

```text
draft_model_hash
draft_runtime_hash
speculation_enabled
max_draft_tokens
verification batching strategy
```

If speculation is known to preserve byte-identical output under a given path, exact verification can use that profile.

If not, verification may need to operate at the stage-computation level rather than the final generated string.

---

# 28. Verification granularity

Several granularities are possible.

### Full request

Recompute the entire request.

Strong but expensive.

### Stage

Recompute one stage for one request.

Natural fit for DAN.

### Token-position interval

Recompute:

```text
stage S
positions [a,b)
```

More granular.

### Random internal challenge

Recompute selected internal work units.

Potentially efficient, but requires more instrumentation.

First implementation recommendation:

> verify complete stage executions for bounded request segments.

It aligns with the current pipeline architecture and signed replica assignment.

---

# 29. Required evidence for stage verification

To independently verify a stage computation, the verifier needs:

```text
model/range weights
execution profile
stage input activation
position/session metadata
relevant KV state
```

The KV dependency is important.

A transformer stage output is generally not only:

\[
y=f(x)
\]

but effectively:

\[
(y, K_{t+1}, V_{t+1})
=
f(x, K_t, V_t, t)
\]

Therefore arbitrary after-the-fact recomputation is not trivial unless the verifier can reconstruct the same stage state.

Possible approaches:

1. verify from session start;
2. checkpoint KV state;
3. issue hidden challenge sessions specifically designed for verification;
4. duplicate selected work live while the necessary state exists.

For v1 economic verification, **live sampled redundant execution** is likely much simpler than arbitrary forensic recomputation after the fact.

---

# 30. Live verifier architecture

A practical design:

```text
normal:
A -> B -> C

verification of B:
             +--> B ----+
A output ----|          |--> compare
             +--> V ----+
```

Both B and verifier V receive equivalent:

```text
input activation
session/position metadata
compatible KV state
```

The compare point checks their resulting activation commitments.

This is conceptually easy but requires the verifier to maintain compatible session state.

That may mean a verifier is attached from request/session start rather than appearing only at the challenged token.

---

# 31. Alternative: hidden challenge sessions

Another practical mechanism avoids duplicating a real user's entire KV history.

The network creates synthetic verification sessions:

```text
known prompt / generated state
        |
        +--> target provider
        |
        +--> trusted/independent reference
```

This checks:

- stage correctness;
- claimed model;
- basic execution capability;
- malicious garbage output.

It does not prove every real request was computed honestly.

It is therefore complementary to live sampled verification.

---

# 32. Reputation evidence classes

Track different dimensions separately.

Example:

```text
CorrectnessEvidence
ReliabilityEvidence
ProtocolEvidence
AccountingEvidence
```

### Correctness

```text
verified_match
verified_mismatch
```

### Reliability

```text
timeout
disconnect_mid_request
failure_rate
```

### Protocol

```text
invalid_frame
invalid_signature
replay_attempt
```

### Accounting

```text
overclaim
duplicate_claim
membership_mismatch
```

Do not collapse all four immediately into one number.

Settlement/admission policy can derive a score later.

---

# 33. False accusation resistance

A verifier can also lie.

Therefore:

```text
Verifier V says:
"A was wrong"
```

must not automatically punish A.

A dispute should retain:

```text
target claim
target output commitment
verifier output commitment
execution profiles
input commitment
independent signatures
```

Escalation can trigger:

```text
third execution
```

Conceptually:

```text
A result = X
V result = Y

if X != Y:
    send same verification unit to W

if W == X:
    evidence against V

if W == Y:
    evidence against A

if all differ:
    mark numerical/protocol ambiguity
```

This is majority evidence for a specific work unit, not global blockchain consensus.

---

# 34. Collusion

Two colluding providers can agree on a fake result.

Therefore independent verification should prefer a verifier selected outside the target's control.

Future verifier selection may consider:

```text
different operator identity
different network / ASN
no recent repeated pairing
no obvious payment-address linkage
```

These are heuristic anti-collusion measures, not cryptographic guarantees.

---

# 35. Sybil resistance

PeerID creation is cheap:

\[
\text{cost(new PeerID)} \approx 0
\]

Therefore:

```text
100 PeerIDs
```

does not mean:

```text
100 independent economic actors
```

Signatures solve impersonation, not Sybil attacks.

Potential future Sybil costs:

- reputation accumulated over time;
- minimum bond/stake;
- delayed access to high-value jobs;
- payment history;
- hardware attestation;
- operator identity verification for certain pools;
- higher verification rate for new identities.

Do not make stake mandatory merely because a token exists.

Introduce economic friction only when the threat model demonstrates a need.

---

# 36. Payout identity

Network identity and payout identity should be separable.

Example:

```text
PayoutBinding {
    peer_id
    payout_chain
    payout_address
    key_epoch
    valid_from
    valid_until
    nonce
}
```

Signed by the provider identity.

If required by the settlement rail, optionally require proof from the payout key as well:

```text
provider signs binding
wallet signs binding
```

Then the binding proves control of both identities.

This prevents an attacker from replacing another provider's payout address in an unsigned database.

---

# 37. Fixed payment envelope

Client authorization defines:

\[
P_{\max}
\]

The settlement function must satisfy:

\[
0 \le P_{\text{settled}} \le P_{\max}
\]

regardless of provider claims.

Provider-side inflation becomes a redistribution/dispute problem, not an unlimited-charge problem.

This sharply limits economic blast radius.

---

# 38. Accounting weights

For heterogeneous stage sizes, equal provider splitting is not appropriate.

Eventually define a deterministic work-cost function:

\[
w_i = g(
\text{model},
\text{layer range},
\text{tokens},
\text{prefill/decode},
\text{hardware-independent compute estimate}
)
\]

Then:

\[
p_i =
P_{\text{available}}
\frac{w_i}{\sum_j w_j}
\]

for uncontested valid work.

The exact pricing function should remain separate from cryptographic validity.

A valid signed claim can still have a price of zero if policy rejects it.

---

# 39. Dispute states

Suggested work-claim state machine:

```text
SUBMITTED
   |
   v
STRUCTURALLY_VALID
   |
   +---- invalid ----> REJECTED
   |
   v
PENDING_VERIFICATION
   |
   +---- not sampled ----> ACCEPTED
   |
   +---- sampled
            |
            +---- match ----> VERIFIED
            |
            +---- mismatch --> DISPUTED
                                  |
                                  +--> ESCALATED
                                  |
                                  +--> PROVIDER_FAULT
                                  |
                                  +--> VERIFIER_FAULT
                                  |
                                  +--> INCONCLUSIVE
```

Settlement policy can decide which states are payable.

---

# 40. No immediate slashing

A first mismatch should not automatically destroy stake.

Reasons:

- numerical nondeterminism;
- implementation bugs;
- verifier bugs;
- corrupted network payloads;
- ABI mismatch;
- hardware faults.

Initial response:

```text
withhold disputed payment
increase verification rate
collect evidence
```

Only later, after failure modes are understood, consider:

```text
bond forfeiture
slashing
temporary exclusion
```

---

# 41. Malformed / invalid cryptographic messages

Any object failing:

```text
canonical decode
domain/version check
signature validation
expiry check
nonce/replay check
identity binding
```

should be rejected **before** expensive inference or verification.

This is both a correctness and DoS-defense rule.

---

# 42. Resource limits

Cryptographic verification itself can become a DoS vector.

Apply bounds to:

```text
number of signatures per object
replica member count
claim count per request
serialized object size
verification retries
dispute depth
nonce cache size
```

All length-prefixed protocol structures must be size-bounded before allocation.

---

# 43. Key rotation

Long-lived providers eventually need key rotation.

If PeerID identity changes, reputation continuity becomes nontrivial.

Two choices:

### Simple v1

```text
new PeerID = new provider identity
```

No reputation transfer.

### Future

Old identity signs:

```text
KeyMigration {
    old_peer_id
    new_peer_id
    effective_time
}
```

and the new identity countersigns.

Reputation policy decides how much history transfers.

Do not implement this before necessary.

---

# 44. Compromised keys

If a provider private key is stolen, signatures remain cryptographically valid.

Therefore future economic deployment may need:

```text
key revocation
short-lived payout bindings
optional recovery identity
reputation freeze
```

Cryptography cannot distinguish:

```text
legitimate owner
```

from:

```text
attacker holding legitimate private key
```

---

# 45. DHT advertisement signatures

Replica DHT records should be signed.

Example:

```text
ReplicaAdvertisement {
    replica_snapshot_hash
    owner_peer_id

    model_hash
    free_sessions
    endpoint metadata

    advertised_at
    expires_at
    sequence
}
```

Owner signs the advertisement.

Clients should still perform a live owner check.

Why both?

Signature proves:

```text
owner published this record
```

Live check proves:

```text
replica appears reachable and currently alive
```

DHT data alone is staleable.

---

# 46. Capability advertisements

Provider capability records can also be signed:

```text
ProviderCapability {
    peer_id
    runtime_abi_hash
    GPU metadata
    available_memory
    loaded_model_ranges
    timestamp
    expiry
}
```

But signatures only prove:

> this provider claimed these capabilities.

They do not prove the GPU/VRAM claim is true.

Reservation/load/warm-up and future challenge mechanisms still provide actual evidence.

---

# 47. Replica READY as evidence

READY should not be merely a local boolean.

For future accounting, READY can correspond to a committed lifecycle event:

```text
all leases acquired
all stage assignments accepted
all members loaded
all ring edges linked
warm-up succeeded
snapshot signed
```

The owner may then emit:

```text
ReplicaReadyReceipt {
    snapshot_hash
    warmup_request_hash
    timestamp
}
```

signed by the owner.

This is operational evidence, not proof that every future request will be correct.

---

# 48. Failure receipts

When possible, preserve signed/local evidence of terminal failures.

Example categories:

```text
MEMBER_DISCONNECTED
ROUTE_BROKEN
CONTEXT_FULL
INVALID_FRAME
REQUEST_CANCELLED
TIMEOUT
```

Failure evidence helps distinguish:

```text
provider fraud
```

from:

```text
ordinary network failure
```

Do not make synchronous signature generation mandatory on every hot-path failure if it harms inference latency. Evidence can be assembled asynchronously where safe.

---

# 49. Hot-path cryptography

Do not sign every activation frame individually unless benchmarking proves the cost negligible and the evidence is necessary.

A better approach may be:

```text
hash every relevant frame
accumulate transcript
sign periodic/request-level root
```

For example:

\[
T_{k+1}=H(T_k \parallel F_k)
\]

At request completion:

```text
provider signs final transcript root
```

This amortizes signature cost.

Potential later optimization:

```text
Merkle tree of frame commitments
```

then sign one Merkle root.

Merkle proofs allow selective disclosure of individual events without signing every event.

This is optional, not required for the first implementation.

---

# 50. Merkle batching

If a provider processes many work units:

\[
C_1,C_2,\dots,C_n
\]

build a Merkle tree:

\[
R=\operatorname{MerkleRoot}(C_1,\dots,C_n)
\]

Provider signs:

\[
\sigma=\operatorname{Sign}(sk,R)
\]

Then any individual claim \(C_i\) can be proven included using an \(O(\log n)\) Merkle proof.

This may become useful for settlement batches.

It is unnecessary before claim volume makes per-claim signatures expensive.

---

# 51. Verification economics

Let:

- \(C\) = value gained by cheating on one unchecked unit;
- \(p\) = verification probability;
- \(L\) = expected loss if caught.

A crude deterrence condition is:

\[
(1-p)C - pL < 0
\]

which rearranges to:

\[
p > \frac{C}{C+L}
\]

This does not prescribe policy, but it shows the relationship between:

```text
verification rate
and
penalty magnitude
```

If penalties are tiny, verification must be frequent.

If reputation/bond loss is substantial, lower sampling rates may deter cheating.

This should be calibrated from observed economics, not guessed now.

---

# 52. Verification cost

Let:

- \(W\) = normal useful compute cost;
- \(p\) = sampled verification fraction;
- \(r\) = relative cost of reproducing a checked work unit.

Approximate total compute:

\[
W_{\text{total}} \approx W(1+pr)
\]

If full duplicate verification has \(r=1\) and:

\[
p=0.05
\]

then raw compute overhead is approximately:

\[
5\%
\]

before networking and scheduler overhead.

This is why sampled verification is attractive relative to universal duplication.

---

# 53. Do not call signatures "proof of work"

Terminology should stay precise.

Recommended:

```text
signature
authorization
commitment
receipt
work claim
verification record
```

Reserve:

```text
proof
```

for mechanisms that actually establish the claimed property under a defined security assumption.

A signed `WorkClaim` is not itself a proof of useful work.

---

# 54. zkML / succinct proof systems

The theoretically stronger future target is:

\[
\pi = \operatorname{Prove}(f,x,y)
\]

such that:

\[
\operatorname{Verify}(f,x,y,\pi)=1
\]

implies, under the proof system assumptions:

\[
y=f(x)
\]

without recomputing the full model stage.

Potential approaches:

- SNARKs;
- STARKs;
- specialized verifiable matrix multiplication;
- zkML;
- polynomial commitment based inference proofs.

Current concerns:

```text
prover overhead
GPU integration
quantized model support
floating-point semantics
KV-state representation
proof size
verification latency
engineering complexity
```

Do not block DAN's economic prototype on this.

---

# 55. Trusted execution environments

Another possible future path:

```text
GPU/CPU TEE
+
remote attestation
```

could attest that approved inference code ran on specific hardware.

This changes the trust assumption from:

```text
trust probabilistic peer verification
```

to:

```text
trust hardware vendor + attestation stack
```

It may be useful for some deployment classes but should not become a mandatory DAN dependency.

---

# 56. Consensus is not currently required

DAN replica formation already uses:

```text
first valid reservation wins
```

through worker leases.

Security/accounting does not automatically imply a need for blockchain consensus.

If Alice and Bob need to settle a specific signed job, they need agreement about:

```text
that job
```

not a globally ordered history of every DAN event.

A global consensus layer should only be introduced if an actual protocol invariant requires one.

---

# 57. No global ledger requirement

Most evidence can be scoped:

```text
provider
replica
job
request
settlement
```

A globally replicated ledger is unnecessary for:

- authenticated inference;
- signed replica membership;
- work receipts;
- sampled verification.

A payment rail may later use an external blockchain or another settlement service without making DAN's control plane blockchain-dependent.

---

# 58. Minimum cryptographic implementation before payments

Recommended minimum:

```text
1. Stable provider PeerID identity
2. Canonical signed-object format
3. Domain-separated hashing
4. Replica snapshot hash
5. Member signatures over snapshot
6. Stable replica/session/request IDs
7. Replay-safe client job authorization
8. Signed provider work receipts
9. Execution profile commitment
10. Auditable frame/request commitments
```

But this should follow successful real-WAN replica validation unless an item is needed immediately for network correctness.

---

# 59. First verification implementation

Recommended first real verifier:

```text
sampled live redundant stage execution
```

Properties:

- no zkML;
- no blockchain;
- no global consensus;
- no universal duplication;
- uses the same useful inference computation;
- produces independently signed comparison evidence.

Prototype sequence:

```text
1. select target stage
2. duplicate compatible session state to verifier
3. send same stage input
4. compute independently
5. compare according to execution profile
6. record signed VerificationRecord
7. update evidence counters
```

---

# 60. Adversarial test suite

Before payments, explicitly inject faults.

## Identity / signature

```text
wrong signer
forged signature
modified signed field
wrong domain
old protocol version
expired authorization
replayed nonce
```

## Replica

```text
owner invents member
member signs different snapshot
stale generation
range mismatch
```

## Compute

```text
one-byte activation corruption
random activation
old activation replay
wrong model weights
wrong layer range
FP16/FP32 profile mismatch
```

## Accounting

```text
duplicate claim
overlapping claim
missing stage
inflated compute units
claim after expiration
claim for another request
```

## Verifier

```text
malicious verifier reports mismatch
two verifiers disagree
nondeterministic kernel produces drift
```

## Network

```text
member disconnect
owner disconnect
relay expiry
duplicate delivery
reordered control messages
timeout during dispute
```

Every injected case should have an expected protocol outcome.

---

# 61. Suggested C++ conceptual interfaces

```cpp
struct SignedEnvelope {
    uint32_t protocol_version;
    ObjectType type;
    std::vector<uint8_t> canonical_payload;
    PeerId signer;
    Signature signature;
};

Hash domain_hash(
    std::string_view domain,
    std::span<const uint8_t> canonical_payload);

bool verify_envelope(const SignedEnvelope&);
```

Replica:

```cpp
struct ReplicaSnapshot {
    ReplicaId replica_id;
    Generation generation;

    Hash model_hash;
    Hash runtime_abi_hash;
    Hash execution_profile_hash;

    PeerId owner;
    std::vector<MemberAssignment> members;

    uint32_t context_length;
    uint32_t sessions_max;

    Timestamp created_at;
    Timestamp expires_at;
};
```

Job:

```cpp
struct JobAuthorization {
    JobId job_id;
    PeerId client;

    Hash replica_snapshot_hash;
    Hash prompt_commitment;
    Hash execution_profile_hash;

    uint32_t max_output_tokens;
    uint64_t max_compute_units;
    Amount max_payment;

    Nonce nonce;
    Timestamp created_at;
    Timestamp expires_at;
};
```

Claim:

```cpp
struct WorkClaim {
    JobId job_id;
    RequestId request_id;

    Hash replica_snapshot_hash;

    PeerId provider;
    StageIndex stage;

    uint32_t layer_begin;
    uint32_t layer_end;

    uint64_t sequence_begin;
    uint64_t sequence_end;

    Hash input_commitment;
    Hash output_commitment;

    uint64_t compute_units;
    Nonce nonce;
};
```

Verification:

```cpp
enum class VerificationResult {
    Match,
    Mismatch,
    Inconclusive,
};

struct VerificationRecord {
    Hash work_claim_hash;

    PeerId verifier;
    Hash verifier_execution_profile_hash;

    Hash claimed_output;
    Hash verifier_output;

    VerificationResult result;
    Timestamp completed_at;
};
```

These are conceptual schemas, not final wire definitions.

---

# 62. State-machine invariants

## Replica

```text
FORMING
  -> READY
  -> DISSOLVING
  -> DEAD
```

Economically relevant jobs should only bind to:

```text
READY
```

replica snapshots.

A claim from a dead/stale generation must still be auditable but cannot be rebound to the new replica.

## Job

```text
AUTHORIZED
 -> RUNNING
 -> COMPLETED
 -> SETTLED
```

or:

```text
AUTHORIZED
 -> RUNNING
 -> FAILED
```

Authorization must precede billable work.

## Claim

Use the dispute state machine from section 39.

---

# 63. Security invariants

The implementation should preserve at least these invariants.

### Identity

1. No peer can produce another peer's valid application signature without its private key.
2. Application signing keys, if separate, are cryptographically bound to PeerID.

### Replica

3. Every economically recognized member signed the exact replica snapshot.
4. A stale replica generation cannot authorize current work.
5. Owner authority does not substitute for member signatures.

### Jobs

6. A job cannot be executed as billable work without valid client authorization.
7. Authorization is replay-safe.
8. A provider claim cannot increase the client's maximum payment.

### Claims

9. Every claim is bound to one job, request, replica generation, provider, and work unit.
10. Ordinary claims cannot be paid twice for the same work unit.
11. Structural overclaims are rejected before expensive verification.

### Verification

12. Signed hashes are not treated as correctness proofs.
13. Verification comparison rules are explicitly bound to an execution profile.
14. Exact hash equality is used only under a validated deterministic profile.
15. A single verifier accusation is not automatically equivalent to provider guilt.

### Economics

16. Disputed work can be withheld without invalidating unrelated valid work.
17. Penalties derive from auditable evidence.
18. DAN remains operational without a token or blockchain.

---

# 64. Recommended implementation order

```text
PHASE 0 — current network
|
|-- persistent self-forming replicas
|-- real WAN validation
|-- failure/reformation validation
|
v
PHASE 1 — cryptographic substrate
|
|-- canonical object encoding
|-- domain-separated hashing
|-- stable IDs
|-- execution profile hash
|-- signed replica snapshot
|-- member signatures
|
v
PHASE 2 — accounting substrate
|
|-- client JobAuthorization
|-- signed WorkClaim
|-- replay cache
|-- structural claim validator
|-- fixed payment envelope
|
v
PHASE 3 — adversarial verification
|
|-- injected corruption tests
|-- sampled live redundant execution
|-- signed VerificationRecord
|-- dispute state machine
|
v
PHASE 4 — reputation
|
|-- correctness evidence
|-- reliability evidence
|-- protocol abuse evidence
|-- adaptive verification probability
|
v
PHASE 5 — settlement
|
|-- payout binding
|-- external payment rail
|-- verifier compensation
|-- disputed-payment handling
|
v
PHASE 6 — only if justified
|
|-- bonds/staking
|-- stronger Sybil resistance
|-- slashing
|-- Merkle receipt batching
|-- hardware attestation
|-- zkML / succinct proofs
```

---

# 65. Final architecture

The intended trust stack is:

```text
+------------------------------------------------+
| Settlement / payment rail                      |
+------------------------------------------------+
| Reputation / dispute policy                    |
+------------------------------------------------+
| Probabilistic correctness verification         |
| - sampled redundant execution                  |
| - challenge sessions                           |
+------------------------------------------------+
| Signed work claims / receipts                  |
+------------------------------------------------+
| Signed client job authorization                |
+------------------------------------------------+
| Signed replica membership / execution profile  |
+------------------------------------------------+
| Authenticated libp2p identity / PeerID         |
+------------------------------------------------+
| DAN replica + inference data plane             |
+------------------------------------------------+
```

The core distinction is:

\[
\boxed{
\text{Signature}
\Rightarrow
\text{who committed to a statement}
}
\]

while:

\[
\boxed{
\text{Independent verification}
\Rightarrow
\text{evidence that the computation was correct}
}
\]

and:

\[
\boxed{
\text{Settlement policy}
\Rightarrow
\text{what economic consequence follows}
}
\]

Those layers should remain separate.

That separation gives DAN a practical path from today's authenticated P2P inference network to a permissionless economic compute network without prematurely introducing blockchain consensus, wasteful hash Proof-of-Work, or heavyweight zero-knowledge inference proofs.
