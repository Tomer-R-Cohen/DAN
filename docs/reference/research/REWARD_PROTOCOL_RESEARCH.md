# DAN reward protocol: research and design pressure test

*Research only, nothing implemented. 2026-09-25.*

> **Partly superseded** by `PAYMENT_ARCHITECTURE_PRESSURE_TEST.md` (§7 there): burning at
> deposit time and the usage rebate pool are dropped; the capacity pool becomes the only
> bootstrap issuance.

This document checks the proposed DAN reward idea ("useful work earns lottery tickets for
newly minted DAN, fake jobs must lose money") against systems that have tried similar
things in production. Where a claim comes from a source it is linked. Where it is my own
reasoning it says **(analysis)**.

---

## 1. Summary: how the strongest systems solve this

1. **No production system pays a subsidy for demand-driven work without being farmed.**
   Helium (data transfer before HIP-10), Filecoin Plus (the 10× multiplier), Pocket
   (fixed mint per relay) and io.net (rewards for claimed GPUs) all paid more for
   "useful" activity than it cost to fake. All were farmed. The fixes that worked either
   cap the reward at the fee actually burned (Helium HIP-10) or stop minting per unit of
   activity (Pocket WAGMI).
2. **Systems that survive pay for work with fees and mint for something else.** Livepeer
   and Ethereum mint to *stake*. Filecoin mints to *proven storage capacity*. Bitcoin
   mints to *burned energy*. Clients pay for the work itself. None of them mints in
   proportion to client jobs.
3. **Lotteries are a settlement tool, not a security tool.** Livepeer's probabilistic
   micropayments, Filecoin's election and Pocket's relay mining all use a hash lottery so
   that only a few winning events touch the chain. The lottery's security comes from how
   the ticket is built: randomness that the recipient commits to in advance (Livepeer),
   unique signatures over a public beacon (Filecoin: BLS VRF + drand), or a hash taken
   only after the work is done (Pocket).
4. **Correctness is checked by sampling plus a penalty, not by running everything
   twice.** In production this is done on inference by Prime Intellect (TOPLOC validators
   during INTELLECT-2). Hyperbolic (PoSP), Gensyn (Verde) and VeriLLM describe it in
   papers. All rely on the check being unpredictable and the penalty being larger than
   the gain.
5. **Settlement lives on an existing chain; reward calculation is often centralized.**
   Livepeer runs on Arbitrum. Helium, Render and io.net run on Solana. Helium's reward
   calculation is done by oracles run by Nova Labs
   ([HIP-70](https://github.com/helium/HIP/blob/main/0070-scaling-helium.md)): an
   honest admission that computing rewards in a decentralized way is hard.

**Most important finding (analysis).** Suppose a fixed emission of value `E` per epoch is
shared out in proportion to fees burned, and clients burn `B` in total. A self-dealer who
burns `b` receives `E·b/B`. Faking jobs is profitable exactly when **`E/B > 1 + c/b`**,
where `c` is the fake job's compute cost. So the burn guard holds only while the mint for
usage is worth **no more than the fees burned**. The subsidy therefore cannot exceed 1×
at the bootstrap stage, which is exactly when a subsidy is wanted. Under that constraint,
a usage lottery is economically just a rebate on the burn. Any real bootstrap subsidy has
to come from something that can't be self-dealt (proven capacity, verification work),
and each of those has its own problems (§3).

---

## 2. Mechanisms, by type

### Works in production
| Mechanism | Where | What it solves | How it works (primary source) |
|---|---|---|---|
| Probabilistic micropayments | Livepeer | Many tiny payments without a transaction for each | A ticket wins if `keccak256(senderSig, recipientRand) < winProb`. The recipient commits to `recipientRand` by publishing its hash in advance. The sender backs tickets with a deposit plus a reserve and can only withdraw after an unlock period ([spec](https://github.com/livepeer/wiki/blob/master/spec/streamflow/pm.md)). |
| Stake-targeted inflation | Livepeer | Paying for security, not work | Each round, inflation moves up or down by a fixed step toward a target bonding rate (50%) ([docs](https://docs.livepeer.org/v2/delegators/concepts/purpose), [LIP-34/35](https://github.com/livepeer/LIPs)). |
| Difficulty retarget | Bitcoin | Fixed issuance whatever the hash power | Every 2016 blocks, `new = old × actual/target`, with the ratio clamped to between ¼ and 4× ([pow.cpp](https://github.com/bitcoin/bitcoin/blob/master/src/pow.cpp)). |
| VRF election with a public beacon | Filecoin | Leader lottery with no grinding and no gain from splitting identities | `VRF(sk, drand_value …)`. The number of wins comes from a Poisson draw with λ ∝ power share, so splitting into Sybils changes nothing. Power is looked back further than finality ([spec](https://spec.filecoin.io/algorithms/expected_consensus/)). |
| Baseline minting | Filecoin | Emission running ahead of real growth | 30% simple exponential decay, 70% released only as network storage keeps up with a baseline that grows 100% a year ([spec](https://spec.filecoin.io/systems/filecoin_token/minting_model/)). |
| Burn capped 1:1 | Helium HIP-10 | Self-dealing on data traffic | Data rewards are paid at most 1:1 with the Data Credits burned; the surplus goes to other rewards ([HIP-10](https://github.com/helium/HIP/blob/main/0010-usage-based-data-transfer-rewards.md)). |
| Burn-and-mint with capped re-mint | Helium HIP-20, Render RNP-001 | Keeps a supply cap while activity continues | Burned HNT is re-minted up to a small cap per epoch ([HIP-20](https://github.com/helium/HIP/blob/main/0020-hnt-max-supply.md)). Render burns job payments and pays operators from an emission pool according to their share of work ([RNP-001](https://github.com/rendernetwork/RNPs/blob/main/RNP-001.md)). |
| Relay mining | Pocket (Shannon) | Measuring huge numbers of RPC calls cheaply | `hash(signed request + response)` is compared with a difficulty, and `p ← T / R_ema` keeps the number of claims near a target. Servicers commit a Merkle-sum root on-chain, then a later block hash picks which leaf they must prove. Apps are limited by their stake ([paper](https://arxiv.org/abs/2305.10672)). |
| Target inflation instead of a fixed rate per unit | Pocket PUP-11 | Mint growing with activity | 0.01 POKT per relay was replaced with a multiplier adjusted to hit a target annual inflation ([forum](https://forum.pokt.network/t/pup-11-wagmi-inflation/1369)). |
| Threshold randomness beacon | drand (League of Entropy) | Public, verifiable, unpredictable randomness | Threshold BLS. `quicknet` is unchained, runs every 3 s and supports timelock encryption; it started with 18 organizations running 22 nodes ([blog](https://docs.drand.love/blog/2023/10/16/quicknet-is-live/)). |
| Commit-reveal via timelock | Bittensor | Validators copying other validators' scores | Weights are encrypted to a future drand round ([docs](https://docs.learnbittensor.org/concepts/commit-reveal)). |
| Sampled inference checks with eviction | Prime Intellect INTELLECT-2 | Permissionless rollout workers | TOPLOC validators check rollouts on a sample; a node that fails is "slashed and evicted" ([paper](https://arxiv.org/html/2505.07291v1)). |
| Optimistic challenge window | Arbitrum BoLD | Verifying off-chain results cheaply | Assertions confirm after 6.4 days unless challenged, then an interactive bisection game resolves disputes. Bonds can be pooled ([docs](https://docs.arbitrum.io/how-arbitrum-works/bold/gentle-introduction)). |
| Decentralized pool | Monero P2Pool | Payout variance for small miners without a pool operator | A sharechain of lower-difficulty shares; each block pays everyone in the last 6 hours' window (PPLNS) directly ([announcement](https://www.getmonero.org/2021/10/05/p2pool-released.html)). |
| Signed hardware attributes | Akash | Tenants trusting what providers claim | Auditors sign provider attributes on-chain. There is no verification of compute ([docs](https://akash.network/docs/providers/operations/provider-audit/)). |

### Only in papers (or in limited deployment)
| Mechanism | Source | Key point for DAN |
|---|---|---|
| TOPLOC activation fingerprints | [arXiv 2501.16007](https://arxiv.org/abs/2501.16007) | Top-128 values of the last hidden state every 32 tokens, encoded as a polynomial: 258 bytes per 32 tokens. Validation is a single prefill pass. Tested across A100 and 4090, TP 1/2 and three attention kernels with no false positives or negatives. **Can't detect** cheaper-model decoding (speculative decoding with a different model), sampling manipulation or small tweaks. Used in production by Prime Intellect. |
| Proof of Sampling (PoSP / spML) | [arXiv 2405.00295](https://arxiv.org/html/2405.00295) | A pure Nash equilibrium when `p > C / [(1−r)S + (1−2r)R]`. The validator is chosen from a beacon **after** submission. **Needs deterministic execution** (fixed point or soft float). |
| Verde refereed delegation + RepOps | [arXiv 2502.19405](https://arxiv.org/html/2502.19405) (Gensyn) | Bisection to the first operator where results diverge; the referee recomputes only that one operator. RepOps gives bit-identical results across GPUs by fixing the order of float operations, at +30% (matmul) to +126% (8B fine-tune). Assumes one honest party; single GPU only. |
| VeriLLM | [arXiv 2509.24257](https://arxiv.org/pdf/2509.24257) | Recompute via prefill with tolerances. Pipeline splits are not handled clearly. |
| Truebit forced errors | [paper](https://people.cs.uchicago.edu/~teutsch/papers/truebit.pdf) | Solves the **verifier's dilemma** (nobody checks if cheating is rare) by planting known errors with jackpots. |
| Ofelimos PoUW | [CRYPTO 2022](https://eprint.iacr.org/2021/1379.pdf) | A provably secure blockchain built on useful work, but only for a narrow class of optimization problems. It does not generalize to "someone's chat request". |

### Failed or gamed
| System | What happened | Lesson |
|---|---|---|
| **Helium data transfer (before HIP-10)** | One hotspot spent **$145 of Data Credits and earned $21,000 of HNT** ([HIP-10](https://github.com/helium/HIP/blob/main/0010-usage-based-data-transfer-rewards.md)). | Reward per unit of usage above the cost of that usage invites self-dealing. The fix was a 1:1 cap. |
| **Helium Proof-of-Coverage** | Location spoofing. The denylist reached ~25k hotspots (~5%, more than $12M invested in fakes); in March 2022 over 90% were then removed from it ([3roam](https://3roam.com/helium-hotspot-spoofing-and-the-deny-list/), [Tobback](https://medium.com/@tomtobback/the-helium-denylist-going-after-the-scammers-maybe-part1-613b0a8af89c)). | Rewards for "existing somewhere" can't be verified physically. Policing by humans with a denylist is slow, noisy and centralized. |
| **Filecoin Plus** | "Room for SPs to commit fraud and pay bribes." Fil+ reached **56% of consensus power against 11.5% of physical storage**, and "the subsidy crowds out paying deals" ([FIP-0080 discussion](https://github.com/filecoin-project/FIPs/discussions/774); notaries removed for abuse: [#811](https://github.com/filecoin-project/notary-governance/issues/811)). | A multiplier for "real/useful" demand, judged by people, gets captured. Words like "useful" can't be measured. |
| **Pocket fixed mint per relay** | Inflation approaching **~200% APR** projected at 100M relays a day ([PUP-11](https://forum.pokt.network/t/pup-11-wagmi-inflation/1369)). | Never mint a fixed amount per unit of activity. |
| **Bittensor weight copying** | Validators copy the consensus weights and earn without evaluating anything ([docs](https://docs.learnbittensor.org/concepts/weight-copying-in-bittensor)). Commit-reveal helps only if scores change faster than the reveal delay. | Rewarding agreement with consensus rewards copying. Verifiers must commit before they see anyone else's answer. |
| **io.net** | Mass GPU spoofing around the April 2024 token launch, through a leaked universal auth token and a metadata API ([incident report](https://ionet.medium.com/25th-april-incident-report-176e5fb5c576), [Cointelegraph](https://cointelegraph.com/news/io-net-responds-to-gpu-metadata-attack)). Reported numbers are ~1.8M fake GPUs, and after cleanup ~5,350 cluster-ready ([The Block](https://www.theblock.co/post/291315/solana-based-depin-io-net-ceo-claims-network-was-attacked-in-detailed-postmortem); secondary). | Never reward capacity someone *reports*. Rewards need proof by computation, bound to a key. |
| **Livepeer on-chain verification** | Truebit-based verification was dropped once transcoding moved to GPUs, which don't give deterministic output. Checks moved to ML classifiers on the broadcaster side ([blog](https://medium.com/livepeer-blog/livepeer-transcoding-verification-improvements-the-next-level-of-network-security-328b47350e24)). | Checks that need bit-exact results break on GPUs. Plan for tolerance from the start. |

### Depends on trusted hardware
TEE-based checks (Intel SGX/TDX, NVIDIA H100 confidential computing) are cheap where they
exist, but **not on consumer GeForce GPUs**. They are not usable for DAN's target users.

### Practical on ordinary consumer GPUs
Signed receipts; probabilistic micropayments; VRF or BLS tickets over drand; sampled
recompute with **tolerance** (TOPLOC style); bisection down to a single layer; stake plus
escrowed earnings; Poisson tickets that are neutral to Sybils. All of these run on an RTX
2070. RepOps-style determinism would need custom llama.cpp kernels and costs 30–126%;
DAN should avoid it.

---

## 3. DAN problems mapped to solutions

| # | DAN problem | Best existing approach | Tradeoff / verdict |
|---|---|---|---|
| 1 | Unit of work | A protocol formula from the model's shape (like Filecoin's quality-adjusted power: defined by protocol, never by time the provider reports) | Must be fixed per model hash in the catalog. See §6.2. |
| 2 | Client payment split | Livepeer: direct payment by tickets, no protocol fee. Render/Helium: full burn, then mint to operators | A full burn needs a price oracle and ties provider income to the emission pool. **Recommend direct payment plus a small burn** (§8). |
| 3 | Fake demand | Helium HIP-10 cap; Pocket stake-based app budget; Livepeer max float | **Only the cap is proven.** Everything else just limits the rate. |
| 4 | Burn-and-mint? | Helium, Render | Works for stable pricing. It does **not** prevent self-dealing unless mint ≤ burn. |
| 5 | Issuance/difficulty | Bitcoin retarget; Pocket `p ← T/R_ema`; Filecoin baseline | Use a target number of winners per epoch (Pocket). Deferring emission (Filecoin baseline) handles low demand. |
| 6 | Supply shape | Bitcoin cap; Filecoin decay + baseline; Monero tail; Ethereum uncapped | DAN's emission doesn't secure consensus (the L2 does), so **a capped smooth decay** is enough. |
| 7 | Ticket anti-grinding | Filecoin (unique BLS/VRF over drand); Livepeer (recipient commits to randomness first); Pocket (hash after service, commit root, then a later block hash) | Must also stop **backdating**: commit receipts on-chain before the beacon that decides them (§6.3). |
| 8 | Randomness | drand quicknet; chain randomness (RANDAO, block hash) | drand: trusts a threshold of about 18 organizations, 3 s rounds. Chain randomness can be biased a little by block producers. Use drand, with the L2 block hash as a fallback. |
| 9 | VRF/signatures | ECVRF (RFC 9381) or BLS (unique signatures) | **Ed25519 (libp2p keys) is not unique**: a signer can make many valid signatures for one message. That is grinding, so it can't be the lottery source. |
| 10 | Tolerant verification | TOPLOC; prefill recompute; Verde bisection | TOPLOC style plus bisection per layer fits DAN's stages well. |
| 11 | Stake sizing | PoSP inequality; Filecoin reward vesting; Arbitrum bonds | Stake plus escrowed earnings must exceed the gain from cheating ÷ detection probability (§7.4). |
| 12 | Verifier selection/collusion | PoSP (beacon picks the validator after submission); Bittensor commit-reveal; Truebit forced errors | Few nodes hold any given layer range, so collusion risk is higher. Escalate to single-layer referees anyone can become. |
| 13 | Pipeline split | None directly. Pocket and Livepeer pay per service event | Split by each stage's protocol work units, with receipts signed per stage. |
| 14 | Relay / hosting / uptime rewards | Helium PoC (gamed); Filecoin PoSt (capacity, works); Pocket relays (fee-backed) | Uptime and presence claims get gamed. Reward only what is proven by computation or backed by fees. |
| 15 | Sybil without KYC | Resource-proportional Poisson tickets (Filecoin); stake (Livepeer); fee backing | No mechanism gives "one person, one node", and none is needed if rewards scale with proven resource or burned fees. |
| 16 | Reputation | Akash auditors (centralized); Bittensor (gamed) | Keep reputation **local to clients** for choosing providers. The protocol only records audits passed and slashes. |
| 17 | Low demand | Filecoin baseline (deferral); Helium surplus redistribution | Defer the unused usage allowance. Never hand it out as extra rewards per job. |
| 18 | Variance/pools | Poisson win counts; lower difficulty; P2Pool | Tune the target winner count so a median node wins at least weekly. |
| 19 | Settlement without own chain | Livepeer (Arbitrum), Helium/Render/io.net (Solana) | Reward *calculation* was centralized (Helium oracles). Lottery tickets avoid needing an oracle. |
| 20 | Real attacks | See §4 | |

---

## 4. Attacks that matter most for DAN

1. **Self-dealing (fake demand).** The attacker is both client and provider. This is
   the Helium HIP-10 case. It's the main threat to any "mint for useful work" design.
2. **Grinding by choosing the job.** Pick prompts whose ticket wins and compute only those
   **(analysis)**. Fixed if the ticket depends on the finished output plus a later beacon.
3. **Backdating receipts** **(analysis)**. A self-dealer who sees beacon `R+k` forges
   receipts that claim to come from `R` and keeps only winners. Fixed by committing the
   receipt root on-chain before `R+k` (Pocket's claim/proof).
4. **Non-unique signatures used as a lottery.** Ed25519 signers can re-sign with other
   nonces **(analysis)**. Use a VRF or BLS.
5. **Claiming capacity that doesn't exist.** io.net, Helium spoofing. Never reward
   claimed hardware; reward only proven computation.
6. **Lazy computation.** Returning junk, a smaller model, lower precision, or skipping
   layers. TOPLOC catches model swaps and precision changes, but not all subtle changes.
7. **Lazy or copying verifiers.** Truebit's verifier's dilemma, Bittensor's weight
   copying.
8. **Verifier and provider collusion.** Worse in DAN because few nodes hold a given
   layer range.
9. **Farming the capacity pool with idle hardware.** This is Filecoin committed capacity:
   the network ends up paying for capacity nobody uses **(analysis; FIP-0080 shows the
   related crowding-out)**.
10. **Exploiting the tolerance.** Small perturbations under the threshold on every layer
    **(analysis)**. The harm is bounded; accept it.
11. **Infrastructure and key compromise.** io.net's universal token. Every reward right
    must be bound to a per-node key, with no shared secrets.

---

## 5. What to keep from the current DAN design

- AI work as the expensive operation; no hash grinding.
- A ticket that depends on the **completed output**, **provider identity** and **later
  randomness**. That's exactly Pocket and Filecoin combined.
- "Fake jobs must lose money" as the invariant. §6.4 makes it hold.
- Expected value = probability × amount × value. The cap below is written in token units,
  so it holds at any token price.
- Normal payment for work, with the lottery as extra.
- Automatic difficulty and emission not driven by activity (the Pocket PUP-11 lesson).
- Sampled checks + commitments + stake instead of running everything twice.
- Tolerant verification (TOPLOC style) and challenges per stage.
- Rewards for verifiers.
- Existing chain for settlement only.
- Permissionless joining with a local key.
- The order: receipts → points → verification → adversarial testing → testnet → token.

## 6. What to change or remove

1. **Cap the usage mint at the burn.** Per epoch, `mint_usage ≤ κ · burn` (κ < 1, in
   token units). Without this the design is Helium before HIP-10. With it, the usage
   lottery is a partial rebate of the burn, not a subsidy. Be honest about that.
2. **Weight tickets by fee burned, not by work units.** A self-dealer sets its own price
   per work unit. Work units set prices and split payments between stages; burned value
   decides the lottery (or use a burn floor per work unit, which is equivalent).
3. **Burn at deposit time, not on a winning ticket.** The burn must be certain for every
   job. Probabilistic burns can be gamed with probabilistic payments.
4. **Add an on-chain commitment of receipts** before the deciding beacon (stops
   backdating).
5. **Use a VRF/BLS key for tickets**, not the Ed25519 libp2p key. Register it with the
   stake.
6. **Drop "rewards for relay/uptime/hosting" at first.** Each one is claimed rather than
   proven and has been farmed elsewhere. Revisit hosting and relays only as fee-backed
   services.
7. **Don't count on the usage lottery to bootstrap supply.** If a subsidy is wanted, it
   has to come from a separate, small, fast-decaying **capacity pool** based on proven
   computation (§8), with its idle-capacity risk accepted.
8. **Drop the adaptive burn rate at first.** No production system adapts burn to hit an
   inflation target. EIP-1559 adapts to congestion, which is a different problem.
   Adjusting mint against the burn (the cap) does the same job more simply.

### 6.2 Work unit (proposal)
Fixed per model SHA-256 and published with the manifest. It's used for pricing and for
splitting a job's payment across stages. The lottery weight comes from burned fees
(change 2).

```
WU(stage, job) = Σ_layers∈[begin,end) [ w_pf·T_prefill·A_l + w_dec·T_decode·A_l
                                         + w_attn·Σ_t ctx_t·d_l ]
A_l   = active parameters in layer l (dense: all; MoE: shared + top-k experts)
ctx_t = context length when token t is processed (attention cost)
head/embedding counted for the first/last stage; speculative tokens counted as verified
tokens only (rejected guesses are the provider's own choice)
```
`w_pf < w_dec` per token (prefill is batched and compute-bound; decode is memory-bound).
The weights are calibrated once on reference GPUs and fixed per protocol version. They're
never measured per node.

### 6.3 Ticket (proposal)
```
receipt_i  = client_sig( session, stage, model_sha, layer range, token counts,
                         H(input frames), H(output frames), fee, beacon_round_at_start )
root_e     = MerkleSum root of a provider's receipts for epoch e → committed on-chain
             before beacon round R_e + k
ticket_i   = VRF_vrfkey( "DAN-ticket" ‖ receipt_i ‖ root_e ‖ drand(R_e + k) )
wins        = Poisson draw with λ = (fee_burned_i / F_target) · T   (Filecoin style)
```
`H(output frames)` of stage i equals `H(input frames)` of stage i+1 **byte for byte**,
because the frames travel unchanged (§10.1 of PROJECT.md). Neighbours therefore co-sign a
hash chain, and a stage can't later swap its input.

### 6.4 The self-dealing invariant, in token units
For a self-dealer: mint received ≤ κ × its own burn (by the cap and pro-rata weighting),
while its unrecoverable cost is its burn + compute + verification fee. The attacker loses
at least `(1−κ)·burn + compute` per fake job, **whatever the token price** (mint and burn
are counted in the same token). If clients pay in a stablecoin, the protocol fee buys
DAN and burns it, and the cap is counted in the DAN burned.

---

## 7. Proposed verification and slashing protocol

1. **Commit.** Each stage keeps the input frames of every receipt for the challenge
   window W (for example 24 h; about 20 KB per token per hop at f32 for 14B, and only
   frames that get sampled are ever fetched). The neighbours' co-signed hash chain fixes
   inputs and outputs.
2. **Sample.** After `root_e` is committed, `hash(drand(R_e+k) ‖ receipt_id) < p`
   selects audits (p ≈ 1–5%, never below a protocol floor). The provider can't know which
   ones.
3. **Assign.** `hash(beacon ‖ receipt ‖ stage)` picks a verifier, weighted by stake,
   from nodes that hold that layer range. If fewer than m exist, a referee downloads the
   range (range-backed weights, §9.5 of PROJECT.md).
4. **Recompute blind.** The verifier gets the input frames, runs the stage as one prefill
   batch (as in TOPLOC), and **commits its own fingerprint** (top-k per layer boundary
   and per chunk) before seeing the provider's. drand timelock or commit-reveal prevents
   copying.
5. **Compare with tolerance.** Thresholds on exponent and mantissa, TOPLOC style, are
   calibrated per model on DAN's real mix (2070, 3090, A4500, A5000, CPU). For the last
   stage, the output token must be the argmax, unless the gap to the runner-up is below ε.
6. **Escalate by bisection (Verde).** If they disagree, split the stage layer by layer
   down to the first diverging layer. Several randomly drawn referees recompute only that
   **one layer**, which any node can download. The majority decides; the loser is slashed.
7. **Keep verifiers honest.** Some audits go to two verifiers at once (probability q),
   and a verifier whose fingerprint is wrong is slashed. This does the job of Truebit's
   forced errors without faking provider errors.
8. **Slash.** Slash stake plus **all escrowed earnings in the window**, and share part
   of it with the verifier and referees. Earnings vest over W, as Filecoin vests block
   rewards.

**Stake sizing.** A cheat on a job worth g (fee + mint + compute saved) is caught with
probability p. The penalty is the stake S plus the escrow `Esc`. The requirement is
`p·(S + Esc) > g`, with a safety margin. High-volume nodes have large escrow and so
insure themselves. New nodes need a minimum stake **and a throughput limit**: value in
flight ≤ `p·(S+Esc)/margin` (Livepeer's "max float" idea). Reputation lets a node grow
that limit through its escrow, never by lowering p below the floor.

---

## 8. Proposed mint, burn and difficulty mechanism

- **Client payment.** Probabilistic micropayment tickets (Livepeer design) sent
  **directly to each stage**, priced in WU. Out of each payment: ~85–92% to the stages
  (by WU share), ~3–5% verification fee pool, ~5–10% protocol fee **bought back and
  burned** at deposit time. These numbers are placeholders for the testnet to tune.
- **Emission schedule.** A fixed, smoothly decaying curve `S(e)` (e.g. exponential with
  a 4-year half-life) with a hard cap. There's no tail at first; DAN's security comes from
  the settlement chain, not from its own issuance.
- **Split of `S(e)`:**
  - **Usage rebate pool:** `min(share_u·S(e), κ·Burn(e))`, as fee-weighted Poisson
    tickets. Whatever is left over goes to a **deferred reserve** (Filecoin baseline
    idea) and never to other pools in the same epoch.
  - **Verification pool:** paid per completed audit, including the double-check audits.
  - **Capacity pool (optional, small, decays fast):** Poisson tickets from **beacon
    challenges**. At a random beacon round, a node must run a stage on beacon-derived
    input for the layers it advertises, within a deadline. The ticket is its VRF over the
    result. It's proven by computation and can't be self-dealt, but it buys idle capacity
    (Filecoin CC). Keep it only as a bootstrap, with its weight dropping toward zero as
    the usage burn grows.
- **Difficulty.** For each pool, a target winner count `T` per epoch.
  `winProb ← T / N_ema` (Pocket), with the change per epoch clamped to ¼–4× (Bitcoin).
  Reward per win = pool / T. Differences between actual and target winners go into the
  reserve.
- **Low demand.** The usage pool is small, because of the cap. The capacity pool keeps
  nodes around during bootstrap. The deferred reserve is released later only as the
  burn grows. There's never "more mint because there's less demand".

## 9. Concrete anti-fake-demand mechanism (all together)
1. Burn is certain, at deposit time, in DAN units.
2. `mint_usage(e) ≤ κ·Burn(e)`, with κ < 1.
3. Tickets are weighted by fee burned, not by WU or job count.
4. The receipt root is committed on-chain before the deciding drand round.
5. Tickets use a VRF/BLS key over the finished output hash plus that round.
6. Fake jobs are audited like real ones, so the self-dealer also pays for compute and
   verification.
7. Result: every fake job loses at least `(1−κ)·burn + compute`. A real client's burn
   is a cost it pays anyway.

---

## 10. Open questions without good answers

1. **A bootstrap subsidy that can't be farmed.** Every known option is either farmable
   (a demand subsidy) or buys idle hardware (a capacity subsidy). This is the central
   unsolved problem. Grant programmes for known users are what networks actually do,
   and those are centralized (Fil+ shows the risk).
2. **Privacy against auditability.** Audits show real users' activations (and the first
   stage sees prompts). Making audits opt-in would let cheaters skip audited sessions.
   No good answer yet.
3. **Tolerance thresholds** for DAN's mix of GPU, CPU, f16 and fp8 wire formats, and for
   MoE routing ties (a flipped expert choice changes everything after it). This needs
   measurement.
4. **Speculative decoding.** TOPLOC can't tell a cheap model's decode from the real
   one's. DAN's verification must check each emitted token against the target model.
5. **Few holders of each layer range**, which makes collusion between verifier and
   provider cheaper. Single-layer referees help; how big should the referee pool be?
6. **Who pays gas** for small providers' claims and commits? That decides whether
   epochs are hourly or daily.
7. **Replicas.** The owner is the front door; members need clients' signatures
   directly, not the owner's word.
8. **Sampling other than greedy** (seeded sampling makes it checkable; not designed yet).
9. **Price oracle** for WU pricing in USD if clients pay in DAN (Helium uses one; it's
   another trust point).
10. **Governance** of WU weights, κ, p, T and the model list without a central
    authority.
11. **Client onboarding.** Clients need a wallet and a deposit, which clashes with "free
    chat for friends".
12. **Legal and tax treatment** of a token earned for work, which varies by country.
    Needs a lawyer before a real token.
13. **Capacity challenges vs outsourcing.** Is forwarding a challenge to a rented cloud
    GPU cheating, or just real capacity?
14. **Exit scams.** A node builds up good history, then cheats once at high value. The
    escrow window must cover the largest job.

---

## Sources
Primary: [Livepeer PM spec](https://github.com/livepeer/wiki/blob/master/spec/streamflow/pm.md) ·
[Bitcoin pow.cpp](https://github.com/bitcoin/bitcoin/blob/master/src/pow.cpp) ·
[Filecoin consensus](https://spec.filecoin.io/algorithms/expected_consensus/) ·
[Filecoin minting](https://spec.filecoin.io/systems/filecoin_token/minting_model/) ·
[FIP-0080 discussion](https://github.com/filecoin-project/FIPs/discussions/774) ·
[Helium HIP-10](https://github.com/helium/HIP/blob/main/0010-usage-based-data-transfer-rewards.md) ·
[HIP-20](https://github.com/helium/HIP/blob/main/0020-hnt-max-supply.md) ·
[HIP-70](https://github.com/helium/HIP/blob/main/0070-scaling-helium.md) ·
[Render RNP-001](https://github.com/rendernetwork/RNPs/blob/main/RNP-001.md) ·
[Pocket Relay Mining](https://arxiv.org/abs/2305.10672) ·
[Pocket PUP-11](https://forum.pokt.network/t/pup-11-wagmi-inflation/1369) ·
[TOPLOC](https://arxiv.org/abs/2501.16007) · [INTELLECT-2](https://arxiv.org/html/2505.07291v1) ·
[PoSP](https://arxiv.org/html/2405.00295) · [Verde](https://arxiv.org/html/2502.19405) ·
[VeriLLM](https://arxiv.org/pdf/2509.24257) · [Truebit](https://people.cs.uchicago.edu/~teutsch/papers/truebit.pdf) ·
[Ofelimos](https://eprint.iacr.org/2021/1379.pdf) · [drand quicknet](https://docs.drand.love/blog/2023/10/16/quicknet-is-live/) ·
[Bittensor commit-reveal](https://docs.learnbittensor.org/concepts/commit-reveal) ·
[Arbitrum BoLD](https://docs.arbitrum.io/how-arbitrum-works/bold/gentle-introduction) ·
[Monero P2Pool](https://www.getmonero.org/2021/10/05/p2pool-released.html) ·
[Akash audits](https://akash.network/docs/providers/operations/provider-audit/) ·
[Livepeer verification](https://medium.com/livepeer-blog/livepeer-transcoding-verification-improvements-the-next-level-of-network-security-328b47350e24) ·
[io.net incident](https://ionet.medium.com/25th-april-incident-report-176e5fb5c576).
Secondary (numbers not confirmed from primary sources): [Helium denylist](https://3roam.com/helium-hotspot-spoofing-and-the-deny-list/),
[io.net fake GPU counts](https://www.theblock.co/post/291315/solana-based-depin-io-net-ceo-claims-network-was-attacked-in-detailed-postmortem),
[Helium data-only farming](https://cassiopeia.hk/the-next-helium-scam-earning-hnt-by-generating-data-traffic-over-low-cost-data-only-hotspots/).
