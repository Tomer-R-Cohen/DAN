# DAN payment architecture: pressure test of "full burn, then mint"

*Research only, nothing implemented. 2026-09-25. Follows `REWARD_PROTOCOL_RESEARCH.md`
and corrects it where noted (§7).*

**The hypothesis tested:** clients pay a quoted price in DAN, the payment is burned
(possibly all of it), and providers are paid later from protocol issuance according to
valid contribution per epoch, with `mint_usage ≤ κ·burn` and κ < 1. Pricing, accounting,
verification and issuance are separate. The client never deals with workers.

**Verdict: reject the full-burn, epoch-pool version.** It doesn't remove the hard parts
(per-stage accounting, failures, speculation). It moves them into the issuance system,
where they are adversarial, pooled and harder to verify. On top of that it adds
unpredictable income, demand dilution, the need for a protocol-set price with an oracle,
and refund problems. The three large networks that ran pool-style or fixed-issuance
rewards have each moved to **paying each provider for the specific paid work it did**
(§2). The UX you want (pay once, never see the workers) comes from *one deposit + an
automatic payer*, not from burning.

Claims from sources are linked. My own reasoning is marked **(analysis)**.

---

## 1. The key fact: κ < 1 plus full burn is just a fee with extra risk (analysis)

If clients burn `B` and providers receive at most `κ·B`, providers are paid **out of
client money minus a (1−κ) fee**, exactly as with direct payment. The mint adds nothing
economically. What it does add:

- **Price risk.** Value is burned at time t₁ and minted at t₂, and the provider sells at
  t₃.
- **Pooling.** Each provider's pay depends on everyone else's work in the epoch.
- **Accounting risk.** Contribution now has to be proven to the *protocol* (adversarial,
  on-chain), instead of to the *payer* (who saw the work and chose the workers).

If instead the mint per job equals that job's burn times κ and goes to that job's
providers, it is **direct payment in DAN with a (1−κ) fee**. Pocket arrived at exactly
this, below.

The only way for providers to receive more than clients paid is a subsidy (mint > burn),
and a subsidy that follows demand is fake-demand mining (previous report, §1). **So there
is no version of "burn, then mint for usage" that is both safe and better than direct
payment.**

## 2. What real networks converged on

| Network | Model | What happened | Source |
|---|---|---|---|
| **Pocket (Shannon, May 2025)** | Burn per relay; mint to *the suppliers who served it* | "Users pay for data, providers are paid by users, no tokens printed from thin air". Then 2.5% of the mint was cut (PIP-41): **direct payment with a 2.5% fee**, written as burn and mint. For its inflation-funded subsidy, apps are *overcharged by the inflation amount* to "prevent self-dealing", and the foundation reimburses them **off-chain**, a permissioned subsidy. | [pocket.network](https://pocket.network/pokt-token/), [TLM Go docs](https://pkg.go.dev/github.com/pokt-network/poktroll/x/tokenomics/token_logic_module) |
| **Helium (HIP-149, 2026)** | Fixed issuance split by work (PoC + data) | Rewardable data grew **~4× (24K → 97K GB/day)** while HNT issuance stayed fixed and the price "more than halved". Operators' income was "decoupled from the rate carriers actually pay". The fix is **revenue-linked earnings**: at least 0.5× and at most 3.0× the payer's per-GB rate. PoC is being retired. | [HIP-149](https://github.com/helium/HIP/blob/main/0149-helium-utility-and-emissions-realignment.md) |
| **Akash (AEP-23 → AEP-76, live 2026-03-23)** | First stablecoin payments; then BME with USD credits | Stablecoin payments grew revenue but "reduced AKT demand". Under BME, a tenant burns AKT into non-transferable, USD-pegged **ACT**. At each lease settlement ACT is burned and **AKT is minted to that provider at the oracle price**. The oracle is a 30-min TWAP median of two feeds with a 1.5% outlier cut and circuit breakers at collateral ratio 0.95 / 0.90. ACT refunds any time. | [AEP-76](https://akash.network/roadmap/aep-76/), [discussion](https://github.com/orgs/akash-network/discussions/1034) |
| **Livepeer** | Probabilistic tickets straight to the orchestrator; stake inflation separate | In production for years. Orchestrators set their own price (`pricePerUnit`); broadcasters deposit once. Fees grew ~3× in 2025, >72% from AI inference (secondary). | [PM spec](https://github.com/livepeer/wiki/blob/master/spec/streamflow/pm.md), [docs](https://docs.livepeer.org/orchestrators/guides/get-started) |
| **Render (RNP-001)** | Jobs priced in USD, RENDER burned; operators paid from a fixed emission pool per epoch, by work share | The RNP text describes pool-based pay. Secondary sources also describe escrow paid out per job plus separate availability and completion emissions. **Unclear; not verified which dominates in practice.** | [RNP-001](https://github.com/rendernetwork/RNPs/blob/main/RNP-001.md) |
| **Golem** | Direct pay-as-you-go: debit notes during work, invoice at the end, batched ERC-20 payments on Polygon | Simple and direct. The requestor can walk away; the provider stops on unaccepted debit notes. | [docs](https://docs.golem.network/docs/golem/payments) |
| **Filecoin Onchain Cloud (2025)** | "Payment rails": streaming client → provider payments in FIL or USDFC, with lockup and an optional arbiter contract | Direct payment. Block-reward issuance stays separate (capacity). | [docs](https://docs.filecoin.cloud/cookbooks/filecoin-pay/) |
| **Bittensor** | All pay is protocol emission, split by scores | Subnet emissions went price-based → flow-based (Nov 2025) → back to price-based (Jun 2026) as each got gamed; weight copying. | [emissions](https://docs.learnbittensor.org/learn/emissions), [secondary](https://www.tao.media/the-ultimate-guide-to-bittensor-2026/) |

**Pattern:** pay the provider for the paid work it did, at a known price. Issuance goes
to stake, capacity or a bounded, often permissioned, subsidy. Pure emission-pool systems
(Helium PoC, Bittensor) keep having to redesign.

---

## 3. Answers to the 35 questions

### Direct payment: is it really that ugly?
**1.** Less than it looks for DAN, because **the party that plans the route already knows
every worker, retry and speculation depth**. That's `dan-client`, or the replica owner.
Livepeer does solve the settlement part cleanly. Its node software is non-trivial (deposit
+ reserve, unlock period, committed randomness, nonces, max float, face value vs gas), but
the *user* experience is "deposit once". One piece doesn't carry over: Livepeer's reserve
is split among a fixed **active set**, and DAN has none, so that part must be redesigned
(§6). Abandoning direct payment would be a mistake.

### Full burn
**2. New problems with 100% burn:**
- a protocol-set price, which needs governance and an oracle;
- refunds for failed jobs;
- provider pay that depends on the pool, the token price and dilution;
- the pool is shared, so an accounting error or fraud by one provider takes from all;
- every contribution must be proven to the protocol instead of the payer;
- providers can't set prices, which breaks "workers decide for themselves";
- clients must buy DAN.

**3.** They can't know exactly. They can only guess the epoch's total work and burn.
Helium's operators saw exactly this.

**4.** Yes. Income = (own share of epoch work) × pool × token price. Three of the four
factors are outside the provider's control. See HIP-149.

**5.** Yes **(analysis)**:
- Timing work or claims toward quiet epochs.
- Delaying receipt submission to whichever epoch pays best, if receipts aren't bound to
  their epoch.
- Inflating WU by choosing expensive work shapes (long contexts, prefill-heavy jobs).
- In a pool, fake work also **dilutes everyone else**, so the damage is shared and the
  gain private.

**6.** Yes, with any fixed pool. Demand up → more work units → less per unit. That's the
Helium 4× case. Providers are pushed away exactly when they're needed. (With a mint tied
to burn it scales, but then it's direct payment, §1.)

### Accounting
**7.** In every model, WU per stage (layers × active params × tokens, weighted for
prefill/decode). MoE counts active experts for compute. Holding the whole expert bank is a
memory cost, reflected in the stage's own price. The payer controls speculation depth, so
**the payer pays for computed rows**, and rejected guesses are the payer's cost.
Verification is paid from a separate fee (§6).

**8.** Pay for **work the payer received and acknowledged**, in small chunks (Golem's
debit notes). Work lost when a *stage* fails is that stage's small loss, bounded by one
chunk. Speculative rows: paid, because the payer asked for them. Don't pay for "valid but
discarded" work that nobody acknowledged, because nobody can verify it existed.

**9–12.** With pay-as-you-go tickets these questions largely disappear. Nothing is burned
at deposit. The deposit stays the client's (withdrawable after an unlock delay). Each
acknowledged chunk is paid by a ticket, and the protocol fee is taken **when a winning
ticket is redeemed** (a progressive burn). A job that fails completely has cost the
client only the chunks it received; there's nothing to refund. **Burning at deposit does
create bad UX for failed jobs.** An escrow with a burn on completion needs someone to rule
on "completed", which is a dispute system. Chunked tickets avoid that.

**13.** In the recommended model there's **no protocol reward tied to usage**, so there is
nothing to maximize by manipulating a quote. The client's software computes WU itself
from the tokens it sent and received and signs only that. A colluding client and worker
can only move their own money in a circle.

**14.** For payment: the stage's own **price × WU**, set by the market. For any issuance:
never measured compute or runtime (can be faked), never raw WU (a self-dealer sets its own
price per WU). If a pool must follow usage, weight it by **market value paid** with the
per-job cap. Capacity issuance is weighted by proven capacity (§5).

**15.** With direct pay: price × expected WU, known before accepting. A ticket's expected
value (`faceValue × winProb`) is checked before the work, as in Livepeer. With an epoch
pool this can't be answered (Q3).

### Surplus and bootstrap
**16.** It makes the system safe, and it makes the mint pointless (§1). It's unattractive
only if you expect the mint to be the income.

**17.** Only from **clients paying more than providers' costs**: a real market price.
Beyond that: token appreciation (speculation) or a subsidy. There is no fourth source.

**18.** For a public market starting from zero demand, some subsidy or altruism is
unavoidable. **For the friends beta it isn't needed:** providers are also users, so
"serve to earn chat credit" (model G2, §4) is a complete economy with no token.

**19.** Only if the subsidized work is **created by the protocol, not by a client**. Use
beacon-driven capacity challenges: your original lottery, applied to protocol-issued jobs
(§5). Every subsidy tied to demand is farmable up to its size. The known alternatives are
permissioned: Pocket reimburses off-chain, and Helium's revenue-linked multiplier (up to
3×) works because its payers are known carriers.

### Demand extremes
**20. Low demand.** Direct pay: idle providers earn nothing but the capacity pool; price
competition lowers prices. Full burn with a fixed pool: providers are overpaid per unit,
which is the best time to farm. With a burn-capped pool: providers earn little.

**21. High demand.** Direct pay: workers raise prices and the planner picks the free and
cheap ones, so the market clears. Fixed pool: per-unit pay falls, so capacity leaves
(Q6).

### Tokens, stablecoins, oracles
**22.** Yes. With full BME, clients carry quote risk (unless there's an oracle) and
providers carry risk between burn, mint and sale. Stablecoin direct pay removes both.

**23–24.** Yes, and it's cleaner. Clients pay USDC. The protocol fee (a % of each
redeemed ticket) collects in a contract that **buys DAN through a periodic auction and
burns it**. MakerDAO ran surplus auctions like this: sell surplus stablecoin for the
governance token, burn it. **No price oracle is needed**, because the market sets the
price. Akash's history shows why a burn is kept at all: plain stablecoin payments cut
demand for the token (AEP-23).

**25.** Oracle-priced minting (Akash BME) is dangerous for a **new, thin-liquidity
token**. Push the price down briefly and more DAN is minted per dollar owed. Mango Markets
(Oct 2022, ~$110M) is the classic manipulation of a thin token's oracle. Akash needs TWAP,
medians, outlier cuts and circuit breakers even with an established token. A buy-and-burn
auction's worst case is overpaying for the burn, and no one can extract minted tokens.

### Comparisons
**26–28.** See §2. Failures and unintended incentives of BME and epoch pools:
- Helium: fixed issuance against 4× usage (HIP-149), the pre-cap $145 → $21k arbitrage
  (HIP-10), PoC spoofing of a fixed pool.
- Pocket: fixed mint per relay heading toward ~200% inflation.
- Bittensor: its emission allocation was gamed and redesigned twice in 8 months.
- Filecoin Plus: a multiplier captured by self-dealing.

**29–30.** Yes. The client pays into **one deposit**, and payment to providers happens
off-chain as tickets, which are the "internal accounting". A central off-chain accountant
settling batches (model F) is either a trusted operator (Helium's Nova-run oracles) or its
own chain or rollup (Pocket settles claims on its own Cosmos chain). Probabilistic
tickets are the decentralized form of F: one deposit, any number of payees,
statistically exact, and the chain is touched only by winners.

**31–33.** See the matrix (§4) and the table below. Only B, E and G2 keep the chain
completely off the inference hot path *without* an aggregator.

| Design | Chain actions |
|---|---|
| A: on-chain per job | 1+ transaction per job per stage. Not viable. |
| B/E: tickets | Client: deposit and top-ups. Worker: one redemption per win (tuned to e.g. daily). Protocol: periodic auction. |
| Payment channels (per pair) | Open/close per client–worker pair, with the deposit split across pairs. Bad for many short-lived peers. |
| State channels / hubs | Need hubs with liquidity (Lightning-style), i.e. operators. |
| C: burn + epoch pool | Burn per job or deposit, a receipt-root commit per worker per epoch (against backdating), a pool computation (an aggregator, or tickets again), a claim per epoch. More than B. |
| F: batched off-chain accounting | One batch per epoch, but someone trusted computes it, or you run a rollup. |
| G1: Akash BME | Its own chain, per-lease settlement, an oracle. |

**34.** B/E (and G2 now). Any key can receive tickets; no registration. Redemption gas
can come from the winnings through a relayer. Stake is needed only for a trust tier or
for throughput, not to get paid. Caveat: clients need stablecoins, which usually means a
KYC on-ramp. That's a UX barrier, not a protocol one.

**35.** B/E. The payment path has **no protocol-funded reward to game**: value moves from
real clients to workers they chose. Gaming is limited to (a) cheating the client, handled
by spot checks and slashing, (b) the capacity pool, which is bounded and protocol-created,
and (c) the ticket mechanics, which Livepeer has run in production for years.

---

## 4. Model comparison

A direct on-chain · B Livepeer tickets · C full burn → epoch mint · D direct + burn +
usage subsidy · **E tickets in stablecoin + fee buy-and-burn** · F escrow + off-chain
accounting + batches · G1 Akash-style BME (USD credit, oracle mint per job) · G2 reciprocal
credits (serve to earn chat, off-chain, no token)

| | A | B | C | D | **E** | F | G1 | G2 |
|---|---|---|---|---|---|---|---|---|
| Fake-demand resistance | full (no mint) | full | only with κ<1, then pointless | **weak** (subsidy follows demand) | full | full if no mint | full (mint = paid value) | partial: self-serving earns credit for your own chat only |
| Provider revenue predictability | high | high (± ticket variance) | **low** | medium | **high, in USD** | high | high at settlement, token after | n/a (credits) |
| Client UX | bad (tx per job) | deposit once, needs token | buy DAN, burn per job, no refunds | deposit once | **deposit USDC once** | deposit once | card or USD credits | none: just serve |
| Settlement complexity | low logic, huge volume | medium (node software) | high (pool, commits, claims) | high | medium | high or trusted | high (own chain, oracle) | low |
| On-chain cost | prohibitive | low | medium–high | medium | low | low but trusted | own chain | zero |
| Failure handling | per job | per chunk, natural | refunds needed | per chunk | per chunk, natural | aggregator decides | per lease | per chunk |
| Speculation | awkward | payer pays rows | protocol must define | payer pays | payer pays rows | aggregator | per lease | payer pays |
| Pipeline accounting | per stage tx | payer → each stage | protocol-wide, adversarial | payer | **payer → each stage** | aggregator | per lease | receipts |
| Sybil resistance | n/a | n/a (clients choose) | needs stake or proofs | weak | n/a + stake for trust | n/a | n/a | weak (fine among friends) |
| Token value capture | none unless paid in DAN | if paid in DAN | strong narrative | medium | **via fee buy-and-burn** | none | strong | none |
| Bootstrapping | none | none | pool (farmable if > burn) | farmable | + capacity pool (§5) | none | none | **reciprocity** |
| Price volatility | client or provider | same | **both, plus pool** | both | **none on pay path** | none | the oracle's job | none |
| Decentralization | high | high | medium (pool + price governance) | medium | high | **low** | medium (oracle, own chain) | high |
| Attack surface | small | ticket mechanics | largest | large | ticket mechanics + auction | the operator | oracle + chain | credit forging (receipts) |

---

## 5. Keeping the "mining": capacity challenges (analysis)

Your original lottery design (one ticket per unit of valid work, tied to identity and
later randomness, difficulty-adjusted) is sound. It failed only because **clients**
create the work. Point it at work the **protocol** creates:

1. Each epoch, drand round `R` derives a challenge for each advertised layer range:
   input activations from `H(R ‖ model ‖ range)`.
2. The node runs its stage on that input within a deadline set from reference GPU
   speeds. The deadline proves the layers are resident on a real accelerator.
3. It publishes a fingerprint of its output (TOPLOC style) and
   `ticket = VRF(key, R ‖ fingerprint)`. Wins are a Poisson draw weighted by resident
   layer-GiB; the difficulty targets T winners per epoch; the pool follows a fixed,
   decaying schedule.
4. Challenges are spot-checked by the same verifiers as client work.

Nobody can self-deal (the protocol picked the job), grinding is impossible (the input
comes from the beacon), and splitting into Sybils is neutral (Poisson on resource).
Bonus: the answers are **honest speed measurements** for the planner.

The cost is that it pays for idle capacity, as Filecoin's capacity rewards did. So:
small, decaying to zero over a fixed period, and never the main income.

---

## 6. Recommended architecture (simplest viable)

**Phase 0, now (friends beta): G2, reciprocal credits, no chain, no token.**
Workers sign per-chunk receipts, and payers countersign them (payer = `dan-client`, or
the replica owner). Credits earned by serving are spent on chatting. All of it is local
and off-chain. It tests exactly the accounting that every later design needs (per-stage
WU, speculation rows, failures) with nothing at stake.

**Phase 1+, public network: E, stablecoin tickets paid by the planner, fee
buy-and-burn.**
1. **Workers set prices** per WU in `provider_available`; the planner already chooses
   workers and now also sees prices. "Workers decide, clients plan" is kept.
2. **The payer is whoever plans the route.** `dan-client` pays each stage it planned. On
   a replica, the client pays the **owner only** (one payee), and the owner pays its
   members from its own deposit. A client never needs to know members, retries or
   replacements.
3. **The client deposits USDC once** on an L2 contract (unlock delay before withdrawal).
   It is billed per delivered token at the quoted price. Internally, the payer sends each
   stage a probabilistic ticket per acknowledged chunk of computed rows.
4. **The protocol fee** is a small % of each **redeemed winning ticket**. It collects in
   a contract that buys DAN through a periodic auction and burns it. No oracle and no
   burn at deposit, so there are no refunds.
5. **DAN token** is used for: worker stake (trust tier, throughput limit, slashable),
   verifier rewards, the capacity pool (§5) and governance. **No usage-linked mint at
   all**, so the κ question disappears.
6. **Verification** (previous report, §7) protects *clients*. A slash compensates the
   victim and the verifier. Earnings vest over the challenge window, acting as automatic
   collateral.
7. **To redesign from Livepeer:** its reserve depends on an active set that DAN lacks.
   Replace it with a **per-worker max float** (outstanding expected value ≤ a share of
   the deposit that the worker checks) plus a **penalty on the deposit if it's
   overspent**.

### Why it beats the alternatives
- **Than C (full burn):** same fake-demand safety, but with predictable USD income, no
  pool dilution, no refunds, no oracle, no protocol pricing, and less on-chain work.
  C's "simplicity" was only moved into the mint accounting.
- **Than D:** D's usage subsidy is the farmable part. Removing it leaves E.
- **Than F:** F needs a trusted accountant or a chain of its own; tickets do the same
  job with neither.
- **Than G1 (Akash BME):** no oracle, which matters most for a new, illiquid token. Akash
  needs a whole chain, an oracle and circuit breakers for what the auction does
  trustlessly.
- **Than A/B in DAN:** stablecoins remove price risk for both sides. The fee buy-and-burn
  keeps the value capture that Akash lost with plain stablecoins.
- **Starting with G2:** it needs no token, chain, lawyer or on-ramp, and it builds the
  receipts every later phase uses.

---

## 7. Corrections to `REWARD_PROTOCOL_RESEARCH.md`
- §6 item 3 ("burn at deposit time") is **withdrawn**. It was needed only to protect a
  usage mint, and with no usage mint the burn can be taken progressively from redeemed
  tickets.
- The **usage rebate pool** (§8 there) is **dropped**. Under the safety cap it's a fee
  rebate that adds only risk.
- The capacity pool moves from "optional" to **the only bootstrap issuance**, reframed as
  your lottery applied to protocol-created jobs (§5).

## 8. Still open
1. Redesigning the reserve without an active set (max float + penalty): needs a proper
   game-theory check.
2. Replica owners need working capital to pay members before the client's tickets
   arrive. Alternative: the client pays members directly, with the owner giving the list
   (members sign `stage_ready`).
3. How big the capacity pool should be and how fast it should decay; accepting that it
   pays for idle GPUs.
4. Stablecoin on-ramps are KYC'd, which conflicts with anonymous clients (it doesn't
   affect providers).
5. Legal: taking a protocol fee and running buy-and-burn may count as operating a
   business in some countries. Needs a lawyer.
6. Whether a DAN token is worth having at all, rather than stablecoins plus stake in
   USDC. The token is justified only by stake, the capacity pool and governance.

## Sources
[Pocket token](https://pocket.network/pokt-token/) ·
[Pocket TLMs](https://pkg.go.dev/github.com/pokt-network/poktroll/x/tokenomics/token_logic_module) ·
[Helium HIP-149](https://github.com/helium/HIP/blob/main/0149-helium-utility-and-emissions-realignment.md) ·
[Helium HIP-10](https://github.com/helium/HIP/blob/main/0010-usage-based-data-transfer-rewards.md) ·
[Akash AEP-76](https://akash.network/roadmap/aep-76/) ·
[AEP-76 discussion](https://github.com/orgs/akash-network/discussions/1034) ·
[Livepeer PM spec](https://github.com/livepeer/wiki/blob/master/spec/streamflow/pm.md) ·
[Livepeer orchestrator docs](https://docs.livepeer.org/orchestrators/guides/get-started) ·
[Render RNP-001](https://github.com/rendernetwork/RNPs/blob/main/RNP-001.md) ·
[Golem payments](https://docs.golem.network/docs/golem/payments) ·
[Filecoin Pay](https://docs.filecoin.cloud/cookbooks/filecoin-pay/) ·
[Bittensor emissions](https://docs.learnbittensor.org/learn/emissions) ·
[Pocket PUP-11](https://forum.pokt.network/t/pup-11-wagmi-inflation/1369) ·
[FIP-0080 discussion](https://github.com/filecoin-project/FIPs/discussions/774).
Secondary: [Bittensor 2026 guide](https://www.tao.media/the-ultimate-guide-to-bittensor-2026/),
[Helium Q4 2025 figures](https://tokenomics.com/articles/helium-tokenomics-how-hnt-distributes-100-of-network-fees-to-operators).
Mango Markets and MakerDAO surplus auctions are cited from general knowledge, not
re-fetched here.
