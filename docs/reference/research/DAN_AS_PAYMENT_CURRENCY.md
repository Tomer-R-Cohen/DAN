# Should DAN be the payment currency for compute?

*Research only, nothing implemented. 2026-09-26. Follows
`PAYMENT_ARCHITECTURE_PRESSURE_TEST.md`.*

**Idea tested:** providers price work in DAN. When DAN falls, compute gets cheaper, usage
rises and DAN demand rises. When DAN rises, compute gets expensive, providers lower their
DAN prices. The market sets the price, with no oracle.

**Verdict: the loop doesn't stabilize anything.** It breaks on its own logic (§1).

## 1. Why the loop breaks

1. **Providers' costs are in dollars** (electricity, hardware), so rational providers
   reprice in DAN to keep their dollar income. Sia's host software does exactly this: it
   "pins" prices to a fiat exchange rate so hosts are "insulated from fluctuations"
   ([Sia Central](https://medium.com/sia-central-blog/hosting-with-sia-host-manager-from-0-to-100-62e8eef1ae2a)).
   After repricing, compute costs the same in dollars whatever DAN does, so usage doesn't
   change and there is no feedback. The loop exists only while providers are slow to
   reprice, and during that time they're the ones losing money.
2. **Payment demand isn't holding demand.** Users buy DAN just before using it; providers
   sell just after, to pay their bills. Buying and selling roughly cancel out. Vitalik's
   analysis: market cap ≈ value transacted × holding time, and a payment-only token is
   "quite brittle" with "an unavoidable risk of collapsing"
   ([MoE tokens](https://vitalik.eth.limo/general/2017/10/17/moe.html)).
3. **Selling is pro-cyclical.** When DAN falls, providers must sell *more* DAN to cover
   the same dollar costs, which pushes the price down further. The stabilizer runs in
   reverse.
4. **Demand reacts too slowly.** Crypto prices can move 2–5× within weeks. Chat demand
   depends on model quality and availability and reacts over months. It can't keep up.
5. **Volatility is a hidden fee.** A user holding a DAN balance pays for any drop: a
   falling token raises the "de-facto fee" and drives users away, which speeds the fall
   (Vitalik, same post).
6. **Real case: usage up, token down.** Helium's usage grew ~4× while the HNT price
   "more than halved"
   ([HIP-149](https://github.com/helium/HIP/blob/main/0149-helium-utility-and-emissions-realignment.md)).
   Usage alone didn't hold the price up.

## 2. Accepting both DAN and USDC
Users pay in whatever is stable, and demand for DAN goes away. Akash saw this: paying in
USDC "inadvertently reduced AKT demand", even though USDC payers paid a 20% take rate
against 4% for AKT. Akash had to add burn-and-mint to restore it
([AEP-76](https://akash.network/roadmap/aep-76/),
[discussion](https://github.com/orgs/akash-network/discussions/147)). A DAN discount is a
subsidy, and users still buy DAN only just in time.

## 3. Other precedents
- **Livepeer** pays fees in ETH, not in LPT. The token is for stake only
  ([PM spec](https://github.com/livepeer/wiki/blob/master/spec/streamflow/pm.md)).
- **Helium, Render, Akash** all quote prices in dollars and use the token through burning
  and minting, not as the price unit.
- **Sia** hosts price in SC but fix it to fiat in software: the native-token design falls
  back to dollar pricing in practice.

## 4. Tracking compute value without an oracle
This works in **any** design, as long as the protocol never mints based on a price.
Providers each use a private exchange rate to set their own prices (as Sia does). A wrong
rate only hurts that provider, and no protocol funds are at risk. The "no oracle" goal
doesn't need DAN to be the payment currency.

## 5. Comparison
| | DAN only | USDC only | Both |
|---|---|---|---|
| User UX | Buy a thin, volatile token first | Simple | Users pick USDC |
| Provider income | Volatile; sells into a thin market | Stable | Mostly USDC |
| Token demand | Circular (buy, then sell again) | None, unless fee buy-and-burn | ≈ USDC case |
| Stability | Pro-cyclical | n/a | n/a |
| Oracle needed | No | No | No |

## 6. Simplest good design (unchanged)
Providers set **USDC prices** and clients pay in USDC (tickets). A small fee **buys and
burns DAN** through an auction. DAN is used for **stake**, verifier rewards and
protocol-created challenge rewards. Value capture comes from the burn and from stake that
is locked up, not from being a payment currency.
