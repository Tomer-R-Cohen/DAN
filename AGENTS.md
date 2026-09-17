# Instructions for agents

Read [`docs/PROJECT.md`](docs/PROJECT.md) first: what DAN is, why, how it works, the
current state, rules, and roadmap.

Short rules (details in PROJECT.md §3):
- Never run `git commit`; the owner commits. Suggest a message instead.
- Keep replies short, clear and simple.
- Inspect before changing; build and test after; keep `docs/PROJECT.md` current.
- Keep the design decentralized: no scheduling authority in infrastructure, clients plan
  their own routes, workers decide for themselves, clients never send download URLs.
- Payments, reputation, Sybil resistance, consensus, verification, failover and
  latency-aware placement are deferred unless the owner asks.
