# Instructions for agents

Read [`docs/PROJECT.md`](docs/PROJECT.md) first: what DAN is, why, how it works, the
current state, rules, and roadmap.

Short rules (details in PROJECT.md §3):
- Never run `git commit`; the owner commits. Suggest a message instead.
- Never add a co-author, "Co-Authored-By", "Generated with" or any AI attribution line to
  commit messages or pull requests. The owner is the only author.
- Keep replies short, clear and simple.
- Network latency is the first optimization priority: first token and time between
  useful tokens for one interactive request take precedence over aggregate throughput.
- Prefer llama.cpp implementations even when DAN has an equivalent. Keep custom
  inference code only for a necessary DAN capability or a demonstrated advantage;
  preserve the decentralized WAN integration. See PROJECT.md §3.
- Latency takes precedence over upstream reuse. Reject latency regressions; if an
  optimization's latency tradeoff remains unclear, ask the owner before adopting it.
- The owner handles UI testing. Do not repeatedly inspect, render, screenshot or
  launch the UI, or install UI dependencies just to verify it, unless explicitly
  asked. Keep necessary build/backend checks focused on the changed behavior.
- Inspect before changing; build and test after; keep `docs/PROJECT.md` current.
- Keep the design decentralized: no scheduling authority in infrastructure, clients plan
  their own routes, workers decide for themselves, clients never send download URLs.
- Payments, reputation, Sybil resistance, consensus, result verification and failover are
  deferred unless the owner asks. Continuous batching was authorized as opt-in on
  2026-10-06; keep it disabled by default.
- After changing a node's code, test on a real network when you can (the owner's PC plus a
  RunPod GPU pod has served as the second machine); §12 of the guide shows how.
