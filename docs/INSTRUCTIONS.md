# Working Instructions

- Build the smallest working distributed primitive, understand it, document it,
  and only then add the next layer.
- Use C++23, raw POSIX TCP sockets, CMake, and the standard library.
- Preserve the current poll-based coordinator, persistent providers, and framed text protocol.
- Do not add AI, cryptocurrency, blockchain, marketplaces, distributed training,
  or scalability infrastructure until explicitly requested.
- Before changing code, inspect the repository and read this `docs/` directory.
- After a meaningful change, build, test, update the relevant documents, and
  record exact progress in `STATE.md`.
- Keep architecture documentation aligned with what actually works.
- Treat model weights and model choice as registry configuration, never as
  family-specific networking or scheduling logic.

`STATE.md` is the concise handoff for current progress. `ROADMAP.md` separates
completed, current, next, and future work.
