# DAN launch readiness

This is the release gate for the provider-owned service. A checked item needs a
repeatable test or physical receipt; a feature existing in source is not enough.

## Private beta gate

| Gate | State | Evidence or remaining work |
|---|---|---|
| Provider-owned split inference | Done | Qwen2.5 32B ran across RTX 2070 + RTX A5000; local automatic two-stage CUDA ring also passed. |
| Persistent workers and range-backed weights | Done | Workers reuse loaded stages and verified sparse range caches across coordinator reconnects. The pinned default artifact returned its declared 491,400,032-byte length, honored a one-byte range request, and matched the cached full-file SHA-256. |
| Automatic formation and recovery | Done | Idle failure detection, replacement formation, and inference after recovery passed on CUDA. |
| Bounded concurrent API | Done | Fair queue, backpressure, timeouts, cancellation, buffered replies, and token SSE pass acceptance. |
| API readiness and horizontal routing | Done | Health checks replica state; repeated coordinator endpoints round-robin independent replicas and fail over from unreachable, reforming, failed, or saturated replicas before output begins. The runbook keeps coordinator binary ports private and scales independent gateway/replica groups behind a standard TLS load balancer. |
| One-command coordinator service | Done | Extracted Windows coordinator and contributor packages completed encrypted onboarding, CUDA generation, SSE, provider loss, cached rejoin, and post-reformation generation. The launcher holds a native Windows system-required execution state while running. |
| Encrypted automatic activation ring | Done | The extracted libp2p packages formed a direct authenticated provider-to-provider ring with a checked tail return, balanced 12/12 stages, real CUDA generation, and recovery after provider loss. Tailscale mode wires the same ring explicitly. |
| NAT relay operation | Done | The sidecar has an optional bounded circuit-v2 relay-service mode. Its test proves an allowlisted reservation carries an authenticated circuit stream and an unlisted reservation is rejected. Provider packages now reserve and advertise relay addresses for provider-to-provider ring fallback; Windows/Linux cross-build and real provider preflight pass. Cross-NAT physical acceptance remains separate. |
| CUDA graph and speculative acceleration | Partial | CUDA graph reuse is active in provider workers. An opt-in `--draft-model --pipeline-depth 4` run completed buffered and SSE requests through the direct encrypted ring (8 rounds, 27/32 accepted draft tokens); the standard package does not ship a draft model or enable speculation by default. |
| Stranger-machine acceptance | Open | Run the packaged archives on clean Windows hosts behind different NATs and attach the receipt required by [P2P transport](P2P_TRANSPORT.md#clean-machine-acceptance). |
| Production soak and churn | Open | The first long run reached 3,300 requests, then correctly failed when Windows suspended the host. The checker now holds a native Windows system-required execution state and requires an exact expected outage count. A fresh 24-hour run started 2026-09-12 22:41 +03:00 from 3,411 completed / 3 failed / 3 reformations; its deliberate provider churn recovered with exactly +1 failure and +1 reformation and zero resident-session/KV residue. |
| Distribution license and notices | Done | DAN is Apache-2.0 licensed and both packages include its license/notice. Builders also include pinned llama.cpp/sidecar notices, the exact pinned Qwen2.5 Apache-2.0 license, the installed CUDA license, and a tested 109-file Go dependency bundle (including reciprocal source). This machine has a complete, non-preview Visual Studio Community 2026 installation; [Microsoft's Community terms](https://visualstudio.microsoft.com/vs/community/) permit individual developers to build free or paid apps, and its [current redistribution list](https://learn.microsoft.com/visualstudio/releases/2026/redistribution) permits validly licensed users to redistribute files under `VC\Redist` unmodified. Organization eligibility remains the release builder's responsibility. |
| Reachable dependency vulnerabilities | Done | `govulncheck` v1.8.0 found three reachable Go advisories; DTLS, WebTransport, and QUIC were upgraded to fixed versions and the repeat scan reported zero reachable vulnerabilities. Package creation now reruns this gate. |
| Operations | Done | Authenticated per-replica Prometheus metrics, packaged alert rules, backup, upgrade, rollback, and incident actions are documented. |

## Shard parity and c0mpute integration

These are not private-beta blockers. Shard is the inference engine; worker-account
policy and payments come from its surrounding c0mpute platform and are listed
separately instead of being misrepresented as engine features.

| Capability | DAN today | Gap to Shard |
|---|---|---|
| Execution evidence | Authenticated peer identity or trusted Tailscale; capacity and completed work are self-reported | Signed per-stage activation-chain receipts with complete layer coverage, plus randomized comparison against a trusted copy of the assigned block |
| Weight distribution | Verified HTTPS range fetch and local reuse | Content-addressed peer seeding and mirror-independent recovery |
| Failure semantics | Ring reforms and the next request succeeds | Preserve committed tokens and resume the interrupted request on a warm spare |
| Per-replica throughput | Fair bounded queue; one request executes on a replica at a time; opt-in pipelined speculative chunks | Continuous/batched verification across concurrent requests and production-tuned speculation |
| Privacy | Providers see their boundary activations | Trusted boundary placement and sensitive-job routing policy |
| Model/runtime breadth | Dense Qwen2 GGUF and greedy decoding | Multiple tuned model engines, tool semantics, lossless sampling, and proven long context |
| Swarm management | One replica per coordinator; a stateless gateway balances independent replica groups | One control plane managing multiple swarms with topology-aware placement and live rebalancing |

c0mpute-only integration still absent from DAN: worker-account admission,
reputation/ejection policy, job accounting, pricing, refunds, and payouts. Those
are required for a paid permissionless network, but not for honest Shard engine parity.

## Naming and data boundary

The public process is `dan-api-gateway`. It implements DAN's local JSON/SSE API
and never contacts an external inference vendor. Prompts and generated tokens
flow only through the configured DAN gateway, coordinator, and providers.
