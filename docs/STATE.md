# Project State

## What Works

- CMake builds the C++23 `coordinator`, `provider`, `managed_provider`, and
  `distributed_model_experiment` executables and the `distributed_runtime` static library.
- The coordinator accepts multiple persistent providers while reading prompts.
- Length-prefixed framing reliably transfers complete messages over TCP.
- Every provider keeps its own llama.cpp child and GGUF model loaded.
- Prompts run concurrently across providers, with one active request per provider.
- Waiting prompts are held in a FIFO queue and dispatched when providers free up.
- Request IDs correlate and label responses that finish out of order.
- Disconnected providers are removed and are not scheduled again.
- Providers report ID, device, GPU, VRAM, model, and backend metadata.
- The coordinator stores and displays metadata for every live provider.
- Completed requests update per-provider elapsed time, request count, and average.
- The coordinator stores both normal providers and manually configured
  distributed-model groups as execution targets.
- `/group <id> <prompt>` dispatches through a group's llama.cpp RPC workers and
  records request-ID-correlated latency and running averages.
- `/model <model-name> <prompt>` automatically selects a compatible available
  provider or distributed group and round-robins for ties.
- An optional registry classifies test, main, and distributed models and stores
  family, path, runtime name, quantization, memory, context, requirements, and
  permitted execution modes.
- Provider-side inference failures return a short `ERROR` message.
- Coordinator port and provider host/port are configurable arguments.
- Provider Control Plane v1 loads one `dan-main` manifest with any number of
  required shards, assigns one shard per eligible managed provider, and tracks
  `ASSIGNED`, `DOWNLOADING`, `CACHED`, `LOADING`, `READY`, and `ERROR` reports.
- Managed providers send heartbeats. Timed-out providers remain visible as
  `OFFLINE`; their shard no longer counts toward replica readiness.
- `/providers` shows capabilities, sticky shard assignments, last-seen age, and
  aggregate `dan-main` replica readiness.
- A reconnecting provider keeps its assignment and can advertise an exact
  model/version/shard/hash cache match before returning to `READY`.
- Managed Worker Runtime v1 downloads local/file or HTTP(S) artifacts ahead of
  requests, verifies real SHA-256 and optional size, atomically publishes them
  under a configurable cache directory, and re-verifies inventory on restart.
- The provider owns one llama.cpp RPC worker process, waits for its TCP endpoint
  before reporting `READY`, detects unexpected exit, and supports idempotent
  `/load <id>` plus `/unload <id>` without deleting cached bytes.

## Most Recent Change

Managed Worker Runtime v1 now connects the control plane to real provider-side
actions. `managed_provider` consumes each arbitrary-N assignment, prepares and
verifies its artifact, starts an owned RPC worker with `fork`/`execvp`, checks the
configured private TCP endpoint, and reports real state. Cache identity is
model/version/shard/SHA-256; corrupt files are rejected and exact reconnects skip
`DOWNLOADING`. Existing persistent whole-model providers and manual llama.cpp RPC
groups remain unchanged.

This milestone still does not connect managed readiness to user inference. It
prepares N healthy RPC workers, but does not create a persistent distributed
client/runtime or route prompts through the managed replica.

The previously prepared rental is deferred behind Persistent Managed Distributed Serving.
The staged rental plan was an RPC disk-cache and persistent-runtime experiment.
About 13 minutes of repeated Qwen3 model preparation motivates testing reuse
before spending another session on a larger model. The instructions are
[Pod A](POD_A_NEXT_TEST_PROMPT.md), [Pod B](POD_B_NEXT_TEST_PROMPT.md), and the
[shared runbook](NEXT_GPU_EXPERIMENT.md). Cache reuse, retained-cache worker
restart, and persistent distributed serving remain unproven. The planned
llama-server phase tests the backend; it does not connect control-plane assignment
to persistent DAN groups. The [provider lifecycle](PROVIDER_LIFECYCLE.md) now
separates implemented v1 state tracking from the missing data/runtime path.

The two-GPU smoke test passed on 2026-09-05 across RunPod private networking:
Pod A RTX 3090 and Pod B RTX A4500 both participated in standalone and DAN
distributed-group inference using Qwen2.5-1.5B-Instruct Q4_K_M, tensor split
`6,5`, and 4,096 context. Aggregate-VRAM necessity remains unvalidated because
the model fits on either GPU. See [TWO_GPU_SMOKE_REPORT.md](TWO_GPU_SMOKE_REPORT.md).

A follow-up Qwen3-30B-A3B Q4_K_M two-GPU run also passed through both entry
points at 32,768 context. RPC0 mapped to the RTX 3090 and RPC1 to the RTX A4500;
both allocated roughly 10–12 GiB, showed substantial utilization, and returned
to baseline after shutdown. Standalone latency was 806,286 ms and DAN group
latency was 804,709.8 ms, dominated by roughly 13 minutes of TCP model
distribution. This proves participation with a serious model, but still does
not prove aggregate-VRAM necessity because the model may fit on one worker.
See [QWEN3_TWO_GPU_REPORT.md](QWEN3_TWO_GPU_REPORT.md).

Committed evidence: [dan-qwen3-30b-a3b-results.tar.gz](../dan-qwen3-30b-a3b-results.tar.gz),
including `results/qwen3-30b-a3b/report.md`, successful `dan-gpu-32768/`, and
the earlier failed `dan-gpu/` run. The benchmark used DAN `c5b3bfd` and llama.cpp
`95ef7fc16054e63b427a3ef00188e055ef7586d8`; commit `e2cbd2b` added the archive.
SmolLM remains infrastructure-test-only.

General bug review fixed buffered-stdin/poll stalls, EINTR handling in poll,
inherited provider sockets in distributed child jobs, provider SIGPIPE failure,
trailing slash/backslash runtime input commands, and permissive registry number/
boolean/role parsing. Core network framing and target-selection policy remain.
The new real CPU validation initially failed on request 8 with an empty answer.
Inspection found that the pinned interactive runtime carries its positive token
budget across turns. Persistent providers now default to end-of-turn generation
(`--n-predict -1`); the validation driver adds an external request timeout.

## Verification

Managed Worker Runtime v1: the CMake build and all ten CPU-only regression tests
pass. Integration coverage starts four managed providers with different VRAM,
real local artifacts and lightweight TCP workers. It verifies cold download,
SHA-256, `READY`, unexpected worker exit/`NOT_READY`, cached `/load` recovery,
duplicate-load PID stability, provider restart, and no second download. Separate
cases cover corrupt cache replacement, wrong hash, failed source, missing/early
worker executable, and malformed load messages. Existing control-plane coverage
still starts four fake GPU providers with different VRAM,
assigns four manifest shards, reaches `READY`, stops one provider's heartbeats,
observes `OFFLINE`/`NOT_READY`, then reconnects the same identity from its saved
cache inventory and returns to `READY` without a second `DOWNLOADING` report. It
also verifies that the real C++ provider reports a matching cached, loaded runtime
through `CACHED`, `LOADING`, and `READY`.

Next-test documentation checks: Bash blocks parse successfully, whitespace
checks passed, CMake build succeeded, and the then-four regression tests passed.
This validates preparation only; no new GPU/cache/server experiment was run.

The two-pod reports below are operator-supplied summaries. Their raw pod logs
and the reported Pod A evidence commit `fb0ce58` have not been imported into
this checkout. Do not describe the documentation commit as transfer of those
raw artifacts. The earlier A40 archive is present in the repository.

- GPU preparation: CMake build and strict `-Werror` syntax checks passed.
- The original four CPU-only regression tests passed, covering runtime-option forwarding,
  ten sequential framed responses, piped prompt bursts/EOF shutdown, invalid
  configuration rejection, and failure-evidence preservation.
- Real rented NVIDIA A40 CUDA validation passed with Qwen3-30B-A3B Q4_K_M
  at 32,768 context: ten persistent-provider responses, coordinator/provider
  exit codes 0, 3,844.7 ms average DAN latency, 5,500.4 ms runtime ready time,
  and 21,227 MiB observed peak VRAM.
- Direct 32,768-context inference reported 4,191.91 ms load time and
  164.87 evaluation tokens/s. This token rate is a direct-runtime measurement,
  not a DAN throughput measurement.
- The earlier 4,096-context persistent run completed eight responses, then
  stalled on request 9 as Qwen3 thinking traces filled the context; it was not
  an OOM. The 32,768-context rerun passed. Answers were reviewed in the report;
  this is a small acceptance workload, not a broad quality benchmark.
- The local VMware environment remains CPU-only. An L4-specific memory-fit
  test remains unverified; the separate two-pod CUDA smoke test is recorded below.
- Real two-pod CUDA RPC smoke test passed with RPC0=`10.0.23.9:50052` on an
  RTX 3090 and RPC1=`10.1.115.60:50052` on an RTX A4500. Qwen2.5-1.5B-Instruct
  Q4_K_M produced responses through both the standalone experiment and DAN
  distributed group. Pod A peak VRAM was 890 MiB at 5% utilization; Pod B
  peak VRAM was 1,093 MiB at 3% utilization. Standalone latency was 55,279 ms
  and group latency was 57,223 ms; load dominated both measurements.
- The generated Pod B hostname did not resolve; its numeric private IPv4
  address worked. Numeric private IPv4 endpoints are required operationally
  for this pinned RPC client in the observed RunPod setup.
- Latest real SmolLM CPU rerun after the token-budget fix completed ten
  responses; `results.json` confirmed both process exit codes were zero.
  Runtime ready time was 987.2 ms. Local temporary evidence is in
  `/tmp/dan-gpu-prep-cpu-results-eos/` (not included in the repository).
  This verifies persistent CPU inference, not GPU execution or answer quality.
- DAN configured and built successfully with CMake 3.28.3 and GCC 13.2.0.
- All four source files also compiled with `-Werror` before the CMake build.
- The framed subprocess path passed a stand-in executable test.
- Real CPU inference passed with llama.cpp commit `95ef7fc` and
  `SmolLM2-360M-Instruct-Q4_K_M.gguf`.
- A ten-prompt acceptance run returned ten responses without restarting the
  coordinator, provider, connection, or model process.
- `exit` sent `BYE`; both programs then exited successfully.
- A three-provider real-model run dispatched four prompts in order `1, 2, 3, 1`.
- Provider 2 was then terminated; the coordinator detected and removed it.
- Two more prompts went to providers `3, 1`; both remaining providers exited
  successfully after `BYE`.
- Two providers registered with distinct CPU and CUDA/GPU metadata in a live
  real-model test; the coordinator displayed all reported fields.
- Round-robin responses updated the correct provider with measured elapsed time,
  completed request count, and running average.
- In a three-provider real-model test, requests 1, 2, and 3 were all dispatched
  before any response completed, proving inference overlap.
- Requests 4, 5, and 6 queued while all providers were busy and were dispatched
  as providers became available.
- Responses completed out of submission order and every response printed its
  matching request ID and provider ID.
- `exit` drained all six requests before sending `BYE`; all providers exited
  successfully.
- Built llama.cpp commit `95ef7fc` with `GGML_RPC=ON` and ran two localhost CPU
  RPC workers at `127.0.0.1:50061` and `127.0.0.1:50062`.
- Both RPC workers participated in an equal `--tensor-split 1,1` run and one
  model-generated response completed successfully.
- A follow-up run explicitly selected only `RPC0,RPC1` as offload devices
  and completed successfully with activity on both workers. This selection
  does not eliminate llama.cpp's normal client CPU work or CPU buffer fallbacks.
- A fixed 256-token localhost RPC run took 23.55 seconds end-to-end and reported
  12.10 generated tokens/second. The comparable local run took 28.82 seconds
  and reported 9.56 generated tokens/second. Localhost TCP was not the observed
  bottleneck in this test.
- The main coordinator was started with one normal persistent provider and one
  `rpc-pair` group backed by two RPC workers.
- Request 1 ran on the normal provider while request 2 ran concurrently through
  the distributed group; both responses printed the correct target, request ID,
  latency, request count, and average.
- An unknown-model request was rejected clearly without disrupting later work.
- Two requests for `SmolLM2-360M-Instruct-Q4_K_M.gguf` were automatically sent
  to a normal provider and `rpc-auto` group without `/group`; both completed
  with correct request IDs and metrics.
- Loaded the three-role example registry and displayed all model definitions.
- `/model smollm2-test` resolved to the provider runtime name and completed a
  real inference request successfully.

## Known Limitations

- llama.cpp diagnostics are printed on the provider terminal
- Each individual provider remains limited to one active request
- The request queue is memory-only and has no configured size limit
- Conversation history is provider-local, so round-robin prompts do not share
  one conversation context
- Capability values are self-reported and are not verified against hardware
- Model compatibility uses registry IDs or exact runtime names; aliases do not exist
- Registry memory requirements are declared estimates, not live enforcement
- The example serious models are not downloaded or validated on this 8 GB-class
  CPU test environment
- Timing is coordinator-observed request latency and includes TCP transport
- DAN does not calculate tokens per second; capture llama.cpp evaluation
  diagnostics separately where available
- Parallel CPU providers can contend for cores; useful throughput scaling
  requires sufficient CPU/GPU resources
- llama.cpp RPC is proof-of-concept, fragile, and unsafe outside a trusted LAN
- Each distributed-group request currently starts a fresh inference process, so
  its latency includes startup and model distribution/loading
- Distributed runtime selects one RPC device per configured endpoint; expose
  exactly one CUDA device per worker. It fixes offload at 99 layers and
  generation at 256 tokens, uses non-conversation mode, and does not forward
  registry context. See SETUP.md for explicit runtime environment settings.
- The two-GPU smoke tests prove remote participation, mapping, allocation, and
  activity, but not aggregate GPU-memory necessity. DAN has no per-worker GPU
  telemetry; the reports rely on external `nvidia-smi` and RPC evidence.
- No request timeout, cancellation, streaming, authentication, or encryption
- Protocol text is not UTF-8 validated
- Model quality depends entirely on the selected GGUF model
- Control Plane v1 supports one managed model (`dan-main`), one replica, and one
  shard assignment per provider. Assignments are sticky; v1 does not rebalance.
- Cache eviction/resume and persistent distributed request serving are not
  implemented. HTTP(S) uses the installed `curl`; SHA-256 uses `sha256sum`.
  The example manifest contains placeholders and is not deployment metadata.
- Heartbeat state is coordinator-local and memory-only. Reconnect identity and
  all capability/cache claims are self-reported without authentication.

## Next Intended Task

Build Persistent Managed Distributed Serving: when the managed replica is ready,
create and retain one distributed inference runtime, route many sequential
`dan-main` requests through it, and prove there is no artifact re-download,
worker restart, or repeated full weight transfer per request.

Only after that integration should the Pod A/B hardware instructions be executed.
The next meaningful rental validates real cache verification, restart recovery,
heartbeat/offline behavior, persistent loaded weights, and repeated DAN requests
across N managed providers. Aggregate-VRAM necessity remains separate.
