# Project State

## What Works

- CMake builds the C++23 `coordinator`, `provider`, and
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

## Most Recent Change

Documentation now reflects the completed 2026-09-05 A40 benchmark and the
two-GPU readiness audit. The standalone RPC experiment and DAN distributed
groups are ready for a controlled remote two-worker test with configuration;
no source changes are required for the intended test. Actual remote two-GPU
execution and aggregate-VRAM fit have not yet been validated. See
[SETUP.md](SETUP.md#integrated-distributed-model-over-llamacpp-rpc).

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

- GPU preparation: CMake build and strict `-Werror` syntax checks passed.
- Four CPU-only regression tests passed, covering runtime-option forwarding,
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
  test and real remote two-GPU inference remain unverified.
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
- Neither the local CPU RPC results nor the A40 benchmark proves aggregate
  GPU-memory fit across two machines. DAN has no per-worker GPU telemetry.
- No request timeout, cancellation, streaming, authentication, or encryption
- Protocol text is not UTF-8 validated
- Model quality depends entirely on the selected GGUF model

## Next Intended Task

Run one inference across two separately provisioned CUDA RPC workers, first
with the standalone experiment and then through a DAN distributed group.
Verify device mapping, layer placement, memory, and activity on both nodes.
Then select a model/context exceeding either worker's available VRAM but fitting
across both. Preserve single-worker allocation failures and two-worker success
under identical settings, with no unintended CPU layer placement. Qwen3 already
fits on one A40, so running it on two A40s proves participation only. Retrieve
all evidence before releasing rental storage.
