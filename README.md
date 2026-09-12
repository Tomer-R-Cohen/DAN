# DAN

DAN is building provider-owned distributed LLM inference: providers retain and
execute their assigned transformer stages while a metadata-only coordinator
routes activations without loading the model.

Supported paths:

- Runtime-formed provider-owned Qwen2 replicas over an ordered list of stages.
  This is the main path and uses a C++23 metadata-only coordinator and workers.
- Single providers, each keeping a complete GGUF model loaded.
- Manually configured distributed groups using llama.cpp RPC. This remains the
  legacy/reference fallback and currently starts a runtime for each request.
- One managed `dan-main` replica over arbitrary-N providers, with artifact cache,
  managed RPC workers, persistent serving, and automatic provider replacement.

Providers register model/hardware metadata; responses include request IDs and
latency measurements. Models are registry configuration, not networking logic:
SmolLM2 is infrastructure-test-only, `dan-main` is the replaceable serious-model
candidate, and `dan-large` represents distributed-model candidates.

## Architecture and algorithm

**Provider-owned execution.** The coordinator never loads a GGUF. It reads
only model metadata (layer count, hidden size, tensor byte sizes) and routes
FP32 activation frames between providers, each of which loads and keeps
resident only its own assigned contiguous range of transformer layers plus
its own KV cache. A provider physically cannot leak weights it never
downloaded outside its assigned range.

**Runtime replica formation.** Providers connect out to the coordinator
(`--coordinator`) and register GPU name + free VRAM; the coordinator
(`--provider-listen`) range-reads the target GGUF's metadata, computes exact
per-layer tensor bytes, and searches provider orderings for the smallest
count whose free VRAM (minus KV and a runtime safety margin) covers a
contiguous stage split. Each assigned provider then range-downloads only its
own layers via HTTP byte-range requests into a sparse-cache file — no
provider ever holds the complete model. See
[Generalized Replica Formation v1](docs/GENERALIZED_REPLICA_FORMATION_V1.md).

```text
tokens -> first stage (embedding) -> activation -> middle stage(s)
       -> activation -> last stage (head) -> sampled token
```

**Speculative decoding.** A small draft model proposes several tokens ahead;
the full distributed replica verifies all of them in one batched forward
pass and greedily accepts the longest matching prefix, rejecting and
resuming from the first mismatch. This amortizes the network round trip
across multiple tokens instead of paying it once per token.
This is opt-in via `--draft-model`; the standard Windows service does not ship a
draft model or enable it. Live metrics expose the configured state and acceptance.

**Pipelining.** Rather than waiting for one full draft-verify round trip
before starting the next, the coordinator keeps several speculative chunks
in flight at once (sender/relay/receiver threads), truncating a stage's KV
to a lower position on receipt of a correction frame instead of an explicit
rollback round-trip. See
[Pipelined Speculative Decoding v1](docs/PIPELINED_SPECULATION_V1.md) for
the full design, including why this doesn't change output versus the
existing K-chunk speculative path (floating-point non-associativity in
batched verification, not a pipelining bug, already exists at K>1 without
any of this).

**Ring topology + chunked prefill.** Manual mode without ring endpoints relays
activations through the coordinator (hub-and-spoke: 2 network legs per hop).
Ring mode forwards stage-to-stage directly (`--next`/`--ring-listen`), with
only the tail stage returning to the coordinator (`--ring-return`) — D+1
legs instead of 2D, removing the coordinator from the per-token hot path.
Chunked prefill (`--prefill-chunk`) splits a long prompt into pieces
forwarded through the ring as soon as each is ready, instead of waiting for
the whole prompt before the next stage can start. Automatic formation accepts
provider-advertised ring endpoints and assigns every selected stage's next hop;
the packaged peer-network mode carries these links over authenticated libp2p;
fixed `--provider`/`--model` addressing remains available. See
[the physical multi-GPU test doc](docs/PIPELINED_RING_PHYSICAL_TEST.md) for
why and what's proven versus still untested.

## Build and test

The main coordinator/runtime uses C++23 and CMake 3.20+. Python is still used by
legacy integration tests, but normal provider-owned inference does not require
it. Ubuntu 24.04 is the documented Linux baseline.

```bash
cmake -S . -B build
cmake --build build -j 2
python3 -m unittest discover -s tests -v
```

The provider also builds natively with Visual Studio 2022/MSVC on Windows
10/11 x64:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
powershell -ExecutionPolicy Bypass -File .\scripts\release_windows_provider.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\release_windows_coordinator.ps1
```

The release commands create separate self-contained provider and coordinator
archives under `build`. Extract the relevant ZIP and double-click its DAN executable.
The coordinator archive also includes `dan-api-gateway.exe` for authenticated
DAN chat completions, including SSE streaming.

llama.cpp and model weights are external dependencies. See
[the provider-owned setup guide](docs/PROVIDER_OWNED_SETUP.md) for the current
path's runtime installation and provider/coordinator commands, or
[the legacy path](docs/LEGACY_PATH.md) for the managed/whole-model/RPC path's.

## Status and documentation

The provider-owned v2 path now serves concurrent stateless and persistent-session
clients through a bounded fair queue while loading each stage once. It reuses
llama.cpp sequence IDs for isolated provider-owned KV; its C++ coordinator owns
no GGUF and routes only validated control, token, and activation frames. The managed legacy
path downloads and verifies assigned artifacts, owns RPC workers,
keeps one distributed `llama-server` alive across requests, and automatically
replaces a missing provider with an eligible spare.

The [provider lifecycle](docs/LEGACY_PATH.md#provider-lifecycle) documents
current boundaries; repeated full-model transfer per user request is not the
target design.

Real single-GPU CUDA validation passed on NVIDIA A40 with Qwen3-30B-A3B Q4_K_M
at 32,768 context: 10/10 persistent-provider requests, 3.845 s average DAN
latency, 21,227 MiB observed peak VRAM, and clean shutdown. Evidence is committed
in [the benchmark archive](dan-qwen3-30b-a3b-results.tar.gz).

The legacy RPC two-GPU smoke test passed across an RTX 3090 and RTX A4500 using
Qwen2.5-1.5B-Instruct Q4_K_M. Both RPC devices showed allocation and GPU
activity through the standalone experiment and a DAN distributed group.
Those legacy runs did not prove aggregate-VRAM necessity because each model may
fit on one worker. The provider-owned engine later proved it with Qwen2.5 32B
split across RTX 2070 and RTX A5000 providers. See the [project status and test results](docs/PROJECT_STATUS.md) and
[distributed setup and acceptance requirements](docs/LEGACY_PATH.md#integrated-distributed-model-over-llamacpp-rpc).

- [Current status, results, and roadmap](docs/PROJECT_STATUS.md)
- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/DECISIONS.md)
- [Legacy path: managed dan-main, whole-model providers, RPC groups, wire protocol, model registry, GPU validation](docs/LEGACY_PATH.md)
- [Provider-owned two-stage execution prototype](docs/PROVIDER_OWNED_EXECUTION_V0.md)
- [Persistent provider-owned runtime v1](docs/PROVIDER_OWNED_RUNTIME_V1.md)
- [Concurrent multi-session runtime v2](docs/PROVIDER_OWNED_RUNTIME_V2.md)
- [Pipelined speculative decoding v1](docs/PIPELINED_SPECULATION_V1.md)
- [Pipelined speculative decoding: first real-WAN result](docs/PIPELINED_WAN_RESULTS.md)
- [Pipelined speculative decoding + ring: physical multi-GPU test](docs/PIPELINED_RING_PHYSICAL_TEST.md)
- [Range-backed provider model storage](docs/RANGE_BACKED_PROVIDER_STORAGE.md)
- [Generalized replica formation v1](docs/GENERALIZED_REPLICA_FORMATION_V1.md)
- [Aggregate-VRAM physical test: 14B (superseded) to 32B (proven)](docs/AGGREGATE_VRAM_TEST.md)
- [Windows contributor release v1.0.1](docs/CONTRIBUTOR_RELEASE_V1.0.1.md)
- [Windows coordinator release v1.0.1](docs/COORDINATOR_RELEASE_V1.0.1.md)
- [Windows + Linux CUDA physical test](docs/PROVIDER_OWNED_V2_WINDOWS_LINUX_TEST.md)
- [Provider-owned build and run guide](docs/PROVIDER_OWNED_SETUP.md)

The provider-owned libp2p path authenticates stable PeerIDs and encrypts control
and activation traffic, but it does not independently verify returned
computation. Keep raw coordinator/worker ports private and put the keyed API
behind TLS. Legacy llama.cpp RPC is unauthenticated and must not be public;
blocking peer reads and subprocess marker framing remain known limitations.

## License

DAN is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Bundled dependencies and separately downloaded model weights retain their own
licenses.
