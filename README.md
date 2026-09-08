# DAN

DAN is building provider-owned distributed LLM inference: providers retain and
execute their assigned transformer stages while a metadata-only coordinator
routes activations without loading the model.

Supported paths:

- Persistent provider-owned execution over two fixed Qwen2 stages. This is the
  main development direction and uses a C++23 coordinator and workers.
- Single providers, each keeping a complete GGUF model loaded.
- Manually configured distributed groups using llama.cpp RPC. This remains the
  legacy/reference fallback and currently starts a runtime for each request.
- One managed `dan-main` replica over arbitrary-N providers, with artifact cache,
  managed RPC workers, persistent serving, and automatic provider replacement.

Providers register model/hardware metadata; responses include request IDs and
latency measurements. Models are registry configuration, not networking logic:
SmolLM2 is infrastructure-test-only, `dan-main` is the replaceable serious-model
candidate, and `dan-large` represents distributed-model candidates.

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
powershell -ExecutionPolicy Bypass -File .\scripts\release_windows_provider.ps1 `
  -BuildDirectory .\build\Release
```

The last command creates the self-contained package at
`build\DAN-Provider-Windows-x64.zip`. Extract it and double-click
`dan-provider.exe`.

llama.cpp and model weights are external dependencies. See [setup](docs/SETUP.md)
for runtime installation and provider/coordinator commands.

## Status and documentation

The provider-owned v2 path now serves concurrent stateless and persistent-session
clients through a bounded fair queue while loading each stage once. It reuses
llama.cpp sequence IDs for isolated provider-owned KV; its C++ coordinator owns
no GGUF and routes only validated control, token, and activation frames. The managed legacy
path downloads and verifies assigned artifacts, owns RPC workers,
keeps one distributed `llama-server` alive across requests, and automatically
replaces a missing provider with an eligible spare. Gamer Provider Testnet v1 adds
Linux NVIDIA detection, persistent identity, VRAM headroom, private-network
validation, and reconnecting one-command provider startup. See the Linux
[friends testnet guide](docs/FRIENDS_TESTNET.md) or the Windows
[friends testnet guide](docs/FRIENDS_TESTNET_WINDOWS.md).

The next [Pod A instructions](docs/POD_A_NEXT_TEST_PROMPT.md) and
[Pod B instructions](docs/POD_B_NEXT_TEST_PROMPT.md) describe the gated real CUDA
validation. The [provider lifecycle](docs/PROVIDER_LIFECYCLE.md) documents current
boundaries; repeated full-model transfer per user request is not the target design.

Real single-GPU CUDA validation passed on NVIDIA A40 with Qwen3-30B-A3B Q4_K_M
at 32,768 context: 10/10 persistent-provider requests, 3.845 s average DAN
latency, 21,227 MiB observed peak VRAM, and clean shutdown. Evidence is committed
in [the benchmark archive](dan-qwen3-30b-a3b-results.tar.gz).

The two-GPU smoke test also passed across an RTX 3090 and RTX A4500 using
Qwen2.5-1.5B-Instruct Q4_K_M. Both RPC devices showed allocation and GPU
activity through the standalone experiment and a DAN distributed group.
Aggregate-VRAM necessity remains unverified because this small model fits on
either GPU. A subsequent Qwen3 two-GPU run also passed with both GPUs active,
but Qwen3 may fit on one worker and therefore still does not prove aggregate
memory necessity. See the [project status and test results](docs/PROJECT_STATUS.md) and
[distributed setup and acceptance requirements](docs/SETUP.md#integrated-distributed-model-over-llamacpp-rpc).

- [Current status, results, and roadmap](docs/PROJECT_STATUS.md)
- [Friends testnet setup](docs/FRIENDS_TESTNET.md)
- [GPU deployment guide and checklist](docs/GPU_VALIDATION.md)
- [GPU results template](docs/GPU_RESULTS_TEMPLATE.md)
- [Architecture](docs/ARCHITECTURE.md), [protocol](docs/PROTOCOL.md), and [decisions](docs/DECISIONS.md)
- [Model registry and strategy](docs/MODELS.md)
- [Provider-owned two-stage execution prototype](docs/PROVIDER_OWNED_EXECUTION_V0.md)
- [Persistent provider-owned runtime v1](docs/PROVIDER_OWNED_RUNTIME_V1.md)
- [Concurrent multi-session runtime v2](docs/PROVIDER_OWNED_RUNTIME_V2.md)
- [Range-backed provider model storage](docs/RANGE_BACKED_PROVIDER_STORAGE.md)
- [Generalized replica formation v1](docs/GENERALIZED_REPLICA_FORMATION_V1.md)
- [Aggregate-VRAM 14B physical test](docs/AGGREGATE_VRAM_14B_TEST.md)
- [Aggregate-VRAM 32B RTX A5000 test](docs/AGGREGATE_VRAM_32B_A5000_TEST.md)
- [Windows + Linux CUDA physical test](docs/PROVIDER_OWNED_V2_WINDOWS_LINUX_TEST.md)
- [Provider-owned build and run guide](docs/PROVIDER_OWNED_SETUP.md)

Use only in a trusted environment. DAN has no authentication or encryption;
blocking peer reads and subprocess marker framing remain known limitations.
Do not expose DAN or llama.cpp RPC ports publicly.

## License

A project license has not yet been selected. No open-source license grant is
provided by this repository. llama.cpp and model weights have their own licenses.
