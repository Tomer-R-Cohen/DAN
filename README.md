# DAN

DAN is an experimental inference coordinator for persistent llama.cpp providers.
It routes model-specific requests to compatible execution targets, queues work,
and runs requests concurrently across providers, one request per provider.

Supported paths:

- Single providers, each keeping a complete GGUF model loaded.
- Manually configured distributed groups using llama.cpp RPC. This path is
  experimental and currently starts a runtime for each request.
- One managed `dan-main` replica over arbitrary-N providers, with artifact cache,
  managed RPC workers, persistent serving, and automatic provider replacement.

Providers register model/hardware metadata; responses include request IDs and
latency measurements. Models are registry configuration, not networking logic:
SmolLM2 is infrastructure-test-only, `dan-main` is the replaceable serious-model
candidate, and `dan-large` represents distributed-model candidates.

## Build and test

Requires Linux, CMake 3.20+, a C++23 compiler and Python 3 for regression tests.
Ubuntu 24.04 is the documented deployment baseline.

```bash
cmake -S . -B build
cmake --build build -j 2
python3 -m unittest discover -s tests -v
```

llama.cpp and model weights are external dependencies. See [setup](docs/SETUP.md)
for runtime installation and provider/coordinator commands.

## Status and documentation

The managed path now downloads and verifies assigned artifacts, owns RPC workers,
keeps one distributed `llama-server` alive across requests, and automatically
replaces a missing provider with an eligible spare. Gamer Provider Testnet v1 adds
Linux NVIDIA detection, persistent identity, VRAM headroom, private-network
validation, and reconnecting one-command provider startup. See the
[friends testnet guide](docs/FRIENDS_TESTNET.md).

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
memory necessity. See the [smoke-test report](docs/TWO_GPU_SMOKE_REPORT.md),
[Qwen3 report](docs/QWEN3_TWO_GPU_REPORT.md), and
[distributed setup and acceptance requirements](docs/SETUP.md#integrated-distributed-model-over-llamacpp-rpc).

- [Current state and known limitations](docs/STATE.md)
- [Friends testnet setup](docs/FRIENDS_TESTNET.md)
- [GPU deployment guide and checklist](docs/GPU_VALIDATION.md)
- [GPU results template](docs/GPU_RESULTS_TEMPLATE.md)
- [Architecture](docs/ARCHITECTURE.md), [protocol](docs/PROTOCOL.md), and [decisions](docs/DECISIONS.md)
- [Model registry and strategy](docs/MODELS.md)
- [Roadmap](docs/ROADMAP.md)

Use only in a trusted environment. DAN has no authentication or encryption;
blocking peer reads and subprocess marker framing remain known limitations.
Do not expose DAN or llama.cpp RPC ports publicly.

## License

A project license has not yet been selected. No open-source license grant is
provided by this repository. llama.cpp and model weights have their own licenses.
