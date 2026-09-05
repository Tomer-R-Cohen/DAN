# DAN

DAN is an experimental inference coordinator for persistent llama.cpp providers.
It routes model-specific requests to compatible execution targets, queues work,
and runs requests concurrently across providers, one request per provider.

Supported paths:

- Single providers, each keeping a complete GGUF model loaded.
- Manually configured distributed groups using llama.cpp RPC. This path is
  experimental and currently starts a runtime for each request.

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

The latest persistent SmolLM CPU smoke run completed ten prompts and clean
shutdown. Real rented-GPU validation has **not been completed**. The CUDA
runbook prepares a single NVIDIA L4-class provider test; it is not evidence of
GPU memory fit, performance, or successful multi-GPU execution.

- [Current state and known limitations](docs/STATE.md)
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
