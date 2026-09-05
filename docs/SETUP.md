# Build and Run

For the rented NVIDIA L4 session, follow [GPU_VALIDATION.md](GPU_VALIDATION.md).
It includes the checklist, pinned CUDA build, candidate download, direct test,
ten-request driver, and evidence collection. Actual GPU validation is pending.

## Build DAN

```bash
cmake -S . -B build
cmake --build build
```

## Prepare llama.cpp

Build the official llama.cpp `llama-completion` target outside this repository:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
git -C ~/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DBUILD_SHARED_LIBS=OFF
cmake --build ~/llama.cpp/build --config Release -j --target llama-completion
```

After building DAN, run the CPU-only deployment regressions with
`python3 -m unittest discover -s tests -v`. These use a stand-in runtime and do
not validate CUDA or model quality.

Download a compatible instruction-tuned GGUF model from a source whose license
you accept. Keep model weights outside the DAN repository.

## Run with One or More Providers

Start the coordinator first:

```bash
./build/coordinator
```

Start a provider with explicit runtime and model paths:

```bash
./build/provider /path/to/llama-completion /path/to/model.gguf
```

By default its reported ID is `<hostname>-<process-id>`, its model name is the
GGUF filename, and its device/backend are CPU. Override display metadata when
starting a different configuration:

```bash
./build/provider /path/to/llama-completion /path/to/model.gguf \
  192.0.2.10 9000 \
  --id gpu-node-1 \
  --device GPU \
  --gpu "NVIDIA RTX 4090" \
  --vram "24 GiB" \
  --model-name "My Model" \
  --backend CUDA
```

Available metadata flags are `--id`, `--device`, `--gpu`, `--vram`,
`--model-name`, and `--backend`. They are self-reported and do not change
llama.cpp runtime configuration; pass any actual runtime configuration through
the supplied llama.cpp build/environment or the explicit provider options:
`--gpu-layers N`, `--runtime-device CUDA0`, `--ctx-size N`, `--n-predict N`.
Persistent providers default to `--n-predict -1` (end-of-turn generation).
Positive budgets in the pinned interactive runtime span turns and can cause
empty responses; they are not reliable per-request caps. Manual runs have no
generation timeout; the GPU validation driver supplies an external timeout.
These options configure inference; `--backend CUDA` alone is only metadata.

Start additional providers with the same command in additional terminals. Each
provider loads and retains its own model process. The coordinator reports each
connection as `Provider 1`, `Provider 2`, and so on.

When `Prompt (or exit):` appears, type prompts normally. The coordinator logs
`Dispatching to provider N` and selects providers in round-robin order. With
three providers, the first three prompts are dispatched immediately and run at
the same time. Additional prompts wait in a FIFO queue until a provider becomes
available. Input does not wait for earlier inference to finish.

Responses include their request IDs because completion order can differ from
submission order. Each response is followed by that provider's latest elapsed
time, completed request count, and average elapsed time. Type `exit` to stop
input; DAN finishes queued and active requests, sends `BYE` to every live
provider, and shuts down cleanly.

Real throughput depends on hardware. Multiple CPU providers on a machine with
too few cores can contend and run more slowly even though their inference
processes overlap. Separate GPUs or adequately provisioned provider machines are
the intended way to obtain parallel throughput.

For different machines, make TCP port `9000` reachable and pass the coordinator
address to the provider:

```bash
./build/provider /path/to/llama-completion /path/to/model.gguf 192.0.2.10 9000
```

Additional machines can run the same provider command with the coordinator's
reachable address. The current protocol is unencrypted and unauthenticated, so
use only a trusted network at this stage.

## Integrated Distributed Model over llama.cpp RPC

This target type is configured manually in the coordinator and remains separate
from normal whole-model provider scheduling. llama.cpp describes its RPC backend
as proof-of-concept, fragile, and insecure. Never
expose an RPC port to the internet or an untrusted network.

Build llama.cpp with RPC and the appropriate accelerator backend on both worker
machines. For NVIDIA CUDA workers:

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-rpc \
  -DGGML_RPC=ON -DGGML_CUDA=ON
cmake --build /path/to/llama.cpp/build-rpc --config Release -j \
  --target ggml-rpc-server llama-completion
```

Start one RPC worker on each provider machine, binding only a trusted LAN
address. The server prints the exposed device and available memory:

```bash
# Provider A
/path/to/build-rpc/bin/ggml-rpc-server \
  --host 192.168.1.21 --port 50052 --device CUDA0

# Provider B
/path/to/build-rpc/bin/ggml-rpc-server \
  --host 192.168.1.22 --port 50052 --device CUDA0
```

On the coordinator machine holding the GGUF file, configure the group at
startup. The final argument is a tensor split such as `1,1`, or `auto`:

```bash
./build/coordinator 9000 \
  --group large-model-group \
  /path/to/build-rpc/bin/llama-completion \
  /path/to/large-model.gguf \
  192.168.1.21:50052,192.168.1.22:50052 \
  1,1
```

Repeat `--group` to configure more groups. The coordinator displays groups at
startup alongside normal providers. Route a prompt explicitly:

```console
Prompt (or exit): /group large-model-group Explain distributed inference.
```

Ordinary prompts continue to use normal DAN providers. Group prompts receive
the same request IDs and may run concurrently with normal provider requests.
The response identifies the group and prints its latency and running average.

The standalone `distributed_model_experiment` command remains available for
isolated diagnostics and uses the same shared runtime implementation.

For normal operation, specify the model and let DAN choose a target:

```console
Prompt (or exit): /model large-model.gguf Explain TCP briefly.
```

Selection applies model compatibility, availability, then round-robin. A
provider matches its reported `--model-name`; a group matches its configured
path or filename. If no target serves the model, DAN reports the request ID and
continues other work. `/group` remains a diagnostic override. Plain prompts are
backward compatible and use normal providers only.

## Use the Model Registry

Copy and edit the example rather than changing source code:

```bash
cp config/models.example.conf config/models.conf
./build/coordinator 9000 --models config/models.conf
```

Then request stable IDs:

```console
Prompt (or exit): /model smollm2-test Check the scheduler.
Prompt (or exit): /model dan-main Answer the real user question.
```

The example paths are placeholders and the serious models are not bundled.
Providers must load the configured weights and report the matching
`runtime_name`. A distributed group may reference a registry model path:

```bash
./build/coordinator 9000 --models config/models.conf \
  --group large-group /path/to/rpc-llama-completion @dan-large \
  192.168.1.21:50052,192.168.1.22:50052 auto
```

See `docs/MODELS.md` for the schema, role policy, and main-model promotion
checklist.

For the target acceptance test, record each RPC server's reported free memory
and select a GGUF whose required device allocation exceeds either value alone
but is below their combined capacity. Confirm that both worker logs receive RPC
activity and retain the complete experiment output.

Compare an identical fixed-token prompt locally, with one RPC endpoint, and
with both endpoints. Record end-to-end latency and llama.cpp's prompt/evaluation
tokens per second. Also record LAN link speed and utilization; if generation
speed falls substantially while the link is saturated or latency-sensitive,
network communication is the likely bottleneck.
