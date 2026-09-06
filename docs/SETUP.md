# Build and Run

Single-GPU validation passed on NVIDIA A40 with Qwen3-30B-A3B Q4_K_M at
32,768 context and ten persistent-provider responses. See [STATE.md](STATE.md)
for measurements and evidence. [GPU_VALIDATION.md](GPU_VALIDATION.md) retains
the original L4/Qwen2.5 example and the single-provider validation procedure.
The two-pod CUDA RPC smoke test has passed with Qwen2.5-1.5B-Instruct Q4_K_M:
both RPC workers participated in standalone and DAN distributed-group inference.
Qwen3-30B-A3B also passed through both paths at 32,768 context, with roughly
10–12 GiB allocated and nonzero utilization on each GPU. Its long startup was
dominated by TCP model distribution.
The next hardware session is deferred until the Provider Control Plane v1
managed-worker adapter exists. It will then measure RPC disk-cache reuse, worker
restart with retained cache, and repeated requests through a persistent
RPC-backed runtime under DAN control.
Start Codex on each fresh clone and tell it to read
[POD_A_NEXT_TEST_PROMPT.md](POD_A_NEXT_TEST_PROMPT.md) or
[POD_B_NEXT_TEST_PROMPT.md](POD_B_NEXT_TEST_PROMPT.md); each points to the complete
[shared runbook](NEXT_GPU_EXPERIMENT.md). Aggregate-VRAM necessity remains pending
after these reuse experiments. Existing instructions below remain reference
commands for the process-per-request implementation.

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

## Local Provider Control Plane v1

Start the coordinator with the example four-shard `dan-main` manifest:

```bash
./build/coordinator 9000 \
  --managed-model config/dan-main.example.manifest
```

In four other terminals, vary ID/GPU/VRAM/cache path:

```bash
python3 scripts/fake_provider.py --id node-a --port 9000 \
  --gpu RTX3090 --vram-mib 24576 --cache-file /tmp/dan-node-a.cache
python3 scripts/fake_provider.py --id node-b --port 9000 \
  --gpu A4500 --vram-mib 20480 --cache-file /tmp/dan-node-b.cache
python3 scripts/fake_provider.py --id node-c --port 9000 \
  --gpu Fake16G --vram-mib 16384 --cache-file /tmp/dan-node-c.cache
python3 scripts/fake_provider.py --id node-d --port 9000 \
  --gpu Fake12G --vram-mib 12288 --cache-file /tmp/dan-node-d.cache
```

Enter `/providers` at the coordinator. Stop one process long enough to exceed
the default ten-second heartbeat timeout, then restart the same command. Its
cache file causes an exact cached identity to be advertised; the sticky shard
returns through `CACHED`/`LOADING` to `READY` without `DOWNLOADING`.

The manifest format is `model|dan-main|<version>` followed by any number of
`shard|<id>|<size-bytes>|<content-hash>|<source>|<minimum-vram-mib>` lines.
The committed manifest is simulation-only placeholder metadata. Real providers
can opt into heartbeats/assignments with `--control-plane --vram-mib N` and may
repeat `--cached-shard 'dan-main|version|shard|hash'`. V1 trusts this inventory;
it does not download or verify files, and the managed replica does not yet serve
inference. `--heartbeat-timeout <seconds>` overrides the coordinator default for
testing.

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
machines. Use revision `95ef7fc16054e63b427a3ef00188e055ef7586d8` on workers
and client. For NVIDIA CUDA workers:

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-rpc \
  -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=OFF
cmake --build /path/to/llama.cpp/build-rpc --config Release -j \
  --target ggml-rpc-server llama-completion
```

Start one RPC worker on each provider machine, binding only a trusted LAN
address. The server prints the exposed device and available memory. Replace
the example addresses with reachable private IPv4 addresses assigned to each
worker. The pinned server defaults to loopback and requires numeric IPv4 for
binding; clients support IPv4 hostnames, but IPv6 endpoints are unsupported.
Allow client access to TCP port 50052 on both workers through private networking
and firewall rules. Verify routing before provisioning additional hardware.

Expose exactly one CUDA device per endpoint: DAN generates `RPC0,RPC1` from
endpoint count, whereas llama.cpp numbers every exposed device. Multiple devices
on worker A could otherwise cause both selected devices to belong to A.

```bash
# Provider A
/path/to/build-rpc/bin/ggml-rpc-server \
  --host 192.168.1.21 --port 50052 --device CUDA0

# Provider B
/path/to/build-rpc/bin/ggml-rpc-server \
  --host 192.168.1.22 --port 50052 --device CUDA0
```

On the coordinator machine holding the GGUF file, configure the group at
startup. The final argument is a tensor split such as `1,1`, or `auto`.

The client needs an RPC-enabled `llama-completion`; CUDA on the client is
optional (`-DGGML_RPC=ON -DGGML_CUDA=OFF` is sufficient for a CPU client).
Supply absolute runtime/model paths and all GGUF shards locally on the client.
Workers do not require their own GGUF copies. Reserve sufficient client RAM
and disk for model loading. Runtime invocation uses `execvp`, not shell parsing.

For a controlled 32,768-context test, set the inherited llama.cpp environment
on the client before starting either DAN entry point:

```bash
export LLAMA_ARG_CTX_SIZE=32768
export LLAMA_ARG_SPLIT_MODE=layer
export LLAMA_ARG_FIT=off
export GGML_RPC_NO_RDMA=1
```

Set `GGML_RPC_NO_RDMA=1` on the workers too when measuring plain TCP. Context
and fit settings are supported by the pinned llama.cpp; DAN does not forward
registry context. Choose a context that fits the intended deployment and keep
it identical in comparison runs. Use two distinct endpoints without spaces
and a positive finite split such as `1,1`; unequal GPUs may need another ratio.

Before inference, verify device mapping from the client:

```bash
/path/to/build-rpc/bin/llama-completion \
  --rpc 192.168.1.21:50052,192.168.1.22:50052 --list-devices
```

Require `RPC0` on worker A and `RPC1` on worker B. Then start the group:

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

```bash
set -o pipefail
printf '%s\n' 'Explain TCP reliability in two sentences.' |
  timeout --kill-after=30s 30m ./build/distributed_model_experiment \
  /path/to/build-rpc/bin/llama-completion /path/to/large-model.gguf rpc-pair \
  192.168.1.21:50052,192.168.1.22:50052 --tensor-split 1,1 \
  2>&1 | tee standalone-rpc.log
```

Use an external timeout for coordinator test runs too. DAN has no internal
distributed request timeout; `exit` drains work and may wait indefinitely on a
stalled runtime. Keep RPC servers alive throughout the client runs. Inspect
response IDs and error logs, not just coordinator exit status. Avoid concurrent
groups/providers using the same GPUs: DAN does not reserve shared worker memory.

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

The two-GPU smoke test passed without source changes. Its evidence is summarized
in [TWO_GPU_SMOKE_REPORT.md](TWO_GPU_SMOKE_REPORT.md). The following acceptance
requirements remain for aggregate-VRAM validation:

- First prove participation with a supported model through the standalone
  experiment and DAN group. Qwen3 fits on one A40; two A40s running it do not
  establish aggregate-memory necessity.
- For aggregate VRAM, choose a model with at most 99 offloadable layers,
  including its output layer. The shared runtime hard-codes 99 GPU layers,
  a 256-token cap, and non-conversation mode. It is not the persistent chat path.
- llama.cpp's layer split distributes weights and KV across selected RPC
  devices. Client CPU work and CPU buffer fallbacks still exist. Inspect
  placement/offload logs for unintended host-resident layers; endpoint labels
  and successful output alone cannot prove full intended offload.
- Run identical model/context settings against A alone and B alone using
  `llama-completion --rpc <one-endpoint> --device RPC0 --n-gpu-layers 99`
  with the same prompt, generation cap and fit settings. Both DAN entry points
  require at least two endpoint entries, so use direct llama.cpp for controls.
  Preserve attributable GPU allocation failures and then two-worker success.
- Capture server startup/device logs, client placement logs, and timestamped
  GPU samples on both nodes. In a separate terminal on each worker, run:

```bash
nvidia-smi -i 0 \
  --query-gpu=timestamp,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv --loop-ms=200 > gpu-samples.csv
```

Stop sampling after the test and preserve each node's file separately. Require
VRAM increases and activity on both GPUs correlated with the same inference;
layer splitting need not show simultaneous utilization in every sample.
DAN has no built-in per-GPU telemetry. Record load time and generation token
rates from llama.cpp diagnostics, and network RTT/traffic using external tools.
Each group request starts a fresh runtime: measured latency includes process
startup, model distribution/loading, prompt evaluation and generation, but
excludes queue wait. It is not comparable directly to the 3.845 s persistent
A40 provider average. RPC cache and OS cache state also affect load timing.

Compare an identical fixed-token prompt locally, with one RPC endpoint, and
with both endpoints. Record end-to-end latency and llama.cpp's prompt/evaluation
tokens per second. Also record LAN link speed and utilization; if generation
speed falls substantially while the link is saturated or latency-sensitive,
network communication is the likely bottleneck.
