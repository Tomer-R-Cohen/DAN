# Qwen3 Two-GPU Distributed Inference Result

**PASS — both GPUs participated.**

Qwen3-30B-A3B Q4_K_M generated responses through both
`distributed_model_experiment` and DAN's `/group` path using two llama.cpp RPC
workers. The run used llama.cpp `95ef7fc16054e63b427a3ef00188e055ef7586d8`,
TCP (`GGML_RPC_NO_RDMA=1`), layer splitting, tensor split `6,5`, 32,768 context,
`RPC0,RPC1`, and 99 requested GPU layers. The GGUF remained on Pod A.

| RPC device | Endpoint | GPU |
|---|---|---|
| RPC0 | `10.0.23.9:50052` | NVIDIA GeForce RTX 3090 |
| RPC1 | `10.1.115.60:50052` | NVIDIA RTX A4500 |

The model identity was `Qwen3-30B-A3B-Q4_K_M.gguf`, 18,556,685,824 bytes,
SHA256 `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.

The standalone run produced a response with 781,425.76 ms model
load/distribution, 24.64 prompt-evaluation tokens/s, 13.73 generation tokens/s,
255 generated evaluation tokens, and 806,286 ms total latency. The DAN group
run produced a response for request ID 1 with 783,180.33 ms load/distribution,
40.56 prompt-evaluation tokens/s, 16.94 generation tokens/s, 255 generated
evaluation tokens, and 804,709.8 ms total latency. Both paths shut down cleanly.

External telemetry at approximately 200 ms intervals recorded:

| GPU | Peak VRAM | Peak utilization | After shutdown |
|---|---:|---:|---:|
| RTX 3090 | 11,958 MiB | 34% | 1 MiB |
| RTX A4500 | 9,839 MiB | 43% | 1 MiB |

The peaks occurred 219 ms apart. Both workers accepted RPC connections, showed
CUDA graph warmups, and reported no RPC, CUDA, network, disconnect, or OOM
errors. This proves two-GPU participation and distributed execution. It does
not prove aggregate-VRAM necessity: Qwen3 may fit on one worker. The next test
must preserve single-worker allocation failures and two-worker success for a
model/context that exceeds either GPU's individual capacity.

The Pod A evidence was collected under
`results/two-gpu-qwen3/`; Pod B evidence was collected under
`results/two-gpu-qwen3-b/`. The raw evidence is from the test pods and is not
included in this repository unless copied separately. The supplied report lists
`FINAL-REPORT.md`, RPC mapping and worker logs, per-GPU telemetry, inference
logs/metrics, and Pod B RPC/network/error summaries.

The earlier small-model test encountered an unresolvable generated Pod B
hostname. The Qwen3 run used numeric private IPv4 successfully and reported no
network failures. Numeric private IPv4 remains the tested operational choice.

Next operational instructions are in [NEXT_GPU_EXPERIMENT.md](NEXT_GPU_EXPERIMENT.md):
measure disk-cache reuse and persistent-runtime requests before the separate
aggregate-memory experiment. The summary above is operator-supplied; raw evidence
and the reported Pod A commit `fb0ce58` have not been transferred into this checkout.
