# Two-GPU Distributed Inference Smoke Test Report

## Result

**PASS — two-GPU participation proven.**

On 2026-09-05, Qwen2.5-1.5B-Instruct Q4_K_M executed through llama.cpp RPC
across two RunPod pods. Both GPUs received allocations, showed nonzero
utilization, and handled RPC activity during the same tests. The standalone
`distributed_model_experiment` and DAN's integrated `distributed_group` path
both produced responses.

This validates two-pod distributed-inference plumbing. It does not prove
aggregate-VRAM necessity because the approximately 940 MiB model fits on either
GPU. The large Qwen3 model was not downloaded or tested for this smoke test.

## Configuration

- Pod A: NVIDIA GeForce RTX 3090, 24,576 MiB physical / 24,124 MiB RPC capacity,
  private endpoint `10.0.23.9:50052`
- Pod B: NVIDIA RTX A4500, 20,042 MiB RPC capacity, private endpoint
  `10.1.115.60:50052`
- llama.cpp: `95ef7fc16054e63b427a3ef00188e055ef7586d8`
- Build: Release, `GGML_CUDA=ON`, `GGML_RPC=ON`, `BUILD_SHARED_LIBS=OFF`
- Transport: TCP with `GGML_RPC_NO_RDMA=1`
- Context: 4096; split mode: layer; tensor split: `6,5`
- Selected devices: `RPC0,RPC1`; DAN requested 99 GPU layers
- Model SHA256: `1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370`

The verified mapping was:

| RPC device | Endpoint | Worker GPU |
|---|---|---|
| RPC0 | 10.0.23.9:50052 | RTX 3090 |
| RPC1 | 10.1.115.60:50052 | RTX A4500 |

## Results

The standalone experiment produced a response with 55,279 ms distributed
latency, 51,984.54 ms model load, 40.28 prompt-evaluation tokens/s, and
17.67 generation tokens/s. The DAN group request produced a response with
57,223 ms group latency, 51,139.01 ms model load, 68.05 prompt-evaluation
tokens/s, and 17.98 generation tokens/s. These latencies include fresh process
startup, model distribution/loading, prompt evaluation, and generation.

Telemetry at approximately 200 ms intervals recorded 890 MiB peak and 5%
peak utilization on Pod A, and 1,093 MiB peak and 3% peak utilization on Pod B.
Pod B returned to 1 MiB after clean worker shutdown. Both workers accepted RPC
connections and no RPC errors occurred after switching to Pod B's numeric
private IPv4 endpoint.

The generated Pod B hostname did not resolve and failed `getent hosts` and
`nc`; the numeric private endpoint succeeded. Numeric private IPv4 endpoints
are therefore an operational requirement for this pinned RPC client in the
observed RunPod setup.

## Evidence

The raw evidence was collected on Pod A under:

`/workspace/DAN/results/two-gpu-smoke/`

and on Pod B under:

`results/two-gpu-smoke-b/`

Key files include `FINAL-REPORT.md`, `rpc-device-map.log`,
`rpc-worker-a.log`, `gpu-a-samples.csv`, `standalone-distributed.log`,
`dan-distributed-group.log`, and Pod B's `rpc-server.log`,
`nvidia-smi-200ms.csv`, and `telemetry-summary.txt`.

## Next milestone

The subsequent Qwen3 test is summarized in [QWEN3_TWO_GPU_REPORT.md](QWEN3_TWO_GPU_REPORT.md).
The planned cache/runtime rental in [NEXT_GPU_EXPERIMENT.md](NEXT_GPU_EXPERIMENT.md)
is gated on managed-worker integration. This smoke report is an
operator-supplied summary; its raw pod evidence is not included in this checkout.

Later run an aggregate-VRAM test with a model/context whose required GPU allocation
exceeds either worker individually but fits across both. Preserve single-worker
allocation failures and two-worker success under identical settings, and verify
placement and telemetry. This smoke-test result should not be described as
aggregate-VRAM validation.
