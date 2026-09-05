# Pod A: DAN Two-GPU Distributed Inference Smoke Test

You are Pod A for DAN's two-GPU distributed inference smoke test.

## Role and hardware

- Hardware: RTX 3090, 24 GB VRAM
- Role: orchestrator, RPC worker A, and RPC client
- Pinned llama.cpp commit: `95ef7fc16054e63b427a3ef00188e055ef7586d8`

Do not redesign DAN or refactor source code. Do not download the large Qwen3
model for this smoke test.

## Goal

Prove that one small instruct GGUF can execute across Pod A and Pod B, with
both GPUs participating in one inference.

## Procedure

1. Verify the RTX 3090, driver, CUDA toolkit, and RunPod private networking.
2. Build the pinned llama.cpp revision with CUDA and RPC support.
3. Start Pod A's `ggml-rpc-server` on `CUDA0`, TCP port `50052`, bound only to
   a RunPod-private reachable interface. Do not expose RPC publicly.
4. Start GPU telemetry at approximately 200 ms intervals and save it under
   `results/two-gpu-smoke/`.
5. Wait for Pod B to provide `RPC_B_ENDPOINT`.
6. Verify Pod B from Pod A with both commands:

   ```bash
   getent hosts <pod-b-host>
   nc -vz <pod-b-host> 50052
   ```

7. Build or invoke the pinned llama.cpp client and verify device mapping:

   ```text
   RPC0 = RTX 3090
   RPC1 = RTX A4500
   ```

8. Use a small compatible instruct GGUF, preferably
   Qwen2.5-1.5B-Instruct Q4_K_M or another approximately 1–2 GB model. Keep
   the GGUF on Pod A; Pod B must not download model weights.
9. Run `distributed_model_experiment` across both RPC endpoints with tensor
   split `6,5`.
10. Require evidence that both GPUs received model allocation and showed VRAM
    and/or utilization activity during the inference.
11. If standalone inference succeeds, run one DAN `distributed_group` request
    with the same model and endpoints. Do not alter DAN source code.
12. Save all logs, telemetry, device output, model hashes, commands, and
    results under `results/two-gpu-smoke/`.

## Required evidence

Preserve, at minimum:

- Pod A GPU/CUDA and network checks
- llama.cpp revision and build configuration
- Pod A RPC server log and listening-port check
- Pod A telemetry with timestamps at roughly 200 ms intervals
- Pod B endpoint and Pod A DNS/connectivity checks
- llama.cpp RPC device listing proving `RPC0` and `RPC1` mapping
- GGUF filename, source, size, and SHA256
- standalone experiment stdout/stderr and exit status
- DAN distributed-group stdout/stderr and exit status, if attempted
- Pod A and Pod B telemetry and RPC logs copied into the result set

Do not treat endpoint labels alone as proof. Correlate allocation/offload logs,
VRAM growth, and GPU utilization with the inference timestamps.

## Final report

Report:

- `PASS` or `FAIL`
- `RPC0`/`RPC1` mapping
- whether a response was produced
- Pod A peak VRAM and peak utilization
- Pod B peak VRAM and peak utilization, if evidence is available
- latency and tokens/second, when llama.cpp reports them
- network failures, if any
- exact evidence file paths

If the test fails, preserve partial evidence and identify whether the failure
was caused by build, DNS, connectivity, RPC startup, device mapping, model
allocation, inference, or DAN group routing.
