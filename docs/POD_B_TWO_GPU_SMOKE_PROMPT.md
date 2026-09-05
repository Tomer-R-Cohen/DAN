# Pod B: DAN Two-GPU Distributed Inference Smoke Test

You are Pod B for DAN's two-GPU distributed inference smoke test.

## Role and hardware

- Hardware: RTX A4500, 20 GB VRAM
- Role: RPC worker B and independent verifier only
- Pinned llama.cpp commit: `95ef7fc16054e63b427a3ef00188e055ef7586d8`

Do not modify DAN source code. Do not download any model. The complete GGUF is
held on Pod A.

## Procedure

1. Verify the RTX A4500, driver, CUDA toolkit, and RunPod private networking.
2. Build the pinned llama.cpp revision with CUDA and RPC support.
3. Start `ggml-rpc-server` using only `CUDA0` on TCP port `50052`, bound to a
   RunPod-private reachable interface. Do not expose RPC publicly.
4. Start `nvidia-smi` telemetry at approximately 200 ms intervals and save it
   under `results/two-gpu-smoke-b/`.
5. Determine this Pod's reachable RunPod private hostname or IPv4 address.
6. Print the endpoint exactly in this form:

   ```text
   RPC_B_ENDPOINT=<reachable-private-host>:50052
   ```

7. Verify that the RPC server is listening on TCP port `50052`.
8. Keep the RPC server running while Pod A performs the distributed test.
9. During inference, record VRAM before, during, and after the request, GPU
   utilization, and RPC activity or errors.
10. Save all Pod B logs, telemetry, network identity, revision/build details,
    and checks under `results/two-gpu-smoke-b/`.

## Required evidence

Preserve, at minimum:

- GPU identity, driver, CUDA toolkit, and total/free VRAM
- llama.cpp revision and build configuration
- private hostname/address and the exact `RPC_B_ENDPOINT` line
- RPC server command, startup log, and listening-port check
- timestamped `nvidia-smi` telemetry at approximately 200 ms intervals
- RPC activity, errors, and process status during Pod A's inference
- before/during/after VRAM observations

Model weights must not be downloaded on Pod B. Endpoint reachability and server
startup do not prove participation; participation requires correlated RPC
activity, VRAM growth, or GPU utilization during Pod A's inference.

## Final report

Report:

- `RPC_B_ENDPOINT`
- GPU identity and VRAM
- RPC server status
- peak VRAM
- peak GPU utilization
- whether Pod B participation is `PROVEN` or `NOT PROVEN`
- exact evidence file paths

If the test fails or no activity is observed, keep the RPC server logs and
telemetry, state the failure category, and leave the evidence directory intact.
