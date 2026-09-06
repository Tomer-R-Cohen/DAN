# Codex instructions: Pod B, next staged GPU experiment

**Software gate: do not provision or use a GPU rental from this prompt yet.**
Managed Worker Runtime v1 now verifies/caches assigned artifacts and owns healthy
RPC workers. Persistent Managed Distributed Serving is still missing: DAN does
not retain a distributed client or route repeated `dan-main` requests through
the ready replica. Implement it and add exact commands before renting GPUs.

After that gate, B independently verifies its assigned manifest shard, reported
state transitions, persistent participation, heartbeat-loss behavior, retained
cache restart without another download, and replica readiness restoration. The
evidence and safety requirements below remain the baseline.

You are Pod B: RTX A4500 20 GB, RPC worker B and independent verifier.
Read this file and [NEXT_GPU_EXPERIMENT.md](NEXT_GPU_EXPERIMENT.md) completely,
then execute their B instructions. Both are in the cloned DAN repository; no
previous chat is needed. A has the RTX 3090, full model, clients and orchestrator.
Do not modify DAN/llama.cpp source, run inference clients, download the GGUF,
or provision hardware. Model weights reach B only as assigned RPC tensors.

Previous two-pod Qwen2.5 and Qwen3 runs proved participation. This rental measures
disk-cache reuse and ten requests through a persistent RPC-backed runtime. It
does not implement persistent DAN groups or test aggregate-VRAM necessity.

## 1. Prepare, measure, announce

Set `POD_ROLE=B`. Execute common preparation/build/telemetry in the runbook.
Build only `ggml-rpc-server`, using exact llama.cpp revision
`95ef7fc16054e63b427a3ef00188e055ef7586d8`, Release, CUDA+RPC enabled, static
libraries. Require working driver/nvcc, A4500 identity, sufficient RAM/disk and
a numeric private IPv4 on podnet1. Record actual resources and volume persistence.

The model on A is Qwen3-30B-A3B-Q4_K_M.gguf from official Qwen GGUF, size
18,556,685,824 bytes, SHA256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.
A must verify it and share the verification result. B verifies local cache file
inventory/hashes for diagnostics, not the complete GGUF; RPC's FNV keys are not
cryptographic shard-manifest verification.

Use a new `$CACHE_ROOT` on `/workspace/dan-data` as specified in the runbook;
preserve it across all worker restarts. No data deletion. Start the 200 ms GPU
telemetry and one-second podnet1 byte counters before starting the worker.
Evidence lives in `results/next-gpu-B-<UTC-run-id>/` under `$EVIDENCE`.

Define the runbook's `start_worker` function, then `start_worker baseline off`.
Verify listener binds only `$PRIVATE_IP:50052`, worker exposes only CUDA0=A4500,
PID stays alive, cache is disabled, and log shows TCP. Print:

```text
RPC_B_ENDPOINT=<current-numeric-private-ip>:50052
READY phase=baseline
```

Print the actual IP, not the placeholder. The previous generated hostname failed
DNS; do not reuse previous pod addresses. Ask for A's current private IP, record
`getent hosts <A-IP>`, `ip -4 route get <A-IP>`, and `nc -vz -w 10 <A-IP> 50052`.
Require private routing. Never publish or bind an RPC listener publicly.

## 2. Follow the staged handshake

The human operator relays phase messages; wait explicitly rather than assume
timing. Keep logs and record UTC phase boundaries, baseline VRAM, file count/
bytes under `$CACHE_ROOT/rpc`, private RX/TX snapshots, and process status.

1. Baseline: leave cache-disabled worker running while A runs one completion.
2. On `RESTART phase=populate cache=on retain-cache=yes`, require A confirms no
   active client. Stop only your verified worker PID; wait and verify port closes.
   Start `start_worker populate on` with the same empty CACHE_ROOT. Verify
   log shows `$CACHE_ROOT/rpc/`, TCP and CUDA0. Print `READY phase=populate`.
   First cache-enabled inference must create cache files; preserve its RX delta.
3. Warm: do not restart. Observe A's warm-1 and warm-2 fresh clients, collecting
   hash-reuse logs, network deltas, and GPU activity for each separately.
4. On `RESTART phase=restart cache=on retain-cache=yes`, with no active client,
   stop and restart via `start_worker restart on` and the same CACHE_ROOT.
   Verify files survived and print `READY phase=restart`. Observe restart load.
5. DAN: keep worker running for A's current `/group` request. Record separately.
6. Persistent: keep worker running while A starts one llama-server and performs
   ten serial HTTP requests. Record initial load separately from request windows.
   Check weight allocation remains between requests and no per-request full
   tensor payload stream or model reload is observed. Nonzero utilization need
   not occur in every 200 ms sample.

No overlapping clients, model downloads, archive transfers or unrelated network
tests during measured windows. Do not reset counters/drop OS caches. Diagnostic
cache hashes may be computed after timed phases to avoid competing disk I/O.
If RPC dies, save errors/status and notify the operator; do not silently restart
and mark the interrupted phase successful. Cache failures do not authorize fixes
to source. Follow runbook FAIL/INCONCLUSIVE rules.

## 3. Independent acceptance and preservation

Report cache population, warm-1, warm-2, restart, DAN, and persistent windows
separately. Cache reuse requires hash-based load evidence plus warm/restart RX
at most 20% of populate RX, allowing for idle traffic; otherwise explain failure
or attribution limits. Persistent evidence requires retained VRAM, compute/RPC
activity across the request sequence and absence of repeated full loads. A owns
response/HTTP metrics; do not infer them from B's GPU activity.

Write FINAL-REPORT.md with endpoint/GPU, cache mount/path/durability, phase peak
VRAM/utilization, RX/TX deltas, cache count/size, worker PIDs, errors, model hash
provenance from A, and exact raw file paths. Supply this and an archive/checksum
to A or the operator. Follow the shared runbook's reviewed evidence commit/push
instructions using a distinct B validation branch, or export a Git bundle if
credentials are unavailable. Do not edit main or force-push.

Keep the worker alive until A explicitly prints `CLIENTS_STOPPED`. Then stop
the verified PID, verify port closes, record after-shutdown VRAM, stop telemetry,
and finalize evidence hashes and archive. Do not delete the cache. Confirm raw
artifacts and bundle (if needed) are copied off this pod and destination checksums
match before permitting rental shutdown. If no transfer destination is available,
ask for it and preserve the Pod. Never assume a stopped/terminated Pod retains
its filesystem. Report endpoint, per-phase verdicts, evidence paths, publication/
bundle status, backup receipt, and safe-to-terminate status to the operator.
