# Codex instructions: Pod A, next staged GPU experiment

**Software is ready; do not provision hardware unless explicitly authorized.**
Persistent serving, automatic replacement, and Linux NVIDIA gamer onboarding
passed locally. This role validates the real CUDA path using the exact managed
coordinator/provider commands in
[NEXT_GPU_EXPERIMENT.md](NEXT_GPU_EXPERIMENT.md).

This role validates the integrated path: manifest hash checks,
N-provider assignment/readiness, persistent repeated DAN requests, required-node
heartbeat loss, cached process restart with no re-download, and readiness
restoration. The detailed environment/evidence requirements below remain the
baseline and should not be weakened.

Replacement validation requires a third eligible provider beyond A and B. An
ordinary local gaming PC may fill that role. If no third provider is present,
record replacement as NOT TESTED.

You are Pod A: RTX 3090 24 GB, DAN orchestrator, RPC client, and RPC worker A.
Execute this prompt and its repository-local prerequisite
[NEXT_GPU_EXPERIMENT.md](NEXT_GPU_EXPERIMENT.md) in full. That runbook supplies
build, networking, worker control, telemetry, phase gates, acceptance, publication,
and cleanup commands shared by both roles. No previous chat is needed. These are
instructions for an already provisioned disposable Pod, not permission to rent
hardware. Do not modify DAN or llama.cpp source.

The managed acceptance at the top of the shared runbook is authoritative. Do not
run the later unmanaged direct-completion, `/group`, or external llama-server
commands; they remain historical measurement reference only.

The completed tests proved two-GPU Qwen2.5 and Qwen3 participation, with Qwen3
requiring about 13 minutes of load/distribution per fresh DAN request. Next test
disk-cache reuse and a persistent backend in one rental, with one model download
and one build. Neither is yet validated. Aggregate-VRAM necessity remains later.

## 1. Prepare A and wait for B

Set `POD_ROLE=A` and execute common preparation and telemetry from the runbook.
Use llama.cpp `95ef7fc16054e63b427a3ef00188e055ef7586d8`, Release,
`GGML_CUDA=ON`, `GGML_RPC=ON`, `BUILD_SHARED_LIBS=OFF`. Build
`ggml-rpc-server`, `llama-completion`, and `llama-server` once. Build/test DAN.
Keep all output under `$EVIDENCE` (`results/next-gpu-A-<UTC-run-id>/`).
Require 3090 identity and private podnet1 routing. Reuse verified persistent
weights if available, otherwise download once. Require the checksum below.

```bash
export MODEL="$DATA_ROOT/models/Qwen3-30B-A3B-Q4_K_M.gguf"
export MODEL_SHA=0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5
if ! test -f "$MODEL"; then
  curl -fL --retry 3 -C - \
    https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf \
    -o "$MODEL.partial"
  test "$(stat -c %s "$MODEL.partial")" = 18556685824 || exit 1
  printf '%s  %s\n' "$MODEL_SHA" "$MODEL.partial" | sha256sum -c - || exit 1
  mv "$MODEL.partial" "$MODEL"
fi
test "$(stat -c %s "$MODEL")" = 18556685824 || exit 1
printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | sha256sum -c - | tee "$EVIDENCE/model-check.txt" || exit 1
```

Abort on any size/checksum failure; preserve the mismatching file and report it.
The official model version is defined by those bytes and SHA256. Never download
another model or ask B to download the GGUF. Prepare `$CACHE_ROOT` on the selected
volume as described in the runbook; its `rpc/` child is populated by llama.cpp.

Start A with `start_worker baseline off`. Verify its private-only listener and
CUDA0. Print `RPC_A_ENDPOINT=$PRIVATE_IP:50052`. Ask the operator for B's
`RPC_B_ENDPOINT` and `READY phase=baseline`; wait for both. Set the received
endpoint literally, with one numeric IPv4 and port 50052, then:

```bash
export RPC_A_ENDPOINT="$PRIVATE_IP:50052"
# Replace B_IP with the current verified numeric private IPv4, not a historical IP.
export B_IP='REPLACE_WITH_B_PRIVATE_IPV4'
test "$B_IP" != REPLACE_WITH_B_PRIVATE_IPV4 || exit 1
export RPC_B_ENDPOINT="$B_IP:50052"
export RPC_ENDPOINTS="$RPC_A_ENDPOINT,$RPC_B_ENDPOINT"
getent hosts "$B_IP" > "$EVIDENCE/b-getent.txt"
ip -4 route get "$B_IP" > "$EVIDENCE/b-route.txt"
nc -vz -w 10 "$B_IP" 50052 > "$EVIDENCE/b-connectivity.txt" 2>&1
export GGML_RPC_NO_RDMA=1
export LLAMA_ARG_CTX_SIZE=32768
export LLAMA_ARG_SPLIT_MODE=layer
export LLAMA_ARG_FIT=off
timeout --kill-after=10s 60s "$BIN/llama-completion" \
  --rpc "$RPC_ENDPOINTS" --list-devices > "$EVIDENCE/devices.log" 2>&1
```

The B_IP assignment requires the received address. Require RPC0=A/RTX 3090
and RPC1=B/RTX A4500, each
worker exposing only CUDA0. The client may list local CUDA0 as well, but no
inference command may select it directly. Do not proceed on failed checks.

## 2. Measure phases 0–3

Use one fixed raw prompt and deterministic sampling. This compares disk/model
reuse; it is not an answer-quality benchmark. Define:

```bash
run_direct() {
  label="$1"
  date -u +%FT%T.%NZ > "$EVIDENCE/${label}-begin.txt"
  /usr/bin/time -f 'wall_seconds=%e exit_status=%x' \
    -o "$EVIDENCE/${label}-wall.txt" \
    timeout --kill-after=30s 30m "$BIN/llama-completion" \
    --model "$MODEL" --rpc "$RPC_ENDPOINTS" --device RPC0,RPC1 \
    --n-gpu-layers 99 --ctx-size 32768 --split-mode layer --tensor-split 6,5 \
    --fit off --seed 42 --temp 0 --n-predict 256 --no-conversation \
    --single-turn --simple-io --no-display-prompt --color off \
    --prompt 'Explain TCP reliability in two sentences.' \
    > "$EVIDENCE/${label}.stdout" 2> "$EVIDENCE/${label}.stderr"
  result=$?
  printf '%s\n' "$result" > "$EVIDENCE/${label}-exit.txt"
  date -u +%FT%T.%NZ > "$EVIDENCE/${label}-end.txt"
  return "$result"
}
```

Do not blindly run all phases after a failed command. After each run inspect
output, exit status, worker health, offload logs and both pods' phase evidence.
Record and relay phase boundaries; wait for B readiness after every restart.

1. With cache disabled on both workers: `run_direct baseline` once.
2. No active client: stop A worker by verified PID. Tell B
   `RESTART phase=populate cache=on retain-cache=yes`; wait for B ready.
   Start A using `start_worker populate on`. Confirm both caches initially empty
   and logs show their actual cache paths. Run `run_direct populate` once.
3. Preserve both workers/caches. Run `run_direct warm-1` and then
   `run_direct warm-2`. Each is a fresh client process. Record reuse evidence,
   cache bytes/file count, and private-network RX on B for each.
4. With no active client, restart both workers with cache enabled and the same
   CACHE_ROOT. Relay `RESTART phase=restart cache=on retain-cache=yes`; wait for
   B ready and start A via `start_worker restart on`. Run `run_direct restart`.
   Verify reuse survived worker process restart; do not claim a physical cold
   disk test or cross-pod persistence from this result.

## 3. Existing DAN group with warm disk cache

Leave both workers running. Capture the following run, including stderr and
pipeline status. DAN's runtime sampling differs from the direct seed/temp flags,
so treat generation timing as descriptive, not an exact A/B benchmark.

```bash
date -u +%FT%T.%NZ > "$EVIDENCE/dan-begin.txt"
printf '%s\n' \
  '/group qwen3-cache Explain TCP reliability in two sentences.' 'exit' |
  timeout --kill-after=30s 30m "$DAN_ROOT/build/coordinator" 9000 \
    --group qwen3-cache "$BIN/llama-completion" "$MODEL" "$RPC_ENDPOINTS" 6,5 \
    > "$EVIDENCE/dan.log" 2>&1
printf '%s\n' "$?" > "$EVIDENCE/dan-exit.txt"
date -u +%FT%T.%NZ > "$EVIDENCE/dan-end.txt"
```

Keep port 9000 private/firewalled. Require response for ID 1, performance output,
and drain completion. It still launches a new model runtime; record that fact.

## 4. Persistent RPC-backed runtime, ten requests

No other inference client may be active. Keep the cache-enabled workers running.
This uses external llama-server directly, not the DAN /group adapter.

```bash
date -u +%FT%T.%NZ > "$EVIDENCE/server-start.txt"
nohup env GGML_RPC_NO_RDMA=1 "$BIN/llama-server" \
  --model "$MODEL" --rpc "$RPC_ENDPOINTS" --device RPC0,RPC1 \
  --n-gpu-layers 99 --ctx-size 32768 --split-mode layer --tensor-split 6,5 \
  --fit off --parallel 1 --host 127.0.0.1 --port 18080 \
  > "$EVIDENCE/server.log" 2>&1 < /dev/null &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$EVIDENCE/server.pid"
```

Poll `curl -fsS --max-time 5 http://127.0.0.1:18080/health` at 2-second intervals,
checking the recorded PID remains alive, for at most 30 minutes. Save health
output and ready timestamp. Abort the request stage if readiness never succeeds;
do not restart silently. Then perform ten sequential completions:

```bash
for i in $(seq 1 10); do
  kill -0 "$SERVER_PID" || break
  date -u +%FT%T.%NZ > "$EVIDENCE/http-${i}-begin.txt"
  curl --fail-with-body --max-time 180 --silent --show-error \
    http://127.0.0.1:18080/completion \
    -H 'Content-Type: application/json' \
    --data '{"prompt":"Explain TCP reliability in two sentences.","n_predict":256,"temperature":0,"seed":42,"cache_prompt":false,"stream":false}' \
    -o "$EVIDENCE/http-${i}.json" \
    -w 'http_code=%{http_code} wall_seconds=%{time_total}\n' \
    > "$EVIDENCE/http-${i}-wall.txt" 2> "$EVIDENCE/http-${i}.stderr"
  rc=$?
  printf '%s\n' "$rc" > "$EVIDENCE/http-${i}-exit.txt"
  date -u +%FT%T.%NZ > "$EVIDENCE/http-${i}-end.txt"
  test "$rc" = 0 || break
  jq -e '.content | type == "string" and length > 0' "$EVIDENCE/http-${i}.json" || break
done
```

Require ten successful responses, same PID, one load lifecycle, sustained weight
allocation on both GPUs between requests, and no full tensor retransmission.
Extract timings/token counts and stop reason from JSON; report missing metrics
as unavailable. Record median/range request latency separately from server
startup, disk-cache reload and previous DAN group latency. Identical prompts
are intentional; `cache_prompt:false` separates weight reuse from prompt reuse.
Do not claim conversational quality or production DAN persistence from this test.

## 5. Report, publish, preserve, then release

Apply all runbook acceptance and cleanup rules. Obtain B's phase summaries and
raw artifact archive/checksum through operator relay or supplied SSH access;
do not fabricate B measurements. Include exact relative evidence paths in
FINAL-REPORT.md and compare phases in a table. Update STATE/ROADMAP with actual
outcomes. Commit curated docs/evidence to the A validation branch and push if
authenticated, otherwise export a Git bundle with the raw archive. Do not claim
publication if push failed. Never commit model weights or caches.

Stop the HTTP server using its verified PID, confirm port 18080 closed, then
print `CLIENTS_STOPPED` so B can shut down safely. Stop A worker, record VRAM
release, stop telemetry, and finalize hashes/archives. Verify off-pod copies of
both pods' results before saying the rental is safe to terminate. If transfer
access is missing, request the destination and keep results/storage intact.

Final response: cache PASS/FAIL/INCONCLUSIVE (including restart), DAN group result,
persistent-runtime result, metric table, both GPUs' evidence, outstanding limits,
commit/branch/push-or-bundle status, off-pod receipt, and cleanup status.
