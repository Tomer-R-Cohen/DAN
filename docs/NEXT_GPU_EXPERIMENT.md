# Next rental: RPC disk-cache reuse, then persistent runtime

Status: planned; no cache or persistent distributed serving results yet.
Entrypoints: [Pod A](POD_A_NEXT_TEST_PROMPT.md), [Pod B](POD_B_NEXT_TEST_PROMPT.md).
Read this entire runbook with the selected role prompt. All context is in this
repository. Old `POD_*_TWO_GPU_SMOKE_PROMPT.md` files describe the completed small
test and are not the next-session instructions.

## Scope and invariants

Use one RTX 3090 24 GB on A (client, orchestrator, worker A) and one RTX A4500
20 GB on B (worker B and verifier). Same pinned llama.cpp on both:
`95ef7fc16054e63b427a3ef00188e055ef7586d8`. Record actual `--version`, Git SHA,
compiler, CUDA, driver and build flags; a printed RPC protocol version alone
does not establish the source revision. No moving-head runtime upgrade.

Use Qwen3-30B-A3B-Q4_K_M.gguf from
[Qwen's publisher repository](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/tree/main),
18,556,685,824 bytes, SHA256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.
This hash is the model version gate even if the download URL's branch moves.
Do not substitute another quantization or hash. Only A downloads the GGUF.
Both workers populate their RPC tensor cache from the first cache-enabled run.

Keep context 32768, split mode layer, split 6,5, devices RPC0,RPC1, GPU layers
99, generation cap 256, and TCP throughout. Do not change DAN or llama.cpp source,
registry schema, networking protocol, scheduling, or production architecture.
Do not run aggregate-VRAM experiments, download another model, or provision
additional hardware in this session. Stage failures are evidence, not permission
to refactor. Test scripts and report files under results are permitted.

## Common preparation on each fresh pod

Start from `git clone https://github.com/Tomer-R-Cohen/DAN.git` and `cd DAN`.
Use Ubuntu 24.04 x86-64 CUDA development image. Commands assume root; otherwise
use sudo for package installation. Require a working driver and `nvcc`; do not
replace a working driver. If CUDA toolkit is absent, report the image blocker.

```bash
apt-get update
apt-get install -y build-essential cmake git curl ca-certificates python3 \
  libcurl4-openssl-dev libssl-dev pkg-config iproute2 netcat-openbsd \
  jq time procps tar
export DAN_ROOT="$PWD"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
# Set POD_ROLE=A on A, B on B, before the next lines.
export EVIDENCE="$DAN_ROOT/results/next-gpu-${POD_ROLE}-${RUN_ID}"
mkdir -p "$EVIDENCE"
git rev-parse HEAD > "$EVIDENCE/dan-revision.txt"
nvidia-smi > "$EVIDENCE/nvidia-smi.txt"
nvcc --version > "$EVIDENCE/nvcc.txt"
ip -4 addr > "$EVIDENCE/addresses.txt"
ip -4 route > "$EVIDENCE/routes.txt"
export PRIVATE_IP="$(ip -4 -o addr show dev podnet1 | awk '{split($4,a,"/");print a[1]}')"
test -n "$PRIVATE_IP"
```

Require exactly one assigned private IPv4 and a private route through podnet1;
if the interface is absent, stop for networking setup. Never reuse the previous
session's addresses. Bind only this address, never `0.0.0.0`. Permit TCP 50052
only from the peer through the private network; no public RPC port mapping.
Do not flush firewall rules. Numeric IPv4 worked previously when generated
RunPod hostnames did not resolve. Verify route and listener on both nodes.

Choose `/workspace/dan-data` on an attached volume. Inspect `findmnt -T
/workspace` and `df -h /workspace`; record the storage type and whether the volume
survives stopping/termination. If only disposable storage exists, record that
cross-process cache persistence can be tested but cross-pod persistence cannot.
Reserve at least 80 GiB free disk and 32 GiB host RAM on A, 40 GiB disk and
16 GiB RAM on B; record actual free resources. Do not delete existing caches.

```bash
export DATA_ROOT=/workspace/dan-data
mkdir -p "$DATA_ROOT" "$DATA_ROOT/models" "$DATA_ROOT/cache"
export CACHE_ROOT="$(mktemp -d "$DATA_ROOT/cache/rpc-${POD_ROLE}.XXXXXX")"
printf '%s\n' "$CACHE_ROOT" > "$EVIDENCE/cache-root.txt"
export LLAMA_ROOT="$DATA_ROOT/llama-95ef7fc"
```

If LLAMA_ROOT is absent, clone it; otherwise inspect its SHA and clean status
and reuse only a clean matching checkout. Do not overwrite unrelated work.

```bash
if ! test -d "$LLAMA_ROOT"; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_ROOT"
  git -C "$LLAMA_ROOT" checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
fi
test "$(git -C "$LLAMA_ROOT" rev-parse HEAD)" = 95ef7fc16054e63b427a3ef00188e055ef7586d8
test -z "$(git -C "$LLAMA_ROOT" status --porcelain)"
set -o pipefail
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build-rpc" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=OFF 2>&1 | tee "$EVIDENCE/configure.log"
# A builds all three once; B needs only ggml-rpc-server.
```

Use `cmake --build "$LLAMA_ROOT/build-rpc" -j 4 --target ggml-rpc-server
llama-completion llama-server` on A, and the same command with only
`ggml-rpc-server` on B. Capture build output; retry `-j 2` for build memory
pressure without changing features. Copy CMakeCache.txt and Git SHA to evidence.
Define `BIN="$LLAMA_ROOT/build-rpc/bin"` and export it. On A also configure/build
DAN and run `python3 -m unittest discover -s tests -v`, saving the output.

Keep the controlling shell alive or save an allowlisted session file containing
only role, run ID, evidence/data/cache/runtime paths, private IPs and owned PIDs.
On resume re-source those values and verify each process command line. Never
generate a new CACHE_ROOT or EVIDENCE path halfway through a run, and never dump
the entire environment (it may contain authentication tokens).

## Telemetry and worker control (both pods)

Start before any client loads the model. Keep PID files and preserve stderr.

```bash
nohup nvidia-smi -i 0 \
  --query-gpu=timestamp,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv --loop-ms=200 \
  > "$EVIDENCE/gpu.csv" 2> "$EVIDENCE/gpu.stderr" < /dev/null &
GPU_PID=$!
printf '%s\n' "$GPU_PID" > "$EVIDENCE/gpu.pid"
nohup bash -c 'while :; do
  printf "%s," "$(date -u +%FT%T.%NZ)"
  paste -d, /sys/class/net/podnet1/statistics/rx_bytes /sys/class/net/podnet1/statistics/tx_bytes
  sleep 1
done' > "$EVIDENCE/network.csv" 2> "$EVIDENCE/network.stderr" < /dev/null &
NET_PID=$!
printf '%s\n' "$NET_PID" > "$EVIDENCE/network.pid"
```

Record idle baseline and UTC clock offsets on both pods. Interface counters
include unrelated traffic: download the model before measured phases and pause
artifact copies/builds during runs. B's private RX is the main remote weight
transfer signal; A-to-A RPC uses the local host even with its private address.
This is not a pure cross-network measurement for both workers. Keep debug level
constant throughout and record idle traffic rather than claiming exact RPC bytes.

Define this function in the controlling shell on each pod:

```bash
start_worker() {
  phase="$1"; cache_mode="$2"
  args=()
  if test "$cache_mode" = on; then args+=(--cache); fi
  nohup env LLAMA_CACHE="$CACHE_ROOT" GGML_RPC_NO_RDMA=1 GGML_RPC_DEBUG=1 \
    "$BIN/ggml-rpc-server" --host "$PRIVATE_IP" --port 50052 --device CUDA0 \
    "${args[@]}" > "$EVIDENCE/rpc-${phase}.log" 2>&1 < /dev/null &
  RPC_PID=$!
  printf '%s\n' "$RPC_PID" > "$EVIDENCE/rpc-${phase}.pid"
}
```

After each start, verify `kill -0 "$RPC_PID"`, `ss -ltnp`, and the log's CUDA0,
private endpoint, and cache directory. With cache enabled the pinned server uses
`$CACHE_ROOT/rpc/`. If any check fails, do not announce ready. Never run two
workers on the same port. Before restart, wait for A to confirm no active client;
verify the recorded PID command line, `kill -TERM "$RPC_PID"`, `wait
"$RPC_PID"`, and confirm the listener closed. Do not use broad pkill commands.

## Synchronization and exact phase order

A controls phase order. The human operator relays phase lines between the two
Codex sessions (no agent-to-agent channel is assumed). B prints
`RPC_B_ENDPOINT=<numeric-private-ip>:50052` and `READY phase=<name>` only after
checks. A must wait for matching readiness; never infer it from elapsed time.
Both record BEGIN/END UTC timestamps and cache file count/bytes at each phase.
No cache directory deletion or rotation after phase 1 starts.

| Phase | Workers A and B | A action |
|---|---|---|
| 0 baseline | start_worker baseline off | One direct completion; no disk cache |
| 1 populate | restart, start_worker populate on; same empty CACHE_ROOT | One identical completion; verify caches gain files |
| 2 warm | keep both workers and cache | Two fresh completion processes, serially |
| 3 restart | stop both workers with no client; restart with same CACHE_ROOT, cache on | One fresh completion, prove disk reuse survives worker restart |
| 4 DAN | keep restarted workers | One current DAN /group request; check ID and response |
| 5 persistent | keep workers, no other model client | Start one llama-server; ten HTTP completions; stop once at end |

If networking, hash or device mapping fails, stop before inference. If a phase
fails, preserve it, diagnose operational causes only, and mark its outcome FAIL
or INCONCLUSIVE. The persistent phase may still run after a disk-cache failure
if two-worker inference remains healthy: report separate outcomes. Do not repeat
13-minute baselines unnecessarily or rebuild/download between phases.

## Acceptance and interpretation

- All successful phases must return nonempty output with successful runtime
  status, correct RPC0=A/3090 and RPC1=B/A4500, allocation and compute evidence
  on both GPUs, and no hidden offload/fit changes.
- Cache pass: files populate both caches; server logs show hash-based tensor
  reuse (inspect `set_tensor_hash` diagnostics), and B's private RX in warm and
  restart phases is at most 20% of cache-population RX after accounting for idle
  traffic. This is a declared experimental threshold, not a runtime guarantee.
  Report individual measurements; if background traffic prevents attribution,
  label INCONCLUSIVE. Faster time alone is not cache proof. Do not drop OS caches;
  report filesystem page-cache effects separately.
- Record no-cache, populate, each warm, and restart wall/load times and token
  rates. Disk reuse still incurs client model reads/hash work, allocation,
  disk-to-VRAM loading, and activation traffic; it is not loaded-model persistence.
- Persistent pass: one server PID and one model-load lifecycle across ten serial
  HTTP 200 nonempty completions; both GPU allocations remain between requests;
  no per-request full weight retransmission or reload evidence. Report per-request
  network deltas and compare to initial load. Use `cache_prompt:false` to avoid
  confusing prompt/KV reuse with weight reuse. A 256-token cap may truncate output;
  record stop reason. This is backend validation, NOT persistent DAN /group support.
- DAN phase passes only if request ID 1 returns a real response and drains cleanly;
  coordinator exit status alone is insufficient. `/group` still reloads each time.
- No phase proves aggregate-VRAM necessity. Do not sum sampled GPU peaks and
  describe that as a model memory requirement or a single-device failure proof.

## Results, publication, and cleanup (both roles)

Write FINAL-REPORT.md under EVIDENCE with phase PASS/FAIL/INCONCLUSIVE, commands,
environment allowlist (never credentials), versions, hashes, endpoint mapping,
cache directory/mount/bytes/files, wall/load/token metrics, per-GPU peaks,
phase network deltas, errors, PID/load-count evidence, and provenance for B data.
Missing B evidence means joint participation/reuse is INCONCLUSIVE, not assumed.
Keep full raw logs, JSON responses, telemetry, summaries, and partial failures.
Create a SHA256 manifest for evidence files (exclude the manifest itself).

After timed phases, record cache inventory without deleting it:

```bash
du -sb "$CACHE_ROOT" > "$EVIDENCE/cache-bytes-final.txt"
find "$CACHE_ROOT" -type f -printf '%P %s\n' > "$EVIDENCE/cache-files-final.txt"
find "$CACHE_ROOT" -type f -exec sha256sum {} + > "$EVIDENCE/cache-sha256-final.txt"
```

Also capture size/file counts at each phase boundary. These local cache hashes
are diagnostic inventory, not independently supplied expected shard hashes.

After A finishes phase 5, stop its HTTP server and confirm its listener is gone.
A prints `CLIENTS_STOPPED`; only then may either role stop workers. Verify each
owned PID, terminate/wait, capture `nvidia-smi` after VRAM release and `ss -ltnp`,
then stop telemetry using its recorded PIDs. Retain cache and GGUF on the chosen
volume; do not delete weights, caches, unrelated processes, or whole directories.

Commit only curated reports/docs and small evidence, on separate branches
`validation/next-gpu-A-<RUN_ID>` and `validation/next-gpu-B-<RUN_ID>`. `/results/`
is ignored: explicitly `git add -f` each reviewed evidence file intended for Git.
Never add GGUF, RPC caches, build products, credentials, or oversized raw logs.
Keep large raw artifacts in a tar.gz outside Git and compute its SHA256. Update
STATE/ROADMAP with measured results on A only; do not mark target architecture
implemented. Recommended result commit: `Record RPC cache and persistent runtime validation`.
Push the role branch with `git push -u origin HEAD` when credentials exist;
never force-push or put credentials in URLs/logs. Do not merge to main from both
pods independently. If auth fails, create `git bundle create <export>.bundle HEAD`
and export it with artifacts. A local commit alone is not backup.

Copy each pod's evidence archive, checksum, and any bundle to the user's local
machine or confirmed durable external storage. Use provided authenticated SSH/
storage access, verify the destination file checksum, and record the receipt.
If no destination is available, ask for one and leave the pod/storage intact.
Git summaries do not back up raw logs. Do not stop/terminate the Pod until receipt
is confirmed. After receipt, report it is safe for the operator to stop/terminate
the rental in RunPod; do not assume stopping preserves its volume or ends billing.

Once all evidence writers have stopped, an export can be prepared as follows
(archive is outside EVIDENCE so it cannot include itself):

```bash
export ARTIFACT="$DATA_ROOT/next-gpu-${POD_ROLE}-${RUN_ID}.tar.gz"
tar -czf "$ARTIFACT" -C "$(dirname "$EVIDENCE")" "$(basename "$EVIDENCE")"
sha256sum "$ARTIFACT" > "$ARTIFACT.sha256"
```

Provide these absolute paths for the operator's `scp`/storage copy. After any
subsequent evidence edits, regenerate the archive/checksum and verify the new
off-pod copy before termination. Export the Git bundle separately if needed.

## Primary implementation references

Pinned [RPC usage](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/tools/rpc/README.md),
[cache implementation](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/ggml/src/ggml-rpc/ggml-rpc.cpp),
and [HTTP server API](https://github.com/ggml-org/llama.cpp/blob/95ef7fc16054e63b427a3ef00188e055ef7586d8/tools/server/README.md).
RPC disk cache is not a cryptographic shard distribution system; see
[the intended provider lifecycle](PROVIDER_LIFECYCLE.md).
