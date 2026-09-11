# Legacy Path: Managed dan-main, Whole-Model Providers, and RPC Groups

This merges what used to be seven separate documents (`SETUP.md`, `PROTOCOL.md`,
`PROVIDER_LIFECYCLE.md`, `MODELS.md`, `FRIENDS_TESTNET.md`, `GPU_VALIDATION.md`,
`GPU_RESULTS_TEMPLATE.md`) into one, since they all describe the same thing:
DAN's original architecture — a text-protocol coordinator, whole-model
providers, the managed `dan-main` control plane, and manually configured
llama.cpp RPC groups. **This is not DAN's current main direction.** The
[README](../README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) describe the
current path: runtime-formed provider-owned replicas, a C++23 metadata-only
coordinator, and range-backed sparse GGUF storage — see
[PROVIDER_OWNED_SETUP.md](PROVIDER_OWNED_SETUP.md) to build and run that path.
Everything below remains accurate for the legacy path specifically, which
README still lists as a supported fallback.

## Status

Single-GPU validation passed on NVIDIA A40 with Qwen3-30B-A3B Q4_K_M at 32,768
context: ten persistent-provider responses, averaging 3,844.7 ms per request,
21,227 MiB peak VRAM, clean exit. See [PROJECT_STATUS.md](PROJECT_STATUS.md)
and the [committed benchmark archive](../dan-qwen3-30b-a3b-results.tar.gz) for
the full report, registries, and logs — including the failed 4,096-context
attempt (Qwen3's default thinking traces exhausted it after eight replies)
before the successful 32,768-context rerun.

Two remote CUDA RPC workers (RTX 3090 + RTX A4500) separately passed a small-model
smoke test and a Qwen3-30B-A3B run through both the standalone experiment and a
DAN distributed group, at roughly 10-12 GiB allocated with nonzero utilization
on each GPU; startup was dominated by TCP model distribution. Qwen3 fits on a
single A40, so neither of these establishes aggregate-VRAM necessity — no
aggregate-VRAM candidate has been validated yet. The persistent
`llama-server`-backed managed path and automatic replacement have passed
locally; the cache/persistent-runtime rental experiment remains deferred until
a managed-worker adapter connects Provider Control Plane v1 to verified cache
and persistent serving.

## Wire protocol

Managed providers and coordinators advertise protocol version 2 (adds download
progress reporting; intentionally incompatible with older managed binaries).
Each message is one frame: a 4-byte unsigned payload length (network byte
order) followed by that many bytes of UTF-8-compatible text. Maximum payload
16 MiB; text content is treated as bytes, UTF-8 validation is not performed.

**`HELLO`** — provider to coordinator, identifies the expected peer after
connecting. Payload: `HELLO`. Provider sends `CAPABILITIES` next.

**`CAPABILITIES`** — provider to coordinator, registers hardware/model/backend:

```text
CAPABILITIES
provider_id=<provider identifier>
provider_name=<optional friendly name>
device_type=<device type>
gpu_name=<GPU name or not available>
vram=<VRAM or not available>
vram_mib=<numeric usable VRAM MiB, 0 if unspecified>
model_name=<model name>
backend=<CPU, CUDA, or another backend>
control_plane=<0 or 1>
cached_shard=<model ID>|<version>|<shard ID>|<content hash>
worker_endpoint=<managed worker host:port, when configured>
```

Values are self-reported display strings; newline/carriage-return are replaced
with spaces. `cached_shard` may repeat and is accepted only when all four
identity fields are present. Existing providers send `control_plane=0` and
need no managed fields. Coordinator responds `ASSIGN_SHARD` (managed
providers), `PROMPT` (legacy scheduler selection), or `BYE`.

**`ASSIGN_SHARD`** — coordinator to managed provider, assigns one required
shard of the single `dan-main` replica. Payload: newline-separated `model_id`,
`version`, `shard_id`, `size_bytes`, `hash`, `source`. Healthy assignments stay
stable; offline ownership releases so an eligible spare can replace it. A
reconnect may reclaim an unowned shard but never preempts a healthy
replacement.

**`SHARD_STATE`** — managed provider to coordinator, reports actual local
preparation state. Payload: newline-separated `model_id`, `version`,
`shard_id`, `hash`, `state`. States: `ASSIGNED`, `DOWNLOADING`, `CACHED`,
`LOADING`, `READY`, `ERROR`. The coordinator rejects malformed fields, a
mismatched assignment identity/hash, `UNASSIGNED` reports, and invalid state
transitions. A provider may report `CACHED` immediately when its inventory
exactly matches the manifest; replica readiness still requires `READY`.

**`DOWNLOAD_PROGRESS`** — managed provider to coordinator, at most once per
second while downloading. Payload: newline-separated `downloaded_bytes`,
`total_bytes`, `bytes_per_second`. Accepted only from the provider currently
downloading the assigned artifact; requires `downloaded_bytes <= total_bytes`
and the reported total to match the manifest.

**`LOAD_SHARD` / `UNLOAD_SHARD`** — coordinator to managed provider. Payload:
newline-separated `model_id`, `version`, `shard_id`, `hash`. Starts/health-checks
or stops the DAN-owned worker for the exact assignment. `LOAD_SHARD` is sent
automatically after `CACHED` and may repeat safely (an already-healthy worker
isn't duplicated). `UNLOAD_SHARD` stops only the owned worker, leaves the
verified artifact cached. State reports: `LOADING` then `READY`, or `CACHED`
after unload; failures report `ERROR`.

**`HEARTBEAT`** — managed provider to coordinator, twice per second. Payload:
`HEARTBEAT\nused_vram_mib=<MiB>` (`HEARTBEAT` alone remains valid for legacy
providers). Refreshes last-seen time and reports DAN-worker VRAM use. Missing
the coordinator's default ten-second timeout marks the provider offline,
removes its shard from replica readiness, and releases live ownership without
deleting history or cached bytes.

**`PROMPT`** — coordinator to provider, requests inference. Payload:
`PROMPT\n<request ID>\n<prompt text>` (e.g. `PROMPT\n42\nExplain TCP briefly.`).
Expects `RESPONSE` or `ERROR`.

**`RESPONSE`** — provider to coordinator, returns generated text. Payload:
`RESPONSE\n<request ID>\n<generated text>`.

**`ERROR`** — provider to coordinator, reports failed inference. Payload:
`ERROR\n<request ID>\n<short description>`. No response expected; that
provider connection is removed.

**`BYE`** — coordinator to provider, ends the session cleanly. Payload: `BYE`.
No response expected. Legacy providers exit; reconnect-enabled gamer providers
treat coordinator shutdown as a temporary disconnect.

Each provider has its own framed TCP connection: `HELLO` then `CAPABILITIES`
after connecting, optionally assignments/state reports/heartbeats for a
managed connection, then any number of sequential `PROMPT`/`RESPONSE` pairs
before `BYE`. One active request per provider connection at a time. Request
IDs are unique per coordinator run and correlate responses that can arrive out
of order.

Distributed groups are coordinator-local configured targets, not DAN TCP
providers — their RPC traffic uses llama.cpp's own protocol, with an internal
framed socket returning the request ID and result from the group job; no new
network-visible message type is needed. The requested model is coordinator
scheduling metadata, not a new provider message field — after selection the
existing request-ID/prompt frame is sent unchanged, with supported-model
information coming from `CAPABILITIES`.

## Model registry and strategy

Models are deployment configuration and data, not DAN networking logic. Start
the coordinator with `--models <registry-file>` and request a stable ID with
`/model <id> <prompt>`. Replacing the main model means editing the registry
and provider/group startup configuration, not coordinator scheduling code.

Registry context/memory fields are metadata only — providers need
`--ctx-size` explicitly (the driver supplies it), and distributed groups don't
forward registry context; set `LLAMA_ARG_CTX_SIZE` in their inherited runtime
environment (see "Integrated distributed model over llama.cpp RPC" below).

**Roles:**

- `test` — small, cheap models used only to verify networking, framing,
  queues, concurrency, and RPC plumbing. Output quality is irrelevant. The
  included `smollm2-test` entry is test-only and must not be presented as
  DAN's intelligence.
- `main` — the best open-weight model the deployment can serve at acceptable
  latency and quality; this is the model for real DAN use and demonstrations.
- `distributed` — useful models whose memory or execution requirements justify
  a manually configured multi-provider RPC group.

**File format** — pipe-delimited, eleven fields:

```text
id|role|family|path|runtime_name|quantization|memory_mib|context_length|requirements|single_provider|distributed
```

`runtime_name` must match provider capability metadata; `path` is used for
configured distributed groups; execution flags are `yes`/`no`; lines starting
with `#` and blank lines are ignored. See `config/models.example.conf`.

**Choosing the main model.** Rankings and runtime support change quickly —
before promoting a model to `main`, verify license, llama.cpp support, prompt
template, memory at the desired context, quality on DAN workloads, latency,
and failure behavior on the actual hardware. Keep the stable ID `dan-main`;
update family/path/runtime name/quantization/resources. As a starting point,
OpenAI documents `gpt-oss-20b` (Apache-2.0, ~16 GB deployments) and
`gpt-oss-120b` (stronger, ~80 GB, 131,072 token context) — the sample registry
uses the former as a main-model candidate and the latter as a
large/distributed candidate; these are replaceable examples, not permanent
choices. ([Announcement](https://openai.com/index/introducing-gpt-oss/),
[gpt-oss-120b docs](https://developers.openai.com/api/docs/models/gpt-oss-120b).)

**Promotion checklist:** benchmark candidates on representative prompts and
the intended llama.cpp build; measure model + KV-cache memory at the chosen
context; select a quantization meeting quality/latency needs; set
`single_provider`/`distributed` from tested execution support; start
providers with matching `--model-name` or configure groups with the registry
path using `@model-id`; change the `dan-main` entry only after the deployment
passes these checks.

## Provider lifecycle

**Repeated full-model streaming during user requests is NOT the intended DAN
production architecture.** Model preparation must be amortized across many
requests, or download/transfer cost dominates latency, wastes bandwidth,
increases rental expense, and makes interactive use impractical. Each DAN
group request currently launches a new runtime — the measured ~781-783s of a
~805s total Qwen3 distributed run identifies model preparation as the
dominant cost without isolating network/disk/hashing/allocation individually.

For a Linux gaming PC, `dan-provider` performs the steps before `Join`: load
minimal local config, detect NVIDIA GPU/index/VRAM with `nvidia-smi`, subtract
the configured reserve, load or create a persistent random identity, select a
private RPC endpoint, and enter the managed-provider lifecycle. Coordinator or
network loss reconnects with capped backoff and re-advertises verified cache.

```text
provider joins
  -> capabilities measured
  -> model/shard assignment
  -> shard downloaded ahead of inference
  -> hash verified
  -> cached on disk
  -> optionally loaded into VRAM
  -> provider advertises cached/loaded state
  -> scheduler forms execution target
  -> many inference requests reuse weights
```

1. **Join** — identity, capabilities, cached inventory register on the
   existing connection. Identity is self-reported; sessions/leases and
   authorized connectivity are not implemented.
2. **Measure** — report available GPU VRAM, RAM, disk, backend revision,
   network reachability; distinguish total capacity from capacity reserved by
   other jobs.
3. **Assign** — the coordinator loads one versioned `dan-main` manifest and
   assigns its shard collection one-per-provider. The manifest includes
   quantization, compatible runtime, context, tensor mapping, size, and
   cryptographic checksums. Placement is a control-plane operation.
4. **Prepare** — `managed_provider` copies local/`file://` bytes or downloads
   HTTP(S) bytes before user work, verifies optional length and SHA-256, and
   atomically publishes the completed cache entry. Failed verification
   becomes `ERROR`.
5. **Cache** — verified bytes live under model/version/shard/hash identity.
   Startup scans and re-hashes inventory. HTTP downloads keep a stable
   partial file, resume across restarts, and report size/speed. Eviction is
   not implemented.
6. **Load** — `LOAD_SHARD` starts one owned llama.cpp RPC worker and polls its
   TCP endpoint, reporting `READY` only after the child is alive and reachable.
7. **Advertise** — v1 tracks `UNASSIGNED`, `ASSIGNED`, `DOWNLOADING`,
   `CACHED`, `LOADING`, `READY`, `ERROR`, plus online/offline heartbeat state.
   Cached is not loaded; busy/draining control states are future work.
8. **Form target** — the replica is `READY` only when every required shard
   has an online `READY` provider. The coordinator then starts one persistent
   server and separately health-gates runtime `READY`.
9. **Serve** — `/model dan-main` sends sequential HTTP completion requests to
   the same server PID; request traffic doesn't restart workers, reacquire
   artifacts, or redistribute the model. Session/KV-cache ownership is
   separate from weight residency.
10. **Recover** — withdraw readiness on a lost worker, stop the runtime, move
    only the missing shard to an eligible spare, run normal preparation, and
    automatically recreate the runtime once every required provider is ready.

Automatic Provider Replacement / Reassignment v1 implements step 10 for the
one replica: provider loss stops the runtime and rejects requests while
recovery selects and prepares a spare; runtime startup resumes automatically
after replica readiness. Leases, in-flight replay, proactive rebalancing, and
multiple replicas remain unimplemented. Manual groups remain a separate
process-per-request path.

Assignments are a collection, not named A/B slots. Online unassigned
providers are spares. For each missing shard the coordinator prefers an exact
model/version/shard/hash cache match, then highest sufficient VRAM, then
provider ID. Offline ownership releases while sticky history remains — a
reconnect reclaims an unowned shard through normal selection but never evicts
a healthy replacement. A provider reporting `ERROR` is excluded for that shard
until reconnect.

**Bridge experiments vs. production design.** RPC `--cache` stores large
tensor payloads on each worker and can satisfy later loads from disk, but
still needs initial population, client-side GGUF/tensor metadata, hashing,
allocation, and disk-to-VRAM loading after restart — it is not a portable,
ahead-of-time shard distribution protocol. The pinned implementation uses a
64-bit FNV cache key, not a cryptographic shard manifest; do not describe it
as production integrity verification (the experiment does verify the source
GGUF's SHA-256 and records local cache file hashes for diagnostics). A
persistent `llama-server` now supplies DAN's managed serving path; local
tests prove process reuse and automatic replacement, but real CUDA
tensor-transfer reuse remains a hardware question.

## Build and run

```bash
cmake -S . -B build
cmake --build build
```

Build llama.cpp outside this repository:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
git -C ~/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DBUILD_SHARED_LIBS=OFF
cmake --build ~/llama.cpp/build --config Release -j \
  --target llama-completion llama-server rpc-server
```

Run the CPU-only deployment regressions after building DAN with
`python3 -m unittest discover -s tests -v` — these use a stand-in runtime and
do not validate CUDA or model quality.

### One-command gaming-PC provider

On a trusted private overlay, configure once with the coordinator address,
this machine's private address, and the built llama.cpp RPC worker:

```bash
./scripts/setup_provider.sh \
  100.80.10.1:9000 "$(tailscale ip -4)" \
  "$HOME/llama.cpp/build/bin/rpc-server"
```

Then start or restart contribution with `./build/dan-provider`. It reads
`~/.dan/provider.conf`, detects NVIDIA GPU memory, reserves 1536 MiB by
default, persists `~/.dan/provider-id`, chooses a worker port, and reconnects
automatically. Use `--check` to validate without connecting. See "Friends
testnet guide" below for configuration, status, cache, troubleshooting, and
first-run validation.

### Local Provider Control Plane simulator

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
python3 scripts/fake_provider.py --id node-e --port 9000 \
  --gpu SpareGPU --vram-mib 12288 --cache-file /tmp/dan-node-e.cache
```

Enter `/providers` at the coordinator. Four providers are assigned, the fifth
is `SPARE`. Stop an assigned process past the default ten-second heartbeat
timeout — the spare automatically receives the missing shard; restarting the
original identity does not evict the healthy replacement.

The manifest format is `model|dan-main|<version>` followed by any number of
`shard|<id>|<size-bytes>|<content-hash>|<source>|<minimum-vram-mib>` lines
(the committed manifest is simulation-only placeholder metadata).
`--heartbeat-timeout <seconds>` overrides the default for testing. With no
eligible spare, `/providers` reports `replacement: NONE_ELIGIBLE`; a later
eligible join is assigned automatically.

### Real managed provider

Use a manifest with an exact 64-hex SHA-256 and a local path, `file://`,
HTTP, or HTTPS source:

```bash
./build/managed_provider --id node-a --gpu RTX3090 --vram-mib 24576 \
  --cache-dir /var/lib/dan/models \
  --worker /path/to/llama.cpp/build/bin/rpc-server \
  --worker-host 10.0.0.10 --worker-port 50052 --worker-device CUDA0 \
  --host 10.0.0.1 --port 9000
```

Default cache: `~/.dan/models`, layout
`<model>/<version>/<shard>/<sha256>.artifact`. Wildcard worker bindings are
rejected. `CACHED` automatically triggers a load. `/unload <id>` and
`/load <id>` stop/restart only the owned worker, retaining verified disk
cache. HTTP(S) requires `curl`; hashing requires `sha256sum`. Run the local
four-provider validation with `python3 -m unittest -v tests.test_managed_worker`.
Without `--managed-runtime`, the control plane stays readiness-only and
legacy provider/manual-group routing is unchanged.

### Persistent managed serving

```bash
./build/coordinator 9000 \
  --managed-model /path/to/dan-main.manifest \
  --managed-runtime /path/to/llama-server /path/to/dan-main.gguf 18080 \
  --managed-runtime-context 32768
```

Port `18080` binds to loopback. Each managed provider advertises a numeric
private RPC endpoint; once all shards reach `READY`, DAN constructs the RPC
list, `RPC0..RPCN-1` device list, and VRAM-proportional tensor split in shard
order. Wait for `/providers` to show both `replica: READY` and
`runtime: READY`, then:

```text
/model dan-main Explain TCP reliability briefly.
/runtime restart
/runtime stop
/runtime start
```

`/runtime start` is idempotent. Startup/request timeouts default to 30
minutes (`--managed-runtime-timeout`, `--managed-request-timeout`); repeat
`--managed-runtime-arg <value>` for extra llama-server options. Local
persistent acceptance: `python3 -m unittest -v tests.test_persistent_serving`.
The coordinator-local GGUF may be distributed to RPC workers once at server
start and is not resent per user request — managed provider artifacts and
llama.cpp's internal RPC tensor/cache behavior remain distinct layers. If an
assigned provider disappears, DAN stops this runtime, prepares an eligible
spare, and starts a new runtime automatically after replica readiness
returns. Download a compatible instruction-tuned GGUF from a source whose
license you accept; keep weights outside the DAN repository.

### Run with one or more providers

```bash
./build/coordinator
```

```bash
./build/provider /path/to/llama-completion /path/to/model.gguf
```

Default reported ID is `<hostname>-<process-id>`, model name is the GGUF
filename, device/backend default to CPU. Override display metadata:

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

`--id`/`--device`/`--gpu`/`--vram`/`--model-name`/`--backend` are self-reported
and don't change llama.cpp runtime configuration; pass actual runtime
configuration through the llama.cpp build/environment or `--gpu-layers N`,
`--runtime-device CUDA0`, `--ctx-size N`, `--n-predict N`. Persistent
providers default to `--n-predict -1` (end-of-turn generation) — positive
budgets in the pinned interactive runtime span turns and can return empty
responses; they are not a reliable per-request cap. Manual runs have no
generation timeout; the GPU validation driver supplies an external one.

Start additional providers with the same command in additional terminals;
each loads and retains its own model process. The coordinator reports
`Provider 1`, `Provider 2`, etc., logs `Dispatching to provider N`, and
selects round-robin — with three providers the first three prompts dispatch
immediately and run concurrently; further prompts queue FIFO. Responses
include request IDs (completion order can differ from submission order),
followed by that provider's latest elapsed time, completed count, and
average. Type `exit` to drain queued/active requests, send `BYE` to every
live provider, and shut down cleanly.

Real throughput depends on hardware — too few cores can make CPU providers
contend even though inference overlaps; separate GPUs or well-provisioned
machines are the intended way to get parallel throughput. For different
machines, make TCP port `9000` reachable:

```bash
./build/provider /path/to/llama-completion /path/to/model.gguf 192.0.2.10 9000
```

The current protocol is unencrypted and unauthenticated — use only a trusted
network.

### Integrated distributed model over llama.cpp RPC

Configured manually in the coordinator, separate from normal whole-model
provider scheduling. llama.cpp describes its RPC backend as
proof-of-concept, fragile, and insecure — never expose an RPC port to the
internet or an untrusted network.

Build llama.cpp with RPC and the accelerator backend on both worker machines,
revision `95ef7fc16054e63b427a3ef00188e055ef7586d8` on workers and client:

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-rpc \
  -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=OFF
cmake --build /path/to/llama.cpp/build-rpc --config Release -j \
  --target ggml-rpc-server llama-completion
```

Start one RPC worker per machine, binding only a trusted LAN address (the
pinned server defaults to loopback and requires numeric IPv4; clients support
IPv4 hostnames but not IPv6 endpoints). Allow client access to TCP port 50052
through private networking/firewall rules. Expose exactly one CUDA device per
endpoint — DAN generates `RPC0,RPC1` from endpoint count, while llama.cpp
numbers every exposed device, so multiple devices on one worker could put
both selected devices on the same machine:

```bash
# Provider A
/path/to/build-rpc/bin/ggml-rpc-server \
  --host 192.168.1.21 --port 50052 --device CUDA0

# Provider B
/path/to/build-rpc/bin/ggml-rpc-server \
  --host 192.168.1.22 --port 50052 --device CUDA0
```

The client needs an RPC-enabled `llama-completion` (CUDA on the client is
optional — `-DGGML_RPC=ON -DGGML_CUDA=OFF` suffices for a CPU client), with
all GGUF shards local. Workers don't need their own GGUF copies. Runtime
invocation uses `execvp`, not shell parsing.

For a controlled 32,768-context test, set the inherited environment on the
client before starting either DAN entry point (and `GGML_RPC_NO_RDMA=1` on
the workers too, when measuring plain TCP):

```bash
export LLAMA_ARG_CTX_SIZE=32768
export LLAMA_ARG_SPLIT_MODE=layer
export LLAMA_ARG_FIT=off
export GGML_RPC_NO_RDMA=1
```

Verify device mapping before inference (require `RPC0` on worker A, `RPC1`
on worker B):

```bash
/path/to/build-rpc/bin/llama-completion \
  --rpc 192.168.1.21:50052,192.168.1.22:50052 --list-devices
```

Then start the group (final argument is a tensor split like `1,1`, or `auto`):

```bash
./build/coordinator 9000 \
  --group large-model-group \
  /path/to/build-rpc/bin/llama-completion \
  /path/to/large-model.gguf \
  192.168.1.21:50052,192.168.1.22:50052 \
  1,1
```

Repeat `--group` for more groups. Route explicitly with
`/group large-model-group Explain distributed inference.`; ordinary prompts
keep using normal providers, and group prompts share the same request-ID
space and may run concurrently with them. A registry model may back a group
too: `--group large-group /path/to/rpc-llama-completion @dan-large ...`.

The standalone `distributed_model_experiment` command remains available for
isolated diagnostics, using the same shared runtime implementation:

```bash
set -o pipefail
printf '%s\n' 'Explain TCP reliability in two sentences.' |
  timeout --kill-after=30s 30m ./build/distributed_model_experiment \
  /path/to/build-rpc/bin/llama-completion /path/to/large-model.gguf rpc-pair \
  192.168.1.21:50052,192.168.1.22:50052 --tensor-split 1,1 \
  2>&1 | tee standalone-rpc.log
```

Use an external timeout for coordinator test runs too — DAN has no internal
distributed request timeout, and `exit` drains work and may wait indefinitely
on a stalled runtime. Keep RPC servers alive throughout client runs; inspect
response IDs and error logs, not just exit status. Avoid concurrent
groups/providers on the same GPUs (DAN doesn't reserve shared worker memory).

For normal operation, `/model large-model.gguf Explain TCP briefly.` selects
by model compatibility, then availability, then round-robin — a provider
matches its reported `--model-name`; a group matches its configured path or
filename. If nothing serves the model, DAN reports the request ID and
continues other work. `/group` remains a diagnostic override; plain prompts
stay backward compatible.

**Aggregate-VRAM acceptance requirements** (the two-GPU smoke test passed
without source changes; full record in [PROJECT_STATUS.md](PROJECT_STATUS.md)):

- First prove participation with a supported model through the standalone
  experiment and DAN group — a model that fits on one GPU alone doesn't
  establish aggregate-memory necessity.
- Choose a model with at most 99 offloadable layers including its output
  layer — the shared runtime hard-codes 99 GPU layers, a 256-token cap, and
  non-conversation mode; it is not the persistent chat path.
- Inspect placement/offload logs for unintended host-resident layers —
  llama.cpp's layer split distributes weights and KV across selected RPC
  devices, but client CPU work and buffer fallbacks still exist; endpoint
  labels and successful output alone don't prove full intended offload.
- Run identical model/context settings against A alone and B alone
  (`llama-completion --rpc <one-endpoint> --device RPC0 --n-gpu-layers 99`,
  same prompt/cap/fit) as controls — both DAN entry points require at least
  two endpoints, so use direct llama.cpp for the single-GPU comparison.
- Capture server startup/device logs, client placement logs, and timestamped
  GPU samples on both nodes:

```bash
nvidia-smi -i 0 \
  --query-gpu=timestamp,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv --loop-ms=200 > gpu-samples.csv
```

Require VRAM increases and activity on both GPUs correlated with the same
inference (layer splitting need not show simultaneous utilization in every
sample — DAN has no built-in per-GPU telemetry). Record load time and
generation token rates from llama.cpp diagnostics, network RTT/traffic with
external tools. Each group request starts a fresh runtime, so measured
latency includes process startup, model distribution/loading, prompt
evaluation and generation but excludes queue wait — not directly comparable
to a persistent-provider average; RPC cache and OS cache state also affect
load timing. Compare an identical fixed-token prompt locally, with one RPC
endpoint, and with both; also record LAN link speed/utilization — if
generation speed falls substantially while the link is saturated or
latency-sensitive, network communication is the likely bottleneck.

## Friends testnet guide

Gamer Provider Testnet v1 supports Linux PCs with an NVIDIA GPU on a trusted
private network. It does not expose DAN safely to the public internet — do
not forward coordinator or RPC ports through a router.

**What you need:** a Linux PC with a working NVIDIA driver and `nvidia-smi`; a
private overlay connection to the coordinator (such as Tailscale); CMake, a
C++23 compiler, Git, `curl`, `sha256sum`; a llama.cpp `rpc-server`
executable; free disk space at least as large as any assigned artifact.

```bash
nvidia-smi
tailscale ip -4
```

The advertised address must be numeric loopback, RFC1918 private IPv4, or the
Tailscale `100.64.0.0/10` range — DAN rejects wildcard and public advertised
addresses. Firewall policy must allow the coordinator to reach the selected
RPC worker port over the private overlay.

**Build once:**

```bash
git clone <TOMERS_DAN_REPOSITORY_URL> DAN
cd DAN
cmake -S . -B build
cmake --build build -j
```

Build the pinned llama.cpp RPC worker outside DAN if not given a compatible
executable:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
git -C ~/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build -j --target rpc-server
```

**Configure and start.** Get the coordinator's private `HOST:PORT`, then:

```bash
./scripts/setup_provider.sh \
  100.80.10.1:9000 \
  "$(tailscale ip -4)" \
  "$HOME/llama.cpp/build/bin/rpc-server"
```

The script checks prerequisites, builds DAN, creates `~/.dan/provider.conf`
only when absent, validates GPU detection, and prints the launch command — it
does not install drivers, alter networking, require root, or overwrite an
existing config. Start later with `./build/dan-provider`, which reads
`~/.dan/provider.conf`, chooses the lowest NVIDIA GPU index, reserves
1536 MiB by default, chooses an available worker port, creates a stable
random ID in `~/.dan/provider-id`, and joins the pool. `Ctrl-C` stops
contributing; DAN stops only its owned RPC worker.

Expected startup:

```text
DAN Provider

Provider ID: node-7f3a12c04b91
Name: tomer-pc
GPU: NVIDIA GeForce RTX 2070
GPU UUID: GPU-...
Device: CUDA0
VRAM total: 8192 MiB
Reserved: 1536 MiB
Available to DAN: 6656 MiB
Coordinator: 100.80.10.1:9000
Cache: /home/tomer/.dan/models
Worker: STOPPED (100.80.10.4:54321)
Coordinator: CONNECTED
Role: SPARE
```

If assigned, later output shows `Assignment`, `DOWNLOADING`, `CACHED`,
`LOADING`, `READY`. `/providers` at the coordinator shows persistent ID,
optional name, GPU, advertised VRAM, `ASSIGNED`/`SPARE` role, shard state,
worker endpoint, last-seen age.

**Configuration** (`~/.dan/provider.conf`, one `key=value` per line):

```ini
coordinator=100.80.10.1:9000
provider_name=tomer-pc
cache_dir=/home/tomer/.dan/models
rpc_worker=/home/tomer/llama.cpp/build/bin/rpc-server
advertise_host=100.80.10.4
device=0
reserve_vram_mib=1536
reconnect_seconds=2
```

`device`/`provider_name` are optional (without `device`, the lowest reported
GPU index is chosen deterministically); `worker_port` is optional for a fixed
private-overlay port. `./build/dan-provider --help` lists CLI overrides;
`./build/dan-provider --check` validates without connecting or starting a
worker. Reconnect delay doubles up to 30 seconds during an outage and resets
after successful registration — same process, ID, verified cache, and
compatible running worker are reused after recovery.

**Cache and disk use.** Verified artifacts live under
`~/.dan/models/<model>/<version>/<shard>/<sha256>.artifact` and survive
restarts (stopping DAN does not delete them). Inspect with `du -sh
~/.dan/models`. Tell each participant the maximum expected artifact size
beforehand. Cache eviction is manual in v1 — stop `dan-provider` before
deleting files.

**Changing contribution limits.** Edit `reserve_vram_mib` and restart — the
advertised value is detected total VRAM minus this reserve, which must be
smaller than total VRAM (static; DAN doesn't yet notice a game starting or
dynamically reduce allocation). Set `device` to another installed GPU's
`nvidia-smi` index.

**Troubleshooting:** `NVIDIA GPU detection failed` → run `nvidia-smi`, repair
the driver outside DAN. `Configured CUDA device ... was not reported` →
remove/correct `device=`. `VRAM reserve must be smaller` → lower
`reserve_vram_mib`. Private-address error → use the numeric overlay address,
never `0.0.0.0` or a public address. Can't connect → confirm coordinator
address, overlay connectivity, firewall (reconnection is automatic). Worker
failure → confirm `rpc_worker` is executable and CUDA llama.cpp libraries are
available. Collect a shareable log without credentials:

```bash
mkdir -p ~/.dan/logs
./build/dan-provider 2>&1 | tee ~/.dan/logs/provider-$(date +%Y%m%d-%H%M%S).log
nvidia-smi -q > ~/.dan/logs/nvidia-smi-q.txt
```

Share the provider log, `nvidia-smi` output, DAN/llama.cpp commits, and
`du -sh ~/.dan/models` — never VPN credentials or private keys.

**First home-GPU validation.** Use one machine as A, at least one friend's
NVIDIA GPU as B, and (when available) a third as an eligible spare C — keep
all RPC endpoints on the private overlay:

1. Start the coordinator, record its commit and private address.
2. Run one `dan-provider` command on each PC.
3. Record detected total/advertised VRAM, persistent IDs, roles.
4. Wait for every assigned artifact and worker, then replica/runtime `READY`.
5. Send ten real prompts, retain request latency and tokens/second evidence.
6. Confirm runtime/worker PIDs stay unchanged and nothing re-downloads.
7. Stop one required provider — with C, require automatic replacement and
   recovery; without it, record `NONE_ELIGIBLE`.
8. Restart the original provider, confirm it returns as a spare with the
   same ID/cache without stealing the healthy assignment.
9. Compare a strong GPU alone, strong+weaker, and all available GPUs — record
   VRAM contribution, CUDA utilization, throughput, network traffic, recovery
   time. Do not assume the weaker GPU improves performance.

This is a trusted friends testnet, not a production security boundary.

## Rented GPU validation: one CUDA provider

Use Ubuntu 24.04 x86-64 with a working NVIDIA driver and CUDA development
toolkit (including `nvcc`). Keep coordinator and provider on the rental;
connect by SSH; do not expose DAN ports publicly. Allow 30 GB+ temporary disk
space for sources, build products, weights, and evidence.

The commands below use the original Qwen2.5-7B/L4 recipe as an example
deployment, not a reproduction of the A40 result in "Status" above — an
L4-specific run is unverified.

**Before renting:** use `https://github.com/Tomer-R-Cohen/DAN.git` and record
the checked-out revision (the A40 benchmark used `c5b3bfd`; `e2cbd2b` added
its evidence archive). Choose a CUDA **development** image, not just one with
the driver. Choose a candidate/quantization/download location/license
beforehand. Keep SmolLM2 test-only — the recipe below uses
Qwen2.5-7B-Instruct Q4_K_M as a conservative, serious GPU acceptance baseline,
not a best-quality claim; swap the candidate through the registry for later
comparisons. Reserve time to copy logs off before ending the rental.

### 1. Build prerequisites and DAN

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git curl ca-certificates \
  libcurl4-openssl-dev libssl-dev python3 python3-venv pkg-config time
git clone https://github.com/Tomer-R-Cohen/DAN.git DAN
cd DAN
export DAN_ROOT="$PWD"
mkdir -p results
git rev-parse HEAD | tee results/dan-revision.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

### 2. Verify hardware and build the pinned llama.cpp revision

```bash
nvidia-smi | tee results/nvidia-smi.txt
nvcc --version | tee results/nvcc.txt
```

`nvidia-smi`'s CUDA version is driver compatibility, not proof of an
installed compiler. If `nvcc` is missing, pick the rental's CUDA development
image or, on Ubuntu 24.04 with a working driver: `sudo apt-get install -y
nvidia-cuda-toolkit`. Build this workspace's current llama.cpp revision, not
a moving upstream head — a later runtime revision requires rerunning both
direct and DAN tests:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
export LLAMA_ROOT="$(realpath ../llama.cpp)"
git -C "$LLAMA_ROOT" checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C "$LLAMA_ROOT" rev-parse HEAD | tee results/llama-revision.txt
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build-cuda" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build "$LLAMA_ROOT/build-cuda" -j 4 --target llama-completion
export LLAMA_BIN="$LLAMA_ROOT/build-cuda/bin/llama-completion"
"$LLAMA_BIN" --list-devices 2>&1 | tee results/devices.txt
```

Require a CUDA device (normally `CUDA0`). Retry with `-j 2` if build memory
is exhausted. Reference:
[official CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda).

### 3. Place weights and configure dan-main

```bash
mkdir -p ../models
export MODEL_DIR="$(realpath ../models)"
for part in 00001 00002; do
  curl -fL --retry 3 -C - \
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-${part}-of-00002.gguf" \
    -o "$MODEL_DIR/qwen2.5-7b-instruct-q4_k_m-${part}-of-00002.gguf"
done
export MODEL_PATH="$MODEL_DIR/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
export MODEL_NAME="$(basename "$MODEL_PATH")"
sha256sum "$MODEL_DIR"/qwen2.5-7b-instruct-q4_k_m-*.gguf | tee results/weights.sha256
du -ch "$MODEL_DIR"/qwen2.5-7b-instruct-q4_k_m-*.gguf | tee results/weights-size.txt
python3 - <<'PY'
import os
from pathlib import Path
test = Path('config/models.example.conf').read_text().splitlines()[1]
main = '|'.join(['dan-main', 'main', 'Qwen2.5', os.environ['MODEL_PATH'],
                 os.environ['MODEL_NAME'], 'Q4_K_M', '8192', '4096',
                 'CUDA single GPU validation; memory estimate only', 'yes', 'no'])
Path('config/models.gpu.conf').write_text(test + '\n' + main + '\n')
PY
cp config/models.gpu.conf results/models.gpu.conf
```

`4096` is the **configured test context**, not the model's advertised
maximum; the memory field is an estimate, not a fit guarantee. All shards are
required — no weights ship with DAN.

### 4. Direct CUDA inference first

```bash
set -o pipefail
"$LLAMA_BIN" --model "$MODEL_PATH" --device CUDA0 --gpu-layers 999 \
  --ctx-size 4096 --n-predict 256 --conversation --single-turn --simple-io \
  --no-display-prompt --color off \
  --prompt 'Explain TCP reliability in two sentences.' \
  2>&1 | tee results/direct.log
```

Require a useful response, exit 0, CUDA/offload evidence in logs, and
increased VRAM — a CUDA metadata label alone doesn't prove GPU execution.
Record llama.cpp's load/eval timings when emitted. On OOM, lower context or
quantization, record the change, rerun. Don't count an unnoticed CPU
fallback as GPU success.

### 5. Ten sequential prompts through DAN (recommended)

```bash
python3 scripts/validate_gpu.py --registry config/models.gpu.conf \
  --runtime "$LLAMA_BIN" --output results/dan-gpu
```

Starts the real coordinator and provider, waits for registration, sends one
`/model dan-main` request at a time, retains answers, provider startup logs,
per-request latencies, exit codes, GPU info, and one-second VRAM/utilization
samples; sends `exit` after ten responses. Failures/timeouts leave partial
results and terminate process groups. Use a new output directory per run —
the session retains conversation history, so these are session timings, not
ten independent cold prompts. Persistent runs use `--n-predict -1`; the
direct one-shot test above still caps at 256 tokens. Do not equate ten
nonempty answers with a quality pass.

### 6. Manual terminal operation (alternative)

Do not run alongside the driver above. Terminal 1:

```bash
./build/coordinator 9000 --models config/models.gpu.conf 2>&1 | tee results/coordinator-manual.log
```

Terminal 2 (same `LLAMA_BIN`/`MODEL_PATH`/`MODEL_NAME` as above):

```bash
./build/provider "$LLAMA_BIN" "$MODEL_PATH" 127.0.0.1 9000 \
  --id l4-validation --model-name "$MODEL_NAME" --backend CUDA --device GPU \
  --gpu "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" \
  --vram "$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader)" \
  --runtime-device CUDA0 --gpu-layers 999 --ctx-size 4096 --n-predict -1 \
  2>&1 | tee results/provider-manual.log
```

Enter `/model dan-main Explain TCP reliability.` in terminal 1; the ten
representative prompts are in `scripts/validate_gpu.py`. `exit` drains work
and shuts down the provider.

### 7. Review evidence and shut down

- [ ] GPU name, total/free VRAM, driver and CUDA compiler recorded.
- [ ] DAN revision, llama.cpp revision, GGUF hashes recorded.
- [ ] Model ID/family/quantization, total shard size, actual context recorded.
- [ ] Direct CUDA inference succeeded; provider startup and offload verified.
- [ ] Provider loaded once and received ten prompts; ten IDs have real responses.
- [ ] Latencies and observed VRAM peak recorded; answers reviewed for truncation/errors.
- [ ] Token rate recorded from direct llama.cpp diagnostics, or explicitly unavailable.
- [ ] Runtime exits and any crashes/errors recorded, including failed attempts.
- [ ] Results copied off temporary storage before ending the rental.

Copy the "Results report template" block below into `results/report.md` and
fill it in, then:

```bash
tar -czf dan-gpu-results.tar.gz results
```

Use `vram.csv` for observed peak memory (one-second sampling may miss short
peaks). DAN doesn't currently return generated-token counts, so don't infer
tokens/s from words or characters — direct llama.cpp token rates and DAN
latency measure different things; provider ready time includes
loading/warmup, not pure disk I/O.

From your own machine: `scp USER@RENTAL:/path/to/DAN/dan-gpu-results.tar.gz .`,
confirm the download, check `nvidia-smi` for leftover model processes, then
terminate the rented instance through its control panel — DAN does not stop
billing on its own.

**Known review limitations:** the interactive subprocess adapter separates
answers using llama.cpp's `\n> ` display marker; a generated identical marker
can confuse it. Handshake and frame reads still block — stalled peers have no
core request timeout; the validation driver's timeout bounds a test run, and
these limitations must be recorded if encountered. RPC memory-fit and network
bottleneck claims remain outside this single-GPU test.

### Results report template

Copy the block below to `results/report.md` and fill it in before setting its
status away from `NOT RUN`. A CPU smoke run or self-reported CUDA metadata is
not a GPU acceptance pass.

```text
Status: NOT RUN (change only after collecting evidence)
Date / operator / rental image:
DAN revision / llama.cpp revision:
GPU name / driver / CUDA compiler:
Total VRAM / free VRAM before load / observed peak VRAM:
Model ID / role / family:
GGUF filename(s) / source / SHA256 / quantization / total bytes:
Configured context / token cap / offloaded layers:
Direct load success or failure / load time / generated tokens per second:
DAN runtime ready time / CUDA offload evidence:
DAN responses received (out of 10) / process exit codes:
Per-request latency: attach results.json and coordinator.log
Answer quality / truncation / context carryover observations:
Crashes / inference errors / OOM / retries and configuration changes:
Artifacts: devices.txt, direct.log, provider.log, coordinator.log, vram.csv
Conclusion: pass / fail / inconclusive; explain GPU execution evidence.
```
