# Operations

How to build, run, and operate DAN's provider-owned path. For why it's built
this way, see [ARCHITECTURE.md](ARCHITECTURE.md); for current status and the
beta launch gate, [PROGRESS.md](PROGRESS.md); for physical-test evidence,
[TESTS_AND_STATS.md](TESTS_AND_STATS.md).

The coordinator is a C++23 metadata/router process and never opens a GGUF.
Stage A owns the embedding and Qwen2 layers `0..11`; stage B owns layers
`12..23`, final norm, output head, and greedy sampling. Each worker loads its
assigned tensors once and uses llama.cpp sequence IDs to keep independent KV
state in one shared context. DAN's existing llama.cpp RPC path remains
available as a legacy/reference fallback (see
[`reference/design/legacy-path.md`](reference/design/legacy-path.md)); normal
provider-owned inference does not require Python.

## Build

The script checks out the pinned llama.cpp revision, applies the existing small
provider-owned patch, and builds the worker, C++ coordinator, and protocol test.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_provider_owned.ps1 -Cuda
```

Linux uses the same CMake project:

```bash
git clone https://github.com/ggml-org/llama.cpp.git build/provider-owned-v1/llama.cpp
git -C build/provider-owned-v1/llama.cpp checkout 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C build/provider-owned-v1/llama.cpp apply "$PWD/patches/llama-provider-owned.patch"
cmake -S . -B build/provider-owned-v1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR="$PWD/build/provider-owned-v1/llama.cpp" \
  -DGGML_CUDA=ON -DGGML_CCACHE=OFF
cmake --build build/provider-owned-v1 -j"$(nproc)" \
  --target dan-stage-worker dan-provider-owned-coordinator provider_owned_protocol_test provider_owned_concurrency_client
ctest --test-dir build/provider-owned-v1 --output-on-failure \
  -R '^provider_owned_protocol_test$'
```

The DAN targets compile as C++23. The pinned llama.cpp dependency retains its
own supported C++17 compilation mode. Note: `git -C <dir> apply <patch>`
resolves the patch path relative to `<dir>`, not your cwd — always pass an
absolute path (`"$PWD/patches/..."`) or it silently looks in the wrong place.

## Run manually

Start each worker once. They remain READY across requests and coordinator
disconnects until they receive an explicit shutdown frame.

For range-backed storage, start with a model path that does not exist and pass
the pinned source identity from the manifest. The worker fetches only its GGUF
header/index and assigned tensor ranges, then reuses the verified sparse cache
on later starts:

```powershell
$Revision = '9217f5db79a29953eb74d5343926648285ec7e67'
$Sha256 = '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
$Url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/$Revision/qwen2.5-0.5b-instruct-q4_k_m.gguf"

.\dan-stage-worker.exe --model .\cache\provider-a\qwen.gguf `
  --model-url $Url --model-revision $Revision --model-sha256 $Sha256 `
  --stage-start 0 --stage-end 12 --host 127.0.0.1 --port 50101 `
  --ctx 512 --gpu-layers 999 --max-sessions 8
```

Provider B uses a different cache path, `--stage-start 12 --stage-end 24`, and
its own listening address. Never point range mode at an existing untracked full
GGUF; the worker refuses to overwrite it.

```powershell
# Provider A
.\dan-stage-worker.exe --model .\qwen.gguf --stage-start 0 --stage-end 12 `
  --host 127.0.0.1 --port 50101 --ctx 512 --gpu-layers 999 --max-sessions 8

# Provider B (use its private Tailscale address when remote)
.\dan-stage-worker.exe --model .\qwen.gguf --stage-start 12 --stage-end 24 `
  --host 127.0.0.1 --port 50102 --ctx 512 --gpu-layers 999 --max-sessions 8
```

Run stateless requests from the metadata-only coordinator:

```powershell
.\dan-provider-owned-coordinator.exe `
  --manifest .\config\provider-owned-qwen2.5-0.5b-q4km.json `
  --provider-a 127.0.0.1:50101 --provider-b 127.0.0.1:50102 `
  --prompt 'The capital of France is' --tokens 20 --requests 100 `
  --report .\provider-owned-v1-report.json
```

Use one session for follow-ups by repeating `--prompt` and adding
`--persistent`. Use `--resident-sessions N` to retain multiple independent
sessions and route requests round-robin. `--reset-between` resets each reused
session before its next request. `--shutdown-workers` gracefully stops both
workers after the run.

The trusted-Tailscale assumption remains for this static-address mode: never
expose worker ports publicly. Only one request executes at a time; GPU
batching, simultaneous replica execution, provider replacement, physical GGUF
shards, and KV migration remain out of v2.

Start bounded concurrent serving instead of batch CLI mode:

```powershell
.\dan-provider-owned-coordinator.exe `
  --manifest .\config\provider-owned-qwen2.5-0.5b-q4km.json `
  --provider-a 127.0.0.1:50101 --provider-b 100.68.128.88:50102 `
  --listen 127.0.0.1:50100 --queue-capacity 128 --client-threads 64
```

A provider can host that coordinator role alongside its own GPU stage through
the normal launcher. Other providers join the advertised private-network address
on port 50201; clients use port 50100:

```powershell
.\dan-provider.exe `
  --host-coordinator .\config\provider-owned-qwen2.5-0.5b-q4km.json `
  --provider-listen 0.0.0.0:50201 --serve 0.0.0.0:50100 `
  --advertise-host 100.64.0.10 --provider-name head-provider
```

The provider owns the coordinator child process and stops it when the provider
exits. This is role co-location, not election: run one host coordinator per swarm.

### Ring topology (activation hot-path bypass)

To form the stage ring automatically, expose one reachable listener per provider
and a return listener on the coordinator host:

```powershell
dan-provider-owned-coordinator.exe ... --provider-listen 0.0.0.0:50201 `
  --listen 0.0.0.0:50100 --ring-return 0.0.0.0:50205
dan-stage-worker.exe ... --coordinator COORDINATOR_IP:50201 `
  --ring-listen THIS_PROVIDER_IP:50202
```

Each worker advertises its ring endpoint and receives its next hop in the stage
assignment. Use private or authenticated libp2p-forwarded endpoints, not public
unauthenticated worker sockets. One configuration of this is proven over real
WAN (SSH tunnel, fixed addressing); another is still broken (libp2p relay
circuit with auto-registration and `--provider-peer-auth`). Check
[TESTS_AND_STATS.md](TESTS_AND_STATS.md#the-ring-mode-gap-precisely) before
relying on ring mode over an untrusted network.

With packaged `network=libp2p`, the launcher creates the local ring listener and
proxy automatically. Assignments contain successor multiaddresses and expected
predecessor PeerIDs, so activation traffic goes directly between authenticated
provider sidecars instead of through the coordinator's control tunnel.

The Windows contributor launcher automatically supplies
`--ring-listen TAILSCALE_IP:50202` when `network=tailscale`; `ring_port` can
override that port in `provider.conf`. Start the packaged coordinator with
`Start-DAN-Service.ps1 -ProviderNetwork tailscale` to bind its encrypted direct
ring return on port `50205`.

### Acceptance client and gateway

The v2 acceptance client command is:

```powershell
.\provider_owned_concurrency_client.exe --coordinator 127.0.0.1:50100 `
  --clients 20 --sessions 50 --requests 1000
```

Expose the binary service through the DAN HTTP API gateway:

```powershell
$env:DAN_API_KEY = 'replace-with-a-long-random-secret'
.\dan-api-gateway.exe --listen 0.0.0.0:8080 `
  --coordinator 127.0.0.1:50100 --model qwen2.5-0.5b-instruct-q4-k-m
```

Repeat `--coordinator` to round-robin across independently formed replicas.
Before response content is emitted, the gateway also skips unreachable,
reforming, failed, or saturated replicas:

```powershell
.\dan-api-gateway.exe --listen 127.0.0.1:8080 `
  --coordinator 127.0.0.1:50100 --coordinator 127.0.0.1:50101 `
  --model qwen2.5-0.5b-instruct-q4-k-m
```

It serves `GET /health`, authenticated `GET /metrics`, `GET /v1/models`, and
`POST /v1/chat/completions`; set `"stream": true` for token-by-token SSE.
`GET /health` returns `200` only while the coordinator has an available replica,
and `503` while the ring is unavailable or reforming.
`GET /metrics` exports per-replica Prometheus samples for availability, queues,
latency, requests, reformations, and generated-token throughput.
Non-loopback listening fails closed unless an API key is configured. Put TLS at
the reverse proxy; do not expose plaintext HTTP directly to the internet.
Provider packages include the same executable under `runtime` so a provider
hosting the coordinator can expose that service too.

See [the v1 implementation report](reference/design/runtime-v1.md) for
protocol, lifecycle, correctness, and benchmark details, [the v2 report](reference/design/runtime-v2.md)
for concurrency results, and [the range-backed storage report](reference/design/range-backed-storage.md)
for storage and integrity acceptance results.

## Running the packaged service

Run `Start-DAN-Service.cmd`. Each run opens an interactive picker
(`Select-DAN-Model.ps1`) showing every bundled model config (`config\models\*.json`)
so you can choose the target model (split across providers, served over the WAN)
and, optionally, a smaller same-family draft model for speculative decoding —
press Enter on either prompt to keep the last choice, so a routine restart is
just two Enters. The chosen target becomes `config\active-model.json`; a chosen
draft model is downloaded once (checksum-verified) into `data\draft-model\` and
wired in automatically via `--draft-model`/`--draft-tokens 4`/`--pipeline-depth`.
Picking "none" (or a download failure) falls back to non-speculative decoding
without failing the launch. Pass `-NoPicker` to skip the prompts entirely and
reuse the last saved selection (`config\selection.json`) — use this for
unattended restarts. `dan_speculative_enabled`, `dan_pipeline_depth`, and the
draft-token counters confirm whether speculation is actually active; ring
readiness alone does not. CUDA graph reuse is compiled in by default for every
release build (`build_provider_owned.ps1 -Cuda` sets `GGML_CUDA_GRAPHS=ON`);
there is no separate flag to enable it.

After the picker, the coordinator and gateway launch in the background and, by
default, a live terminal dashboard (`Show-DAN-Dashboard.ps1`) takes the
foreground, refreshed once a second: readiness, per-replica queue depth,
tokens/sec, time-to-first-token, p50/p95 latency, queue wait, a compute-vs-network
per-token time breakdown (which one is the actual bottleneck — GPU or WAN), and
speculative-decoding stats. Pass `-NoDashboard` to fall back to plain logs.
Readiness is `GET /health`; it returns `200` only when at least one configured
replica is available. Authenticated `GET /metrics` uses Prometheus text format
(the dashboard is just a terminal view of the same endpoint, including the new
`dan_queue_wait_ms`, `dan_time_to_first_token_ms`,
`dan_provider_a/middle/b_compute_ms_per_step`, and `dan_network_ms_per_step`
series). Configure an external scraper with the same bearer token as
`DAN_API_KEY`, then load [`dan-alerts.yml`](../deploy/prometheus/dan-alerts.yml).
The Windows launcher prevents automatic system sleep while the service is
running; it does not prevent an operator-requested sleep, hibernation,
shutdown, or power loss.

### Multi-turn conversations reuse the coordinator's resident session

`dan-api-gateway` used to treat every chat request as stateless: each new
message resent the full conversation and the coordinator reprocessed it from
an empty KV cache. It now recognizes when a request's history is exactly a
prior turn's history plus new messages, and continues that turn's coordinator
session instead — sending only the new text and resuming its resident KV cache,
rather than restarting the conversation from scratch on every message. A new
conversation (or one the gateway doesn't recognize, e.g. after a restart)
transparently falls back to starting a fresh session. Idle sessions are
destroyed after 10 minutes to free their KV. This is purely a gateway-side
change (`sidecar/cmd/dan-api-gateway/main.go`); no coordinator or protocol
change was needed since session support already existed there.

### Onboarding a friend's GPU

Send them one command instead of a zip to unpack by hand (requires the release
archives to be published as GitHub Releases assets first — see
[Release packaging notes](#release-packaging-notes)):

```powershell
irm https://raw.githubusercontent.com/Tomer-R-Cohen/DAN/main/scripts/Install-DAN-Provider.ps1 | iex
```

`Install-DAN-Provider.ps1` downloads the latest `DAN-Provider-*-Windows-x64.zip`
from GitHub Releases, verifies its published `.sha256`, extracts it under
`%LOCALAPPDATA%\DAN\app`, adds a desktop shortcut, and launches `dan-provider.exe`
— which runs its own first-run wizard (GPU detection, private-network onboarding,
then the live TUI dashboard already described in
[the contributor package guide](reference/operations/friends-testnet-windows.md)).
They only need to paste the coordinator address you give them.

## Horizontal scale

Run one coordinator package and provider pool per independent replica group,
using the same model and strong API key. Keep each coordinator's binary port on
loopback, expose each DAN gateway only through TLS, and put a standard HTTP load
balancer in front of the gateways. Route traffic only to instances whose
`/health` returns `200`. This is the recommended multi-host layout because a
provider or coordinator failure removes only one replica group.

The gateway can alternatively balance several coordinators itself by repeating
`-coordinator`, but those binary endpoints have no application-layer
authentication and must be reachable only through an authenticated encrypted
private network:

```powershell
.\dan-api-gateway.exe -listen 127.0.0.1:8080 -model MODEL_ID `
  -coordinator 10.0.0.11:50100 -coordinator 10.0.0.12:50100
```

Do not expose coordinator port `50100` to the public internet. DAN chat requests
carry their full message history and are stateless at this gateway boundary;
coordinator-resident sessions are not shared across replica groups.

Release logs are under `data\logs` beside the coordinator package. Provider logs
are under `%LOCALAPPDATA%\DAN\logs`. Preserve the first error and the preceding
formation events when reporting an incident.

## Soak testing

On an otherwise idle release deployment, run the gate for 24 hours while deliberately stopping and restarting
providers. A request interrupted by provider loss is counted as an outage and
retried after readiness returns. The gate fails if readiness does not recover
within three minutes, any unrelated request fails, or a stateless request leaves
resident session/KV allocation behind:

```powershell
$env:DAN_API_KEY = 'the-service-key'
.\Test-DAN-Soak.ps1 -ExpectedOutages 1
```

Use the default of zero when no deliberate churn is scheduled. The gate fails
if the observed outage count differs, so an unrelated transient outage cannot
be silently accepted.

**Known gap**: the checker only writes progress to `Write-Host`, never to a
file. A 24-hour run started 2026-09-12 lost its entire result when the console
closed. Redirect output (`Tee-Object`/`Start-Transcript`) before running it
again — see [PROGRESS.md](PROGRESS.md#private-beta-launch-gate).

## Alert actions

- `DANNoReplica`: inspect coordinator formation logs, then provider sidecar and
  worker logs. Confirm the advertised VRAM covers the active manifest.
- `DANCoordinatorDown`: restart only that coordinator group; other configured
  coordinators continue receiving requests.
- `DANQueueSaturated`: add an independent replica group and repeat its
  `-coordinator` address at the gateway. Raising the queue only delays rejection.
- `DANRequestFailures`: check provider disconnects, queue timeouts, and client
  cancellations separately before changing capacity.
- `DANReplicaChurn`: remove the unstable provider and verify its network path,
  GPU driver, power, and model cache before returning it to service.

## Backup, upgrade, and rollback

Back up `%LOCALAPPDATA%\DAN\coordinator\identity.key`; losing it changes the
coordinator PeerID and invalidates distributed provider packages. Back up
`%LOCALAPPDATA%\DAN\identity.key` and `provider-v1.1.conf` on each provider.
Model ranges are verified caches and can be downloaded again. Active sessions
and queued requests are memory-only and cannot be restored.

To upgrade, keep the old extracted directory, stop the service, extract the new
archive beside it, copy only `config\active-model.json` and `config\relays.txt`
when customized, then start the new service and wait for `/health` to return
`200`. Roll back by stopping it and starting the untouched old directory. The
identities and provider model caches live outside the release directories.

## Provider onboarding (Windows contributor package)

End-user setup, troubleshooting, and package compatibility for the Windows
contributor package are in
[`reference/operations/friends-testnet-windows.md`](reference/operations/friends-testnet-windows.md)
— it's also packaged as that release's `README.txt` (see `CMakeLists.txt`'s
`install(FILES ...)`), so it's kept lean and end-user-facing rather than
merged into this operator-facing document.

## Release packaging notes

See [`reference/releases/coordinator-v1.0.1.md`](reference/releases/coordinator-v1.0.1.md)
and [`reference/releases/contributor-v1.0.1.md`](reference/releases/contributor-v1.0.1.md)
for the Windows Release v1.0.1 packaging process and acceptance checklist.
