# Pipelined Speculative Decoding + Ring Topology: Physical Multi-GPU Test

This is the physical test for the work in
[docs/PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md): pipelined
speculative decoding (`--pipeline-depth`), the served path running it, direct
stage-to-stage ring return (`--next`/`--ring-listen`/`--ring-return`), and
chunked pipelined prefill (`--prefill-chunk`). Everything in that document was
verified on one machine — CPU loopback for phases 1-5, one local GPU for phase
6. Nothing in it has been measured over a real network between two different
machines, which is the entire premise those phases exist for (the round trip
is the thing pipelining hides). That is what this test measures.

Follow the same Windows + Linux CUDA pattern as
[docs/PROVIDER_OWNED_V2_WINDOWS_LINUX_TEST.md](PROVIDER_OWNED_V2_WINDOWS_LINUX_TEST.md):
the Windows RTX 2070 runs the coordinator and stage 0 (first half of the
model); a rented Linux NVIDIA GPU runs stage 1 (second half, and the tail).
Networking uses [`dan-sidecar`](P2P_TRANSPORT.md), DAN's own encrypted
libp2p transport, not Tailscale — see "Why dan-sidecar, and what's untested
about it" below before assuming this part just works.

Every command below is also available through a script —
`scripts\pipelined_ring_test_windows.ps1` on Windows,
`scripts/pipelined_ring_test_linux.sh` on Linux. Run either with no arguments
for the list. Both reduce to three real phases: `prereqs`, `build`, and
`run` (or `run-ring` for section 8). `run`/`run-ring` do everything needed
to execute a config in one command — write the manifest, download weights
if the config needs them, print this machine's PeerID, start the sidecar
tunnel(s) and any local provider in the background, run the coordinator or
provider in the foreground, and clean up the background processes
afterward. `check-chunks`/`gpu-stats`/`stats`/`package` are evaluation
utilities that stay separate. PeerID/address exchange between the two
machines still has to be copy-pasted by a person — that part isn't, and
can't easily be, scripted.

Both scripts also track stats instead of leaving numbers buried in raw logs.
On Windows, each `run`/`run-ring` writes the coordinator's JSON `--report`
(decode_tok_s, p50/p95 latency, speculative rounds, draft acceptance,
network ms/token) and appends one row to `build\pipelined-ring-stats.csv`;
run the `stats` subcommand any time to see every recorded run in a table. On
Linux there's no coordinator to report from, so `gpu-stats` instead samples
`nvidia-smi` every 2s into `build/gpu-stats-linux.csv` while a test runs
(start it in its own terminal alongside `run`/`run-ring`), and `stats`
summarizes that plus the chunked-prefill line count. `package` on both
sides bundles the CSV/JSON files along with the logs.

**baseline and pipelined use auto-registration, not a manually copied
GGUF.** The coordinator listens (`--provider-listen`) and each provider
(`dan-stage-worker --coordinator ...`) connects out to it, registers its
GPU/VRAM, and downloads only the layer range the coordinator assigns it --
the same pattern already proven in
[the 32B aggregate-VRAM test](AGGREGATE_VRAM_32B_A5000_TEST.md). Nobody
manually downloads weights for those two configs; a local weights copy is
only fetched (automatically, by the scripts) for pipelined's `--draft-model`
and for ring mode. Ring mode itself still uses the older fixed-address
`--model`/`--stage-start`/`--stage-end`/`--provider` flow, because the
coordinator rejects `--ring-return` together with auto-registration (see
`coordinator.cpp`'s option validation) -- that is a real code constraint,
not a script choice.

**Before starting:** the pipelining/ring/chunking work is uncommitted at the
time this doc was written. Commit and push it (or otherwise get it onto the
Linux box — this session does not push on its own; confirm with whoever owns
the repo first) before step 2. Record the exact revision you test in the
evidence below — this is new code, not the pinned commit the existing
Windows+Linux doc references.

## 0. What's different from the existing hub-and-spoke test

- **Ring mode needs connectivity in both directions.** Hub-and-spoke only
  needs the coordinator to reach out to both providers. Ring mode also needs
  the Linux stage to reach *back* to the Windows coordinator's ring-return
  listener — Windows→Linux is not enough on its own. If the Windows machine
  has no public, port-forwarded address (typical for a home/office box), that
  back-connection is exactly the case `dan-sidecar`'s relay support exists
  for; see section 5.
- **Three configurations to run, not one:** hub-and-spoke as a baseline
  (already proven by the existing doc; rerun it here only to confirm this
  specific revision/pair of machines still works), pipelined speculative
  decoding over hub-and-spoke (measures the phase 1-3 throughput claim for
  the first time over a real link), and ring + chunked prefill (measures the
  phase 4-5 claim: fewer WAN hops, against the pipelined hub-and-spoke run).
- **A draft model is required** for the pipelined and ring runs. This test
  reuses the same weights as both target and draft (matches how phases 1-6
  were verified) — simplest to set up, not what a production deployment
  would do. A dedicated small draft model would show a different, likely
  better, real acceptance rate; that is out of scope here.

## Why dan-sidecar, and what's untested about it

DAN already has its own encrypted, peer-authenticated transport
(`dan-sidecar`, libp2p-based — see [P2P_TRANSPORT.md](P2P_TRANSPORT.md)), so
this test uses it rather than a third-party VPN. Be aware of what that
actually means here, honestly:

- The transport is generic — a `dan-sidecar` process just forwards one local
  TCP port to one remote peer's exposed port, or exposes one local port to
  authenticated peers. It has no idea what DAN protocol is riding inside it.
  Every `--provider`/`--next`/`--ring-listen`/`--ring-return` flag in this
  doc therefore points at a **local** address; the sidecar processes set up
  in section 5 are what make that local address actually reach the other
  machine.
- `dan-sidecar`'s one documented, previously-used pattern
  ([P2P_TRANSPORT.md](P2P_TRANSPORT.md)) is a single coordinator↔provider
  tunnel behind DAN's automatic `--provider-listen`/`--provider-peer-auth`
  formation flow. **Ring mode does not use that flow** — `--next`/
  `--ring-listen`/`--ring-return` only exist on the explicit `--provider
  HOST:PORT` path (`dan-stage-worker`'s plain `--model`/`--stage-start`
  mode), which is a separate code path this session never wired into the
  automatic formation flow. So this test tunnels the explicit-mode ports
  directly with plain `-inbound`/`-forward` sidecar pairs, one pair per
  logical link, rather than following the documented single-tunnel pattern.
  That combination — ring mode's multiple direct links, each carried over
  its own sidecar tunnel — has not been run before this doc, on any machine,
  by anyone. Section 5 is built directly from `dan-sidecar`'s actual flags
  (`sidecar/main.go`), not copied from a working example.
- Section 8 (ring) needs **three** independent tunnel pairs (six sidecar
  processes, three identities per machine); the hub-and-spoke baseline and
  pipelined runs (sections 6-7) need only **one**. Budget real setup time for
  section 8's networking specifically — it is the newest, least-trodden part
  of this whole test.

If any of this proves unreliable under real NAT conditions, falling back to
Tailscale (or plain public IPs) for the exact same DAN-side commands is a
one-line change: replace the `127.0.0.1:<forwarded-port>` addresses below
with whatever address reaches the other machine directly. Nothing about
`--pipeline-depth`/`--next`/`--ring-listen`/`--ring-return`/`--prefill-chunk`
cares what carries the bytes.

## 1. Linux prerequisites

Rent an x86-64 Ubuntu NVIDIA machine with the driver and CUDA development
toolkit. Do not expose DAN's ports publicly — every DAN process below binds
to `127.0.0.1` or to a sidecar-forwarded port, never directly to a public
interface.

```bash
set -euo pipefail
nvidia-smi
nvcc --version
apt-get update
apt-get install -y build-essential cmake git curl ca-certificates golang-go
go version   # need Go new enough to honor GOTOOLCHAIN=go1.25.7 (Go 1.21+)
```

Both NVIDIA commands must succeed before continuing.

## 2. Clone and build on Linux

```bash
set -euo pipefail
cd /root
git clone https://github.com/Tomer-R-Cohen/DAN.git
cd DAN
git checkout <your pushed branch or commit>
git rev-parse HEAD | tee revision-linux.txt

git clone https://github.com/ggml-org/llama.cpp.git build/pipelined-ring/llama.cpp
git -C build/pipelined-ring/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C build/pipelined-ring/llama.cpp apply patches/llama-provider-owned.patch

cmake -S . -B build/pipelined-ring \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR="$PWD/build/pipelined-ring/llama.cpp" \
  -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CCACHE=OFF
cmake --build build/pipelined-ring --parallel "$(nproc)" \
  --target dan-stage-worker provider_owned_protocol_test provider_owned_range_model_test provider_owned_formation_test
ctest --test-dir build/pipelined-ring --output-on-failure \
  -R '^provider_owned_(protocol|range_model|formation)_test$'
grep -q '^GGML_CUDA:BOOL=ON$' build/pipelined-ring/CMakeCache.txt
grep -q '^GGML_CUDA_GRAPHS:BOOL=ON$' build/pipelined-ring/CMakeCache.txt

bash ./scripts/build_sidecar.sh
```

`-DGGML_CUDA_GRAPHS=ON` is the phase 6 flag — confirm it is actually set in
the cache rather than assuming the CMake command took effect. The sidecar
build runs `go test ./...` before compiling; a test failure there means don't
proceed to section 5 with an unverified transport.

## 3. Build on Windows

Requires Go installed (`winget install GoLang.Go`, or the installer from
[go.dev](https://go.dev/dl/); Go 1.21+ for `GOTOOLCHAIN` to work).

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
git pull --ff-only origin <your pushed branch or commit>
git rev-parse HEAD | Tee-Object revision-windows.txt

cmake --build build-provider-owned-cuda --config Release --target dan-provider-owned-coordinator
cmake --build build-provider-owned-cuda --config Release --target dan-stage-worker
Select-String -Path .\build-provider-owned-cuda\CMakeCache.txt -Pattern '^GGML_CUDA(_GRAPHS)?:BOOL=ON$'

.\scripts\build_sidecar.ps1
```

The existing `build-provider-owned-cuda` tree already has `GGML_CUDA_GRAPHS`
on (confirmed during phase 6 verification); the `Select-String` above should
print both lines. If it doesn't, reconfigure with `-DGGML_CUDA_GRAPHS=ON`
before building. `build_sidecar.ps1` produces both a Windows and a Linux
sidecar binary in `build\sidecar\`; you already built the Linux one natively
in step 2, so only `dan-sidecar-windows-amd64.exe` is needed from this step.

## 4. Write the manifest

Baseline and pipelined (sections 6-7) use auto-registration, the same
proven pattern as [the 32B aggregate-VRAM test](AGGREGATE_VRAM_32B_A5000_TEST.md):
the coordinator listens (`--provider-listen`), each provider connects out to
it and downloads only the layer range it's assigned. Nobody manually copies
a GGUF for those two sections — the coordinator only needs to know the
model's repo/revision/SHA-256, not hold a copy of it. Write that on Windows
(the coordinator always runs there in this test):

```powershell
$Revision = '91cad51170dc346986eccefdc2dd33a9da36ead9'
$Sha256 = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
$Url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$Revision/qwen2.5-1.5b-instruct-q4_k_m.gguf"
@"
{
  "model_id": "qwen25-1-5b-pipelined-ring",
  "architecture": "qwen2",
  "layers": 28,
  "hidden_size": 1536,
  "context_size": 1024,
  "artifact_revision": "$Revision",
  "artifact_sha256": "$Sha256",
  "artifact_url": "$Url"
}
"@ | Set-Content -Encoding utf8 .\build\pipelined-ring-manifest.json
```

The `run`/`run-ring` script steps below write this automatically if it's
missing — no need to run this by hand unless you want to inspect it first.
This is a file write, not a download — no network call happens here.

Section 7 (pipelined) still needs one local weights copy on Windows only,
for the coordinator's own `--draft-model`; section 8 (ring) needs a local
copy on both machines, because ring mode can't use auto-registration (the
coordinator rejects `--ring-return` together with `--provider-listen`). Both
are downloaded automatically by `run -Draft`/`run-ring` when the file is
missing — the commands below show what that download does under the hood.

## 5. Set up dan-sidecar tunnels

Create one identity per sidecar process (server or client role — both need a
stable PeerID). Each `-id` run prints a PeerID; record every one, they're
needed as `-allow`/`-p2p/...` values below.

```bash
# Linux
./build/pipelined-ring/dan-sidecar-linux-amd64 -key provider1.key -id
./build/pipelined-ring/dan-sidecar-linux-amd64 -key ring-server.key -id
./build/pipelined-ring/dan-sidecar-linux-amd64 -key ringreturn-client.key -id
```

```powershell
# Windows
.\build\sidecar\dan-sidecar-windows-amd64.exe -key coordinator.key -id
.\build\sidecar\dan-sidecar-windows-amd64.exe -key ring-client.key -id
.\build\sidecar\dan-sidecar-windows-amd64.exe -key ringreturn-server.key -id
```

### Tunnel pair 1: Linux provider1 → Windows coordinator (:50200)

Needed for sections 6 and 7 (auto-registration). Note the direction: unlike
the old hub-and-spoke test, the *provider* dials out here, not the
coordinator — that's how `--coordinator`/`--provider-listen` registration
works. So the coordinator's own machine (Windows, usually with no reachable
public address) is the one that needs `-inbound`, and the rented Linux box
is the one that needs `-forward`.

Windows (server, exposes the local coordinator port to the authenticated
Linux client):

```powershell
& .\build\sidecar\dan-sidecar-windows-amd64.exe -key coordinator.key `
  -listen /ip4/0.0.0.0/tcp/4100 -inbound 127.0.0.1:50200 `
  -allow <PROVIDER1_PEER_ID_FROM_LINUX> `
  2>&1 | Tee-Object .\build\sidecar-coordinator.log
```

Linux (client — `WINDOWS_ADDR` is a direct public address if Windows has
one, otherwise a relay circuit address; either way it is what the Windows
`-id`/server output resolves to, not something to guess; add one or more
`-relay RELAY_ADDRESS` values if needed, see [P2P_TRANSPORT.md](P2P_TRANSPORT.md)):

```bash
./build/pipelined-ring/dan-sidecar-linux-amd64 -key provider1.key \
  -listen /ip4/0.0.0.0/tcp/4100 \
  -forward "127.0.0.1:50200=<WINDOWS_ADDR>/p2p/<COORDINATOR_PEER_ID_FROM_WINDOWS>" \
  2>&1 | tee build/sidecar-provider1.log
```

The Linux provider will use `--coordinator 127.0.0.1:50200` — the local
forwarded port, not a remote address — throughout sections 6 and 7. The
Windows-side provider (`provider0`) needs no tunnel at all; it registers
over plain loopback (`--coordinator 127.0.0.1:50200`) since it's on the
same machine as the coordinator.

### Tunnel pair 2: stage 0 (Windows) → stage 1 ring-listen (Linux :50103)

Needed only for section 8 (ring).

Linux (server):

```bash
./build/pipelined-ring/dan-sidecar-linux-amd64 -key ring-server.key \
  -listen /ip4/0.0.0.0/tcp/4102 -inbound 127.0.0.1:50103 \
  -allow <RING_CLIENT_PEER_ID_FROM_WINDOWS> \
  2>&1 | tee build/sidecar-ring-server.log
```

Windows (client):

```powershell
& .\build\sidecar\dan-sidecar-windows-amd64.exe -key ring-client.key `
  -listen /ip4/0.0.0.0/tcp/4102 `
  -forward "127.0.0.1:50103=/ip4/<LINUX_PUBLIC_IP>/tcp/4102/p2p/<RING_SERVER_PEER_ID_FROM_LINUX>" `
  2>&1 | Tee-Object .\build\sidecar-ring-client.log
```

Stage 0 will use `--next 127.0.0.1:50103`.

### Tunnel pair 3: stage 1 (Linux) → coordinator ring-return (Windows :50105)

Needed only for section 8. This is the back-connection section 0 calls out —
the coordinator is the server here, so if Windows has no reachable public
address, this pair is the one that needs `-relay`.

Windows (server):

```powershell
& .\build\sidecar\dan-sidecar-windows-amd64.exe -key ringreturn-server.key `
  -listen /ip4/0.0.0.0/tcp/4103 -inbound 127.0.0.1:50105 `
  -allow <RINGRETURN_CLIENT_PEER_ID_FROM_LINUX> `
  2>&1 | Tee-Object .\build\sidecar-ringreturn-server.log
```

Linux (client — `WINDOWS_ADDR` is a direct public address if Windows has
one, otherwise a relay circuit address; either way it is what the Windows
`-id`/server output resolves to, not something to guess):

```bash
./build/pipelined-ring/dan-sidecar-linux-amd64 -key ringreturn-client.key \
  -listen /ip4/0.0.0.0/tcp/4103 \
  -forward "127.0.0.1:50105=<WINDOWS_ADDR>/p2p/<RINGRETURN_SERVER_PEER_ID_FROM_WINDOWS>" \
  2>&1 | tee build/sidecar-ringreturn-client.log
```

Stage 1 will use `--next 127.0.0.1:50105`; the coordinator will use
`--ring-return 127.0.0.1:50105` (listening locally — the sidecar server
above is what makes that reachable from Linux).

Confirm all sidecar processes needed for the section you're running have
logged a connection (or are at least listening without error) before
starting the matching DAN processes.

## 6. Baseline: auto-registration, no draft model

Needs tunnel pair 1 only. Confirms this revision and this pair of machines
still do plain distributed inference correctly before adding pipelining or
ring on top. Start the coordinator's own tunnel first (Windows, from
section 5), then the two providers, then the coordinator.

Windows, first window — local provider (no tunnel, loopback only):

```powershell
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"
& .\build-provider-owned-cuda\Release\dan-stage-worker.exe `
  --coordinator 127.0.0.1:50200 `
  --provider-id windows-provider0 --cache-dir .\build\pipelined-ring-provider-cache `
  2>&1 | Tee-Object .\build\provider0.log
```

Linux, another window — remote provider (through tunnel pair 1's local
forward):

```bash
cd /root/DAN
mkdir -p build/pipelined-ring-provider-cache
./build/pipelined-ring/dan-stage-worker \
  --coordinator 127.0.0.1:50200 \
  --provider-id linux-provider1 --cache-dir build/pipelined-ring-provider-cache \
  2>&1 | tee build/provider1.log
```

Both providers will sit and wait ("Connected. Waiting for useful work...")
until the coordinator below starts and assigns them layers. Windows, third
window — the coordinator itself:

```powershell
& .\build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe `
  --manifest .\build\pipelined-ring-manifest.json `
  --provider-listen 127.0.0.1:50200 --metadata-cache .\build\pipelined-ring-model-index.tmp --provider-peer-auth `
  --prompt 'The capital of France is' --tokens 20 --requests 3 --shutdown-workers `
  --report .\build\baseline-report.json `
  2>&1 | Tee-Object .\build\baseline-report.log
```

Watch each provider's log for `Downloading required model data...` followed
by `range model ... downloaded=...` — that's the auto-download actually
happening, not a stale assumption that it will. Require 20 tokens on every
request, `network_ms/token` clearly nonzero (real WAN activation traffic
through the sidecar tunnel, not pure loopback), and a deterministic response
repeated across the 3 requests. `--shutdown-workers` ends both provider
processes when the run completes, so section 7 restarts them fresh. Save
`baseline-report.log`/`.json`; this is the reference `A_ms`/`B_ms`/
`network_ms` split every later run compares against.

## 7. Pipelined speculative decoding, auto-registration

Still needs only tunnel pair 1. Answers the phase 1-3 question: does keeping
several speculative chunks in flight actually hide the WAN round trip, on a
real link. This is the one place baseline's "no manual weights" claim has an
exception: the coordinator's own `--draft-model` is a local file it loads
directly, not something a provider downloads for it.

Running `.\scripts\pipelined_ring_test_windows.ps1 run -PeerId <...> -Draft`
does the whole section in one command (downloads that one weights copy if
missing, then the tunnel/provider/coordinator sequence below). To do it by
hand instead: relaunch both provider processes exactly as in section 6 (same
commands, fresh logs — `--shutdown-workers` ended them), then on Windows:

```powershell
& .\build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe `
  --manifest .\build\pipelined-ring-manifest.json `
  --provider-listen 127.0.0.1:50200 --metadata-cache .\build\pipelined-ring-model-index.tmp --provider-peer-auth `
  --draft-model .\build\pipelined-ring-weights\qwen2.5-1.5b.gguf --draft-tokens 8 --draft-gpu-layers 999 `
  --pipeline-depth 6 `
  --prompt 'Explain in detail why Paris became the capital of France, including historical, political, and geographical reasons.' `
  --tokens 120 --requests 3 --shutdown-workers `
  --report .\build\pipelined-hubspoke-report.json `
  2>&1 | Tee-Object .\build\pipelined-hubspoke-report.log
```

Record `decode_tok_s`, `draft_accept`, `speculative_rounds`, and
`draft_ms/token` from the `READY` line for each request. Compare
`decode_tok_s` against the section 6 baseline's implied rate
(`1000 / (A_ms + network_ms + B_ms)`) — this is the first real measurement of
whether pipelining pays off over an actual network, which nothing in
[PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md) had before this
test. Try `--pipeline-depth 2` and `--pipeline-depth 12` too if time allows;
the design doc's own model says the optimum sits near `RTT / tau_max`, which
this is the first chance to actually measure rather than project.

## 8. Ring topology + chunked prefill

Ring mode can't use auto-registration — the coordinator rejects
`--ring-return` together with `--provider-listen` (see `coordinator.cpp`'s
option validation), so this section goes back to the older fixed-address
`--model`/`--stage-start`/`--stage-end`/`--provider` flow, and both machines
need a real local weights copy this time (tunnel pair 1 and the two
`provider0`/`provider1` processes from sections 6-7 are not used here).

Running `.\scripts\pipelined_ring_test_windows.ps1 run-ring -LinuxIp ...
-ControlPeerId ... -RingPeerId ... -RingReturnPeerId ...` on Windows and
`./scripts/pipelined_ring_test_linux.sh run-ring <win-addr> ...` on Linux
does the whole section in one command each (downloads weights if missing,
starts all three tunnels and the non-coordinator stage in the background,
runs the coordinator/stage in the foreground). To do it by hand instead,
download the file at section 4's `$Url`/`$MODEL_URL` (Windows:
`Invoke-WebRequest -Uri $Url -OutFile .\build\pipelined-ring-weights\qwen2.5-1.5b.gguf`;
Linux: `curl -fL "$MODEL_URL" -o build/pipelined-ring-weights/qwen2.5-1.5b.gguf`)
and confirm its SHA-256 matches section 4's `$Sha256`/`$SHA256` on both
machines, confirm tunnel pairs 2 and 3 from section 5 are up and connected,
then run the processes below. Answers the
phase 4-5 question: does removing the coordinator from the per-token hot
path reduce the round trip further, on top of pipelining.

Linux (ring listener for stage 0, `--next` pointing at the local forward from
tunnel pair 3):

```bash
cd /root/DAN
./build/pipelined-ring/dan-stage-worker \
  --model build/pipelined-ring-weights/qwen2.5-1.5b.gguf \
  --stage-start 14 --stage-end 28 --host 127.0.0.1 --port 50102 \
  --ring-listen 127.0.0.1:50103 --next 127.0.0.1:50105 \
  --ctx 1024 --gpu-layers 999 --max-sessions 4 \
  2>&1 | tee build/stageB-ring.log
```

Windows, second window (`--next` pointing at the local forward from tunnel
pair 2; `--prefill-chunk` only takes effect on the first stage):

```powershell
& .\build-provider-owned-cuda\Release\dan-stage-worker.exe `
  --model .\build\pipelined-ring-weights\qwen2.5-1.5b.gguf `
  --stage-start 0 --stage-end 14 --host 127.0.0.1 --port 50101 `
  --next 127.0.0.1:50103 --prefill-chunk 24 `
  --ctx 1024 --gpu-layers 999 --max-sessions 4 `
  2>&1 | Tee-Object .\build\stageA-ring.log
```

Windows, third window — the coordinator, with `--ring-return` listening
locally (tunnel pair 3's server sidecar is what makes that reachable from
Linux). This process will not print `READY` until the Linux stage above
connects through the ring, which in turn will not connect until this
listener is open — start this window, confirm the "waiting for the tail
stage" line appears, and only then does the exact start order of the other
two stop mattering:

```powershell
& .\build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe `
  --manifest .\build\pipelined-ring-manifest.json `
  --provider 127.0.0.1:50101 --provider 127.0.0.1:50102 `
  --ring-return 127.0.0.1:50105 `
  --draft-model .\build\pipelined-ring-weights\qwen2.5-1.5b.gguf --draft-tokens 8 --draft-gpu-layers 999 `
  --pipeline-depth 6 `
  --prompt 'Explain in detail why Paris became the capital of France, including historical, political, and geographical reasons.' `
  --tokens 120 --requests 3 `
  2>&1 | Tee-Object .\build\ring-report.log
```

Require: identical committed output to section 7's run (same prompt, same
`--draft-tokens`; ring mode must not change what gets generated, only how it
travels), `A_ms/token` reported as 0 in the metrics (ring mode does not
attribute per-hop compute the way hub-and-spoke does — this is the documented
reduced-fidelity metrics gap, not a bug), and the Linux stage's log showing
`ring: predecessor connected`. Confirm chunking actually fired, not just that
the flag was accepted — the test prompt is long enough at `--prefill-chunk 24`
to produce at least two non-final chunks, but check rather than assume:

```powershell
Select-String -Path .\build\stageA-ring.log -Pattern 'phase=prefill-chunk'
```

```bash
grep 'phase=prefill-chunk' build/stageB-ring.log
```

Both must show more than one match. Compare `decode_tok_s` directly against
section 7 — this is the number
[PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md) never had: ring
vs. hub-and-spoke, both pipelined, on the same real link.

## 9. Evidence checklist

- [ ] Revision under test recorded on both machines (`revision-linux.txt`,
      `revision-windows.txt`) — must match, and must be pushed, not a local
      working-tree-only checkout on one side.
- [ ] `GGML_CUDA` and `GGML_CUDA_GRAPHS` both confirmed `ON` in both CMake
      caches. Sidecar `go test ./...` passed on both machines before use.
- [ ] Weight SHA-256 confirmed identical on both machines (section 7's
      draft copy and section 8's ring copies; baseline needs none).
- [ ] All PeerIDs recorded (six identities); each tunnel pair confirmed
      connected before its matching DAN section started (server log shows an
      accepted inbound connection, or the client log shows no dial error).
- [ ] Section 6 (baseline, tunnel pair 1, auto-registration) log: both
      providers' logs show `range model ... downloaded=...` (the auto-download
      actually firing), 20/20/20 tokens across 3 requests, deterministic
      output, nonzero `network_ms/token`.
- [ ] Section 7 (pipelined, auto-registration, tunnel pair 1 only) log:
      `decode_tok_s`, `draft_accept`, `speculative_rounds`, `draft_ms/token`
      recorded for each request; `decode_tok_s` compared against the
      section 6 baseline rate.
- [ ] Section 8 (ring + chunked prefill, fixed-address, tunnel pairs 2 and 3)
      log: same metrics recorded; output confirmed identical to section 7's;
      Linux stage log confirms ring connections and chunk activity;
      `decode_tok_s` compared against section 7.
- [ ] Any crash, hang, or unexpected disconnect recorded verbatim, including
      which of the three configurations — and, if it happened in section 8,
      whether it was a DAN-side failure or a sidecar tunnel failure. Given
      this exact sidecar-plus-ring combination has never been run before,
      a tunnel-side failure here is expected information, not noise.
- [ ] All log files (DAN and sidecar) and both revision files copied off the
      rented Linux machine before terminating it.

```bash
# on Linux, before terminating the rental
tar -czf pipelined-ring-results.tar.gz build/stageB-*.log build/sidecar-*.log weights-linux.sha256 revision-linux.txt
```

```powershell
# on Windows
Compress-Archive -Path .\build\stageA-*.log,.\build\sidecar-*.log,.\build\baseline-report.log,.\build\pipelined-hubspoke-report.log,.\build\ring-report.log,.\revision-windows.txt -DestinationPath .\pipelined-ring-results.zip -Force
```

Known limitations carried over from
[PIPELINED_SPECULATION_V1.md](PIPELINED_SPECULATION_V1.md): only two stages
are verified (ring and chunking are not hardcoded to two, but untested beyond
it); ring mode does not cover `route_speculative` itself, only `route_step`
— the speculative verify traffic in sections 7 and 8 both travel
hub-and-spoke regardless, so section 8's win, if any, comes entirely from
prefill and the non-speculative parts of the loop, not from ring-ifying the
hot decode path. That gap is real and this test does not close it — see
phase 4's entry in "Phasing" in the design doc for what ring-ifying
`route_speculative` would need. Layered on top of that: the sidecar tunneling
in section 8 is, as of this doc, unverified end to end (see "Why dan-sidecar,
and what's untested about it" above) — a failure specifically in section 8
should be triaged as "which of these two new things broke" before being
treated as a ring-mode code defect.
