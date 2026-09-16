# Aggregate-VRAM Physical Test: 14B (superseded) → 32B (proven)

Date: 2026-09-08

Status: **distributed provider-owned inference passed** on a model that
genuinely requires both GPUs. An earlier 14B attempt is kept here for the
record but was superseded — it fit on one GPU alone and therefore proved
nothing about pooling VRAM across machines.

## The 14B attempt (superseded — kept for the record)

- Model: `bartowski/Qwen2.5-14B-Instruct-GGUF`, `Qwen2.5-14B-Instruct-Q8_0.gguf`
- Logical size: 15,701,598,336 bytes (14.63 GiB); 48 layers, hidden size 5,120
- Windows RTX 2070 (6,656 MiB) + Linux RTX 2000 Ada (~15,852 MiB free)
- Tested plan: RTX 2070 layers 0-5 (2.41 GiB), RTX 2000 Ada layers 6-47
  (12.22 GiB)

This showed the split mechanism working, but the RTX 2000 Ada alone had
enough VRAM to hold the entire 14.63 GiB model — so a passing run here
would say nothing about whether DAN could pool VRAM that neither GPU has
individually. It was superseded by the 32B target below specifically
because it fails this test: use this target for an RTX 2070 offering
6,656 MiB and something in the ~24,000 MiB range; the 14B target does not
force aggregation.

## Target: Qwen2.5-32B, chosen because it can't fit on either GPU alone

- Repository: `bartowski/Qwen2.5-32B-Instruct-GGUF`
- File: `Qwen2.5-32B-Instruct-Q5_K_M.gguf`
- Revision: `2116cbb385b8ce3a4d28cf3bf1cd2039a55821a6`
- SHA-256: `371c9d50b4c2db96c7b6695b81a23c03c10a0ee780a8c3fe9193aac148febdaf`
- Logical size: 23,262,157,696 bytes (21.66 GiB)
- Architecture: dense Qwen2, 64 layers, hidden size 5,120, context 512

Tested planner result (provider order and available VRAM can move the
boundary by a layer):

```text
RTX 2070 (6,656 MiB): layers 0..6, 2.82 GiB weights + 112 MiB KV
RTX A5000 (24,000 MiB): layers 7..63, 18.85 GiB weights + 912 MiB KV
```

Pass requires two stages and neither provider downloading or allocating
the complete GGUF.

## Reproduction

Continuing from an existing build (binaries from commit
`15bcd564e5915834d42da4472f7fff961206ad3f` or later don't need rebuilding
for a configuration-only target change like this):

On Linux:

```bash
cd /DAN
git fetch origin
git checkout --detach origin/main
pgrep -af 'tailscaled|ncat'
export DAN_COORDINATOR=127.0.0.1:50200
```

On Windows:

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
git pull --ff-only origin main
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"
New-Item -ItemType Directory -Force .\build\aggregate-32b\provider-windows | Out-Null
```

Start the Windows provider first:

```powershell
& .\build-provider-owned-cuda\Release\dan-stage-worker.exe `
  --coordinator 127.0.0.1:50200 `
  --provider-id windows-rtx2070 `
  --cache-dir .\build\aggregate-32b\provider-windows `
  --vram-mib 6656 `
  2>&1 | Tee-Object .\build\aggregate-32b\provider-windows.log
```

Then start the coordinator in another Windows terminal. It will wait for
the Linux provider:

```powershell
$Prompt = "<|im_start|>system`nYou are a helpful assistant.<|im_end|>`n<|im_start|>user`nWhat is the capital of France? Answer in one sentence.<|im_end|>`n<|im_start|>assistant`n"

& .\build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe `
  --manifest .\config\provider-owned-qwen2.5-32b-q5km.json `
  --provider-listen 0.0.0.0:50200 `
  --metadata-cache .\build\aggregate-32b\model-index.tmp `
  --prompt $Prompt --tokens 20 --requests 1 --shutdown-workers `
  --report .\build\aggregate-32b\physical-report.json `
  2>&1 | Tee-Object .\build\aggregate-32b\coordinator.log
```

Finally restart the Linux worker. A previous `peer disconnected` is
harmless; it means the Windows coordinator was not listening yet.

```bash
cd /DAN
mkdir -p build/aggregate-32b/provider-linux
./build/aggregate-14b/dan-stage-worker \
  --coordinator "$DAN_COORDINATOR" \
  --provider-id linux-rtx-a5000 \
  --cache-dir "$PWD/build/aggregate-32b/provider-linux" \
  2>&1 | tee build/aggregate-32b/provider-linux.log
```

Do not interrupt range downloads. Allow at least 10 GB free disk on
Windows and 28 GB on Linux. The two providers together download about
23.3 GB plus small duplicated metadata/shared data, not 23.3 GB each.

**Pass evidence**: the coordinator must report `layers=64`,
`N-stage replica READY stages=2`, and 20 generated tokens. It must hold no
GGUF. Record the plan, range-download bytes, sparse physical allocation,
VRAM, Tailscale path/RTT, per-stage compute, network ms/token, activation
bytes/token, and model reload counts. A second run must report
`cache=reused` without downloading ranges again.

## Results: the completed 2026-09-08 run

Status detail: the stricter claim that the raw quantized weights cannot
fit in either physical GPU remains unproven — the 21.66 GiB GGUF is
smaller than the A5000's approximately 23.35 GiB offered VRAM — but DAN's
planner correctly required both providers once runtime, KV, and safety
headroom were included.

```text
Windows coordinator (metadata/routing only)
    |
    +-- Windows RTX 2070 provider: layers 0..6
    |       embedding + early transformer layers
    |       6,656 MiB offered
    |
    +-- activation over Tailscale: FP32, hidden size 5,120
            20,480 bytes per decoded token
            |
            +-- Linux RTX A5000 provider: layers 7..63
                    remaining layers + norm + output head
                    23,899 MiB offered
```

The Linux rental was an Ubuntu 24.04 container with an NVIDIA RTX A5000,
driver 580.159.04, CUDA driver capability 13.0, and CUDA toolkit 12.8. It
had no `/dev/net/tun`, so Tailscale ran in userspace-networking mode.
`ncat` forwarded local port 50200 through Tailscale's local SOCKS5 server
to the Windows coordinator at `100.85.166.77:50200`.

The coordinator derived the following assignment at runtime from model
tensor sizes and provider offers:

| Provider | Layers | Planned model data | KV allocation |
|---|---:|---:|---:|
| Windows RTX 2070 | 0-6 | 2.82 GiB | 112 MiB |
| Linux RTX A5000 | 7-63 | 18.85 GiB | 912 MiB |

Each stage used the range-backed sparse GGUF cache. The coordinator
fetched only GGUF metadata/index ranges and had no GGUF or model-weight
dependency. Exact sparse physical allocation and downloaded-byte totals
were not captured from the successful 32B run, so those remain evidence
gaps rather than inferred measurements.

### Smoke request

Prompt asked for the capital of France. The model stopped naturally at EOS.

```text
tokens:                 8
latency:                4,280.731 ms
output:                 The capital of France is Paris.
decode:                 4.338 tok/s
RTX 2070 compute:       16.460 ms/token
network:                175.727 ms/token
RTX A5000 compute:      38.339 ms/token
activation:             20,480 bytes/token
model reloads:          1 per provider
```

This was correct but did not meet the 20-token acceptance length because
EOS was reached after eight generated tokens.

### 20-token acceptance request

Prompt: "Explain in detail why Paris became the capital of France,
including historical, political, and geographical reasons."

```text
tokens:                 20
latency p50/p95/p99:    5,782.003 ms (one measured request)
decode:                 4.279 tok/s
RTX 2070 compute:       14.156 ms/token
network:                184.791 ms/token
RTX A5000 compute:      34.762 ms/token
activation:             20,480 bytes/token
activation total:       819,200 bytes
model reloads:          1 per provider process
```

Exact captured output: "Paris became the capital of France due to a
combination of historical, political, and geographical factors."

Provider A processed 40 tokens and generated none, as expected for the
first stage. Provider B processed the same 40 tokens and generated all 20
output tokens. Both reported zero resident sessions after the stateless
request was destroyed. Startup was 4,733.113 ms on Provider A and
3,382.096 ms on Provider B. Each provider reported `model_reload_count=1`.

The average measured decode path was 233.709 ms/token:

| Component | ms/token | Share |
|---|---:|---:|
| RTX 2070 stage | 14.156 | 6.1% |
| Tailscale transfer/round trip | 184.791 | 79.1% |
| RTX A5000 stage | 34.762 | 14.9% |

Network latency clearly dominated this run. A later five-ping sample
reached the Linux provider through `DERP(tor)` rather than a direct
Tailscale path: 503 ms for the first ping, then 164, 162, 165, and 163 ms.
Tailscale explicitly reported `direct connection not established`. This
closely explains the approximately 185 ms/token network measurement.

Only one request was measured, so its latency is simultaneously p50, p95,
and p99; these values are not a latency distribution.

### Sparse storage evidence

The successful Windows stage cache reported:

```text
logical bytes:          23,262,157,696
downloaded bytes:        3,031,018,368
allocated bytes:         3,024,814,080
download percentage:             13.03%
tensors present:                     85
stage:                              0-7 (half-open)
sparse flag:                        set
```

Windows had two allocated ranges and holes elsewhere in the full-layout
GGUF. Its sidecar pins the URL, revision, complete-file SHA-256, stage,
logical size, range offsets, sizes, and per-range hashes. The Linux
planner reported 18.85 GiB of assigned model data, but its final sidecar,
tensor count, and physical allocation were not copied into this
repository and must not be presented as measured values.

The second run started from the existing provider cache paths and
proceeded to inference without the earlier 24-minute model-range
transfer. This operationally demonstrated cache reuse. Preserve the
provider `cache=reused` log lines in a future evidence bundle for an
explicit audit trail.

### Failure discovered and fixed

The first Linux download reached 100% of one 18.7 GiB merged HTTP range,
then failed with `range-backed model: could not write sparse model
range`.

Root cause: the implementation first downloaded the entire merged range
to a temporary file, then copied it into the sparse GGUF. Peak storage
was therefore approximately twice the assigned data (about 37.4 GiB),
excluding build files. Failure cleanup removed both the temporary file
and incomplete sparse model, so that transfer was not resumable.

Commit `af43de3` bounds downloads to 256 MiB chunks. Each verified chunk
is written at its original GGUF offset and its temporary file is removed
before the next chunk. This preserves original tensor offsets and sparse
holes while reducing temporary peak storage from 18.7 GiB to 256 MiB. The
Windows range model and formation tests passed after the change.

Other operational findings:

- A worker message `peer disconnected while receiving frame` before
  assignment meant the relay reached Windows while no coordinator was
  listening.
- The coordinator intentionally waits without a stage-download timeout
  after assignment.
- A one-line PowerShell prompt avoids here-string paste/parser mistakes.
- The Linux CUDA build can be reused after configuration-only model
  changes.
- When only `range_model.cpp` changes, rebuilding `dan-stage-worker`
  reuses the already compiled llama.cpp/CUDA objects.

## What this run proves

1. A metadata-only Windows coordinator formed a runtime-defined two-stage
   Qwen2 replica from live provider capabilities.
2. Provider stages loaded and executed different contiguous layer ranges.
3. Intermediate FP32 activations crossed the physical Windows/Linux
   network.
4. The first provider did not execute the output head or generate
   tokens.
5. The final provider generated the response from the received
   activations.
6. KV storage remained local to each provider's owned layers.
7. The coordinator did not download or load model weights.
8. A correct response and a separate 20-token response completed.
9. Stage workers loaded their model ranges once per process and shut
   down cleanly.
10. Network latency, specifically a relayed DERP path, was the main
    performance bottleneck.

## Qualification and remaining evidence

This is a successful distributed, provider-owned, range-backed inference
proof. It is also an aggregate-*planned-capacity* proof: after DAN's
conservative headroom, runtime buffers, and KV allocation, neither
provider was eligible to host the complete replica alone.

It is not yet the strict raw-weight aggregate-VRAM proof originally
requested. The GGUF is 21.66 GiB while the A5000 offered about 23.34 GiB,
so its quantized weights are smaller than that GPU's raw offer. A later
test must use model weights larger than 23,899 MiB but small enough to
fit the combined contributed capacity with runtime/KV overhead, or
explicitly cap the A5000's contribution below the model size while
retaining enough combined planned capacity.

Still outstanding for a complete benchmark package:

- exact-token comparison with a full-model llama.cpp deterministic
  baseline;
- Linux sparse physical allocation, downloaded bytes, tensor count, and
  sidecar;
- GPU utilization and peak physical VRAM measurements during inference;
- a direct Tailscale path test, if the rental network can establish one;
- multiple warm requests for meaningful p50/p95/p99 statistics.

The successful configuration was added in commit `a3d295c`, its
PowerShell prompt correction in `53dafa3`, and bounded range downloads in
`af43de3`. The coordinator's unedited JSON reports are preserved as
[the smoke result](evidence/qwen2.5-32b-smoke.json) and
[the 20-token result](evidence/qwen2.5-32b-20-token.json).
