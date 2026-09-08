# Aggregate-VRAM 32B RTX A5000 Test

The completed 2026-09-08 run and its measured results are documented in
[Qwen2.5 32B Windows/Linux Provider-Owned Result](AGGREGATE_VRAM_32B_RESULTS.md).

Use this target for an RTX 2070 offering 6,656 MiB and an RTX A5000 offering
about 24,000 MiB. The earlier 14B target fits on the A5000 alone and therefore
does not prove aggregate-VRAM execution.

## Target

- Model: `bartowski/Qwen2.5-32B-Instruct-GGUF`
- File: `Qwen2.5-32B-Instruct-Q5_K_M.gguf`
- Revision: `2116cbb385b8ce3a4d28cf3bf1cd2039a55821a6`
- SHA-256: `371c9d50b4c2db96c7b6695b81a23c03c10a0ee780a8c3fe9193aac148febdaf`
- Logical size: 23,262,157,696 bytes (21.66 GiB)
- Model: dense Qwen2, 64 layers, hidden size 5,120, context 512

The tested planner result is approximately:

```text
RTX 2070 (6,656 MiB): layers 0..6, 2.82 GiB weights + 112 MiB KV
RTX A5000 (24,000 MiB): layers 7..63, 18.85 GiB weights + 912 MiB KV
```

Provider order and available VRAM may move the boundary. Pass requires two
stages and neither provider downloading or allocating the complete GGUF.

## Continue from the existing build

This target changes configuration only. Existing binaries built from commit
`15bcd564e5915834d42da4472f7fff961206ad3f` or later do not need rebuilding.
Update both checkouts to current `main` before starting the processes.

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

Then start the coordinator in another Windows terminal. It will wait for the
Linux provider:

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

Finally restart the Linux worker. A previous `peer disconnected` is harmless;
it means the Windows coordinator was not listening yet.

```bash
cd /DAN
mkdir -p build/aggregate-32b/provider-linux
./build/aggregate-14b/dan-stage-worker \
  --coordinator "$DAN_COORDINATOR" \
  --provider-id linux-rtx-a5000 \
  --cache-dir "$PWD/build/aggregate-32b/provider-linux" \
  2>&1 | tee build/aggregate-32b/provider-linux.log
```

Do not interrupt range downloads. Allow at least 10 GB free disk on Windows
and 28 GB on Linux. The two providers together download about 23.3 GB plus
small duplicated metadata/shared data, not 23.3 GB each.

## Pass evidence

The coordinator must report `layers=64`, `N-stage replica READY stages=2`, and
20 generated tokens. It must hold no GGUF. Record the plan, range-download
bytes, sparse physical allocation, VRAM, Tailscale path/RTT, per-stage compute,
network ms/token, activation bytes/token, and model reload counts. A second
run must report `cache=reused` without downloading ranges again.
