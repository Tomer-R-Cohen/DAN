# Provider-Owned v0: Windows RTX 2070 + Linux CUDA runbook

> Historical v0 reproduction only. For the maintained persistent C++ runtime,
> use [Provider-Owned Runtime Setup](PROVIDER_OWNED_SETUP.md).

This runs the existing two-stage experiment without changing DAN's normal RPC
runtime. The Windows machine hosts the metadata-only coordinator and Provider A;
the rented Linux GPU hosts Provider B. Both providers keep the same complete
GGUF, but each worker loads and executes only its assigned stage.

## Fixed experiment values

| Item | Value |
|---|---|
| DAN source commit | `91f6edfd84c620919428f0992ae1120086587887` |
| llama.cpp commit | `95ef7fc16054e63b427a3ef00188e055ef7586d8` |
| Model | Qwen2.5 0.5B Instruct, GGUF `Q4_K_M` |
| Model filename | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| Model bytes | `491400032` |
| Model SHA-256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| Provider A | embedding + layers `0..11` |
| Provider B | layers `12..23` + final norm/head/greedy sampler |
| Context | `512` |
| Prompt | `The capital of France is` |
| Generated tokens | `20` |
| Provider A endpoint | `127.0.0.1:50101` (Windows-local only) |
| Provider B endpoint | `<Linux Tailscale IPv4>:50102` |

Commit `91f6edf` must exist on the GitHub remote before a rented machine can
check it out. It is the frozen source used by the successful local proof.

## 1. Rent the Linux machine and verify CUDA

Use an x86-64 Ubuntu image with an NVIDIA driver and the CUDA **development**
toolkit. A runtime-only image is insufficient because this build requires
`nvcc`. Do not open DAN ports in the provider's public/cloud firewall.

```bash
set -euo pipefail
nvidia-smi
nvcc --version
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl ca-certificates python3
gcc --version
cmake --version
```

Stop here if either `nvidia-smi` or `nvcc --version` fails. Install the toolkit
for the rented image using NVIDIA's
[distribution-specific instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/),
then repeat these checks.

## 2. Clone the frozen DAN source on Linux

```bash
set -euo pipefail
cd "$HOME"
git clone https://github.com/Tomer-R-Cohen/DAN.git
cd DAN
git checkout --detach 91f6edfd84c620919428f0992ae1120086587887
test "$(git rev-parse HEAD)" = 91f6edfd84c620919428f0992ae1120086587887
python3 -m unittest -v experimental.provider_owned.test_protocol
```

## 3. Build Provider B with Linux CUDA

```bash
set -euo pipefail
cd "$HOME/DAN"
LLAMA_DIR="$PWD/build/provider-owned-v0/llama.cpp"
BUILD_DIR="$PWD/build/provider-owned-v0/stage-build-cuda"
PATCH_FILE="$PWD/experimental/provider_owned/llama-provider-owned-v0.patch"

git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
git -C "$LLAMA_DIR" checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
test "$(git -C "$LLAMA_DIR" rev-parse HEAD)" = 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C "$LLAMA_DIR" apply --check "$PATCH_FILE"
git -C "$LLAMA_DIR" apply "$PATCH_FILE"

cmake -S experimental/provider_owned -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SOURCE_DIR="$LLAMA_DIR" \
  -DGGML_CUDA=ON \
  -DGGML_CCACHE=OFF
cmake --build "$BUILD_DIR" --parallel "$(nproc)" --target dan-stage-worker

grep -q '^GGML_CUDA:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"
test -x "$BUILD_DIR/dan-stage-worker"
```

The two final commands are the build gate: do not continue unless both pass.

## 4. Join the same Tailscale tailnet on Linux

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale status
export LINUX_TS_IP="$(tailscale ip -4 | head -n1)"
test -n "$LINUX_TS_IP"
printf 'Linux Tailscale IP: %s\n' "$LINUX_TS_IP"
```

Authenticate with the same tailnet used by the Windows machine. If UFW is
active, allow Provider B only on the Tailscale interface:

```bash
if sudo ufw status | grep -q '^Status: active'; then
  sudo ufw allow in on tailscale0 to any port 50102 proto tcp
fi
```

No public ingress rule for TCP `50102` is required or wanted.
The install command is from Tailscale's official
[Linux installation guide](https://tailscale.com/docs/install/linux).

If the rental is a container without `/dev/net/tun`, run Tailscale in userspace
mode and publish the loopback worker to the tailnet:

```bash
sudo mkdir -p /var/run/tailscale /var/lib/tailscale
sudo nohup tailscaled --tun=userspace-networking \
  --state=/var/lib/tailscale/tailscaled.state \
  --socket=/var/run/tailscale/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 &
sleep 2
sudo tailscale up
tailscale serve --bg --tcp=50102 tcp://127.0.0.1:50102
tailscale serve status
```

For this fallback, use `127.0.0.1` as Provider B's `--host` below. The
coordinator still connects to the Linux Tailscale IPv4 on port `50102`.

## 5. Download and verify the model on Linux

```bash
set -euo pipefail
cd "$HOME/DAN"
MODEL_DIR="$PWD/build/provider-owned-v0/provider-b"
MODEL_FILE="$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
mkdir -p "$MODEL_DIR"

curl -fL --retry 5 --retry-delay 3 \
  -o "$MODEL_FILE" \
  'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'

echo '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db  build/provider-owned-v0/provider-b/qwen2.5-0.5b-instruct-q4_k_m.gguf' | sha256sum --check --strict
test "$(stat -c %s "$MODEL_FILE")" -eq 491400032
```

## 6. Prepare Provider A on Windows

Run from PowerShell in the existing DAN checkout. This uses a separate CUDA
build directory, so the successful CPU/local build remains untouched.

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$DanCommit = '91f6edfd84c620919428f0992ae1120086587887'
if ((git rev-parse HEAD).Trim() -ne $DanCommit) { throw "Wrong DAN commit" }
nvidia-smi
if ($LASTEXITCODE -ne 0) { throw 'NVIDIA driver/GPU is required' }
nvcc --version
if ($LASTEXITCODE -ne 0) { throw 'CUDA Toolkit/nvcc is required' }

$CudaBuild = Join-Path $PWD 'build\provider-owned-v0\stage-build-cuda'
powershell -ExecutionPolicy Bypass -File .\experimental\provider_owned\build.ps1 `
  -Cuda -BuildDirectory $CudaBuild

if (-not (Select-String -Quiet -LiteralPath "$CudaBuild\CMakeCache.txt" -Pattern '^GGML_CUDA:BOOL=ON$')) {
  throw 'CUDA was not enabled in the Windows worker build'
}

$ModelA = Join-Path $PWD 'build\provider-owned-v0\provider-a\qwen2.5-0.5b-instruct-q4_k_m.gguf'
if ((Get-Item -LiteralPath $ModelA).Length -ne 491400032) { throw 'Wrong model size' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $ModelA).Hash.ToLowerInvariant() -ne '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db') {
  throw 'Wrong model SHA-256'
}
```

If the model is absent on Windows, download it before running the size/hash
checks:

```powershell
$ModelDir = Join-Path $PWD 'build\provider-owned-v0\provider-a'
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$ModelA = Join-Path $ModelDir 'qwen2.5-0.5b-instruct-q4_k_m.gguf'
curl.exe -fL --retry 5 --retry-delay 3 -o $ModelA 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'
```

## 7. Confirm the private route before launching

On Linux, print the Provider B address again:

```bash
tailscale status
tailscale ip -4
```

On Windows, replace the example with that Linux Tailscale IPv4:

```powershell
$LinuxTailIP = '100.x.y.z'
tailscale status
tailscale ping $LinuxTailIP
ping.exe $LinuxTailIP
```

`tailscale ping` must identify the rented machine. Do not probe TCP `50101` or
`50102`: each v0 worker accepts exactly one connection, and a port probe would
consume it before the coordinator connects.

## 8. Launch in this exact order

Keep each process in its own terminal. Each worker accepts one coordinator
connection and exits after that request, so restart both workers for every rerun.

Before starting Provider B, begin Linux GPU sampling in another Linux terminal:

```bash
cd "$HOME/DAN"
mkdir -p build/provider-owned-v0
timeout 120s nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
  --format=csv -lms 100 > build/provider-owned-v0/provider-b-gpu.csv &
echo $! > build/provider-owned-v0/provider-b-gpu.pid
```

### Terminal 1 — Linux Provider B

```bash
set -euo pipefail
cd "$HOME/DAN"
export LINUX_TS_IP="$(tailscale ip -4 | head -n1)"
export PROVIDER_B_HOST="$LINUX_TS_IP" # use 127.0.0.1 with userspace Tailscale Serve
./build/provider-owned-v0/stage-build-cuda/dan-stage-worker \
  --model "$PWD/build/provider-owned-v0/provider-b/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --stage-start 12 \
  --stage-end 24 \
  --host "$PROVIDER_B_HOST" \
  --port 50102 \
  --ctx 512 \
  --gpu-layers 999 \
  2>&1 | tee build/provider-owned-v0/provider-b.log
```

Wait for `DAN stage ready: layers 12..23` and `listening on ...:50102`.
The llama.cpp startup log must name a CUDA device; stop if all tensors are
assigned to CPU.

### Terminal 2 — Windows Provider A

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$CudaBuild = Join-Path $PWD 'build\provider-owned-v0\stage-build-cuda'
$env:PATH = "$CudaBuild\bin\Release;$env:PATH"
$ModelA = Join-Path $PWD 'build\provider-owned-v0\provider-a\qwen2.5-0.5b-instruct-q4_k_m.gguf'

& "$CudaBuild\Release\dan-stage-worker.exe" `
  --model $ModelA `
  --stage-start 0 `
  --stage-end 12 `
  --host 127.0.0.1 `
  --port 50101 `
  --ctx 512 `
  --gpu-layers 999 2>&1 | Tee-Object -FilePath .\build\provider-owned-v0\provider-a.log
```

Wait for `DAN stage ready: layers 0..11` and `listening on 127.0.0.1:50101`.
The DLL `PATH` line is required for the Windows build-tree executable.

### Terminal 3 — Windows connectivity check and coordinator

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$LinuxTailIP = '100.x.y.z'

$GpuMonitor = Start-Process -FilePath nvidia-smi.exe -WindowStyle Hidden -PassThru `
  -ArgumentList @('--query-gpu=timestamp,name,memory.used,utilization.gpu', '--format=csv', '-lms', '100') `
  -RedirectStandardOutput .\build\provider-owned-v0\provider-a-gpu.csv

try {
  python .\experimental\provider_owned\coordinator.py `
    --manifest .\experimental\provider_owned\qwen2.5-0.5b-q4km.json `
    --provider-a 127.0.0.1:50101 `
    --provider-b "${LinuxTailIP}:50102" `
    --prompt 'The capital of France is' `
    --tokens 20 `
    --report .\build\provider-owned-v0\physical-report.json
  if ($LASTEXITCODE -ne 0) { throw 'Coordinator test failed' }
} finally {
  Stop-Process -Id $GpuMonitor.Id -ErrorAction SilentlyContinue
}
```

The coordinator has no model argument and does not open a GGUF. It receives
Provider A's FP32 activation and routes it to Provider B.

## 9. Correctness gate and benchmark record

Run after the coordinator exits:

```powershell
$Expected = ' Paris. It is the largest city in Europe and the second largest in the world. It is also'
$Result = Get-Content -Raw .\build\provider-owned-v0\physical-report.json | ConvertFrom-Json
if ($Result.metrics.generated_tokens -ne 20) { throw 'Did not generate 20 tokens' }
if ($Result.output -cne $Expected) { throw "Output differs from frozen greedy baseline: $($Result.output)" }

$Result.metrics | Select-Object `
  prompt_tokens, generated_tokens, prefill_ms, decode_tok_s, `
  a_compute_ms_mean, activation_route_ms_mean, b_compute_ms_mean, `
  total_token_ms_mean, prefill_activation_bytes, decode_bytes_per_token | Format-List
```

That expected string is the frozen all-CUDA baseline. The earlier CPU baseline
matched its first 19 token positions and produced `located` instead of `also`
at position 20; compare CUDA with CUDA for the physical correctness gate.

Expected fixed traffic is `17,920` activation bytes for the five-token prefill
and `3,584` activation bytes per decoded token, excluding the timing field and
40-byte frame header. Save these artifacts before destroying the rental:

```powershell
Get-Content .\build\provider-owned-v0\physical-report.json
Get-Content .\build\provider-owned-v0\provider-a.log
Get-Content .\build\provider-owned-v0\provider-a-gpu.csv
```

On Linux:

```bash
cd "$HOME/DAN"
kill "$(cat build/provider-owned-v0/provider-b-gpu.pid)" 2>/dev/null || true
nvidia-smi
cat build/provider-owned-v0/provider-b.log
cat build/provider-owned-v0/provider-b-gpu.csv
```

Record `tailscale ping`/`ping` RTT, both provider logs, the JSON report, Linux
GPU name/VRAM, and Windows RTX 2070 GPU/VRAM. A failure or disconnect should
abort the coordinator cleanly; it must not affect ordinary DAN RPC mode.

## Known v0 boundaries

- Provider-owned execution is proven; provider-specific storage shards are not
  implemented. Both machines download the same full 491,400,032-byte GGUF.
- Activations route through the coordinator, not directly from A to B.
- One request, two fixed stages, greedy sampling, and one replica only.
- No public ports, scheduler changes, replacement, authentication changes, or
  modifications to the existing llama.cpp RPC runtime are part of this test.
