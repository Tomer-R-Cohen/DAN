# Aggregate-VRAM 14B Physical Test

This runbook tests commit `15bcd564e5915834d42da4472f7fff961206ad3f`
with the Windows RTX 2070 and Linux RTX 2000 Ada. Use a fresh cache: the first
run downloads about 15.7 GB across the two providers and can take a while.

## Target model and expected plan

- Model: `bartowski/Qwen2.5-14B-Instruct-GGUF`
- File: `Qwen2.5-14B-Instruct-Q8_0.gguf` (single file, dense Qwen2)
- Revision: `05244aa5d871c661c80082a15d3bce44714d068d`
- SHA-256: `23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf`
- Logical size: 15,701,598,336 bytes (14.63 GiB)
- Model metadata: 48 layers, hidden size 5,120
- Context: 512; eight resident KV slots

With 6,656 MiB offered by the RTX 2070 and about 15,852 MiB free on the RTX
2000 Ada, the tested plan was:

```text
RTX 2070:       layers 0..5   2.41 GiB weights + 96 MiB KV
RTX 2000 Ada:   layers 6..47 12.22 GiB weights + 672 MiB KV
```

Provider arrival order can reverse the two stage roles, and small free-VRAM
changes can move the boundary by a layer. Success requires two stages and no
provider holding all 14.63 GiB of model weights.

Allow at least 6 GB free disk on Windows and 18 GB on Linux for sparse data,
temporary range verification, build output, and logs. Close other GPU-heavy
programs first.

## 1. Prepare Linux

```bash
set -euo pipefail
nvidia-smi
nvcc --version
apt-get update
apt-get install -y build-essential cmake git curl ca-certificates ncat

cd /root
git clone https://github.com/Tomer-R-Cohen/DAN.git
cd DAN
git checkout --detach 15bcd564e5915834d42da4472f7fff961206ad3f

git clone https://github.com/ggml-org/llama.cpp.git build/aggregate-14b/llama.cpp
git -C build/aggregate-14b/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C build/aggregate-14b/llama.cpp apply patches/llama-provider-owned.patch

cmake -S . -B build/aggregate-14b \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR="$PWD/build/aggregate-14b/llama.cpp" \
  -DGGML_CUDA=ON -DGGML_CCACHE=OFF
cmake --build build/aggregate-14b --parallel "$(nproc)" \
  --target dan-stage-worker provider_owned_protocol_test \
    provider_owned_range_model_test provider_owned_formation_test
ctest --test-dir build/aggregate-14b --output-on-failure \
  -R '^provider_owned_(protocol|range_model|formation)_test$'
grep -q '^GGML_CUDA:BOOL=ON$' build/aggregate-14b/CMakeCache.txt
```

## 2. Connect Linux to Tailscale

For a full VM with `/dev/net/tun`:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up
tailscale status
tailscale ping WINDOWS_TAILSCALE_IP
export DAN_COORDINATOR=WINDOWS_TAILSCALE_IP:50200
```

For a container without `/dev/net/tun`, start the official userspace SOCKS5
mode and expose a localhost relay for the DAN worker:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
mkdir -p /root/.dan-tailscale
nohup tailscaled --tun=userspace-networking \
  --state=/root/.dan-tailscale/tailscaled.state \
  --socket=/root/.dan-tailscale/tailscaled.sock \
  --socks5-server=127.0.0.1:1055 \
  >/root/.dan-tailscale/tailscaled.log 2>&1 &
sleep 2
tailscale --socket=/root/.dan-tailscale/tailscaled.sock up
tailscale --socket=/root/.dan-tailscale/tailscaled.sock status
tailscale --socket=/root/.dan-tailscale/tailscaled.sock ping WINDOWS_TAILSCALE_IP

export WINDOWS_TS_IP=WINDOWS_TAILSCALE_IP
nohup ncat -lk 127.0.0.1 50200 --sh-exec \
  "ncat --proxy 127.0.0.1:1055 --proxy-type socks5 $WINDOWS_TS_IP 50200" \
  >/tmp/dan-tailnet-relay.log 2>&1 &
export DAN_COORDINATOR=127.0.0.1:50200
```

Replace `WINDOWS_TAILSCALE_IP` before running either block.

## 3. Start the generic Linux provider

The worker may be started before the coordinator and will wait for it.

```bash
cd /root/DAN
mkdir -p build/aggregate-14b/provider-linux
./build/aggregate-14b/dan-stage-worker \
  --coordinator "$DAN_COORDINATOR" \
  --provider-id linux-rtx2000-ada \
  --cache-dir "$PWD/build/aggregate-14b/provider-linux" \
  2>&1 | tee build/aggregate-14b/provider-linux.log
```

## 4. Build and start the Windows provider

In PowerShell on the RTX 2070 PC:

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
git pull --ff-only origin main
git checkout 15bcd564e5915834d42da4472f7fff961206ad3f

.\scripts\build_provider_owned.ps1 -Cuda `
  -BuildDirectory "$PWD\build-provider-owned-cuda" `
  -LlamaSource "$PWD\build\provider-owned-aggregate\llama.cpp"

tailscale status
$WindowsTailIP = (tailscale ip -4 | Select-Object -First 1)
$WindowsTailIP
```

Run PowerShell as Administrator once if inbound port 50200 is blocked:

```powershell
New-NetFirewallRule -DisplayName 'DAN provider control 50200' `
  -Direction Inbound -Protocol TCP -LocalPort 50200 -Action Allow
```

Open a normal PowerShell window for Provider A:

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"
New-Item -ItemType Directory -Force .\build\aggregate-14b\provider-windows | Out-Null

& .\build-provider-owned-cuda\Release\dan-stage-worker.exe `
  --coordinator 127.0.0.1:50200 `
  --provider-id windows-rtx2070 `
  --cache-dir .\build\aggregate-14b\provider-windows `
  --vram-mib 6656 `
  2>&1 | Tee-Object .\build\aggregate-14b\provider-windows.log
```

## 5. Start the coordinator and run inference

After both provider terminals say they are waiting, open another Windows
PowerShell window:

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"
$Prompt = @'
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France? Answer in one sentence.<|im_end|>
<|im_start|>assistant
'@

& .\build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe `
  --manifest .\config\provider-owned-qwen2.5-14b-q8.json `
  --provider-listen 0.0.0.0:50200 `
  --metadata-cache .\build\aggregate-14b\model-index.tmp `
  --prompt $Prompt --tokens 20 --requests 1 --shutdown-workers `
  --report .\build\aggregate-14b\physical-report.json `
  2>&1 | Tee-Object .\build\aggregate-14b\coordinator.log
```

The workers display curl's size, speed, and transfer progress while fetching
their assigned ranges. Do not interrupt the first download unless necessary;
an incomplete cache is rejected on restart.

## 6. Acceptance evidence

The coordinator must report:

```text
model metadata ready: layers=48 hidden=5120
N-stage replica READY stages=2
request=1 ... tokens=20
```

Save these files:

```text
build/aggregate-14b/coordinator.log
build/aggregate-14b/provider-windows.log
build/aggregate-14b/physical-report.json
/root/DAN/build/aggregate-14b/provider-linux.log
```

On Linux, record sparse allocation and GPU use:

```bash
cd /root/DAN
MODEL=$(find build/aggregate-14b/provider-linux -name '*.gguf' -print -quit)
stat -c 'logical=%s bytes blocks=%b block_size=%B' "$MODEL"
du -B1 "$MODEL"
nvidia-smi
grep -E 'range model|partial load|DAN stage READY|request=|graceful shutdown' \
  build/aggregate-14b/provider-linux.log
```

On Windows:

```powershell
Get-ChildItem .\build\aggregate-14b\provider-windows\*.gguf |
  ForEach-Object { $_ | Select-Object FullName,Length; fsutil sparse queryflag $_.FullName }
Select-String -Path .\build\aggregate-14b\provider-windows.log `
  -Pattern 'range model|partial load|DAN stage READY|request=|graceful shutdown'
```

Pass requires 20 generated tokens, two persistent worker PIDs, no complete
GGUF on either provider, no GGUF on the coordinator, and the reported stage
weight totals each remaining below that provider's offered VRAM. Record
Tailscale path/RTT, decode tok/s, stage compute, network ms/token, activation
bytes/token, CUDA VRAM, physical cache sizes, and model reload counts.

The first run proves aggregate-VRAM execution. A second identical run must say
`cache=reused` on both providers and must not redownload model ranges.
