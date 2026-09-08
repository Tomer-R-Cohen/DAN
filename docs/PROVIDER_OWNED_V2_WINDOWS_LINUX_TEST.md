# Provider-Owned v2: Windows + Linux CUDA Test

This is the maintained physical test for commit
`56239140ce3231186c015acc26355bf368f6f3eb`. The Windows RTX 2070 runs the
metadata-only coordinator and Provider A; the rented Linux NVIDIA GPU runs
Provider B. Both providers use range-backed sparse GGUF caches.

## 1. Linux prerequisites

Rent an x86-64 Ubuntu NVIDIA machine with both the driver and CUDA development
toolkit. Run:

```bash
set -euo pipefail
nvidia-smi
nvcc --version
apt-get update
apt-get install -y build-essential cmake git curl ca-certificates
curl -fsSL https://tailscale.com/install.sh | sh
```

Both NVIDIA commands must succeed. Do not expose DAN's port publicly.

## 2. Clone and build Provider B

```bash
set -euo pipefail
cd /root
git clone https://github.com/Tomer-R-Cohen/DAN.git
cd DAN
git checkout --detach 56239140ce3231186c015acc26355bf368f6f3eb

git clone https://github.com/ggml-org/llama.cpp.git build/provider-owned-v2/llama.cpp
git -C build/provider-owned-v2/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C build/provider-owned-v2/llama.cpp apply patches/llama-provider-owned.patch

cmake -S . -B build/provider-owned-v2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR="$PWD/build/provider-owned-v2/llama.cpp" \
  -DGGML_CUDA=ON -DGGML_CCACHE=OFF
cmake --build build/provider-owned-v2 --parallel "$(nproc)" \
  --target dan-stage-worker provider_owned_protocol_test provider_owned_range_model_test
ctest --test-dir build/provider-owned-v2 --output-on-failure \
  -R '^provider_owned_(protocol|range_model)_test$'
grep -q '^GGML_CUDA:BOOL=ON$' build/provider-owned-v2/CMakeCache.txt
```

## 3. Join the Windows tailnet

For a full VM with `/dev/net/tun`:

```bash
tailscale up
tailscale status
export LINUX_TS_IP="$(tailscale ip -4 | head -n1)"
```

For a container without `/dev/net/tun`:

```bash
mkdir -p /var/run/tailscale /var/lib/tailscale
nohup tailscaled --tun=userspace-networking \
  --state=/var/lib/tailscale/tailscaled.state \
  --socket=/var/run/tailscale/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 &
sleep 2
tailscale up
tailscale serve --bg --tcp=50102 tcp://127.0.0.1:50102
tailscale status
tailscale serve status
export LINUX_TS_IP="$(tailscale ip -4 | head -n1)"
```

Save the printed Linux Tailscale IPv4 for the Windows command.

## 4. Start Linux Provider B

Use the Tailscale IP as `BIND_IP` on a full VM. Use `127.0.0.1` with the
userspace Tailscale Serve fallback.

```bash
set -euo pipefail
cd /root/DAN
REVISION=9217f5db79a29953eb74d5343926648285ec7e67
SHA256=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/$REVISION/qwen2.5-0.5b-instruct-q4_k_m.gguf"
BIND_IP="$(tailscale ip -4 | head -n1)" # change to 127.0.0.1 for userspace Serve
mkdir -p build/physical-v2/provider-b

./build/provider-owned-v2/dan-stage-worker \
  --model "$PWD/build/physical-v2/provider-b/qwen.gguf" \
  --model-url "$URL" --model-revision "$REVISION" --model-sha256 "$SHA256" \
  --stage-start 12 --stage-end 24 --host "$BIND_IP" --port 50102 \
  --ctx 512 --gpu-layers 999 --max-sessions 8 \
  2>&1 | tee build/physical-v2/provider-b.log
```

Wait for `cache=downloaded`, a CUDA device, `partial load ... 146 of 291`, and
`DAN stage READY`. Provider B should download about 278 MB, not the full file.

## 5. Start Windows Provider A

Open a new PowerShell window:

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$Revision = '9217f5db79a29953eb74d5343926648285ec7e67'
$Sha256 = '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
$Url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/$Revision/qwen2.5-0.5b-instruct-q4_k_m.gguf"
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"

& .\build-provider-owned-cuda\Release\dan-stage-worker.exe `
  --model .\build\range-cache\provider-a\qwen.gguf `
  --model-url $Url --model-revision $Revision --model-sha256 $Sha256 `
  --stage-start 0 --stage-end 12 --host 127.0.0.1 --port 50101 `
  --ctx 512 --gpu-layers 999 --max-sessions 8 `
  2>&1 | Tee-Object .\build\physical-v2-provider-a.log
```

Wait for `cache=reused`, the RTX 2070, `partial load ... 145 of 291`, and
`DAN stage READY`.

## 6. Run the physical inference

Open another Windows PowerShell window and replace the example IP:

```powershell
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'
$LinuxTailIP = '100.x.y.z'
tailscale ping $LinuxTailIP
Test-NetConnection $LinuxTailIP -Port 50102

& .\build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe `
  --manifest .\config\provider-owned-qwen2.5-0.5b-q4km.json `
  --provider-a 127.0.0.1:50101 --provider-b "${LinuxTailIP}:50102" `
  --prompt 'The capital of France is' --tokens 20 --requests 1 `
  --shutdown-workers --report .\build\physical-v2-report.json
```

Success requires 20 tokens, Provider A to Provider B activation traffic, and a
valid deterministic response. The known CPU baseline ends with `It is located`;
the prior all-CUDA baseline differed only at token 20 and ended with `It is
also`. Save the JSON and both worker logs rather than hiding that numerical
boundary.

## 7. Prove sparse storage and cache reuse

After the first worker exits, run on Linux:

```bash
cd /root/DAN
stat -c 'logical=%s bytes' build/physical-v2/provider-b/qwen.gguf
du -B1 build/physical-v2/provider-b/qwen.gguf
grep -E 'range model|cache=|partial load|DAN stage READY' build/physical-v2/provider-b.log
```

Start Provider B again with the identical command from step 4. It must say
`cache=reused` and must not run a new model download. Restart Provider A with
step 5, then rerun step 6 with a new report filename.

Record `tailscale ping`, both GPU names, both startup logs, physical/logical
cache sizes, both inference reports, compute/network time per token, decode
tokens/s, activation bytes/token, VRAM, and model reload count. A provider
disconnect must fail the request cleanly without crashing the coordinator.
