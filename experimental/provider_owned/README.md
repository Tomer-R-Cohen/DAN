# Provider-Owned Execution v0

This is an isolated Qwen2-only research runtime. It does not replace or alter
DAN's existing llama.cpp RPC coordinator/provider mode.

The metadata-only Python coordinator routes one request through two persistent
`dan-stage-worker` processes. Stage A tokenizes and runs the embedding plus
layers `0..11`; stage B accepts FP32 hidden states, runs layers `12..23`, final
norm and the tied output head, then performs greedy sampling. Each llama context
allocates KV only for its 12 owned layers.

## Build

The build script clones the exact pinned llama.cpp revision, applies the small
experimental patch, and builds only the stage worker. A CUDA Toolkit installation
with `nvcc` is required for `-Cuda`; the normal DAN release is unaffected.

```powershell
powershell -ExecutionPolicy Bypass -File .\experimental\provider_owned\build.ps1 -Cuda
```

Linux uses the same sources:

```bash
git clone https://github.com/ggml-org/llama.cpp.git build/provider-owned-v0/llama.cpp
git -C build/provider-owned-v0/llama.cpp checkout 95ef7fc16054e63b427a3ef00188e055ef7586d8
git -C build/provider-owned-v0/llama.cpp apply experimental/provider_owned/llama-provider-owned-v0.patch
cmake -S experimental/provider_owned -B build/provider-owned-v0/stage-build \
  -DLLAMA_SOURCE_DIR="$PWD/build/provider-owned-v0/llama.cpp" -DGGML_CUDA=ON
cmake --build build/provider-owned-v0/stage-build -j4 --target dan-stage-worker
```

Download and SHA-verify the GGUF from the URL in
`qwen2.5-0.5b-q4km.json` separately on each provider. The v0 storage compromise
is deliberate: both providers currently keep the original GGUF file, while the
patched loader creates and executes only tensors required by its assigned stage.

## Run

On provider A (use its Tailscale IP instead of the example):

```powershell
.\dan-stage-worker.exe --model .\qwen.gguf --stage-start 0 --stage-end 12 --host 100.64.0.10 --port 50101 --ctx 512 --gpu-layers 999
```

On provider B:

```powershell
.\dan-stage-worker.exe --model .\qwen.gguf --stage-start 12 --stage-end 24 --host 100.64.0.11 --port 50102 --ctx 512 --gpu-layers 999
```

On the coordinator machine (no GGUF is used or accepted):

```powershell
python .\experimental\provider_owned\coordinator.py `
  --manifest .\experimental\provider_owned\qwen2.5-0.5b-q4km.json `
  --provider-a 100.64.0.10:50101 --provider-b 100.64.0.11:50102 `
  --prompt "The capital of France is" --tokens 20 --report result.json
```

The trusted-Tailscale assumption remains. Do not expose either worker port to
the public internet. A provider disconnect aborts the request; replacement and
resume are intentionally outside v0.

Run the protocol checks with:

```powershell
python -m unittest -v experimental.provider_owned.test_protocol
```
