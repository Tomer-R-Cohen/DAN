# Provider-Owned Runtime v2

This is DAN's provider-owned distributed-inference path. The coordinator is a
C++23 metadata/router process and never opens a GGUF. Stage A owns the embedding
and Qwen2 layers `0..11`; stage B owns layers `12..23`, final norm, output head,
and greedy sampling. Each worker loads its assigned tensors once and uses
llama.cpp sequence IDs to keep independent KV state in one shared context.

DAN's existing llama.cpp RPC path remains unchanged as a legacy/reference
fallback. Normal provider-owned inference does not require Python. The removed
v0 implementation remains available from its frozen Git commit if historical
reproduction is needed.

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
own supported C++17 compilation mode.

## Run

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

The trusted-Tailscale assumption remains. Never expose worker ports publicly.
Only one request executes at a time; GPU batching, simultaneous replica
execution, provider replacement, physical GGUF shards, and KV migration remain
out of v2.

Start bounded concurrent serving instead of batch CLI mode:

```powershell
.\dan-provider-owned-coordinator.exe `
  --manifest .\config\provider-owned-qwen2.5-0.5b-q4km.json `
  --provider-a 127.0.0.1:50101 --provider-b 100.68.128.88:50102 `
  --listen 127.0.0.1:50100 --queue-capacity 128 --client-threads 64
```

The v2 acceptance client command is:

```powershell
.\provider_owned_concurrency_client.exe --coordinator 127.0.0.1:50100 `
  --clients 20 --sessions 50 --requests 1000
```

See [the v1 implementation report](PROVIDER_OWNED_RUNTIME_V1.md) for
protocol, lifecycle, correctness, and benchmark details.
See [the v2 report](PROVIDER_OWNED_RUNTIME_V2.md) for concurrency results.
See [the range-backed storage report](RANGE_BACKED_PROVIDER_STORAGE.md) for
storage and integrity acceptance results.
