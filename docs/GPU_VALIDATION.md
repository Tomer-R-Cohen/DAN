# Rented GPU validation: one CUDA provider

Status: real single-GPU validation passed on NVIDIA A40 with Qwen3-30B-A3B
Q4_K_M at 32,768 context on 2026-09-05. The persistent DAN run returned 10/10
responses, averaged 3,844.7 ms per request, observed 21,227 MiB peak VRAM,
and exited cleanly. See [STATE.md](STATE.md) and
[the committed benchmark archive](../dan-qwen3-30b-a3b-results.tar.gz).

The commands below retain the original Qwen2.5-7B/L4 recipe as an alternative
single-GPU deployment example. An L4-specific run is not verified; these are
not reproduction commands for the A40 result. The archive's
`results/qwen3-30b-a3b/report.md`, registries, runtime commands, and logs record
the actual A40 configuration, including the failed 4,096-context attempt and
successful 32,768-context rerun.

Two remote CUDA RPC workers have since passed both the small-model smoke test
and a Qwen3-30B-A3B run; see [TWO_GPU_SMOKE_REPORT.md](TWO_GPU_SMOKE_REPORT.md)
and [QWEN3_TWO_GPU_REPORT.md](QWEN3_TWO_GPU_REPORT.md). The
[cache and persistent-runtime experiment](NEXT_GPU_EXPERIMENT.md) is deferred
until the managed-worker adapter connects Provider Control Plane v1 to real
verified cache and persistent serving.
Aggregate-VRAM necessity remains a subsequent unproven milestone.

Use Ubuntu 24.04 x86-64 with a working NVIDIA driver and CUDA development toolkit
(including `nvcc`). Keep coordinator and provider on the rental for this first
test; connect by SSH and do not expose DAN ports publicly. Allow 30 GB or more
temporary disk space for sources, build products, weights, and evidence.

## Before renting

- Use `https://github.com/Tomer-R-Cohen/DAN.git` and record the checked-out
  revision. The A40 benchmark used `c5b3bfd`; `e2cbd2b` added its evidence archive.
- Choose a CUDA **development** image, not just an image containing the driver.
- Choose a candidate, quantization, download location, and license beforehand.
- Keep SmolLM2 test-only. The recipe below uses Qwen2.5-7B-Instruct Q4_K_M as a
  conservative, serious GPU acceptance baseline, not a claim of current best
  model quality. Change the candidate through the registry for later comparisons.
- Reserve time to copy logs off the temporary machine before ending the rental.

## 1. Build prerequisites and DAN

Run in bash on the rental:

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

## 2. Verify hardware and build the pinned llama.cpp revision

```bash
nvidia-smi | tee results/nvidia-smi.txt
nvcc --version | tee results/nvcc.txt
```

`nvidia-smi`'s CUDA version is driver compatibility, not proof of an installed
compiler. If `nvcc` is missing, select the rental's CUDA development image or,
on Ubuntu 24.04 with an already working driver, install the Ubuntu toolkit:

```bash
sudo apt-get install -y nvidia-cuda-toolkit
nvcc --version
```

Build this workspace's current llama.cpp revision, not a moving upstream head.
Keep its existing subprocess behavior reproducible. A later runtime revision
requires rerunning direct and DAN tests.

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

Require a CUDA device (normally `CUDA0`). If build memory is exhausted, retry
with `-j 2`. If the compiler/driver combination fails, fix that before downloading
larger candidates. Reference: [official CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda).

## 3. Place weights and configure dan-main

Download both shards of the publisher's GGUF. Alternatively copy existing GGUF
files with `scp` and set `MODEL_PATH` to the first shard (or single GGUF).
Reference: [publisher's model files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main).

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

4,096 is the **configured test context**, not the model's advertised maximum.
The memory field is an estimate, not a fit guarantee. For another candidate,
change its path, family, quantization, memory estimate and runtime name here;
keep `dan-main`. All shards are required. No weights are included with DAN.

## 4. Direct CUDA inference first

```bash
set -o pipefail
"$LLAMA_BIN" --model "$MODEL_PATH" --device CUDA0 --gpu-layers 999 \
  --ctx-size 4096 --n-predict 256 --conversation --single-turn --simple-io \
  --no-display-prompt --color off \
  --prompt 'Explain TCP reliability in two sentences.' \
  2>&1 | tee results/direct.log
```

Require a useful response, exit status 0, CUDA/offload evidence in logs and
increased VRAM. A CUDA metadata label alone does not prove GPU execution.
Record model load and evaluation timings from llama.cpp when emitted. If OOM,
lower context or choose a smaller quantization, record the change and rerun.
Do not count an unnoticed CPU fallback as GPU success.

## 5. Ten sequential prompts through DAN (recommended)

```bash
python3 scripts/validate_gpu.py --registry config/models.gpu.conf \
  --runtime "$LLAMA_BIN" --output results/dan-gpu
```

This starts the actual coordinator and provider, waits for registration, then
sends one `/model dan-main` request at a time, waiting for its response and
performance line. It retains all answers, provider startup logs, per-request
latencies, exit codes, initial GPU information and one-second VRAM/utilization
samples. It sends `exit` after ten responses and waits for both processes.
Failures/timeouts leave partial results and terminate its process groups.

Use a new output directory for each run. The session retains conversation
history; these timings are session timings, not ten independent cold prompts.
Persistent runs use `--n-predict -1` (generate until end-of-turn): the pinned
llama.cpp interactive runtime carries a positive token budget across turns and
can return an empty answer when a later input exhausts it. The direct one-shot
test above still caps output at 256 tokens. The driver's per-request timeout
bounds runaway generation; manual DAN operation has no such timeout. Do not
equate ten nonempty answers with a quality pass.

## 6. Manual terminal operation (alternative)

Do not run this simultaneously with the driver above. Terminal 1, from DAN:

```bash
./build/coordinator 9000 --models config/models.gpu.conf 2>&1 | tee results/coordinator-manual.log
```

Terminal 2, set the same `LLAMA_BIN`, `MODEL_PATH`, and `MODEL_NAME` values as
above, then:

```bash
./build/provider "$LLAMA_BIN" "$MODEL_PATH" 127.0.0.1 9000 \
  --id l4-validation --model-name "$MODEL_NAME" --backend CUDA --device GPU \
  --gpu "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" \
  --vram "$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader)" \
  --runtime-device CUDA0 --gpu-layers 999 --ctx-size 4096 --n-predict -1 \
  2>&1 | tee results/provider-manual.log
```

Enter `/model dan-main Explain TCP reliability.` in terminal 1. Wait for each
response; the ten representative prompts are in `scripts/validate_gpu.py`.
Enter `exit` after completion; it drains work and shuts down the provider.

## 7. Review evidence and shut down

- [ ] GPU name, total/free VRAM, driver and CUDA compiler recorded.
- [ ] DAN revision, llama.cpp revision and GGUF hashes recorded.
- [ ] Model ID/family/quantization, total shard size and actual context recorded.
- [ ] Direct CUDA inference succeeded; provider startup and offload verified.
- [ ] Provider loaded once and received ten prompts; ten IDs have real responses.
- [ ] Latencies and observed VRAM peak recorded; answers reviewed for truncation/errors.
- [ ] Token rate recorded from direct llama.cpp diagnostics, or explicitly unavailable.
- [ ] Runtime exits and any crashes/errors recorded, including failed attempts.
- [ ] Results copied off temporary storage before ending the rental.

Copy `docs/GPU_RESULTS_TEMPLATE.md` to `results/report.md` and fill it in. Use
`vram.csv` for observed peak memory (one-second sampling may miss short peaks).
DAN does not currently return generated-token counts, so do not infer tokens/s
from words or characters. Direct llama.cpp token rates and DAN latency measure
different things. Provider ready time includes loading/warmup, not pure disk I/O.

```bash
cp docs/GPU_RESULTS_TEMPLATE.md results/report.md
tar -czf dan-gpu-results.tar.gz results
```

From your own machine: `scp USER@RENTAL:/path/to/DAN/dan-gpu-results.tar.gz .`
Confirm download, check `nvidia-smi` for leftover model processes, then terminate
the rented instance using its normal control panel. DAN does not stop billing.

Known review limitations: the current interactive subprocess adapter separates
answers using llama.cpp's `\n> ` display marker. A generated identical marker
can confuse it. Handshake and frame reads still block; stalled peers have no
core request timeout. The validation driver's timeout bounds a test run, and
these limitations must be recorded if encountered. RPC memory-fit and network
bottleneck claims remain outside this single-GPU test.
