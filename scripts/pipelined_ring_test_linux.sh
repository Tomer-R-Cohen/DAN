#!/usr/bin/env bash
# Linux-side commands for docs/PIPELINED_RING_PHYSICAL_TEST.md.
#
# Covers each step of that doc as a subcommand instead of typing the full
# command by hand. baseline/pipelined use the same auto-registration +
# auto-download pattern as the proven docs/AGGREGATE_VRAM_32B_A5000_TEST.md
# run: this provider connects out to the Windows coordinator and downloads
# only its assigned layers itself -- no GGUF is copied here by hand for
# those two configs. Only ring mode still needs a pre-downloaded local copy
# and a fixed --model/--stage-start/--stage-end invocation, because the
# coordinator rejects --ring-return together with auto-registration.
#
# Several steps run in separate terminals at the same time, so this is a
# menu of steps, not a single "run everything" script -- open one terminal
# per step that needs to stay running.
#
# Usage: pipelined_ring_test_linux.sh <step> [options]
#   prereqs
#   build [branch-or-commit]
#   sidecar-id <key-name>
#   tunnel-coordinator <win-addr> <win-peer-id>
#   provider1
#   weights
#   tunnel-ring-server <windows-peer-id>
#   tunnel-ringreturn-client <windows-addr> <windows-peer-id>
#   stage1-ring
#   check-chunks
#   gpu-stats
#   stats
#   package
set -euo pipefail

cd "$(dirname "$0")/.."

DAN_BUILD=build/pipelined-ring
SIDECAR_BIN="$DAN_BUILD/dan-sidecar-linux-amd64"
WEIGHTS=build/pipelined-ring-weights/qwen2.5-1.5b.gguf
PROVIDER_CACHE=build/pipelined-ring-provider-cache
GPU_STATS_CSV=build/gpu-stats-linux.csv
STEP="${1:-}"

# Same pinned model as the Windows side's 'manifest'/'weights' steps.
REVISION=91cad51170dc346986eccefdc2dd33a9da36ead9
SHA256=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$REVISION/qwen2.5-1.5b-instruct-q4_k_m.gguf"

usage() {
    cat <<'EOF'
Steps (see docs/PIPELINED_RING_PHYSICAL_TEST.md for what each one means):

  prereqs                                      step 1: check driver/toolchain, install build deps
  build [branch-or-commit]                     step 2: clone/build DAN + llama.cpp + sidecar
  sidecar-id <key-name>                        step 5: create one sidecar identity, print its PeerID
  tunnel-coordinator <win-addr> <win-peer-id>   step 5: Linux side, provider1 -> Windows coordinator
  provider1                                     step 6: auto-registers, auto-downloads its
                                                 assigned layers (needed for baseline and pipelined)
  weights                                       step 8 prereq: download weights for ring mode only
                                                 (ring can't use auto-registration -- see below)
  tunnel-ring-server <win-peer-id>              step 8, pair 2: Linux side (ring only)
  tunnel-ringreturn-client <win-addr> <win-id>  step 8, pair 3: Linux side (ring only)
  stage1-ring                                   step 8: stage 1, fixed-address, with
                                                 --ring-listen/--next
  check-chunks                                  step 8: confirm chunked prefill actually fired
  gpu-stats                                     sample nvidia-smi every 2s into $GPU_STATS_CSV;
                                                 run in its own terminal alongside provider1/stage1-ring
  stats                                         summarize chunk counts and gpu-stats.csv so far
  package                                       step 9: tar up logs, revision file, and stats

Sidecar tunnel processes, provider/stage processes, and gpu-stats are all
long-running -- each of the steps above that starts one blocks its terminal
until you Ctrl+C. Open a separate terminal per step you need running at the
same time. The coordinator (and its own decode_tok_s/latency/draft-accept
report) runs on Windows -- see stats/package there for those numbers; this
side only has GPU utilization/VRAM and chunk counts to record.
EOF
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Expected '$1' to exist. $2" >&2
        exit 1
    fi
}

case "$STEP" in
    prereqs)
        nvidia-smi
        nvcc --version
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git curl ca-certificates golang-go
        go version
        echo 'Both nvidia-smi and nvcc must have printed real output above before continuing.'
        ;;

    build)
        BRANCH="${2:-}"
        if [ -n "$BRANCH" ]; then
            git checkout "$BRANCH"
        else
            echo "No branch/commit given; building whatever is currently checked out." >&2
        fi
        git rev-parse HEAD | tee revision-linux.txt

        if [ ! -d "$DAN_BUILD/llama.cpp" ]; then
            git clone https://github.com/ggml-org/llama.cpp.git "$DAN_BUILD/llama.cpp"
            git -C "$DAN_BUILD/llama.cpp" checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
            git -C "$DAN_BUILD/llama.cpp" apply "$PWD/patches/llama-provider-owned.patch"
        else
            echo "$DAN_BUILD/llama.cpp already exists; not re-cloning."
        fi

        cmake -S . -B "$DAN_BUILD" \
            -DCMAKE_BUILD_TYPE=Release \
            -DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR="$PWD/$DAN_BUILD/llama.cpp" \
            -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CCACHE=OFF
        cmake --build "$DAN_BUILD" --parallel "$(nproc)" \
            --target dan-stage-worker provider_owned_protocol_test provider_owned_range_model_test provider_owned_formation_test
        ctest --test-dir "$DAN_BUILD" --output-on-failure \
            -R '^provider_owned_(protocol|range_model|formation)_test$'

        if grep -q '^GGML_CUDA:BOOL=ON$' "$DAN_BUILD/CMakeCache.txt" \
            && grep -q '^GGML_CUDA_GRAPHS:BOOL=ON$' "$DAN_BUILD/CMakeCache.txt"; then
            echo 'GGML_CUDA and GGML_CUDA_GRAPHS both confirmed ON.'
        else
            echo 'WARNING: GGML_CUDA and/or GGML_CUDA_GRAPHS are not both ON. Reconfigure before continuing.' >&2
        fi

        bash ./scripts/build_sidecar.sh
        ;;

    sidecar-id)
        KEY_NAME="${2:-}"
        [ -n "$KEY_NAME" ] || { echo "Usage: sidecar-id <key-name>  (e.g. provider1, ring-server, ringreturn-client)" >&2; exit 1; }
        "$SIDECAR_BIN" -key "$KEY_NAME.key" -id
        ;;

    tunnel-coordinator)
        WIN_ADDR="${2:-}"
        WIN_PEER_ID="${3:-}"
        [ -n "$WIN_ADDR" ] && [ -n "$WIN_PEER_ID" ] \
            || { echo "Usage: tunnel-coordinator <windows-addr> <coordinator PeerID from Windows>" >&2; exit 1; }
        "$SIDECAR_BIN" -key provider1.key \
            -listen /ip4/0.0.0.0/tcp/4100 \
            -forward "127.0.0.1:50200=$WIN_ADDR/p2p/$WIN_PEER_ID" \
            2>&1 | tee build/sidecar-provider1.log
        ;;

    provider1)
        mkdir -p "$PROVIDER_CACHE"
        "$DAN_BUILD/dan-stage-worker" \
            --coordinator 127.0.0.1:50200 \
            --provider-id linux-provider1 --cache-dir "$PROVIDER_CACHE" \
            2>&1 | tee build/provider1.log
        ;;

    weights)
        mkdir -p build/pipelined-ring-weights
        curl -fL --retry 3 -C - "$MODEL_URL" -o "$WEIGHTS"
        ACTUAL="$(sha256sum "$WEIGHTS" | awk '{print $1}')"
        if [ "$ACTUAL" != "$SHA256" ]; then
            echo "Weight SHA-256 mismatch: expected $SHA256, got $ACTUAL" >&2
            exit 1
        fi
        echo "$ACTUAL  $WEIGHTS" | tee weights-linux.sha256
        echo "This local copy is only for ring mode -- provider1 (baseline/pipelined) downloads its own."
        ;;

    tunnel-ring-server)
        WIN_PEER_ID="${2:-}"
        [ -n "$WIN_PEER_ID" ] || { echo "Usage: tunnel-ring-server <ring-client PeerID from Windows>" >&2; exit 1; }
        "$SIDECAR_BIN" -key ring-server.key \
            -listen /ip4/0.0.0.0/tcp/4102 -inbound 127.0.0.1:50103 \
            -allow "$WIN_PEER_ID" \
            2>&1 | tee build/sidecar-ring-server.log
        ;;

    tunnel-ringreturn-client)
        WIN_ADDR="${2:-}"
        WIN_PEER_ID="${3:-}"
        [ -n "$WIN_ADDR" ] && [ -n "$WIN_PEER_ID" ] \
            || { echo "Usage: tunnel-ringreturn-client <windows-addr> <ringreturn-server PeerID from Windows>" >&2; exit 1; }
        "$SIDECAR_BIN" -key ringreturn-client.key \
            -listen /ip4/0.0.0.0/tcp/4103 \
            -forward "127.0.0.1:50105=$WIN_ADDR/p2p/$WIN_PEER_ID" \
            2>&1 | tee build/sidecar-ringreturn-client.log
        ;;

    stage1-ring)
        require_file "$WEIGHTS" "Run: $0 weights"
        "$DAN_BUILD/dan-stage-worker" \
            --model "$WEIGHTS" \
            --stage-start 14 --stage-end 28 --host 127.0.0.1 --port 50102 \
            --ring-listen 127.0.0.1:50103 --next 127.0.0.1:50105 \
            --ctx 1024 --gpu-layers 999 --max-sessions 4 \
            2>&1 | tee build/stageB-ring.log
        ;;

    check-chunks)
        COUNT="$(grep -c 'phase=prefill-chunk' build/stageB-ring.log || true)"
        echo "prefill-chunk lines in stageB-ring.log: $COUNT"
        if [ "$COUNT" -lt 2 ]; then
            echo "WARNING: expected more than one non-final chunk. Chunking may not have fired -- check --prefill-chunk against the actual prompt length." >&2
        fi
        ;;

    gpu-stats)
        mkdir -p build
        if [ ! -f "$GPU_STATS_CSV" ]; then
            echo 'timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,power_w' > "$GPU_STATS_CSV"
        fi
        echo "Sampling nvidia-smi every 2s into $GPU_STATS_CSV -- Ctrl+C to stop."
        trap 'echo; echo "Stopped. Run: $0 stats"; exit 0' INT TERM
        while true; do
            nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits >> "$GPU_STATS_CSV"
            sleep 2
        done
        ;;

    stats)
        if [ -f build/stageB-ring.log ]; then
            COUNT="$(grep -c 'phase=prefill-chunk' build/stageB-ring.log || true)"
            echo "prefill-chunk lines in stageB-ring.log: $COUNT"
        else
            echo "No build/stageB-ring.log yet -- run stage1-ring first for chunk counts."
        fi
        if [ -f "$GPU_STATS_CSV" ]; then
            LINES="$(tail -n +2 "$GPU_STATS_CSV" | wc -l)"
            echo "$GPU_STATS_CSV: $LINES samples"
            if [ "$LINES" -gt 0 ]; then
                tail -n +2 "$GPU_STATS_CSV" | awk -F', *' '
                    { util+=$2; if ($3>maxmem) maxmem=$3; n++ }
                    END { printf "  avg gpu_util=%.1f%%  peak mem_used=%s MiB (n=%d)\n", util/n, maxmem, n }'
            fi
        else
            echo "No $GPU_STATS_CSV yet -- run '$0 gpu-stats' in a separate terminal during a test."
        fi
        ;;

    package)
        shopt -s nullglob
        FILES=(build/stageB-*.log build/sidecar-*.log build/provider1.log)
        for f in weights-linux.sha256 revision-linux.txt "$GPU_STATS_CSV"; do
            [ -f "$f" ] && FILES+=("$f")
        done
        if [ ${#FILES[@]} -eq 0 ]; then
            echo "No result files found yet -- run the earlier steps first." >&2
            exit 1
        fi
        tar -czf pipelined-ring-results.tar.gz "${FILES[@]}"
        echo "Wrote pipelined-ring-results.tar.gz"
        ;;

    *)
        usage
        ;;
esac
