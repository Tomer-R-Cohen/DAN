#!/usr/bin/env bash
# Linux-side commands for docs/PIPELINED_RING_PHYSICAL_TEST.md.
#
# Covers each step of that doc as a subcommand instead of typing the full
# command by hand. Several steps run in separate terminals at the same time
# (e.g. stage 1 and up to three sidecar tunnels), so this is a menu of
# steps, not a single "run everything" script -- open one terminal per step
# that needs to stay running.
#
# Usage: pipelined_ring_test_linux.sh <step> [options]
#   prereqs
#   build [branch-or-commit]
#   weights
#   sidecar-id <key-name>
#   tunnel-control-server <windows-peer-id>
#   tunnel-ring-server <windows-peer-id>
#   tunnel-ringreturn-client <windows-addr> <windows-peer-id>
#   stage1
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
GPU_STATS_CSV=build/gpu-stats-linux.csv
STEP="${1:-}"

usage() {
    cat <<'EOF'
Steps (see docs/PIPELINED_RING_PHYSICAL_TEST.md for what each one means):

  prereqs                                      step 1: check driver/toolchain, install build deps
  build [branch-or-commit]                     step 2: clone/build DAN + llama.cpp + sidecar
  weights                                       step 4: download weights
  sidecar-id <key-name>                         step 5: create one sidecar identity, print its PeerID
  tunnel-control-server <win-peer-id>           step 5, pair 1: Linux side (needed for every section)
  tunnel-ring-server <win-peer-id>              step 5, pair 2: Linux side (ring only)
  tunnel-ringreturn-client <win-addr> <win-id>  step 5, pair 3: Linux side (ring only)
  stage1                                        step 6/7: stage 1, plain (hub-and-spoke)
  stage1-ring                                   step 8: stage 1, with --ring-listen/--next
  check-chunks                                  step 8: confirm chunked prefill actually fired
  gpu-stats                                     sample nvidia-smi every 2s into $GPU_STATS_CSV;
                                                 run in its own terminal alongside stage1/stage1-ring
  stats                                         summarize chunk counts and gpu-stats.csv so far
  package                                       step 9: tar up logs, revision file, and stats

Sidecar tunnel processes, stage worker processes, and gpu-stats are all
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

    weights)
        REVISION=91cad51170dc346986eccefdc2dd33a9da36ead9
        SHA256=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
        URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$REVISION/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        mkdir -p build/pipelined-ring-weights
        curl -fL --retry 3 -C - "$URL" -o "$WEIGHTS"
        ACTUAL="$(sha256sum "$WEIGHTS" | awk '{print $1}')"
        if [ "$ACTUAL" != "$SHA256" ]; then
            echo "Weight SHA-256 mismatch: expected $SHA256, got $ACTUAL" >&2
            exit 1
        fi
        echo "$ACTUAL  $WEIGHTS" | tee weights-linux.sha256
        ;;

    sidecar-id)
        KEY_NAME="${2:-}"
        [ -n "$KEY_NAME" ] || { echo "Usage: sidecar-id <key-name>  (e.g. control-server, ring-server, ringreturn-client)" >&2; exit 1; }
        "$SIDECAR_BIN" -key "$KEY_NAME.key" -id
        ;;

    tunnel-control-server)
        WIN_PEER_ID="${2:-}"
        [ -n "$WIN_PEER_ID" ] || { echo "Usage: tunnel-control-server <control-client PeerID from Windows>" >&2; exit 1; }
        "$SIDECAR_BIN" -key control-server.key \
            -listen /ip4/0.0.0.0/tcp/4101 -inbound 127.0.0.1:50102 \
            -allow "$WIN_PEER_ID" \
            2>&1 | tee build/sidecar-control-server.log
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

    stage1)
        require_file "$WEIGHTS" "Run: $0 weights"
        "$DAN_BUILD/dan-stage-worker" \
            --model "$WEIGHTS" \
            --stage-start 14 --stage-end 28 --host 127.0.0.1 --port 50102 \
            --ctx 1024 --gpu-layers 999 --max-sessions 4 \
            2>&1 | tee build/stageB-baseline.log
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
        FILES=(build/stageB-*.log build/sidecar-*.log)
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
