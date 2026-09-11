#!/usr/bin/env bash
# Linux-side commands for docs/PIPELINED_RING_PHYSICAL_TEST.md.
#
# Three real phases: prereqs, build, run. 'run' does everything needed to
# actually run baseline/pipelined in one command -- it prints this
# machine's PeerID, starts the sidecar tunnel in the background, and runs
# the provider (which auto-registers and auto-downloads its assigned
# layers) in the foreground. 'run-ring' does the same for ring mode:
# downloads weights if missing, starts both ring tunnels in the background,
# runs the fixed-address stage in the foreground.
#
# check-chunks/gpu-stats/stats/package are evaluation utilities, not part
# of running a test, and stay separate on purpose.
#
# Usage: pipelined_ring_test_linux.sh <step> [options]
#   prereqs
#   build [branch-or-commit]
#   run <win-addr> <win-coordinator-peer-id>
#   run-ring <win-addr> <win-ring-client-peer-id> <win-ringreturn-server-peer-id>
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

# Same pinned model as the Windows side.
REVISION=91cad51170dc346986eccefdc2dd33a9da36ead9
SHA256=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$REVISION/qwen2.5-1.5b-instruct-q4_k_m.gguf"

usage() {
    cat <<'EOF'
Steps (see docs/PIPELINED_RING_PHYSICAL_TEST.md for background):

  prereqs                                       step 1: check driver/toolchain, install build deps
  build [branch-or-commit]                      step 2: clone/build DAN + llama.cpp + sidecar
  run <win-addr> <win-coordinator-peer-id>       step 3: baseline/pipelined. Prints this machine's
                                                  PeerID, runs the tunnel in the background, the
                                                  provider (auto-registers, auto-downloads its
                                                  assigned layers) in the foreground.
  run-ring <win-addr> <win-ring-client-peer-id>  step 3, ring mode: downloads weights if missing,
    <win-ringreturn-server-peer-id>              prints PeerIDs, runs both ring tunnels in the
                                                  background, the fixed-address stage in the
                                                  foreground. Ring can't auto-register -- see the doc.
  check-chunks                                   confirm chunked prefill actually fired (ring only)
  gpu-stats                                      sample nvidia-smi every 2s into $GPU_STATS_CSV;
                                                  run in its own terminal alongside run/run-ring
  stats                                          summarize chunk counts and gpu-stats.csv so far
  package                                        tar up logs, revision file, and stats

'run'/'run-ring' need this machine's printed PeerID given to the matching
Windows command, and the Windows PeerID(s) given back here -- that handshake
is manual and can't be scripted away. Everything else -- starting the tunnel
process, downloading weights when ring needs them, cleaning up afterward --
happens automatically inside the one command. The coordinator (and its own
decode_tok_s/latency/draft-accept report) runs on Windows -- see stats/
package there for those numbers; this side only has GPU utilization/VRAM and
chunk counts to record.
EOF
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Expected '$1' to exist. $2" >&2
        exit 1
    fi
}

get_weights() {
    if [ -f "$WEIGHTS" ]; then return; fi
    echo 'Downloading ring weights (one-time)...'
    mkdir -p build/pipelined-ring-weights
    curl -fL --retry 3 -C - "$MODEL_URL" -o "$WEIGHTS"
    local actual
    actual="$(sha256sum "$WEIGHTS" | awk '{print $1}')"
    if [ "$actual" != "$SHA256" ]; then
        rm -f "$WEIGHTS"
        echo "Weight SHA-256 mismatch: expected $SHA256, got $actual" >&2
        exit 1
    fi
    echo "$actual  $WEIGHTS" | tee weights-linux.sha256
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

    run)
        WIN_ADDR="${2:-}"
        WIN_PEER_ID="${3:-}"
        [ -n "$WIN_ADDR" ] && [ -n "$WIN_PEER_ID" ] \
            || { echo "Usage: $0 run <windows-addr> <coordinator PeerID from Windows 'run' output>" >&2; exit 1; }
        mkdir -p "$PROVIDER_CACHE"

        echo "--- This machine's PeerID (give it to the Windows run command) ---"
        "$SIDECAR_BIN" -key provider1.key -id
        echo '---'

        "$SIDECAR_BIN" -key provider1.key \
            -listen /ip4/0.0.0.0/tcp/4100 \
            -forward "127.0.0.1:50200=$WIN_ADDR/p2p/$WIN_PEER_ID" \
            > build/sidecar-provider1.log 2>&1 &
        TUNNEL_PID=$!
        trap 'kill "$TUNNEL_PID" 2>/dev/null || true' EXIT

        sleep 2
        "$DAN_BUILD/dan-stage-worker" \
            --coordinator 127.0.0.1:50200 \
            --provider-id linux-provider1 --cache-dir "$PROVIDER_CACHE" \
            2>&1 | tee build/provider1.log
        ;;

    run-ring)
        WIN_ADDR="${2:-}"
        WIN_RING_PEER_ID="${3:-}"
        WIN_RINGRETURN_PEER_ID="${4:-}"
        [ -n "$WIN_ADDR" ] && [ -n "$WIN_RING_PEER_ID" ] && [ -n "$WIN_RINGRETURN_PEER_ID" ] \
            || { echo "Usage: $0 run-ring <windows-addr> <ring-client PeerID from Windows> <ringreturn-server PeerID from Windows>" >&2; exit 1; }
        get_weights

        echo "--- This machine's PeerIDs (give them to the matching Windows run-ring command) ---"
        echo 'ring-server:'; "$SIDECAR_BIN" -key ring-server.key -id
        echo 'ringreturn-client:'; "$SIDECAR_BIN" -key ringreturn-client.key -id
        echo '---'

        "$SIDECAR_BIN" -key ring-server.key \
            -listen /ip4/0.0.0.0/tcp/4102 -inbound 127.0.0.1:50103 \
            -allow "$WIN_RING_PEER_ID" \
            > build/sidecar-ring-server.log 2>&1 &
        RING_TUNNEL_PID=$!
        "$SIDECAR_BIN" -key ringreturn-client.key \
            -listen /ip4/0.0.0.0/tcp/4103 \
            -forward "127.0.0.1:50105=$WIN_ADDR/p2p/$WIN_RINGRETURN_PEER_ID" \
            > build/sidecar-ringreturn-client.log 2>&1 &
        RINGRETURN_TUNNEL_PID=$!
        trap 'kill "$RING_TUNNEL_PID" "$RINGRETURN_TUNNEL_PID" 2>/dev/null || true' EXIT

        sleep 2
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
            echo "No build/stageB-ring.log yet -- run 'run-ring' first for chunk counts."
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
        for f in build/provider1.log weights-linux.sha256 revision-linux.txt "$GPU_STATS_CSV"; do
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
