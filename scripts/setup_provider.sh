#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 COORDINATOR_HOST:PORT PRIVATE_OVERLAY_IP /path/to/rpc-server" >&2
    exit 1
fi

for command in cmake c++ nvidia-smi sha256sum curl; do
    command -v "$command" >/dev/null || {
        echo "Missing required command: $command" >&2
        exit 1
    }
done

rpc_worker=$3
[[ -x "$rpc_worker" ]] || { echo "RPC worker is not executable: $rpc_worker" >&2; exit 1; }

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
build_dir="$repo_dir/build"
state_dir="${HOME}/.dan"
config="$state_dir/provider.conf"

cmake -S "$repo_dir" -B "$build_dir"
cmake --build "$build_dir" -j "$(getconf _NPROCESSORS_ONLN)" --target dan-provider managed_provider
mkdir -p "$state_dir/models"

if [[ -e "$config" ]]; then
    echo "Keeping existing config: $config"
else
    umask 077
    printf '%s\n' \
        "coordinator=$1" \
        "advertise_host=$2" \
        "rpc_worker=$rpc_worker" \
        "cache_dir=$state_dir/models" \
        "reserve_vram_mib=1536" \
        "reconnect_seconds=2" >"$config"
    echo "Created $config"
fi

"$build_dir/dan-provider" --config "$config" --check
echo
echo "Start contributing with:"
echo "  $build_dir/dan-provider --config $config"
