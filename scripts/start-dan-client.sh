#!/usr/bin/env bash
# Runs dan-client against the DAN network from any home connection, no coordinator and no
# port forwarding. Starts this machine's client sidecar (its own identity, DHT client,
# relay reservation, candidate API, ring return), runs dan-client --discover, then stops it.
#
#   ./start-dan-client.sh --bootstrap /ip4/VPS_IP/tcp/4001/p2p/VPS_PEERID \
#       --manifest config/provider-owned-qwen2.5-0.5b-q4km.json -- --prompt "Hello" --tokens 32
#
# Options: --bootstrap ADDR (repeatable), --relay ADDR (repeatable; default: bootstrap),
# --manifest FILE, --sidecar PATH, --client PATH, --state-dir DIR. After --, dan-client options.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sidecar="$here/../runtime/dan-sidecar"
client="$here/../dan-client"
state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/dan/client"
manifest=""
bootstrap=()
relay=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap) bootstrap+=("$2"); shift 2 ;;
        --relay) relay+=("$2"); shift 2 ;;
        --manifest) manifest="$2"; shift 2 ;;
        --sidecar) sidecar="$2"; shift 2 ;;
        --client) client="$2"; shift 2 ;;
        --state-dir) state_dir="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ ${#bootstrap[@]} -eq 0 || -z "$manifest" ]]; then
    echo "usage: $0 --bootstrap ADDR --manifest FILE [-- dan-client options]" >&2
    exit 2
fi
[[ ${#relay[@]} -eq 0 ]] && relay=("${bootstrap[@]}")
mkdir -p "$state_dir/logs"

free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}
api_port="$(free_port)"
return_port="$(free_port)"
while [[ "$return_port" == "$api_port" ]]; do return_port="$(free_port)"; done
ready="$state_dir/sidecar-ready-$$.txt"
rm -f "$ready"

args=(-key "$state_dir/identity.key" -listen /ip4/0.0.0.0/tcp/0 -dht client -reachability private
      -candidate-api "127.0.0.1:$api_port" -ring-inbound "127.0.0.1:$return_port"
      -ready-file "$ready" -log "$state_dir/logs/sidecar.log")
for peer in "${bootstrap[@]}"; do args+=(-bootstrap "$peer"); done
for peer in "${relay[@]}"; do args+=(-relay "$peer"); done

"$sidecar" "${args[@]}" >"$state_dir/logs/sidecar.out" 2>&1 &
sidecar_pid=$!
trap 'rm -f "$ready"; kill "$sidecar_pid" 2>/dev/null || true' EXIT

for _ in $(seq 360); do
    [[ -f "$ready" ]] && break
    if ! kill -0 "$sidecar_pid" 2>/dev/null; then
        echo "the client sidecar stopped; see $state_dir/logs/sidecar.log" >&2
        exit 1
    fi
    sleep 0.25
done
[[ -f "$ready" ]] || { echo "the client sidecar did not start; see $state_dir/logs/sidecar.log" >&2; exit 1; }
echo "DAN client identity: $(head -n 1 "$ready")"
"$client" --manifest "$manifest" --discover "127.0.0.1:$api_port" "$@"
