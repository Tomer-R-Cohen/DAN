#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$root/build/sidecar"}
mkdir -p "$out"
cd "$root/sidecar"
export GOTOOLCHAIN=go1.25.7
go test ./...
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -buildvcs=false -o "$out/dan-sidecar-linux-amd64" .
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -buildvcs=false -o "$out/dan-sidecar-windows-amd64.exe" .
