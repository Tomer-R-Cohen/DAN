# Friends Testnet Guide

Gamer Provider Testnet v1 supports Linux PCs with an NVIDIA GPU on a trusted
private network. It does not expose DAN safely to the public internet. Do not
forward coordinator or RPC ports through a router.

## What you need

- A Linux PC with a working NVIDIA driver and `nvidia-smi`
- A private overlay connection to the coordinator, such as Tailscale
- CMake, a C++23 compiler, Git, `curl`, and `sha256sum`
- A llama.cpp `rpc-server` executable
- Free disk space at least as large as any assigned artifact

Confirm the GPU and private address:

```bash
nvidia-smi
tailscale ip -4
```

The advertised address must be numeric loopback, RFC1918 private IPv4, or the
Tailscale `100.64.0.0/10` range. DAN rejects wildcard and public advertised
addresses. Firewall policy must allow the coordinator to reach the selected RPC
worker port over the private overlay.

## Build once

```bash
git clone <TOMERS_DAN_REPOSITORY_URL> DAN
cd DAN
cmake -S . -B build
cmake --build build -j
```

Build the pinned llama.cpp RPC worker outside DAN if Tomer did not give you a
compatible executable:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
git -C ~/llama.cpp checkout --detach 95ef7fc16054e63b427a3ef00188e055ef7586d8
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build -j --target rpc-server
```

## Configure and start

Ask Tomer for the coordinator's private `HOST:PORT`. Replace the three values:

```bash
./scripts/setup_provider.sh \
  100.80.10.1:9000 \
  "$(tailscale ip -4)" \
  "$HOME/llama.cpp/build/bin/rpc-server"
```

The script checks prerequisites, builds DAN, creates `~/.dan/provider.conf` only
when absent, validates GPU detection, and prints the launch command. It does not
install drivers, alter networking, require root, or overwrite an existing config.

Start later with one command:

```bash
./build/dan-provider
```

`dan-provider` reads `~/.dan/provider.conf`, chooses the lowest NVIDIA GPU index,
reserves 1536 MiB by default, chooses an available worker port, creates a stable
random ID in `~/.dan/provider-id`, and joins the pool. Use `Ctrl-C` to stop
contributing; DAN stops only its owned RPC worker.

Expected startup resembles:

```text
DAN Provider

Provider ID: node-7f3a12c04b91
Name: tomer-pc
GPU: NVIDIA GeForce RTX 2070
GPU UUID: GPU-...
Device: CUDA0
VRAM total: 8192 MiB
Reserved: 1536 MiB
Available to DAN: 6656 MiB
Coordinator: 100.80.10.1:9000
Cache: /home/tomer/.dan/models
Worker: STOPPED (100.80.10.4:54321)
Coordinator: CONNECTED
Role: SPARE
```

If assigned, subsequent output shows `Assignment`, `DOWNLOADING`, `CACHED`,
`LOADING`, and `READY`. Tomer can enter `/providers` at the coordinator to see
the persistent ID, optional name, GPU, advertised VRAM, `ASSIGNED`/`SPARE` role,
shard state, worker endpoint, and last-seen age.

## Configuration

`~/.dan/provider.conf` uses one `key=value` per line:

```ini
coordinator=100.80.10.1:9000
provider_name=tomer-pc
cache_dir=/home/tomer/.dan/models
rpc_worker=/home/tomer/llama.cpp/build/bin/rpc-server
advertise_host=100.80.10.4
device=0
reserve_vram_mib=1536
reconnect_seconds=2
```

`device` and `provider_name` are optional. Without `device`, the lowest reported
GPU index is selected deterministically. `worker_port` is also optional; set a
fixed private-overlay port when local firewall policy requires one. Run
`./build/dan-provider --help` for CLI overrides. Validate without connecting or
starting a worker:

```bash
./build/dan-provider --check
```

The reconnect delay doubles up to 30 seconds during an outage and resets after a
successful registration. The same process, ID, verified cache, and compatible
running worker are reused after coordinator or network recovery.

## Cache and disk use

Verified artifacts live under:

```text
~/.dan/models/<model>/<version>/<shard>/<sha256>.artifact
```

They survive provider and coordinator restarts. Stopping DAN does not delete
them. Inspect current consumption with:

```bash
du -sh ~/.dan/models
```

Tomer should tell each participant the maximum expected artifact size before the
test. Cache eviction is manual in v1; stop `dan-provider` before deleting files.

## Changing contribution limits

Edit `reserve_vram_mib` and restart. The advertised value is detected total VRAM
minus this reserve. The reserve must be smaller than total VRAM. To choose another
installed GPU, set its numeric `device` index as shown by `nvidia-smi`.

This reserve is static. DAN does not yet notice a game starting or dynamically
reduce its allocation.

## Troubleshooting and logs

- `NVIDIA GPU detection failed`: run `nvidia-smi`; repair the driver outside DAN.
- `Configured CUDA device ... was not reported`: remove or correct `device=`.
- `VRAM reserve must be smaller`: lower `reserve_vram_mib`.
- Private-address error: use the numeric overlay address, never `0.0.0.0` or a
  public address.
- Cannot connect: confirm the coordinator address, overlay connectivity, and
  firewall. Reconnection happens automatically.
- Worker failure: confirm `rpc_worker` is executable and CUDA llama.cpp libraries
  are available.

Collect a shareable log without credentials:

```bash
mkdir -p ~/.dan/logs
./build/dan-provider 2>&1 | tee ~/.dan/logs/provider-$(date +%Y%m%d-%H%M%S).log
nvidia-smi -q > ~/.dan/logs/nvidia-smi-q.txt
```

Send Tomer the provider log, `nvidia-smi` output, DAN Git commit, llama.cpp commit,
and `du -sh ~/.dan/models`. Do not send VPN credentials or private keys.

## First home-GPU validation

Use Tomer's RTX 2070 as Machine A and at least one friend's NVIDIA GPU as Machine
B. Add Machine C as an eligible spare when available. Keep all RPC endpoints on
the private overlay.

1. Start the coordinator and record its commit and private address.
2. Run one `dan-provider` command on each PC.
3. Record detected total/advertised VRAM, persistent IDs, and roles.
4. Wait for every assigned artifact and worker, then replica/runtime `READY`.
5. Send ten real prompts and retain request latency and tokens/second evidence.
6. Confirm runtime and worker PIDs stay unchanged and no artifact redownloads.
7. Stop one required provider. With Machine C, require automatic replacement and
   serving recovery; without it, record `NONE_ELIGIBLE`.
8. Restart the original provider and confirm the same ID/cache return as a spare
   without stealing the healthy assignment.
9. Compare a strong GPU alone, strong plus weaker GPU, and all available GPUs.
   Record VRAM contribution, CUDA utilization, throughput, network traffic, and
   recovery time. Do not assume the weaker GPU improves performance.

This is a trusted friends testnet, not a production security boundary.
