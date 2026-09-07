# Current Architecture

```text
User
  |
  | prompts
  v
Coordinator -- framed TCP --> Provider 1 <== pipes ==> llama-completion + model
     |       -- framed TCP --> Provider 2 <== pipes ==> llama-completion + model
     |       -- framed TCP --> Provider 3 <== pipes ==> llama-completion + model
     |
     +<-------------- responses may arrive out of order -------------------+
```

## Coordinator

The coordinator listens on TCP port `9000`, accepts any number of providers,
and verifies each provider's `HELLO` and `CAPABILITIES` registration. It stores
the provider's reported ID, device, GPU, VRAM, model, and backend beside its
socket and prints that information on registration. It uses `poll()` to watch
standard input, new connections, and idle-provider disconnections without
adding threads.

The coordinator assigns every submitted prompt a request ID and puts it in a
FIFO queue. It dispatches queued work round-robin to available providers; each
provider can have at most one active request. Standard input remains active
while inference runs. If every provider is busy, prompts remain queued.

Provider response sockets are watched by the same `poll()` event loop. A
completion marks that provider available and immediately triggers another
dispatch. Responses may arrive out of submission order, so output includes both
request and provider IDs. For every successful response the coordinator stores
and prints elapsed time, completed request count, and average elapsed time.
Typing `exit` stops new input, drains queued and active requests, then sends
`BYE` to every connected provider.

Model-qualified input uses `/model <model-name> <prompt>`. The coordinator
filters providers and groups by exact model compatibility, then availability,
then round-robins among remaining targets. An incompatible request is reported
without affecting other work. Plain prompts retain legacy provider scheduling;
`/group` remains a diagnostic override.

With `--models`, the model argument is a registry ID. A model definition holds
role, family, path, runtime name, quantization, approximate memory, context
length, requirements, and allowed execution modes. The scheduler resolves the
ID before target selection, keeping model-family knowledge out of networking
and scheduling code.

## Provider

Each provider starts one interactive `llama-completion` child and waits for its
GGUF model to load, then connects to the configured coordinator. It sends every
received prompt to that child's standard input and captures each generated
answer from its standard output. The child, model, provider, and connection stay
alive between its assigned turns. llama.cpp diagnostics remain on the provider's standard
error stream.

`fork()` creates the inference child process once, `execvp()` replaces that child
with llama.cpp, and pipes carry prompts and answers until shutdown. No shell is
involved, so prompt text is passed as data rather than interpreted as a command.

Providers derive a default ID from hostname and process ID, use the GGUF
filename as the default model name, and default to CPU metadata. Command-line
metadata flags describe GPU or other configurations. This information is
self-reported and is not hardware-attested.

Provider runtime flags separately control GPU layers, runtime device, context
size and generation cap. The GPU validation driver reads the registry context
and uses these flags; normal coordinator requests do not reconfigure runtimes.
Provider startup reports time until ready, including warmup. Real A40 validation
passed with Qwen3-30B-A3B Q4_K_M at 32,768 context and ten persistent requests;
see `PROJECT_STATUS.md` for measurements and committed evidence.

## Communication

TCP is a reliable, ordered byte stream, not a message protocol. Every DAN
message therefore has a four-byte length prefix followed by its text payload.
The shared framing loops over `send()` and `recv()` because either call may
transfer fewer bytes than requested.

## Integrated Distributed-Model Targets

The coordinator supports two execution-target types: a connected single
provider and a manually configured distributed model group. A group spreads one
model across two or more llama.cpp RPC workers:

```text
                         TCP RPC                 accelerator A
                        /       ggml-rpc-server <------------>
DAN experiment -> RPC backend
                        \       ggml-rpc-server <------------>
                         TCP RPC                 accelerator B
```

At startup the coordinator stores and displays each group's name, model path,
RPC endpoints, and tensor split. Model-qualified requests can select it
automatically; `/group <id> <prompt>` remains an override. A group has
available/busy state and at most one active request. Its result uses the same
request ID and updates group latency, request count, and average latency.

For a group request, the coordinator starts an asynchronous child job using the
shared distributed runtime. It launches RPC-enabled `llama-completion`, selects
only `RPC0,RPC1,...`, requests layer offload, and optionally passes
`--tensor-split`. llama.cpp owns tensor placement and RPC transport. The
standalone `distributed_model_experiment` remains as a diagnostic frontend to
the same shared runtime; it is no longer required for coordinator operation.

The current runtime uses `--n-gpu-layers 99`, `--n-predict 256`, and
`--no-conversation`. It launches a fresh process for every group request, so
latency includes model loading/distribution. Registry context is not forwarded;
the pinned runtime accepts `LLAMA_ARG_CTX_SIZE` from the environment. Expose
one CUDA device on each RPC server so `RPC0,RPC1` maps to the intended nodes.
Selecting only RPC devices excludes client accelerators but does not eliminate
normal CPU work or CPU buffer fallbacks. External placement logs and telemetry
are required to prove both GPUs participate and establish aggregate-VRAM fit.

Localhost CPU RPC and a real two-pod CUDA RPC smoke test have passed. The smoke
test mapped RPC0 to an RTX 3090 and RPC1 to an RTX A4500 and produced responses
through both the standalone frontend and DAN group path. It proves participation
but not aggregate-memory necessity because the test model fit on either GPU.
The follow-up Qwen3 run also passed at 32,768 context with both GPUs active; its
model-distribution latency was roughly 13 minutes. See `PROJECT_STATUS.md` and
`SETUP.md` for evidence and deployment requirements.

## Provider Control Plane and Managed Worker Runtime v1

The coordinator can additionally load one `dan-main` manifest. The manifest is
the durable model/version/shard list; every shard has an ID, expected size,
content hash, source metadata, and optional minimum VRAM. It describes exactly
one replica, but the shard collection is arbitrary-length: the same code handles
2, 3, 4, 8, or more managed providers.

Managed providers reuse the existing framed TCP registration connection. They
add numeric VRAM, control-plane support, and zero or more cached identities of
the form `model|version|shard|hash`. The coordinator assigns each required shard
to one eligible unassigned live provider. Exact cached identity is preferred,
followed by greatest reported usable VRAM and provider ID. Healthy assignments
are never displaced by later joins.

```text
provider:  OFFLINE <-> ONLINE (registration and heartbeat timeout)
shard:     UNASSIGNED -> ASSIGNED -> DOWNLOADING -> CACHED -> LOADING -> READY
                                   \-----------------> ERROR
replica:   READY only when every required shard has an online READY provider
```

Every managed provider sends periodic `HEARTBEAT` frames. The coordinator uses
a monotonic last-seen time and marks a provider `OFFLINE` after the configured
timeout. Its live assignment is released for recovery while its provider identity
and cache inventory remain known. `/providers` renders assigned, spare, and offline
providers. An exact cached identity starts a replacement at `CACHED`; stale
version/hash claims do not.

The coordinator is authoritative for connections, assignment and aggregate
readiness. `managed_provider` is authoritative for local bytes and its owned
worker. It stores artifacts as
`<cache>/<model>/<version>/<shard>/<sha256>.artifact`, verifies the optional size
and SHA-256, and uses a temporary file plus rename. Local paths and `file://` use
filesystem copy; HTTP(S) invokes `curl` with fixed argv and no shell.

After `CACHED`, the coordinator sends `LOAD_SHARD`. The provider starts exactly
one configured RPC worker using `fork`/`execvp`, verifies its TCP listener, then
reports `READY`. `UNLOAD_SHARD` stops only that PID and retains disk cache. A
killed worker yields `ERROR`; `/load <provider-id>` retries from verified cache.
Worker endpoints must be explicit non-wildcard addresses. Existing whole-model
providers and manual RPC groups are unchanged. See
[PROVIDER_LIFECYCLE.md](PROVIDER_LIFECYCLE.md).

Repeated full-model streaming during user requests is NOT the intended DAN
production architecture. Current distributed process-per-request execution is
an experimental legacy adapter.

## Persistent Managed Distributed Serving v1

When every assigned shard is online and `READY`, the coordinator starts one
loopback-bound `llama-server`. Providers are ordered by shard index; their private
worker endpoints become the comma-separated `--rpc` list, devices become
`RPC0..RPCN-1`, and reported VRAM becomes the simple `--tensor-split`. No provider
count is fixed. The coordinator-side GGUF path and server executable are explicit
startup configuration because llama.cpp's RPC client still owns model loading.

```text
provider workers READY -> replica READY -> runtime STARTING -> runtime READY
                                                        |           |
                                                        +-> ERROR <-+
```

`/model dan-main <prompt>` is accepted only while both replica and runtime are
ready. One server process retains the loaded/distributed model. Per-request native
HTTP helpers call llama-server's `/completion` endpoint and return the existing DAN
request ID without blocking provider heartbeat processing. They are transports,
not inference runtimes. Runtime startup may distribute tensors once; ordinary
requests do not recreate the server or intentionally redistribute the model.

Provider loss immediately makes the replica not ready and terminates the owned
server. The missing shard is automatically assigned to an eligible spare. When
the replacement reaches `READY`, the server starts again automatically. A server
crash remains `ERROR`; `/runtime start` retries it. `/runtime restart` and
`/runtime stop` affect only DAN's PID.

## Automatic Provider Replacement / Reassignment v1

An online provider with no required assignment is a spare. When an assigned
provider disconnects or times out, ownership is released and the normal managed
preparation path assigns the missing shard. Candidates must be online,
control-plane capable, idle, unassigned, above the shard's minimum VRAM, and not
already failed for that shard. Selection is deterministic: exact cache match,
then VRAM descending, then provider ID ascending.

An `ERROR` owner is replaced when another candidate exists. Failed candidates
are excluded for that shard until they reconnect, preventing tight retry loops.
With no candidate, the shard stays visibly missing and a later eligible join is
considered automatically. Reconnecting former owners remain spare once a healthy
replacement owns the shard, avoiding failback oscillation.

## Gamer Provider Testnet v1

`dan-provider` is a Linux NVIDIA onboarding front end; it does not replace the
managed runtime. It reads a small `key=value` config, queries `nvidia-smi` through
fixed argv, selects the lowest GPU index unless overridden, subtracts static VRAM
headroom, persists a random provider ID, validates a numeric private advertised
IPv4 address, chooses a worker port, and then `exec`s the existing sibling
`managed_provider` with explicit validated arguments.

The managed provider reconnects with capped exponential backoff and re-registers
its verified cache inventory. An unchanged assignment reuses an already healthy
owned worker rather than launching another process. Coordinator BYE/network loss
does not erase cache or identity. SIGINT/SIGTERM stops the provider-owned worker.
The coordinator also records an optional friendly provider name separately from
the stable identity.

This is a trusted 2–5-machine private-overlay testnet. RPC endpoints may use
loopback, RFC1918 IPv4, or Tailscale's `100.64.0.0/10`; public/wildcard advertised
addresses are rejected by `dan-provider`. There is no discovery, NAT traversal,
driver installation, or public authentication.

## Current Limits

- One coordinator; up to one active request per persistent provider
- No coordinator worker threads; readiness is handled with blocking `poll()`
- Registry IDs resolve to exact runtime names; no aliases or fuzzy matching
- No load-aware or performance-aware scheduling
- Conversation context is retained separately by each provider
- Elapsed time includes request/response transport; token throughput is not yet
  available from the subprocess interface
- Queue is in memory and is not persisted
- Distributed-model RPC is experimental, unauthenticated, and LAN/local only
- 16 MiB maximum framed message size
- No authentication, encryption, service registry, or advanced scheduling
- The managed path is limited to one `dan-main` model, one replica, one assignment
  and worker per provider, one sequential persistent runtime, and sticky placement
- Managed artifact files and llama.cpp's coordinator-side GGUF/RPC tensor transfer
  remain separate; proactive rebalancing and managed tensor-cache policy are absent
- Gamer onboarding is Linux/NVIDIA only, reserves a static amount of VRAM, and
  selects only one local GPU per provider process
