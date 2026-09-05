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
Provider startup reports time until ready, including warmup. GPU validation is
pending; see `GPU_VALIDATION.md` for evidence requirements.

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
