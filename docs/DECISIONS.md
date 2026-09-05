# Architectural Decisions

## Keep raw blocking POSIX TCP sockets

The project uses the Linux socket API with `poll()` to multiplex providers,
without coordinator worker threads or a networking framework. Individual frame
and registration reads remain blocking; a stalled partial sender can still
stall the event loop. Inference runs concurrently in separate provider processes.

## Add four-byte length-prefixed framing

Prompts and generated answers can be split across TCP reads. A fixed-width
network-order length followed by payload bytes is the smallest reliable framing
scheme for arbitrary text. Messages are limited to 16 MiB to reject unreasonable
or corrupt lengths before allocating memory.

## Keep one interactive llama.cpp child process

The provider uses the official `llama-completion` executable instead of linking
llama.cpp into DAN. It starts the executable once in conversation and
interactive-first mode, then exchanges turns through pipes. The process and its
loaded GGUF model remain alive for the provider session. `execvp()` is used
directly rather than a shell.

## Use llama-completion's simple interactive I/O

`llama-completion --conversation --interactive-first --simple-io` exposes a
small subprocess interface. The provider waits for its `\n> ` input marker to
separate generated turns. This avoids restarting the runtime while keeping the
current external-runtime integration small.
Persistent generation defaults to `--n-predict -1`: the pinned runtime carries
positive interactive token budgets across turns, which can cause empty answers.
The GPU test driver bounds requests with a timeout; normal DAN does not yet.

## Allow one active request per provider

The coordinator, providers, TCP connections, and llama.cpp children serve
requests concurrently across providers. A provider remains strictly sequential
and never receives new work while busy. Conversation context is retained
independently by each provider's llama.cpp process.

## Use round-robin across live providers

The coordinator stores connected providers in connection order and sends each
queued prompt to the next available provider. After the last provider, selection
wraps to the first. A busy provider is skipped until its response arrives.
`poll()` admits new providers, accepts stdin while inference runs, collects
responses, and notices disconnects. An in-flight request from a disconnected
provider is returned to the front of the queue.

This scheduler deliberately has no worker threads, load-based selection,
priority, persistence, or pricing logic. Model compatibility is checked by the
model-aware execution-target path described below.

## Put request IDs in inference messages

Every coordinator prompt receives a monotonically increasing integer ID. The
provider echoes it in `RESPONSE` or `ERROR`, allowing the coordinator to match
and print out-of-order completions. A response with the wrong ID is treated as a
provider protocol failure.

## Keep a FIFO queue in memory

Prompts that cannot be dispatched immediately wait in a `std::deque`. A provider
completion marks it available and runs dispatch again immediately. On `exit`,
the coordinator stops accepting prompts, finishes queued and active work, and
then shuts providers down cleanly.

## Register simple self-reported capabilities

After `HELLO`, each provider sends one framed `CAPABILITIES` message containing
newline-separated key/value fields. The coordinator stores the fields with the
live socket and prints them. Defaults preserve the short provider command, while
optional command-line flags describe distinct hardware and model deployments.
The model name participates in compatibility checks. Hardware labels remain
informational and do not rank otherwise compatible round-robin targets.

## Measure request time at the coordinator

The coordinator uses a monotonic clock from prompt send through complete
response receipt. It stores each provider's last elapsed time, completed request
count, and cumulative time for an average. This measurement includes TCP
transport but closely represents user-observed request latency. Tokens per
second is omitted because the current llama.cpp stdout integration does not
provide a reliable generated-token count.

## Keep runtime and model outside the repository

For rented GPU validation, pin the current llama.cpp revision and use explicit
provider runtime options. CUDA metadata is not proof of offload. Record direct
llama.cpp diagnostics and nvidia-smi memory samples separately from DAN latency.
The benchmark driver tests real processes and adds no scheduling behavior.

llama.cpp source/build products and GGUF weights are external dependencies, not
vendored project code. Their paths are supplied when starting the provider.

## Treat distributed groups as explicit execution targets

The coordinator stores normal providers and distributed groups as two simple
execution-target types. Groups are configured manually at startup and remain
separate from dynamic provider registration. Ordinary prompts use the normal
provider queue; `/group <id> <prompt>` selects a named group explicitly. This
avoids automatic group formation.

Distributed RPC invocation lives in a shared `distributed_runtime` library.
Both the coordinator and standalone diagnostic executable reuse it, avoiding
two implementations of model splitting and subprocess handling.

## Select by model, availability, then round-robin

`/model <model-name> <prompt>` is the model-aware request form. The coordinator
compares it with provider metadata and configured group models, considers only
available compatible targets, and round-robins across both target types.
Requests wait when compatible targets are busy. With no compatible target, a
request fails clearly without blocking other queued work.

Plain prompts remain backward-compatible provider requests. `/group` remains a
diagnostic override rather than a normal routing requirement.

## Keep models in a replaceable registry

An optional pipe-delimited registry maps stable request IDs to model metadata
and runtime compatibility names. It classifies models as `test`, `main`, or
`distributed` and records memory, context, requirements, and permitted target
types. The coordinator contains no SmolLM, gpt-oss, or other family-specific
branches. Changing `dan-main` is a configuration change.

Registry execution flags constrain selection before normal availability and
round-robin rules. Providers still declare what is actually loaded; configured
groups can reference a registry path with `@model-id`.

## Delegate model splitting to llama.cpp RPC

DAN does not implement tensor or pipeline parallelism. The experiment invokes
an RPC-enabled `llama-completion` with `--rpc`, selects only its `RPC0,RPC1,...`
devices, and uses `--n-gpu-layers 99`. llama.cpp distributes weights and KV
cache according to available device memory by default. The optional
`--tensor-split` setting allows an explicit ratio.

The RPC backend is proof-of-concept and has no authentication or encryption.
It must be restricted to a trusted LAN or local test environment.

## Measure distributed end-to-end latency

Each group request times process startup, model distribution/loading, prompt
evaluation, generation, and RPC traffic together. llama.cpp diagnostics remain
on stderr for prompt/evaluation token rates. A useful network assessment
compares identical fixed-token local, one-RPC, and multi-RPC runs; LAN bandwidth
and latency must be recorded on the target hosts because localhost results do
not characterize a physical network.
