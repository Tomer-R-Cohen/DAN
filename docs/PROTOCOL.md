# Current Protocol

Each message is one frame:

```text
4-byte unsigned payload length (network byte order)
payload length bytes of UTF-8-compatible text
```

The maximum payload is 16 MiB. Text content is currently treated as bytes; UTF-8
validation is not performed.

## `HELLO`

- Direction: Provider to coordinator
- Purpose: Identify the expected peer after connecting
- Payload: `HELLO`
- Expected next message: Provider sends `CAPABILITIES`

## `CAPABILITIES`

- Direction: Provider to coordinator
- Purpose: Register basic provider hardware, model, and backend information
- Payload:

```text
CAPABILITIES
provider_id=<provider identifier>
device_type=<device type>
gpu_name=<GPU name or not available>
vram=<VRAM or not available>
vram_mib=<numeric usable VRAM MiB, 0 if unspecified>
model_name=<model name>
backend=<CPU, CUDA, or another backend>
control_plane=<0 or 1>
cached_shard=<model ID>|<version>|<shard ID>|<content hash>
```

- Expected response: `ASSIGN_SHARD` for managed providers, `PROMPT` when selected
  by the legacy scheduler, or `BYE`

Values are self-reported display strings. Newline and carriage-return characters
are replaced with spaces by the provider. `cached_shard` may be repeated and is
accepted only when all four identity fields are present. Existing providers send
`control_plane=0` and require no managed fields.

## `ASSIGN_SHARD`

- Direction: Coordinator to managed provider
- Purpose: Assign one required shard of the single `dan-main` replica
- Payload: newline-separated `model_id`, `version`, `shard_id`, `size_bytes`,
  `hash`, and `source` fields
- Expected response: one or more `SHARD_STATE` messages

Assignments are sticky to provider identity across an offline/reconnect cycle.
The field collection makes no assumption about the replica's provider count.

## `SHARD_STATE`

- Direction: Managed provider to coordinator
- Purpose: Report the assigned shard's actual local preparation state
- Payload: newline-separated `model_id`, `version`, `shard_id`, `hash`, and
  `state` fields
- States: `ASSIGNED`, `DOWNLOADING`, `CACHED`, `LOADING`, `READY`, or `ERROR`

The coordinator rejects malformed fields, a mismatched assignment identity/hash,
`UNASSIGNED` reports, and invalid state transitions. The provider may report
`CACHED` immediately after assignment when its registration inventory exactly
matches the manifest; replica readiness still requires `READY`.

## `HEARTBEAT`

- Direction: Managed provider to coordinator
- Payload: `HEARTBEAT`
- Purpose: Refresh monotonic last-seen time even while inference is running

Managed providers send heartbeats once per second. Missing the coordinator's
default ten-second timeout marks the provider offline and removes its shard from
replica readiness without deleting the assignment or reported inventory.

## `PROMPT`

- Direction: Coordinator to provider
- Purpose: Request local model inference
- Payload: `PROMPT\n<request ID>\n<prompt text>`
- Example: `PROMPT\n42\nExplain TCP briefly.`
- Expected response: `RESPONSE` or `ERROR`

## `RESPONSE`

- Direction: Provider to coordinator
- Purpose: Return model-generated text
- Payload: `RESPONSE\n<request ID>\n<generated text>`
- Example: `RESPONSE\n42\nTCP is a reliable, ordered transport protocol.`
- Expected response: Another `PROMPT` or `BYE`

## `ERROR`

- Direction: Provider to coordinator
- Purpose: Report that local inference failed
- Payload: `ERROR\n<request ID>\n<short description>`
- Example: `ERROR\n42\nllama.cpp inference failed`
- Expected response: None; that provider connection is removed

## `BYE`

- Direction: Coordinator to provider
- Purpose: End the persistent session cleanly
- Payload: `BYE`
- Expected response: None; both programs exit

Each provider has its own framed TCP connection and sends `HELLO` followed by
`CAPABILITIES` after connecting. A managed connection may additionally carry
assignments, state reports, and heartbeats. A connection then carries any number
of sequential `PROMPT`/`RESPONSE` pairs before `BYE`. The coordinator can have one
active request on every provider connection simultaneously. Request IDs are
unique within one coordinator run and correlate responses that can arrive in a
different order.

Distributed groups are coordinator-local configured targets, not DAN TCP
providers. Their RPC traffic uses llama.cpp's protocol, while an internal framed
socket returns the request ID and result from the asynchronous group job. No new
network-visible DAN message type is required.

The requested model is coordinator scheduling metadata, not a new provider
message field. After selection, the existing request-ID/prompt frame is sent
unchanged; supported-model information comes from `CAPABILITIES`.
