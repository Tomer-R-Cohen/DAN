# Roadmap

## Completed

- Blocking TCP connection between one provider and coordinator
- Bidirectional `HELLO` and fixed-job exchange
- Length-prefixed message framing with complete send/receive loops
- Coordinator prompt input and provider response output
- Provider execution of a local GGUF model through llama.cpp
- First end-to-end CPU inference over TCP
- Persistent coordinator/provider session over one TCP connection
- Repeated inference with one loaded llama.cpp model process
- Multiple persistent providers connected to one coordinator
- Sequential round-robin prompt distribution
- Removal of disconnected providers from scheduling
- Provider capability registration and display
- Per-provider request timing and running averages
- Concurrent inference across persistent providers
- FIFO waiting queue and available/busy provider state
- Request IDs for out-of-order responses
- Separate llama.cpp RPC distributed-model experiment harness
- Two-endpoint local RPC inference and latency comparison
- Distributed groups integrated into the main coordinator as execution targets
- Manual group routing with shared request IDs and performance tracking
- Automatic model-compatible selection across both execution-target types
- Availability filtering and cross-target round-robin scheduling
- Replaceable model registry with test, main, and distributed roles
- Registry-based execution constraints and stable model IDs
- Real A40 CUDA validation: Qwen3-30B-A3B Q4_K_M, 32,768 context,
  10/10 persistent requests, 3.845 s mean latency, 21,227 MiB peak VRAM
- Single-GPU benchmark evidence committed and two-GPU readiness audit completed
- Two-pod CUDA RPC smoke test completed: RTX 3090 + RTX A4500, Qwen2.5-1.5B,
  standalone and DAN distributed-group responses, participation proven
- Qwen3-30B-A3B two-pod CUDA RPC run completed through standalone and DAN group
  paths at 32,768 context; both GPU allocations and activity were observed

## Current

- Select and validate a model/context whose allocation exceeds either GPU's
  capacity but fits across both remote CUDA workers.

## Next

- Validate a model exceeding either GPU's available VRAM across the two workers,
  preserving single-worker failures, successful split placement, and GPU telemetry.
- Broaden candidate quality and performance evaluation beyond the ten-prompt run.
- Improve runtime error details returned to the coordinator

## Future

- Richer resource validation and advanced scheduling (metadata and queues already exist)
- Persistent distributed-model runtime after the RPC feasibility experiment
- Failure recovery, authentication, accounting, and pricing
- Broader distributed GPU workloads and performance characterization
- Token rewards, peer-to-peer operation, and model evolution
