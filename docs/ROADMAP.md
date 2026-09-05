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

## Current

- Existing standalone RPC and DAN distributed-group paths are ready for a
  controlled two-worker CUDA test with configuration; no source changes required.
- Pending hardware validation: two separately provisioned GPUs participate in
  one inference, through both the standalone experiment and DAN coordinator.

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
