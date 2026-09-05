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

## Current

- Prepared single-GPU runbook, benchmark driver, checklist and results template.
- Pending: rent an NVIDIA L4-class machine and verify direct CUDA inference,
  then ten sequential `dan-main` requests through one persistent CUDA provider.

## Next

- Review GPU evidence and candidate quality before promoting a main model.
- Later, validate a model exceeding either GPU's memory across two RPC providers.
- Improve runtime error details returned to the coordinator

## Future

- Richer resource validation and advanced scheduling (metadata and queues already exist)
- Persistent distributed-model runtime after the RPC feasibility experiment
- Failure recovery, authentication, accounting, and pricing
- GPU execution and larger AI workloads
- Token rewards, peer-to-peer operation, and model evolution
