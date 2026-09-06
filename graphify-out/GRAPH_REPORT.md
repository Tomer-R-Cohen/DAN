# Graph Report - DAN  (2026-09-06)

## Corpus Check
- 41 files · ~36,976 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 271 nodes · 490 edges · 24 communities (14 shown, 10 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d7185d3a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Provider
- Capabilities
- Graph Query Flow
- ModelDefinition
- protocol.hpp
- DAN
- Request
- DistributedConfig
- PreparationTests
- Graph Export Formats
- Distributed Runtime Library
- Coordinator
- validate_gpu.py
- Cross-Repository Graph Merge
- Length-Prefixed Framing
- Pod A Two-GPU Smoke Test
- Graph Health Check
- Graphify Workflow
- DAN CMake Build
- control_plane.hpp
- DistributedGroup
- Working Instructions
- Model Registry
- RPC Disk Cache Reuse

## God Nodes (most connected - your core abstractions)
1. `Provider` - 46 edges
2. `DistributedGroup` - 23 edges
3. `main()` - 22 edges
4. `Request` - 18 edges
5. `Capabilities` - 16 edges
6. `ModelDefinition` - 16 edges
7. `ManagedModel` - 15 edges
8. `CachedShard` - 14 edges
9. `DAN` - 14 edges
10. `ShardMetadata` - 12 edges

## Surprising Connections (you probably didn't know these)
- `print_control_plane()` --references--> `CachedShard`  [INFERRED]
  src/coordinator.cpp → include/control_plane.hpp
- `main()` --references--> `ModelRegistry`  [INFERRED]
  src/coordinator.cpp → include/model_registry.hpp
- `has_cached_shard()` --references--> `ShardMetadata`  [EXTRACTED]
  src/coordinator.cpp → include/control_plane.hpp
- `Provider` --references--> `CachedShard`  [EXTRACTED]
  src/coordinator.cpp → include/control_plane.hpp
- `Capabilities` --references--> `CachedShard`  [EXTRACTED]
  src/provider.cpp → include/control_plane.hpp

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Cache and Persistence Experiment** — docs_next_gpu_experiment_experiment, docs_pod_a_next_test_prompt_pod_a, docs_pod_b_next_test_prompt_pod_b, docs_provider_lifecycle_lifecycle [EXTRACTED 1.00]
- **Graphify Extraction Pipeline** — _codex_skills_graphify_skill_structural_extraction, _codex_skills_graphify_skill_semantic_extraction, _codex_skills_graphify_skill_ast_semantic_merge, _codex_skills_graphify_skill_community_analysis [EXTRACTED 1.00]
- **Distributed GPU Validation Documents** — docs_pod_a_two_gpu_smoke_prompt_pod_a, docs_pod_b_two_gpu_smoke_prompt_pod_b, docs_two_gpu_smoke_report_result, docs_qwen3_two_gpu_report_result [INFERRED 0.85]
- **Automated Graph Maintenance** — _codex_skills_graphify_references_add_watch_folder_watcher, _codex_skills_graphify_references_hooks_post_commit_hook, _codex_skills_graphify_references_update_incremental_update [INFERRED 0.85]

## Communities (24 total, 10 thin omitted)

### Community 0 - "Provider"
Cohesion: 0.09
Nodes (53): deque, ManagedModel, id, shards, version, optional, any_busy(), any_group_busy() (+45 more)

### Community 1 - "Capabilities"
Cohesion: 0.11
Nodes (26): socket, Capabilities, backend, cached_shards, control_plane, device_type, gpu_name, model_name (+18 more)

### Community 2 - "Graph Query Flow"
Cohesion: 0.08
Nodes (26): Folder Watcher, URL Ingestion, Confidence Audit Trail, Deterministic Node IDs, Semantic Extraction Contract, CLAUDE.md Graphify Integration, Post-Commit Graph Hook, Breadth-First Traversal (+18 more)

### Community 3 - "ModelDefinition"
Cohesion: 0.13
Nodes (17): size_t, string, vector, ModelDefinition, context_length, distributed, family, id (+9 more)

### Community 4 - "protocol.hpp"
Cohesion: 0.36
Nodes (7): size_t, string, string_view, receive_all(), receive_message(), send_all(), send_message()

### Community 5 - "DAN"
Cohesion: 0.30
Nodes (19): Current Architecture, Architectural Decisions, Validate Weight Reuse, GPU Results Template, Single CUDA Provider Validation, Model Strategy and Registry, RPC Cache and Persistent Runtime Experiment, Persistent RPC Runtime (+11 more)

### Community 6 - "Request"
Cohesion: 0.21
Nodes (11): string, Request, allow_distributed, allow_single, id, model, model_path, prompt (+3 more)

### Community 7 - "DistributedConfig"
Cohesion: 0.19
Nodes (13): DistributedConfig, endpoints, executable, id, model, tensor_split, string, string (+5 more)

### Community 8 - "PreparationTests"
Cohesion: 0.17
Nodes (7): fields(), main(), Small control-plane provider simulator; it never loads or transfers weights., receive(), send(), PreparationTests, CPU-only regression checks for the GPU deployment path (no CUDA claims).

### Community 9 - "Graph Export Formats"
Cohesion: 0.67
Nodes (3): Graph Export Formats, Graphify MCP Server, Token Reduction Benchmark

### Community 10 - "Distributed Runtime Library"
Cohesion: 0.67
Nodes (3): Coordinator Target, Distributed Runtime Library, Distributed Model Targets

### Community 11 - "Coordinator"
Cohesion: 0.67
Nodes (3): Coordinator, Persistent Provider, Round-Robin Scheduling

### Community 19 - "control_plane.hpp"
Cohesion: 0.13
Nodes (29): FieldHandler, cached_shard_value(), CachedShard, hash, model_id, shard_id, version, ShardState (+21 more)

### Community 20 - "DistributedGroup"
Cohesion: 0.14
Nodes (14): DistributedConfig, ExecutionTargetType, pid_t, DistributedGroup, busy, child, completed_requests, config (+6 more)

## Knowledge Gaps
- **107 isolated node(s):** `id`, `size_bytes`, `hash`, `source`, `min_vram_mib` (+102 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 140 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Provider` connect `Provider` to `Capabilities`, `control_plane.hpp`, `DistributedGroup`, `Request`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `CachedShard` connect `control_plane.hpp` to `Provider`, `Capabilities`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `ModelDefinition` connect `ModelDefinition` to `control_plane.hpp`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `main()` (e.g. with `ModelRegistry` and `.find()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `id`, `size_bytes`, `hash` to the rest of the system?**
  _107 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Provider` be split into smaller, more focused modules?**
  _Cohesion score 0.08665269042627533 - nodes in this community are weakly interconnected._
- **Should `Capabilities` be split into smaller, more focused modules?**
  _Cohesion score 0.10887096774193548 - nodes in this community are weakly interconnected._