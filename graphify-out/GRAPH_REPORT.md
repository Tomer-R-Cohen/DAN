# Graph Report - DAN  (2026-09-06)

## Corpus Check
- Corpus is ~32,300 words - fits in a single context window. You may not need a graph.

## Summary
- 208 nodes · 317 edges · 24 communities (11 shown, 13 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.86)
- Token cost: 43,167 input · 6,500 output

## Community Hubs (Navigation)
- Coordinator Scheduling
- Provider Runtime
- Graphify Tooling
- Model Registry
- Distributed Configuration
- Validation Roadmap
- Inference Requests
- RPC Runtime
- GPU Preparation Tests
- Graph Export Tools
- Distributed Execution Targets
- Provider Scheduling
- CUDA Validation Script
- Repository Graph Merge
- Provider Wire Protocol
- Two GPU Smoke Prompts
- Graph Integrity
- Graphify Workflow
- CMake Build
- Architecture Decisions
- GPU Results Reporting
- Agent Instructions
- Registry Overview
- RPC Cache Reuse

## God Nodes (most connected - your core abstractions)
1. `Provider` - 29 edges
2. `DistributedGroup` - 23 edges
3. `Request` - 18 edges
4. `ModelDefinition` - 16 edges
5. `main()` - 16 edges
6. `dispatch_model_requests()` - 11 edges
7. `Capabilities` - 10 edges
8. `LlamaRuntime` - 10 edges
9. `DistributedConfig` - 9 edges
10. `remove_provider()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `models_`  [INFERRED]
  src/coordinator.cpp → include/model_registry.hpp
- `DistributedGroup` --references--> `DistributedConfig`  [EXTRACTED]
  src/coordinator.cpp → include/distributed_runtime.hpp
- `run_distributed_inference()` --references--> `DistributedConfig`  [EXTRACTED]
  src/distributed_runtime.cpp → include/distributed_runtime.hpp
- `DAN` --references--> `Current Architecture`  [EXTRACTED]
  README.md → docs/ARCHITECTURE.md
- `DAN` --references--> `Current Protocol`  [EXTRACTED]
  README.md → docs/PROTOCOL.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Extraction Pipeline** — _codex_skills_graphify_skill_structural_extraction, _codex_skills_graphify_skill_semantic_extraction, _codex_skills_graphify_skill_ast_semantic_merge, _codex_skills_graphify_skill_community_analysis [EXTRACTED 1.00]
- **Automated Graph Maintenance** — _codex_skills_graphify_references_add_watch_folder_watcher, _codex_skills_graphify_references_hooks_post_commit_hook, _codex_skills_graphify_references_update_incremental_update [INFERRED 0.85]
- **Distributed GPU Validation Documents** — docs_pod_a_two_gpu_smoke_prompt_pod_a, docs_pod_b_two_gpu_smoke_prompt_pod_b, docs_two_gpu_smoke_report_result, docs_qwen3_two_gpu_report_result [INFERRED 0.85]
- **Cache and Persistence Experiment** — docs_next_gpu_experiment_experiment, docs_pod_a_next_test_prompt_pod_a, docs_pod_b_next_test_prompt_pod_b, docs_provider_lifecycle_lifecycle [EXTRACTED 1.00]

## Communities (24 total, 13 thin omitted)

### Community 0 - "Coordinator Scheduling"
Cohesion: 0.09
Nodes (47): deque, ExecutionTargetType, optional, any_busy(), any_group_busy(), pid_t, size_t, string_view (+39 more)

### Community 1 - "Provider Runtime"
Cohesion: 0.14
Nodes (21): Capabilities, backend, device_type, gpu_name, model_name, provider_id, vram, capability_message() (+13 more)

### Community 2 - "Graphify Tooling"
Cohesion: 0.08
Nodes (26): Folder Watcher, URL Ingestion, Confidence Audit Trail, Deterministic Node IDs, Semantic Extraction Contract, CLAUDE.md Graphify Integration, Post-Commit Graph Hook, Breadth-First Traversal (+18 more)

### Community 3 - "Model Registry"
Cohesion: 0.13
Nodes (17): size_t, string, vector, ModelDefinition, context_length, distributed, family, id (+9 more)

### Community 4 - "Distributed Configuration"
Cohesion: 0.16
Nodes (14): DistributedConfig, endpoints, executable, id, model, tensor_split, string, size_t (+6 more)

### Community 5 - "Validation Roadmap"
Cohesion: 0.18
Nodes (17): Current Architecture, Validate Weight Reuse, Single CUDA Provider Validation, Model Strategy and Registry, RPC Cache and Persistent Runtime Experiment, Persistent RPC Runtime, Pod A Next Test, Pod B Next Test (+9 more)

### Community 6 - "Inference Requests"
Cohesion: 0.21
Nodes (11): string, Request, allow_distributed, allow_single, id, model, model_path, prompt (+3 more)

### Community 7 - "RPC Runtime"
Cohesion: 0.48
Nodes (6): string, string_view, vector, run_distributed_inference(), split_rpc_endpoints(), trim()

### Community 9 - "Graph Export Tools"
Cohesion: 0.67
Nodes (3): Graph Export Formats, Graphify MCP Server, Token Reduction Benchmark

### Community 10 - "Distributed Execution Targets"
Cohesion: 0.67
Nodes (3): Coordinator Target, Distributed Runtime Library, Distributed Model Targets

### Community 11 - "Provider Scheduling"
Cohesion: 0.67
Nodes (3): Coordinator, Persistent Provider, Round-Robin Scheduling

## Knowledge Gaps
- **89 isolated node(s):** `id`, `executable`, `model`, `endpoints`, `tensor_split` (+84 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 115 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Provider` connect `Coordinator Scheduling` to `Inference Requests`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `DistributedGroup` connect `Coordinator Scheduling` to `Distributed Configuration`, `Inference Requests`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `main()` (e.g. with `.find()` and `.load()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `id`, `executable`, `model` to the rest of the system?**
  _89 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Coordinator Scheduling` be split into smaller, more focused modules?**
  _Cohesion score 0.08599290780141844 - nodes in this community are weakly interconnected._
- **Should `Provider Runtime` be split into smaller, more focused modules?**
  _Cohesion score 0.13675213675213677 - nodes in this community are weakly interconnected._
- **Should `Graphify Tooling` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._