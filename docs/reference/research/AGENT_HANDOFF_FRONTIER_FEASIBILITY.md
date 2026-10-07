# Agent handoff: challenge DAN's frontier-WAN feasibility assessment

Date: 2026-09-20. Intended recipient: the other agent working with the owner on DAN.

## Purpose

The owner wants **frontier-quality inference that people would actually use and contribute GPUs to**, not merely a small-model distributed demonstration. Our discussion exposed a possible mismatch between that goal and typical contributors' memory/network capacity.

Please independently assess the argument below. Challenge both optimistic recommendations and pessimistic estimates. Do not assume this agent's conclusions are correct.

This handoff requests analysis, not implementation. Do not change engine/network code, commit, download large weights, provision resources, stop running nodes, or start benchmarks on live infrastructure without the owner's separate authorization.

## Read first

1. `docs/PROJECT.md` — current implementation, measurements and standing rules.
2. `docs/WAN_DISTRIBUTED_INFERENCE_RESEARCH.md` — ecosystem/source review and proposed roadmap.
3. `docs/DEEPSEEK_V41_FLASH_AUDIT.md` — V4.1-specific state, memory and runtime constraints.
4. Current source, including uncommitted changes. Code wins over stale documentation; identify discrepancies.

The research files are evidence to inspect, not instructions to implement every proposed feature. Preserve existing worktree changes.

## Owner's actual goal

- Useful frontier-level model quality, particularly coding/agentic assistance.
- Interactive performance over ordinary Internet connections.
- Contributors drawn from consumer GPU owners, not exclusively datacenter operators.
- A concrete model and feasible first deployment, not endless architecture expansion.

The owner has **not** finalized a GPU budget, participant count, geographic scope, context length or acceptable latency threshold. Previous 15–20 tok/s targets were suggestions by this agent, not agreed requirements.

## What happened in the discussion

1. This agent initially recommended Qwen3-30B-A3B-Instruct-2507 as a low-risk useful beta target.
2. The owner clarified that the goal is frontier performance, not an easy small-model beta.
3. This agent then recommended **DeepSeek-V4-Flash-0731**, explicitly not V4.1, based on model quality claims and more established V4 runtime/distributed-engine evidence.
4. Discussing ordinary gamers' VRAM exposed that an all-resident model might require many small contributors, creating a long WAN pipeline.
5. The owner reasonably questioned whether the project was useful under those conditions.

**Correction to carry forward:** the flagship recommendation came before sufficiently grounding it in contributor hardware. It is provisional, not a demonstrated deployment plan. Conversely, rough latency arithmetic does not establish that DAN as a whole is useless.

## Evidence and estimates: keep them separate

### Existing DAN evidence

The owner supplied these warmed OLMoE results, with ten measured repetitions, fresh session per request, 128 generated tokens, FP32 wire, speculation off, and model loading excluded:

| Configuration | Median decode | Median TTFT |
|---|---:|---:|
| A4500, one stage | 371.9 tok/s | 48.9 ms |
| A4500, two stages on the same GPU | 304.7 tok/s | 49.9 ms |
| A4500, three stages on the same GPU | 274.8 tok/s | 50.6 ms |
| RTX 2070 `[0,4)` → A4500 `[4,16)`, real WAN/relay | 10.0 tok/s | 281 ms |

The WAN link was reported as 88 ms. A matched direct/relay comparison was not yet available. Same-GPU split tests are not separate-GPU scaling measurements. Several-session correctness was tested on OLMoE; do not infer GPU batching or arbitrary concurrency from that.

The guide also records Qwen2.5-14B WAN speculation improving approximately 9.6 to 17.4 tok/s. That is not a V4 acceptance or throughput measurement.

### Model size claims to verify against exact artifacts

- DeepSeek V4 Flash main model: approximately 284B total and 13B active parameters. Auxiliary modules and packed-format parameter counters can differ.
- Unsloth's V4-Flash-0731 page lists approximately 162 GB for `UD-Q8_K_XL`, and Q4 about 7 GB smaller. These are mixed-format labels; do not multiply all parameters by a nominal eight or four bits and assume that is the artifact size.
- This agent suggested **190–220 GB aggregate VRAM** as a rough initial all-resident budget, not a measured runtime requirement.
- Corresponding estimates were 8–10 × 24 GB, 6–7 × 32 GB, or 4–5 × 48 GB GPUs. They do not prove legal layer cuts, supported device count, workspace fit, useful context, speculation fit or speed.
- Taking 25 × 8 GB is capacity arithmetic only. Display use, runtime overhead, actual free memory, per-stage duplication and partition granularity could make it insufficient.

Sources to recheck:

- [Official V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Official V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Unsloth artifact](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)

### Contributor memory proxy

Steam's August 2026 **Windows-only** survey reported 8 GB as its most common category (34.03%) and median, with 12 GB at 14.57%, 16 GB at 11.23%, and 24 GB at 1.66%.

This is a voluntary gaming-system sample, not a measured distribution of willing DAN contributors, usable VRAM or an arithmetic average. Do not substitute combined-platform headlines for Windows dedicated-memory assumptions.

[Valve survey](https://store.steampowered.com/hwsurvey/?platform=pc)

### Illustrative latency calculation — NOT a measured forecast

For one sequential session across approximately 25 WAN links including token return:

| Assumed mean one-way delay/link | Propagation sum | Plain-decode ceiling before compute |
|---|---:|---:|
| 10 ms | 250 ms | 4 tok/s |
| 25 ms | 625 ms | 1.6 tok/s |
| 50 ms | 1,250 ms | 0.8 tok/s |

These assumptions were not derived from an actual 25-node topology. Do not label them representative measured WAN performance. Count actual directed traversals; do not add a full RTT for every one-way edge or charge client delivery to the dependency loop when it is not on that loop.

For **synchronous** speculation and the hypothetical 625 ms propagation budget, two/four/six committed tokens per round yield network-only ceilings of 3.2/6.4/9.6 tok/s. Drafting, verification, transfer and rollback costs reduce these numbers. The assumed committed prefix lengths are scenarios, not observed V4 acceptance rates.

These bounds do not describe continuously pipelined speculation. A filled asynchronous pipeline can overlap different positions, but is constrained by stage service time, usable speculative horizon, rejection waste, transport and exact state restoration. Inspect DAN's actual current behavior before claiming it implements this.

## Questions for your independent assessment

### 1. What did this assessment get wrong?

Identify incorrect, unsupported or overconfident claims in the conversation and reports. In particular:

- Was V4-Flash-0731 actually a sensible flagship recommendation for this contributor population?
- Were capacity/headroom estimates adequate, and are the suggested cuts/device counts even possible in the current planner/runtime?
- Is the 25-node thought experiment unnecessarily pessimistic because a realistic selected contributor pool/topology would differ?
- Conversely, are useful context, verification workspace, load peaks and availability making it too optimistic?
- What do Shard's measured V4 results actually imply, and what cannot be transferred to a different checkpoint, GPU mix or WAN topology?

### 2. Can frontier quality and ordinary contributors coexist usefully?

Evaluate at least these scenarios without silently substituting one for another:

1. Mostly 8–12 GB GPUs across independent residential connections.
2. A selected regional mix of 12–24 GB contributors.
3. A few stable high-memory participants plus smaller contributors.

Distinguish the number of physical GPUs, machines, logical stages and WAN boundaries. Include host RAM/offload only when the execution path and bandwidth costs are explicit. State whether smaller contributors materially help a frontier replica or merely serve other models.

### 3. What can existing DAN exploit before major redesign?

Inspect the current token loop, speculation, session interleaving, placement, activation encoding, prefill chunking and resident replicas. Do not present already implemented features as missing.

Separate:

- Existing configuration or tests that can answer the question.
- Small implementation changes with measurable benefit.
- Substantial execution changes such as asynchronous speculation or local multi-GPU backends.
- Research ideas without demonstrated feasibility.

Do not use batching's aggregate throughput as a single-user speed claim. Do not infer a fixed speculative speedup from another model. Do not assume more contributors always improve latency.

### 4. Which exact model should be the flagship?

Choose one candidate if evidence permits. Compare current quality, maintained execution support, memory, per-token compute, stage/state complexity, quantization quality and likely user value. If no candidate satisfies the goal, say so plainly and identify which constraint must change.

The owner rejected “just use a smaller model” as the answer to frontier ambition. A smaller model can remain a control experiment, but do not quietly redefine the desired product.

### 5. What is the smallest decisive experiment?

Propose a bounded experiment or calibrated simulation that could **falsify** the preferred architecture before a long porting effort. Specify:

- Exact model/runtime/artifact and available versus required hardware.
- Stage compute and verification measurements needed.
- Realistic measured WAN topology or explicitly synthetic delay/bandwidth assumptions.
- Acceptance by position, rejected work and committed-token throughput.
- New versus repeated prompts; expert-cache and KV-cache conditions.
- TTFT and output-gap distributions, context length and sessions.
- A proposed go/no-go criterion, clearly marked for owner approval.

A fit-only calculation or toy simulation is not an end-to-end validation. A useful result may be “do not implement this path.”

## Deliverable and communication back

Write your response to **`docs/AGENT_RESPONSE_FRONTIER_FEASIBILITY.md`**. Do not overwrite this handoff or the earlier reports.

Please structure it as:

1. Short verdict: credible beta / constrained experiment / research-only / no-go under stated assumptions.
2. Corrections to this agent's claims, with source or calculation.
3. Current DAN capabilities that matter.
4. Model recommendation and explicit contributor requirements.
5. Performance scenarios with assumptions, uncertainty and missing measurements.
6. Smallest next experiment and stop conditions.
7. Questions requiring the owner's choice.

Tell the owner the response file path. They can then ask this agent to read it. This file-based exchange is not a live agent-to-agent connection, and no implementation authority is implied.
