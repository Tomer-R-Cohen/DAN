# Model Strategy and Registry

The first rented-GPU session is specified in [GPU_VALIDATION.md](GPU_VALIDATION.md).
Its replaceable `dan-main` baseline is Qwen2.5-7B-Instruct Q4_K_M with a 4,096
token test context. This choice targets a quick L4 CUDA acceptance run, not a
current best-model ranking. SmolLM2 remains test-only; no serious candidate has
yet passed the rented-GPU benchmark. Registry context/memory are metadata:
the provider must also receive `--ctx-size` explicitly (the driver does this).

Models are deployment configuration and data, not DAN networking logic. Start
the coordinator with `--models <registry-file>` and request a stable registry ID
with `/model <id> <prompt>`. Replacing the main model means editing the registry
and provider/group startup configuration, not changing coordinator scheduling.

## Roles

- `test`: small, cheap models used only to verify networking, framing, queues,
  concurrency, failure handling, and RPC plumbing. Output quality is irrelevant.
- `main`: the best open-weight model that the deployment can serve at acceptable
  latency and quality. This is the model for real DAN use and demonstrations.
- `distributed`: useful models whose memory or execution requirements justify a
  manually configured multi-provider RPC group.

The included `smollm2-test` entry is test-only. It must not be presented as
DAN's intelligence or used to evaluate DAN model quality.

## File Format

The registry is pipe-delimited with eleven fields:

```text
id|role|family|path|runtime_name|quantization|memory_mib|context_length|requirements|single_provider|distributed
```

`runtime_name` must match provider capability metadata. `path` is used for
configured distributed groups. Execution flags are `yes` or `no`. Lines starting
with `#` and blank lines are ignored. See `config/models.example.conf`.

## Choosing the Main Model

Model rankings and runtime support change quickly. Before promoting a model to
`main`, verify license, llama.cpp support, prompt template, memory at the desired
context length, quality on DAN workloads, latency, and failure behavior on the
actual hardware. Keep the stable ID `dan-main`; update its family, path, runtime
name, quantization, and resource metadata.

As a serious starting point, OpenAI documents `gpt-oss-20b` as an Apache-2.0
open-weight reasoning model designed for roughly 16 GB deployments, and
`gpt-oss-120b` as the stronger variant fitting in roughly 80 GB with a 131,072
token context. The sample registry uses the former as a main-model candidate and
the latter as a large/distributed candidate. These are replaceable examples,
not permanent architectural choices or proof that the current host can run them.

Primary references:

- https://openai.com/index/introducing-gpt-oss/
- https://developers.openai.com/api/docs/models/gpt-oss-120b

## Promotion Checklist

1. Benchmark candidates on representative prompts and the intended llama.cpp
   build.
2. Measure model plus KV-cache memory at the chosen context length.
3. Select a quantization that meets quality and latency requirements.
4. Set `single_provider` and `distributed` from tested execution support.
5. Start providers with matching `--model-name`, or configure groups with the
   registry path using `@model-id`.
6. Change the `dan-main` registry entry only after the deployment passes these
   checks.
