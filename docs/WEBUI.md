# Open WebUI on DAN

Use the existing Open WebUI application, with DAN's API running on the user's own
computer. No central inference server or scheduler is added.

## Start

For a newly built DHT installer, open **DAN API**, then **DAN WebUI**. The latter
requires [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
It downloads upstream Open WebUI and Python 3.11 on first use. Keep both windows open.
Visit http://127.0.0.1:3000, create your local account, and choose `dan-auto`.
After updating an existing installation, refresh the page and start a new chat
to pick up the model's default features and agent instructions. Personal overrides
still take precedence.

For a local RTX 2070 test, `config/provider-owned-qwen2.5-7b-q4km.json`
pins a single-file Qwen2.5-7B Q4_K_M artifact (4.68 GB) with 16K context.
Use that manifest in both the worker catalog and API launcher. Whole-model
workers do not send boundary activations; split routes retain the wire-size
context limit. GPU memory fit still depends on the available VRAM.

From a development checkout, build the C++ `dan-client` and Go `dan-api-gateway`
targets, then run (replace the bootstrap address and binary paths):

```powershell
.\scripts\Start-DAN-Client.ps1 -Bootstrap '/ip4/IP/tcp/4001/p2p/PEERID' `
  -Manifest '.\config\provider-owned-qwen2.5-0.5b-q4km.json' `
  -Sidecar 'PATH\dan-sidecar.exe' -Client '.\build-native\dan-client.exe' `
  -Gateway '.\build\gateway\dan-api-gateway.exe'
# In another window:
.\scripts\Start-DAN-WebUI.ps1
```

The connection is `http://127.0.0.1:8080/v1`, key `dan-local`, model `dan-auto`.
`dan-auto` only uses a route predicted to meet the speed target (20 tokens/s per chat,
first answer within 5 s) and otherwise answers with an error naming the slower option;
choose `dan-any` to accept a slower or not yet measured route. A conversation keeps the
model it started with, also after the client's five-minute cache expires.
An optional `DAN_API_KEY` environment variable must be identical in both windows.
The API only binds to loopback. Temperature defaults to 0; temperature, top-p,
top-k, min-p, seed and penalties now use llama.cpp's sampler. Model-owned templates,
required tool calls and `response_format` JSON object/schema constraints are enabled.
Structured response formats cannot be combined with enabled tools in one request.

The API retains one conversation in RAM for up to five idle minutes to reuse its
exact token prefix across chat/tool turns. User/cache-key changes, cancellation and
shutdown release it. Set `prompt_cache_key` per conversation for explicit isolation;
the WebUI launcher forwards user identity. No DAN prompt or KV cache files are added.
The UI's existing local history storage remains unchanged.

`Start-DAN-WebUI.ps1 -Version VERSION` selects a specific upstream release;
the default is the inspected release **0.11.4**. Open WebUI is downloaded separately,
not copied into or relicensed as DAN. See its [license](https://github.com/open-webui/open-webui/blob/main/LICENSE).

## Agent defaults

The launcher configures upstream native function calling and its multi-round tool
execution loop; no custom tool executor or scheduling service is added.

- **Web:** `search_web` and `fetch_url`, enabled for new chats. Upstream DDGS
  (`duckduckgo`, backend `auto`) needs no search account or API key. Native search
  returns snippets without embeddings; page text is capped at 8000 characters.
- **Documents:** uploaded text/PDF/office document extraction, local CPU embeddings
  with upstream `sentence-transformers/all-MiniLM-L6-v2`, and chat-file listing,
  passage retrieval, grep and bounded reads. The first startup downloads the small
  embedding model to the local data folder; later starts reuse it. No Ollama or
  external embedding provider is needed, and this does not use the inference GPU.
- **Python:** upstream Pyodide code interpreter, enabled for new chats. It runs in
  the browser for calculation, analysis and generated files; it is not a host shell.
- **Conversation tools:** clock/date utilities, clarification prompts, task tracking
  in saved chats, and local notes. Instructions permit writing notes when requested.
- **Context:** upstream automatic compaction for saved chats at an estimated 10000
  tokens, leaving room in the current 16K window for tools and answers. Summaries
  preserve recent task state; token estimates and summaries are imperfect, so the
  backend's explicit context-length error still applies.

Agent instructions are supplied through `DEFAULT_INTERFACE_SETTINGS.system`,
which reaches the initial chat request. In 0.11.4, `DEFAULT_MODEL_PARAMS.system`
alone does not insert that prompt into the first provider request. Instructions
require dependent tools to run sequentially and IDs to come from actual results.
Native mode and temperature 0 remain global model parameter defaults.

Automatic title, tag and follow-up generation are disabled to keep extra model
requests out of the local inference queue. Tool loops are capped at 16 iterations.
The current profile does not expose unused knowledge-base, calendar, channel,
automation, sub-agent, image-generation or automatic memory tools. Open Terminal
is a separate optional integration for shell/project access, not configured here.

Sources: upstream [native tools](https://docs.openwebui.com/features/extensibility/plugin/tools/),
[agentic search](https://docs.openwebui.com/features/chat-conversations/web-search/agentic-search/),
[DDGS](https://docs.openwebui.com/features/chat-conversations/web-search/providers/ddgs/),
and the installed 0.11.4 backend sources. Owner verifies browser-side Python and UI.

## Data

Open WebUI stores chats, uploads, accounts and settings in `%LOCALAPPDATA%\DAN\webui`.
`-DataDir` changes this location. Chat export/deletion happens in Open WebUI.
Its launch configuration disables Ollama/cloud-provider defaults, telemetry, update
checks and automatic model updates. Package installation and the first local
embedding-model download use the internet. Settings persist locally; saved admin
settings take precedence over launcher defaults. Explicit web searches send their
queries to the search provider, and page fetches contact the requested website.

DAN's local API sends bounded chat records through stdin to a retained C++ client,
streams results from memory, and retains one scoped session for prefix reuse.
Five idle minutes, a scope change, cancellation/error or shutdown releases it.
It writes no prompt/history database or conversation log. The sidecar retains
its identity and networking diagnostics; model weights/metadata remain cached.
Closing a request cancels generation and closes the client, allowing acknowledged
session destruction before a bounded forced-exit fallback. This is application-level storage
behavior, not a guarantee against OS swap/crash dumps or independently modified workers.

Workers process prompt tokens and activations in memory. DAN does not provide
confidential inference against a worker operator. In particular, the first stage
can inspect the tokenized conversation. Local history does not change that.

## Current limits

- Qwen2 text chat, streaming and full conversation history. The local client selects
  the largest compatible catalog model that fits, preferring a READY replica.
- One active inference per local API, because it shares one sidecar return port.
  Overlapping chat/title/tag requests wait in memory; up to eight requests
  including the active one are admitted. A full queue returns HTTP 429 with
  `Retry-After`. Disconnected waiters are removed. The inference timeout also
  covers queue time. The launcher disables automatic title/tag/follow-up generation.
- Native function tools work through Qwen's tool template: the API returns OpenAI
  `tool_calls`, and Open WebUI executes them and sends results back. Both JSON
  and SSE are supported; normal text streams immediately even with tools enabled.
  Split tool markers and call JSON are held back; validated structured calls are
  sent at completion. `tool_choice` supports
  `auto`, `none` and `required`; named choices remain unsupported.
  Malformed, truncated or unadvertised calls fail instead of being executed.
  Tool quality depends on the selected model; the 0.5B test model can struggle.
- Vision/audio inputs remain unsupported. JSON object/schema output uses upstream
  constrained decoding. This does
  not yet provide a complete coding-agent backend.
- Uploaded-document retrieval is configured locally. Speech and image generation
  still require separate compatible services; this text model cannot interpret
  image/audio content.
- `/health` checks the API process, not GPU availability. `/v1/models` exposes
  `dan-auto`; it does not promise an online worker. Routes persist with the cached client.
- Repeated turns reuse the exact token prefix; changed history is evaluated from
  the first difference, including tool-result turns.
  WebUI compacts long conversations; each resulting request must still fit the
  worker's context. The local API requests a 16384-token window (`-context` to
  change it); the worker must support that size. Increasing HTTP output-token
  limits does not increase worker context.
  Set the provider's `max_context=16384` too; older configurations default to
  4096. Context exhaustion returns `context_length_exceeded`, with guidance to
  shorten history or disable unused tools, rather than a worker-offline error.
  API client sockets allow long CPU prefill; the gateway's overall deadline still
  cancels the process. Worker prefill microbatches are bounded independently of
  context size (`prefill_batch`; 1024 in the current local config).

Launcher regression check (no UI launch or network call):
`powershell -NoProfile -File scripts/tests/webui_launcher_test.ps1`.

Backend checks on 2026-10-06 passed with the live RTX 2070 / Qwen2.5-7B:
upstream clock tool followed by a final model answer; uploaded-file listing/read
followed by the correct answer; local extraction, CPU embedding and retrieval;
DDGS search returning real links; page fetching returning its heading; and
saved-chat compaction retaining a synthetic passphrase in the summary. Temporary
test uploads/chats were removed. Browser Python and the full browser compaction
flow remain owner verification; no UI automation or performance benchmark ran.

Upstream setup reference: [Open WebUI quick start](https://docs.openwebui.com/getting-started/quick-start/).
