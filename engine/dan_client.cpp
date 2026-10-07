// dan-client: runs inference with no coordinator. This process creates the sessions and
// drives the token loop itself, over either
//   --provider ...   an explicit stage route (stages already loaded), or
//   --candidate ...  workers in serve mode: it plans placement, reserves and assigns them, or
//   --discover API   the same, with candidates found by the local sidecar (DHT discovery).

#include "provider_owned/client.hpp"
#include "provider_owned/manifest.hpp"
#include "provider_owned/placement.hpp"
#include "provider_owned/replica.hpp"
#include "provider_owned/selection.hpp"
#include "platform.hpp"

#include <algorithm>
#include <cctype>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <csignal>
#endif
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace po = dan::provider_owned;

namespace {

struct Options {
    std::vector<std::string> manifests;  // several: the largest that fits is used
    std::vector<std::string> providers;
    std::vector<std::string> prompts;
    std::string expected;
    std::string report;
    int tokens = 20;
    int requests = 0;
    bool persistent = false;
    // Direct ring: stages send activations to each other, not back through this client.
    std::string ring_return;
    std::string ring_return_target;
    std::vector<std::string> ring_targets;
    std::vector<std::string> peer_ids;
    bool require_direct = false;
    // Dynamic placement.
    std::vector<std::string> candidates;
    std::vector<std::string> candidate_peers;
    int sessions = 1;
    int minimum_stages = 1;
    int context = 0;  // 0 = the manifest's context
    std::string runtime_abi = DAN_RUNTIME_ABI;
    std::string metadata_cache;
    std::string discover;  // local sidecar candidate API
    int connect_timeout_ms = 45000;
    bool chat = false;  // interactive conversation on one session
    bool api = false;   // raw prompt on stdin; NDJSON events on stdout
    bool api_chat = false; // bounded length-prefixed chat JSON records; one RAM-only session
    bool no_loop = false;  // keep the client in the token loop (diagnostics)
    bool speculate = false;  // draft model on the first stage
    bool tokens_set = false;
    // Persistent replicas.
    bool replica = false;       // use a READY replica if one exists, else place a route
    bool replica_only = false;  // ... and never fall back to placing one
    // Selection (selection.hpp): "target" uses only plans that meet the speed target and
    // otherwise fails with below_target; "any" also accepts slower or unmeasured plans.
    std::string policy = "any";
    std::string pin_model;      // this model (SHA-256) or nothing: a pinned conversation
    bool wait_cold = false;     // --cold-start wait: best plan even if it must load first
    bool form = false;          // run this node's replica owner (never returns)
    std::string self_control;   // --form: this node's worker control listener
    std::string session_listen; // --form: front door the sidecar forwards sessions to
    std::string replica_status; // --form: status file the sidecar advertises from
    int max_edge_rtt_ms = 150;
    bool no_relay_edges = false;
    bool rank_delay = false;
    int warmup_tokens = 8;  // enough to measure the replica's time per token
    bool upgrades = false;  // --form: give an idle replica up for a clearly better one
    long long upgrade_idle_ms = 0, upgrade_check_ms = 0, upgrade_backoff_ms = 0;  // 0 = default
    std::string log;            // append diagnostics (stderr) to this file
    po::DType activations = po::DType::f32le;  // --activations f32|f16|fp8
};

bool read_chat_record(std::string& body, int& tokens) {
    std::string header;
    char byte;
    while (std::cin.get(byte) && byte != '\n') {
        if (header.size() >= 32) throw std::runtime_error("invalid chat record header");
        header += byte;
    }
    if (!std::cin && header.empty()) return false;
    std::istringstream fields(header);
    std::size_t size = 0;
    std::string extra;
    if (!(fields >> tokens >> size) || (fields >> extra) || tokens < 1 || tokens > 4096
        || size == 0 || size > 1024 * 1024) throw std::runtime_error("invalid chat record");
    body.resize(size);
    if (!std::cin.read(body.data(), static_cast<std::streamsize>(size))) throw std::runtime_error("truncated chat record");
    return true;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--persistent") { options.persistent = true; continue; }
        if (option == "--require-direct") { options.require_direct = true; continue; }
        if (option == "--chat") { options.chat = true; continue; }
        if (option == "--api") { options.api = true; continue; }
        if (option == "--api-chat") { options.api = options.api_chat = true; continue; }
        if (option == "--no-loop") { options.no_loop = true; continue; }
        if (option == "--speculate") { options.speculate = true; continue; }
        if (option == "--replica") { options.replica = true; continue; }
        if (option == "--replica-only") { options.replica = options.replica_only = true; continue; }
        if (option == "--form") { options.form = true; continue; }
        if (option == "--no-relay-edges") { options.no_relay_edges = true; continue; }
        if (option == "--rank-delay") { options.rank_delay = true; continue; }
        if (option == "--upgrades") { options.upgrades = true; continue; }
        if (option == "--f16-activations") { options.activations = po::DType::f16le; continue; }
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--manifest") options.manifests.push_back(value);
        else if (option == "--provider") options.providers.push_back(value);
        else if (option == "--prompt") options.prompts.push_back(value);
        else if (option == "--tokens") { options.tokens = std::stoi(value); options.tokens_set = true; }
        else if (option == "--requests") options.requests = std::stoi(value);
        else if (option == "--expected-output") options.expected = value;
        else if (option == "--report") options.report = value;
        else if (option == "--ring-return") options.ring_return = value;
        else if (option == "--ring-return-target") options.ring_return_target = value;
        else if (option == "--ring-target") options.ring_targets.push_back(value);
        else if (option == "--peer-id") options.peer_ids.push_back(value);
        else if (option == "--candidate") options.candidates.push_back(value);
        else if (option == "--candidate-peer") options.candidate_peers.push_back(value);
        else if (option == "--sessions") options.sessions = std::stoi(value);
        else if (option == "--min-stages") options.minimum_stages = std::stoi(value);
        else if (option == "--context") options.context = std::stoi(value);
        else if (option == "--runtime-abi") options.runtime_abi = value;
        else if (option == "--metadata-cache") options.metadata_cache = value;
        else if (option == "--discover") options.discover = value;
        else if (option == "--connect-timeout-ms") options.connect_timeout_ms = std::stoi(value);
        else if (option == "--self-control") options.self_control = value;
        else if (option == "--session-listen") options.session_listen = value;
        else if (option == "--replica-status") options.replica_status = value;
        else if (option == "--max-edge-rtt-ms") options.max_edge_rtt_ms = std::stoi(value);
        else if (option == "--warmup-tokens") options.warmup_tokens = std::stoi(value);
        else if (option == "--upgrade-idle-ms") options.upgrade_idle_ms = std::stoll(value);
        else if (option == "--upgrade-check-ms") options.upgrade_check_ms = std::stoll(value);
        else if (option == "--upgrade-backoff-ms") options.upgrade_backoff_ms = std::stoll(value);
        else if (option == "--log") options.log = value;
        else if (option == "--policy") {
            if (value != "target" && value != "any") throw std::runtime_error("--policy takes target or any");
            options.policy = value;
        }
        else if (option == "--pin-model") {
            if (value.size() != 64 || value.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
                throw std::runtime_error("--pin-model takes a model SHA-256");
            }
            options.pin_model = value;
        }
        else if (option == "--cold-start") {
            if (value != "ready" && value != "wait") throw std::runtime_error("--cold-start takes ready or wait");
            options.wait_cold = value == "wait";
        }
        else if (option == "--activations") {
            if (value != "f32" && value != "f16" && value != "fp8") {
                throw std::runtime_error("--activations takes f32, f16 or fp8");
            }
            options.activations = value == "f16" ? po::DType::f16le
                : value == "fp8" ? po::DType::fp8e4m3 : po::DType::f32le;
        }
        else throw std::runtime_error("unknown option: " + option);
    }
    if (options.api) {
        if (options.chat || options.form || !options.prompts.empty() || options.persistent
            || !options.report.empty() || !options.log.empty() || options.requests != 0) {
            throw std::runtime_error("--api requires a single stdin prompt and no conversation logs");
        }
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        std::string prompt;
        if (options.api_chat) {
            if (!read_chat_record(prompt, options.tokens)) throw std::runtime_error("missing chat record");
        } else prompt.assign((std::istreambuf_iterator<char>(std::cin)), {});
        if (prompt.empty()) throw std::runtime_error("empty stdin prompt");
        options.prompts.push_back(std::move(prompt));
    }
    if (options.form) {
        // The owner of this node's persistent replica: no prompts, it serves others'.
        if (options.manifests.empty() || options.discover.empty() || options.self_control.empty()
            || options.session_listen.empty() || !options.prompts.empty() || options.chat
            || !options.providers.empty() || !options.candidates.empty() || options.replica
            || options.sessions < 1 || options.minimum_stages < 1 || options.context < 0
            || options.max_edge_rtt_ms < 1 || options.warmup_tokens < 1) {
            throw std::runtime_error("usage: dan-client --form --manifest FILE [...] --discover SIDECAR_API "
                "--self-control HOST:PORT --session-listen HOST:PORT [--replica-status FILE] "
                "[--sessions 1] [--min-stages 1] [--context N] [--speculate] [--max-edge-rtt-ms 150] "
                "[--no-relay-edges] [--rank-delay] [--warmup-tokens 2] [--upgrades "
                "[--upgrade-idle-ms 120000] [--upgrade-check-ms 60000] [--upgrade-backoff-ms 600000]]");
        }
        return options;
    }
    if (options.replica && options.discover.empty()) {
        throw std::runtime_error("--replica needs --discover");
    }
    if (options.chat) {
        if (!options.tokens_set) options.tokens = 256;
        if (options.requests == 0) options.requests = 1;
    }
    if (options.requests == 0) options.requests = static_cast<int>(options.prompts.size());
    const bool discovered = !options.discover.empty();
    if (options.ring_return_target.empty() && !discovered) {
        options.ring_return_target = options.ring_return;
    }
    const bool ring = !options.ring_return.empty() || discovered;
    const bool placed = !options.candidates.empty() || discovered;
    const int sources = !options.providers.empty() + !options.candidates.empty() + discovered;
    if (options.manifests.empty() || sources != 1
        || (discovered && !options.candidate_peers.empty())
        || (options.prompts.empty() != options.chat) || options.tokens < 1 || options.requests < 1
        || (options.chat && (options.persistent || !options.report.empty() || !options.expected.empty()))
        || options.sessions < 1 || options.minimum_stages < 1 || options.context < 0
        || options.connect_timeout_ms < 1000
        || (placed && (!options.ring_targets.empty() || !options.peer_ids.empty()
            || (!options.candidate_peers.empty()
                && options.candidate_peers.size() != options.candidates.size())))
        || (!placed && !options.candidate_peers.empty())
        || (!placed && ring && options.ring_targets.size() + 1 != options.providers.size())
        || (!ring && (!options.ring_targets.empty() || !options.peer_ids.empty()
            || options.require_direct))
        || (!options.peer_ids.empty() && options.peer_ids.size() != options.providers.size())) {
        throw std::runtime_error(
            "usage: dan-client --manifest FILE --provider HOST:PORT [--provider HOST:PORT ...] "
            "--prompt TEXT [...] [--tokens N] [--requests N] [--persistent] "
            "[--expected-output TEXT] [--report FILE] "
            "[--ring-return HOST:PORT [--ring-return-target TARGET] --ring-target TARGET "
            "(one per stage after the first) [--peer-id ID (one per stage)] [--require-direct]]\n"
            "   or: dan-client --manifest FILE --candidate HOST:PORT [...] [--candidate-peer PEERID (one per "
            "candidate)] --prompt TEXT [...] [--sessions 1] [--min-stages 1] [--context N] "
            "[--runtime-abi ABI] [--metadata-cache FILE] [--ring-return HOST:PORT "
            "[--ring-return-target TARGET] [--require-direct]] [other options above]\n"
            "   or: dan-client --manifest FILE --discover SIDECAR_API --prompt TEXT [...] "
            "[placement and other options above] [--connect-timeout-ms 45000]\n"
            "   --chat instead of --prompt: an interactive conversation (/new, /quit)\n"
            "   --api instead of --prompt: stdin prompt, NDJSON stdout (no content logs)\n"
            "   --no-loop: keep the client in the per-token loop (slower; diagnostics)\n"
            "   --speculate: let the first stage draft ahead with the smallest offered model\n"
            "   --activations f16|fp8: send activations as f16 (half the bytes) or fp8 (a quarter);\n"
            "      wording can differ from f32 (the default)\n"
            "   --replica: chat through a READY persistent replica when one exists (--replica-only: "
            "never place a route)\n"
            "   --policy target|any: target uses only routes predicted to meet the speed target\n"
            "      (else fails with below_target); any (default) also accepts slower or unmeasured\n"
            "   --pin-model SHA256: this model or nothing (a continuing conversation)\n"
            "   --cold-start ready|wait: prefer a ready replica (default) or the best plan even if\n"
            "      it must load first\n"
            "   or: dan-client --form ... (run this node's replica owner; see --form usage)");
    }
    return options;
}

// Qwen2 chat format; the conversation lives in one session on every stage.
// Qwen's chat markers. Another architecture has its own template, and wrapping its prompt in
// these would quietly feed the model text it was never trained on, so only Qwen2 gets them:
// every other supported model sees the raw message (docs/MOE_SUPPORT_DESIGN.md §7.5).
std::string chat_prompt(const std::string& text, bool first, bool qwen_template) {
    if (!qwen_template) return first ? text : "\n" + text;
    return (first ? "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        : "\n<|im_start|>user\n") + text + "<|im_end|>\n<|im_start|>assistant\n";
}

void run_chat(po::InferenceClient& client, const po::Manifest& manifest, int tokens,
    std::size_t stages, const std::string& architecture) {
    const bool qwen_template = architecture == "qwen2";
    std::printf("\n  DAN chat  |  %s  |  %zu stage%s  |  /new starts over, /quit exits\n\n",
        manifest.model_id.c_str(), stages, stages == 1 ? "" : "s");
    if (!qwen_template) {
        std::printf("  (%s has no DAN chat template yet: your text is sent as a raw prompt)\n\n",
            architecture.c_str());
    }
    std::uint64_t session = client.create_session();
    bool first = true;
    for (std::string line;;) {
        std::printf("you > ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        // Piped input (e.g. from PowerShell) can carry a UTF-8 BOM and CRLF endings.
        if (line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
        if (line.ends_with('\r')) line.pop_back();
        if (line == "/quit" || line == "/exit") break;
        if (line.empty()) continue;
        if (line == "/new") {
            client.reset_session(session);
            first = true;
            std::printf("      (new conversation)\n\n");
            continue;
        }
        const auto stream = [](std::string_view piece) {
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
            return true;
        };
        std::printf("dan > ");
        std::fflush(stdout);
        po::RequestResult result;
        try {
            result = client.generate(session, chat_prompt(line, first, qwen_template),
                tokens, stream);
        } catch (const std::exception& error) {
            if (std::string_view(error.what()).find("context exhausted") == std::string_view::npos) throw;
            // The conversation filled the context: start over with just this message.
            client.reset_session(session);
            std::printf("(context full, starting a new conversation)\n      ");
            result = client.generate(session, chat_prompt(line, true, qwen_template),
                tokens, stream);
        }
        first = false;
        const std::size_t generated = result.metrics.token_ids.size();
        const double decode_ms = result.metrics.latency_ms - result.metrics.ttft_ms;
        std::printf("\n      %zu tokens  |  %.1f tok/s  |  first token %.0f ms\n\n", generated,
            generated > 1 && decode_ms > 0 ? (generated - 1) * 1000.0 / decode_ms : 0.0,
            result.metrics.ttft_ms);
    }
    client.destroy_session(session);
}

// Appends stderr to a file others can read while this process runs.
void redirect_stderr(const std::string& path) {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
#ifdef _WIN32
    FILE* log = _wfsopen(std::filesystem::path(path).c_str(), L"a", _SH_DENYNO);
    if (!log) return;
    _dup2(_fileno(log), _fileno(stderr));
    std::fclose(log);
#else
    if (!std::freopen(path.c_str(), "a", stderr)) return;
#endif
    std::setvbuf(stderr, nullptr, _IONBF, 0);
}

// Every model's shape, largest first, and the placement settings from the command line.
po::PlacementRequest read_models(const Options& options) {
    po::PlacementRequest request;
    request.context = static_cast<std::uint32_t>(options.context);
    request.sessions = static_cast<std::uint32_t>(options.sessions);
    request.minimum_stages = static_cast<std::size_t>(options.minimum_stages);
    request.runtime_abi = options.runtime_abi;
    request.connect_timeout_ms = static_cast<std::uint32_t>(options.connect_timeout_ms);
    request.speculate = options.speculate;
    request.activations = options.activations;
    // Every model's shape comes from its GGUF header; read them at once, since each
    // is a few HTTP range requests.
    const auto metadata_started = po::Clock::now();
    std::vector<po::ModelOption> options_read(options.manifests.size());
    std::vector<std::string> failures(options.manifests.size());
    std::vector<std::thread> readers;
    for (std::size_t index = 0; index < options.manifests.size(); ++index) {
        readers.emplace_back([&, index] {
            try {
                po::ModelOption& option = options_read[index];
                option.manifest = po::load_manifest(options.manifests[index]);
                const std::filesystem::path metadata = options.metadata_cache.empty()
                    ? std::filesystem::temp_directory_path() / "dan-client"
                        / (option.manifest.sha256 + "-" + po::random_route_id() + ".gguf")
                    : std::filesystem::path(options.metadata_cache
                        + "." + option.manifest.sha256.substr(0, 8));
                // The manifest pins the file by SHA-256, so its header never changes:
                // read it over HTTP once, then reuse the saved index.
                const std::filesystem::path cached = dan::platform::data_directory()
                    / "model-index" / (option.manifest.sha256 + ".index");
                if (!po::load_model_index(cached, option.model)) {
                    std::string error;
                    if (!po::inspect_range_model({option.manifest.url,
                            option.manifest.revision, option.manifest.sha256, metadata,
                            0, 1}, option.model, error)) {
                        throw std::runtime_error("model metadata: " + error);
                    }
                    po::save_model_index(cached, option.model);
                }
                // Short manifests (hf_repo form) omit the shape; the header has it.
                if (option.manifest.hidden == 0) option.manifest.hidden = option.model.hidden;
                if (option.manifest.layers == 0) option.manifest.layers = option.model.layers;
                if (option.manifest.hidden != option.model.hidden
                    || option.manifest.layers != option.model.layers) {
                    throw std::runtime_error("manifest shape does not match the model file");
                }
            } catch (const std::exception& failure) {
                failures[index] = failure.what();
            }
        });
    }
    for (std::thread& reader : readers) reader.join();
    for (std::size_t index = 0; index < failures.size(); ++index) {
        if (failures[index].empty()) { request.models.push_back(options_read[index]); }
        else if (options.manifests.size() == 1) throw std::runtime_error(failures[index]);
        else std::fprintf(stderr, "skipping %s: %s\n", options.manifests[index].c_str(),
            failures[index].c_str());
    }
    if (request.models.empty()) throw std::runtime_error("no usable model manifest");
    // Best first: the curated quality tier, then (only between equal tiers) the larger file.
    std::stable_sort(request.models.begin(), request.models.end(),
        [](const po::ModelOption& left, const po::ModelOption& right) {
            return po::preferred_model(static_cast<int>(left.manifest.quality_tier),
                left.model.logical_bytes, static_cast<int>(right.manifest.quality_tier),
                right.model.logical_bytes);
        });
    return request;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
    return value;
}

// A READY persistent replica with a free session for one of the requested models.
struct ReadyChoice {
    std::size_t model = 0;  // index into PlacementRequest::models
    po::ReplicaCandidate replica;
};

// `full`: READY replicas that would fit but have no free session right now.
std::vector<ReadyChoice> usable_replicas(const Options& options, const po::PlacementRequest& request,
    std::size_t& full) {
    full = 0;
    std::vector<std::string> wanted;
    for (const po::ModelOption& option : request.models) wanted.push_back(option.manifest.sha256);
    const auto started = po::Clock::now();
    const std::vector<po::ReplicaCandidate> replicas = po::discover_replicas(options.discover, wanted);
    std::vector<ReadyChoice> usable;
    for (const po::ReplicaCandidate& replica : replicas) {
        for (std::size_t model = 0; model < request.models.size(); ++model) {
            // The context this chat needs, exactly as a newly placed route would get it: a
            // replica formed with less (e.g. 512 positions) never serves it.
            const std::uint32_t needed = options.context != 0
                ? static_cast<std::uint32_t>(options.context) : request.models[model].manifest.context;
            if (replica.context < needed) continue;
            if (lowercase(request.models[model].manifest.sha256) != replica.model_sha256) continue;
            if (replica.sessions_free == 0) ++full;
            else usable.push_back({model, replica});
        }
    }
    std::printf("discovered replicas=%zu usable=%zu discovery_ms=%.0f\n", replicas.size(),
        usable.size(), po::elapsed_ns(started) / 1e6);
    return usable;
}

// A ready replica as the selection policy sees it. Its owner measured time per token and to
// the first token on real requests. Each open session takes turns on the same GPUs, so a new
// chat's time per token is taken as the measured one times the sessions sharing it (with
// batching or idle sessions it is better; this errs on the slow side).
po::PlanEstimate ready_estimate(const ReadyChoice& choice, const po::PlacementRequest& request) {
    const po::ReplicaCandidate& replica = choice.replica;
    const po::Manifest& manifest = request.models[choice.model].manifest;
    const std::uint32_t in_use = replica.sessions_max > replica.sessions_free
        ? replica.sessions_max - replica.sessions_free : 0;
    po::PlanEstimate plan;
    plan.label = "ready " + manifest.model_id + " (replica " + replica.replica_id.substr(0, 8) + ", "
        + std::to_string(replica.stages) + " stage" + (replica.stages == 1 ? "" : "s") + ")";
    plan.quality_tier = static_cast<int>(manifest.quality_tier);
    plan.context = replica.context;
    plan.providers = replica.stages;
    plan.ready = true;
    plan.measured = replica.ms_per_token != 0 && replica.first_token_ms != 0;
    plan.token_ms = static_cast<double>(replica.ms_per_token) * (in_use + 1);
    plan.first_token_ms = replica.first_token_ms == 0 ? 0
        : static_cast<double>(replica.first_token_ms) + replica.rtt_ms;
    plan.cache_warm = true;
    return plan;
}

// A new placement as the selection policy sees it (estimates from placement's preview).
po::PlanEstimate new_estimate(const po::RoutePreview& preview, const po::PlacementRequest& request) {
    const po::Manifest& manifest = request.models[preview.model].manifest;
    po::PlanEstimate plan;
    plan.label = "new " + manifest.model_id + " (" + std::to_string(preview.stages) + " stage"
        + (preview.stages == 1 ? "" : "s") + (preview.from_cache ? ", cached layers)" : ")");
    plan.quality_tier = static_cast<int>(manifest.quality_tier);
    plan.context = preview.context;
    plan.providers = preview.stages;
    plan.ready = false;
    plan.measured = preview.measured && preview.first_token_ms > 0;
    plan.token_ms = preview.token_ms;
    plan.first_token_ms = preview.first_token_ms;
    // Reported only: downloads at an assumed 50 MB/s; loading itself is not estimated.
    plan.load_ms = static_cast<double>(preview.download_bytes) / 50e6 * 1000.0;
    plan.cache_warm = preview.from_cache;
    plan.resource_bytes = preview.download_bytes;
    return plan;
}

std::unique_ptr<po::InferenceClient> open_ready(const ReadyChoice& choice,
    const po::PlacementRequest& request, po::Manifest& manifest, std::size_t& stage_count) {
    const po::ReplicaCandidate& replica = choice.replica;
    po::InferenceRoute route;
    route.stage_endpoints = {replica.control};
    route.hidden = request.models[choice.model].manifest.hidden;
    route.replica = true;
    auto client = std::make_unique<po::InferenceClient>(route);
    // The owner answers once earlier sessions' turns are done: wait, but not forever.
    client->stages().front()->set_timeout(600000);
    manifest = request.models[choice.model].manifest;
    stage_count = replica.stages;
    std::printf("replica=%s owner=%s model=%s stages=%u sessions_free=%u/%u draft=%s "
        "link=%s %ums ms_per_token=%u first_token_ms=%u\n", replica.replica_id.c_str(),
        replica.owner.c_str(), manifest.model_id.c_str(), replica.stages, replica.sessions_free,
        replica.sessions_max, replica.draft_sha256.empty() ? "no" : "yes",
        replica.relayed ? "relay" : "direct", replica.rtt_ms, replica.ms_per_token,
        replica.first_token_ms);
    return client;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    int exit_code = 0;
    FILE* api_output = nullptr;
    try {
        Options options = parse_options(argc, argv);
        // Keep all existing placement diagnostics off the machine-readable channel.
        if (options.api) {
#ifdef _WIN32
            api_output = _fdopen(_dup(_fileno(stdout)), "wb");
            if (!api_output || _dup2(_fileno(stderr), _fileno(stdout)) != 0)
#else
            std::signal(SIGPIPE, SIG_IGN); // closed API output must reach the cancellation sink
            api_output = fdopen(dup(fileno(stdout)), "w");
            if (!api_output || dup2(fileno(stderr), fileno(stdout)) < 0)
#endif
                throw std::runtime_error("could not open API output");
        }
        if (!options.log.empty()) redirect_stderr(options.log);
        if (options.form) {
            po::ReplicaOwnerOptions owner;
            owner.request = read_models(options);
            // Without --context, serve as much context as the GPUs hold (manifest = floor).
            owner.request.largest_context = options.context == 0;
            owner.discover = options.discover;
            owner.self_control = options.self_control;
            owner.session_listen = options.session_listen;
            owner.status_file = options.replica_status;
            owner.max_edge_rtt_ms = static_cast<std::uint32_t>(options.max_edge_rtt_ms);
            owner.allow_relay_edges = !options.no_relay_edges;
            owner.rank_delay = options.rank_delay;
            owner.warmup_tokens = options.warmup_tokens;
            owner.upgrades = options.upgrades;
            if (options.upgrade_idle_ms > 0) owner.upgrade_rules.idle_ms = options.upgrade_idle_ms;
            if (options.upgrade_check_ms > 0) owner.upgrade_rules.check_interval_ms = options.upgrade_check_ms;
            if (options.upgrade_backoff_ms > 0) owner.upgrade_rules.first_backoff_ms = options.upgrade_backoff_ms;
            exit_code = po::run_replica_owner(owner);
#ifdef _WIN32
            WSACleanup();
#endif
            return exit_code;
        }
        po::Manifest manifest = po::load_manifest(options.manifests.front());
        // The GGUF header is authoritative; a manifest may leave the architecture out, and
        // every manifest that does predates OLMoE support, so it is Qwen2.
        std::string architecture = manifest.architecture.empty() ? "qwen2" : manifest.architecture;
        std::unique_ptr<po::InferenceClient> client_holder;
        bool meets_target = false;  // the selected plan met the speed target (selection.hpp)
        std::size_t stage_count = options.providers.size();
        if (!options.providers.empty()) {
            if (options.manifests.size() != 1) {
                throw std::runtime_error("a fixed --provider route takes exactly one --manifest");
            }
            if (manifest.hidden == 0) throw std::runtime_error("manifest must include hidden_size");
            po::InferenceRoute route{options.providers, manifest.hidden};
            if (!options.ring_return.empty()) {
                route.ring_targets.push_back({});
                route.ring_targets.insert(route.ring_targets.end(),
                    options.ring_targets.begin(), options.ring_targets.end());
                route.peer_ids = options.peer_ids;
                route.return_listen = options.ring_return;
                route.return_target = options.ring_return_target;
            }
            client_holder = std::make_unique<po::InferenceClient>(route);
        } else {
            const auto metadata_started = po::Clock::now();
            po::PlacementRequest request = read_models(options);
            manifest = request.models.front().manifest;
            const auto architecture_of = [&](const po::Manifest& chosen) {
                for (const po::ModelOption& option : request.models) {
                    if (option.manifest.sha256 == chosen.sha256
                        && !option.model.architecture.empty()) return option.model.architecture;
                }
                return chosen.architecture.empty() ? std::string("qwen2") : chosen.architecture;
            };
            const double metadata_ms = po::elapsed_ns(metadata_started) / 1e6;
            if (!options.pin_model.empty()) {
                // An explicit model choice (a pinned conversation): this model or nothing.
                std::erase_if(request.models, [&](const po::ModelOption& option) {
                    return lowercase(option.manifest.sha256) != lowercase(options.pin_model);
                });
                if (request.models.empty()) {
                    throw std::runtime_error("model_unavailable: the chosen model is not in this catalog");
                }
            }
            double discovery_ms = 0;
            std::string ring_return = options.ring_return;
            std::string ring_return_target = options.ring_return_target;
            // Workers that may serve these models: from the sidecar (DHT) and/or --candidate.
            const auto find_candidates = [&] {
                std::vector<po::PlacementCandidate> candidates;
                if (!options.discover.empty()) {
                    const auto discovery_started = po::Clock::now();
                    std::vector<std::string> wanted;
                    for (const po::ModelOption& option : request.models) {
                        wanted.push_back(option.manifest.sha256);
                    }
                    po::Discovery found = po::discover_candidates(options.discover, wanted);
                    discovery_ms = po::elapsed_ns(discovery_started) / 1e6;
                    std::printf("discovered candidates=%zu self=%s\n", found.candidates.size(),
                        found.self_peer.c_str());
                    candidates = std::move(found.candidates);
                    if (ring_return.empty()) ring_return = found.return_listen;
                    if (ring_return_target.empty()) ring_return_target = "/p2p/" + found.self_peer;
                    if (ring_return.empty()) {
                        throw std::runtime_error("the sidecar has no ring return (-ring-inbound)");
                    }
                    if (candidates.empty()) {
                        throw std::runtime_error(request.models.size() == 1
                            ? "no workers found for this model" : "no workers found for these models");
                    }
                }
                for (std::size_t index = 0; index < options.candidates.size(); ++index) {
                    candidates.push_back({options.candidates[index], options.candidate_peers.empty()
                        ? std::string{} : options.candidate_peers[index]});
                }
                return candidates;
            };

            // One selection policy for ready replicas and new placements (selection.hpp).
            po::SelectionPolicy policy;
            policy.context = static_cast<std::uint32_t>(options.context);  // 0: each model's own
            policy.cold_start = options.wait_cold ? po::ColdStart::wait : po::ColdStart::ready_first;
            const bool accept_slower = options.policy == "any";
            std::vector<po::PlanEstimate> plans;
            std::vector<ReadyChoice> ready;
            std::vector<po::RoutePreview> previews;  // plans[ready.size() + i] = previews[i]
            po::Selection selection;
            std::size_t full_replicas = 0;
            const auto gather = [&] {
                plans.clear();
                ready.clear();
                previews.clear();
                if (options.replica) {
                    ready = usable_replicas(options, request, full_replicas);
                    for (const ReadyChoice& choice : ready) plans.push_back(ready_estimate(choice, request));
                }
                selection = po::select_plan(plans, policy);
                const bool ready_fallback = selection.fallback && *selection.fallback < ready.size();
                // Look at new placements unless a ready replica settles it: one that meets the
                // target, or (ready-first) a slower one the caller accepted.
                const bool settled = (selection.chosen && plans[*selection.chosen].ready)
                    || (accept_slower && ready_fallback && !options.wait_cold);
                if (!settled && !options.replica_only) {
                    try {
                        po::PlacementRequest preview = request;
                        preview.previews = &previews;
                        po::place_route(find_candidates(), preview);
                    } catch (const std::exception& error) {
                        std::printf("no new placement: %s\n", error.what());
                    }
                    for (const po::RoutePreview& found : previews) plans.push_back(new_estimate(found, request));
                    selection = po::select_plan(plans, policy);
                }
            };
            gather();
            std::optional<std::size_t> pick;
            int looked_again = 0;
            for (;;) {
                pick = selection.chosen;
                if (!pick && accept_slower) {
                    // Accepted slower or unmeasured: ready-first still prefers a ready replica.
                    std::vector<po::PlanEstimate> ready_plans(plans.begin(), plans.begin()
                        + static_cast<std::ptrdiff_t>(ready.size()));
                    const po::Selection among_ready = po::select_plan(ready_plans, policy);
                    pick = !options.wait_cold && among_ready.fallback ? among_ready.fallback
                        : selection.fallback;
                }
                if (!pick) {
                    // Nothing could run at all (e.g. every worker busy) is not a speed problem.
                    const bool any_feasible = std::any_of(plans.begin(), plans.end(),
                        [&](const po::PlanEstimate& plan) {
                            return po::assess_plan(plan, policy).verdict != po::Verdict::ineligible;
                        });
                    // A session or GPU another client just released can still read as taken for
                    // a moment (its owner has not seen the close yet, or is still stopping a
                    // cancelled answer): look again; while a fitting replica is only full, keep
                    // waiting for a session a little longer.
                    const int rounds = full_replicas != 0 ? 8 : 1;
                    if (!any_feasible && looked_again < rounds && !options.discover.empty()) {
                        ++looked_again;
                        std::printf("nothing free yet%s; looking again in 2 s\n",
                            full_replicas != 0 ? " (a fitting replica is full)" : "");
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        gather();
                        continue;
                    }
                    if (plans.empty() && options.replica_only) {
                        throw std::runtime_error("no READY replica with a free session was found");
                    }
                    if (!any_feasible) throw std::runtime_error("no placement fits the available workers");
                    throw std::runtime_error("below_target: " + selection.reason);
                }
                std::printf("selected %s (%s)\n", plans[*pick].label.c_str(),
                    selection.chosen ? "meets the target" : "accepted below target or unmeasured");
                if (selection.better_cold) {
                    std::printf("better model available after loading: %s\n",
                        plans[*selection.better_cold].label.c_str());
                }
                meets_target = selection.chosen.has_value();
                if (*pick >= ready.size()) break;
                try {
                    client_holder = open_ready(ready[*pick], request, manifest, stage_count);
                    architecture = architecture_of(manifest);
                    break;
                } catch (const std::exception& error) {
                    // Gone or full since discovery: drop it (no providers = never eligible) and
                    // choose again among what is left.
                    std::fprintf(stderr, "replica %s unusable: %s\n",
                        ready[*pick].replica.replica_id.c_str(), error.what());
                    plans[*pick].providers = 0;
                    selection = po::select_plan(plans, policy);
                }
            }
            if (*pick >= ready.size()) {
                // Place exactly the selected model at the previewed context.
                const po::RoutePreview& chosen = previews[*pick - ready.size()];
                po::ModelOption option = request.models[chosen.model];
                request.models = {option};
                request.context = chosen.context;
            }
          if (!client_holder) {
            po::PlacedRoute placement;
            // GPUs another client just released can still read as busy for a moment: when
            // nothing fits, search once more after a short wait.
            for (int round = 0;; ++round) {
                try {
                    placement = po::place_route(find_candidates(), request);
                    break;
                } catch (const std::exception& error) {
                    if (options.discover.empty() || round > 0) throw;
                    std::printf("%s; looking again in 2 s\n", error.what());
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
            manifest = placement.manifest;
            architecture = architecture_of(manifest);
            if (request.models.size() > 1) std::printf("model=%s\n", manifest.model_id.c_str());
            if (!placement.draft_model_id.empty()) {
                std::printf("draft=%s\n", placement.draft_model_id.c_str());
            }
            for (std::size_t index = 0; index < placement.stages.size(); ++index) {
                const po::PlacedStage& stage = placement.stages[index];
                std::printf("route=%s stage=%zu worker=%s peer=%s layers=%d..%d link=%s\n",
                    placement.route_id.c_str(), index, stage.worker_id.c_str(),
                    stage.peer_id.empty() ? "-" : stage.peer_id.c_str(), stage.begin, stage.end - 1,
                    (std::string(stage.relayed ? "relay " : "direct ")
                        + std::to_string(stage.rtt_ms) + "ms").c_str());
            }
            const po::PlacementTimings& timings = placement.timings;
            std::printf("timing metadata_ms=%.0f discovery_ms=%.0f capabilities_ms=%.0f plan_ms=%.1f "
                "reserve_ms=%.0f load_ms=%.0f placement_attempts=%d\n", metadata_ms, discovery_ms,
                timings.greeting_ms, timings.plan_ms, timings.reserve_ms, timings.load_ms,
                timings.attempts);
            po::InferenceRoute route = placement.route;
            if (!ring_return.empty()) {
                route.return_listen = ring_return;
                route.return_target = ring_return_target;
            } else {
                route.ring_targets.clear();
                route.peer_ids.clear();
                route.loop_target.clear();
            }
            if (options.no_loop) route.loop_target.clear();
            client_holder = std::make_unique<po::InferenceClient>(route,
                std::move(placement.connections));
            stage_count = placement.stages.size();
          }
        }
        po::InferenceClient& client = *client_holder;
        if (options.api) {
            if (architecture != "qwen2") throw std::runtime_error("API chat currently requires Qwen2");
            // CPU prefill of a tool-rich history can exceed the default 30-second IO timeout.
            // The gateway owns the overall request deadline and cancellation.
            client.set_timeout(600000);
            // The model actually used, so the caller can pin later turns to it.
            std::fprintf(api_output, "{\"model\":\"%s\",\"sha256\":\"%s\",\"meets_target\":%s}\n",
                po::json_escape(manifest.model_id).c_str(), lowercase(manifest.sha256).c_str(),
                meets_target ? "true" : "false");
            std::fflush(api_output);
            const auto session = client.create_session();
            client.set_timeout(600000); // includes the ring return established by create_session
            do {
            po::Frame prepared;
            if (options.api_chat) prepared = client.prepare_chat(session, options.prompts.front());
            const auto result = client.generate(session, options.api_chat ? "chat" : options.prompts.front(), options.tokens,
                [&](std::string_view piece) {
                    // Hex preserves bytes when a token ends in the middle of a UTF-8 character.
                    static constexpr char hex[] = "0123456789abcdef";
                    std::string bytes;
                    for (unsigned char byte : piece) { bytes += hex[byte >> 4]; bytes += hex[byte & 15]; }
                    std::fprintf(api_output, "{\"bytes\":\"%s\"}\n", bytes.c_str());
                    return std::fflush(api_output) == 0;
                });
            std::fprintf(api_output, "{\"done\":true,\"tokens\":%zu,\"eog\":%s,\"prompt_tokens\":%u,\"cached_tokens\":%u}\n",
                result.metrics.token_ids.size(), result.eog ? "true" : "false", prepared.rows, prepared.position);
            std::fflush(api_output);
            } while (options.api_chat && read_chat_record(options.prompts.front(), options.tokens));
            client.destroy_session(session);
            std::fclose(api_output);
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
        if (options.chat) {
#ifdef _WIN32
            SetConsoleOutputCP(CP_UTF8);
#endif
            run_chat(client, manifest, options.tokens, stage_count, architecture);
            std::printf("mode=%s client_activations_received=%llu\n", client.ring() ? "ring" : "hub",
                static_cast<unsigned long long>(client.activations_received()));
#ifdef _WIN32
            WSACleanup();
#endif
            return 0;
        }
        const std::uint64_t persistent_session = options.persistent ? client.create_session() : 0;
        std::vector<std::string> outputs;
        for (int index = 0; index < options.requests; ++index) {
            const std::string& prompt =
                options.prompts[static_cast<std::size_t>(index) % options.prompts.size()];
            const po::RequestResult result = options.persistent
                ? client.generate(persistent_session, prompt, options.tokens)
                : client.generate_once(prompt, options.tokens);
            const std::size_t generated = result.metrics.token_ids.size();
            const double decode_ms = result.metrics.latency_ms - result.metrics.ttft_ms;
            std::printf("request=%d tokens=%zu latency_ms=%.3f ttft_ms=%.1f decode_tok_s=%.2f "
                "route_setup_ms=%.0f output=%s\n", index + 1, generated,
                result.metrics.latency_ms, result.metrics.ttft_ms,
                generated > 1 && decode_ms > 0 ? (generated - 1) * 1000.0 / decode_ms : 0.0,
                client.route_setup_ms(), result.output.c_str());
            outputs.push_back(result.output);
            if (!options.expected.empty() && result.output != options.expected) {
                throw std::runtime_error("deterministic output mismatch on request "
                    + std::to_string(index + 1));
            }
        }
        if (options.persistent) client.destroy_session(persistent_session);
        std::printf("mode=%s client_activations_received=%llu\n", client.ring() ? "ring" : "hub",
            static_cast<unsigned long long>(client.activations_received()));
        if (options.require_direct && client.activations_received() != 0) {
            throw std::runtime_error("intermediate activations passed through the client");
        }
        if (!options.report.empty()) {
            std::ofstream report(options.report, std::ios::binary);
            if (!report) throw std::runtime_error("could not create report");
            report << "{\n  \"runtime\": \"dan-client\",\n  \"outputs\": [";
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                if (index != 0) report << ',';
                report << "\n    \"" << po::json_escape(outputs[index]) << "\"";
            }
            report << "\n  ]\n}\n";
        }
    } catch (const std::exception& error) {
        if (api_output) {
            // Only fixed codes cross the API boundary; exceptions can contain user text.
            const std::string what = error.what();
            const bool context_full = what.find("context exhausted") != std::string::npos;
            const bool invalid_chat = what.find("invalid chat") != std::string::npos;
            if (what.starts_with("below_target: ")) {
                // The reason names only plans (model IDs and numbers), never user text.
                std::fprintf(api_output, "{\"error\":\"below_target\",\"reason\":\"%s\"}\n",
                    po::json_escape(what.substr(14)).c_str());
            } else {
                std::fprintf(api_output, "{\"error\":\"%s\"}\n",
                    what.starts_with("model_unavailable: ") ? "model_unavailable"
                    : context_full ? "context_length_exceeded"
                    : invalid_chat ? "invalid_chat_request" : "inference_failed");
            }
            std::fclose(api_output);
        }
        std::fprintf(stderr, "dan-client: %s\n", error.what());
        exit_code = 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return exit_code;
}
