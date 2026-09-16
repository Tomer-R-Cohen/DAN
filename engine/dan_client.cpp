// dan-client: runs inference with no coordinator. This process creates the sessions and
// drives the token loop itself, over either
//   --provider ...   an explicit stage route (stages already loaded), or
//   --candidate ...  workers in serve mode: it plans placement, reserves and assigns them, or
//   --discover API   the same, with candidates found by the local sidecar (DHT discovery).

#include "provider_owned/client.hpp"
#include "provider_owned/manifest.hpp"
#include "provider_owned/placement.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace po = dan::provider_owned;

namespace {

struct Options {
    std::string manifest;
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
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--persistent") { options.persistent = true; continue; }
        if (option == "--require-direct") { options.require_direct = true; continue; }
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--manifest") options.manifest = value;
        else if (option == "--provider") options.providers.push_back(value);
        else if (option == "--prompt") options.prompts.push_back(value);
        else if (option == "--tokens") options.tokens = std::stoi(value);
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
        else throw std::runtime_error("unknown option: " + option);
    }
    if (options.requests == 0) options.requests = static_cast<int>(options.prompts.size());
    const bool discovered = !options.discover.empty();
    if (options.ring_return_target.empty() && !discovered) {
        options.ring_return_target = options.ring_return;
    }
    const bool ring = !options.ring_return.empty() || discovered;
    const bool placed = !options.candidates.empty() || discovered;
    const int sources = !options.providers.empty() + !options.candidates.empty() + discovered;
    if (options.manifest.empty() || sources != 1
        || (discovered && !options.candidate_peers.empty())
        || options.prompts.empty() || options.tokens < 1 || options.requests < 1
        || options.sessions < 1 || options.minimum_stages < 1 || options.context < 0
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
            "[placement and other options above]");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    int exit_code = 0;
    try {
        const Options options = parse_options(argc, argv);
        const po::Manifest manifest = po::load_manifest(options.manifest);
        if (manifest.hidden == 0) throw std::runtime_error("manifest must include hidden_size");
        std::unique_ptr<po::InferenceClient> client_holder;
        if (!options.providers.empty()) {
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
            po::PlacementRequest request;
            request.manifest = manifest;
            request.context = options.context != 0
                ? static_cast<std::uint32_t>(options.context) : manifest.context;
            request.sessions = static_cast<std::uint32_t>(options.sessions);
            request.minimum_stages = static_cast<std::size_t>(options.minimum_stages);
            request.runtime_abi = options.runtime_abi;
            const std::filesystem::path metadata = options.metadata_cache.empty()
                ? std::filesystem::temp_directory_path() / "dan-client"
                    / (manifest.sha256 + "-" + po::random_route_id() + ".gguf")
                : std::filesystem::path(options.metadata_cache);
            std::string error;
            if (!po::inspect_range_model({manifest.url, manifest.revision, manifest.sha256,
                    metadata, 0, 1}, request.model, error)) {
                throw std::runtime_error("model metadata: " + error);
            }
            std::vector<po::PlacementCandidate> candidates;
            std::string ring_return = options.ring_return;
            std::string ring_return_target = options.ring_return_target;
            if (!options.discover.empty()) {
                po::Discovery found = po::discover_candidates(options.discover, manifest.sha256);
                std::printf("discovered candidates=%zu self=%s\n", found.candidates.size(),
                    found.self_peer.c_str());
                candidates = std::move(found.candidates);
                if (ring_return.empty()) ring_return = found.return_listen;
                if (ring_return_target.empty()) ring_return_target = "/p2p/" + found.self_peer;
                if (ring_return.empty()) {
                    throw std::runtime_error("the sidecar has no ring return (-ring-inbound)");
                }
                if (candidates.empty()) throw std::runtime_error("no workers found for this model");
            }
            for (std::size_t index = 0; index < options.candidates.size(); ++index) {
                candidates.push_back({options.candidates[index], options.candidate_peers.empty()
                    ? std::string{} : options.candidate_peers[index]});
            }
            po::PlacedRoute placement = po::place_route(candidates, request);
            for (std::size_t index = 0; index < placement.stages.size(); ++index) {
                const po::PlacedStage& stage = placement.stages[index];
                std::printf("route=%s stage=%zu worker=%s layers=%d..%d\n",
                    placement.route_id.c_str(), index, stage.worker_id.c_str(),
                    stage.begin, stage.end - 1);
            }
            po::InferenceRoute route = placement.route;
            if (!ring_return.empty()) {
                route.return_listen = ring_return;
                route.return_target = ring_return_target;
            } else {
                route.ring_targets.clear();
                route.peer_ids.clear();
            }
            client_holder = std::make_unique<po::InferenceClient>(route,
                std::move(placement.connections));
        }
        po::InferenceClient& client = *client_holder;
        const std::uint64_t persistent_session = options.persistent ? client.create_session() : 0;
        std::vector<std::string> outputs;
        for (int index = 0; index < options.requests; ++index) {
            const std::string& prompt =
                options.prompts[static_cast<std::size_t>(index) % options.prompts.size()];
            const po::RequestResult result = options.persistent
                ? client.generate(persistent_session, prompt, options.tokens)
                : client.generate_once(prompt, options.tokens);
            std::printf("request=%d tokens=%zu latency_ms=%.3f output=%s\n", index + 1,
                result.metrics.token_ids.size(), result.metrics.latency_ms,
                result.output.c_str());
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
        std::fprintf(stderr, "dan-client: %s\n", error.what());
        exit_code = 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return exit_code;
}
