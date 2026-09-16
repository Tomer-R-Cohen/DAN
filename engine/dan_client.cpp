// dan-client: runs inference over an explicit, already-resolved stage route.
// No coordinator: this process creates the sessions and drives the token loop itself.

#include "provider_owned/client.hpp"
#include "provider_owned/manifest.hpp"

#include <cstdio>
#include <exception>
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
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--persistent") { options.persistent = true; continue; }
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--manifest") options.manifest = value;
        else if (option == "--provider") options.providers.push_back(value);
        else if (option == "--prompt") options.prompts.push_back(value);
        else if (option == "--tokens") options.tokens = std::stoi(value);
        else if (option == "--requests") options.requests = std::stoi(value);
        else if (option == "--expected-output") options.expected = value;
        else if (option == "--report") options.report = value;
        else throw std::runtime_error("unknown option: " + option);
    }
    if (options.requests == 0) options.requests = static_cast<int>(options.prompts.size());
    if (options.manifest.empty() || options.providers.empty() || options.prompts.empty()
        || options.tokens < 1 || options.requests < 1) {
        throw std::runtime_error(
            "usage: dan-client --manifest FILE --provider HOST:PORT [--provider HOST:PORT ...] "
            "--prompt TEXT [...] [--tokens N] [--requests N] [--persistent] "
            "[--expected-output TEXT] [--report FILE]");
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
        po::InferenceClient client({options.providers, manifest.hidden});
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
