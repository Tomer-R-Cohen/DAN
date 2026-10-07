// Two resident sessions on one placed route: do they stay independent?
//
//   provider_owned_sessions_client --provider HOST:PORT [--provider ...] [--ring-target ...]
//       --ring-return HOST:PORT --hidden N [--tokens 24] [--prompt-a T] [--prompt-b T]
//
// Each session is a KV sequence on every stage, so a bug there shows up as one conversation
// seeing another's text. The check is always the same: a request made while the other session
// is live must produce the token ids it produced alone on this same route and hardware.
//
//   1. baseline   each session alone (the other destroyed): two requests per conversation
//   2. interleave both live, requests alternating between them
//   3. reset      session A reset mid-conversation restarts it; B's next request is unchanged
//   4. recreate   A destroyed and created again; B still unaffected
//
// Exit code 0 when every comparison matches. Speculation is not involved.
#include "provider_owned/client.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace po = dan::provider_owned;

namespace {

using Ids = std::vector<std::uint32_t>;

std::string join(const Ids& ids) {
    std::string text;
    for (const std::uint32_t id : ids) text += std::to_string(id) + ",";
    return text;
}

int failures = 0;

void same(const char* what, const Ids& got, const Ids& want) {
    if (got == want) {
        std::printf("  PASS %s (%zu tokens)\n", what, got.size());
        return;
    }
    ++failures;
    std::printf("  FAIL %s\n    got  %s\n    want %s\n", what, join(got).c_str(), join(want).c_str());
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    po::InferenceRoute route;
    int tokens = 24;
    std::string prompt_a = "The capital of France is";
    std::string prompt_b = "Name three colors of the rainbow.";
    std::string follow_a = " Tell me more about it.";
    std::string follow_b = " And one more.";
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--provider") route.stage_endpoints.push_back(value);
        else if (option == "--ring-target") route.ring_targets.push_back(value);
        else if (option == "--ring-return") route.return_listen = value;
        else if (option == "--ring-return-target") route.return_target = value;
        else if (option == "--hidden") route.hidden = std::stoul(value);
        else if (option == "--tokens") tokens = std::stoi(value);
        else if (option == "--prompt-a") prompt_a = value;
        else if (option == "--prompt-b") prompt_b = value;
        else { std::fprintf(stderr, "unknown option %s\n", option.c_str()); return 2; }
    }
    if (route.stage_endpoints.empty() || route.return_listen.empty() || route.hidden == 0) {
        std::fprintf(stderr, "usage: --provider HOST:PORT [...] --ring-return HOST:PORT --hidden N\n");
        return 2;
    }
    // Direct mode: the tail reaches the return listener at the same address (dan-client
    // defaults the same way when no --ring-return-target is given).
    if (route.return_target.empty()) route.return_target = route.return_listen;
    // ring_targets[0] is unused; the client passes one per stage after the first.
    route.ring_targets.insert(route.ring_targets.begin(), std::string{});
    if (route.ring_targets.size() != route.stage_endpoints.size()) {
        std::fprintf(stderr, "expected one --ring-target per stage after the first\n");
        return 2;
    }

    try {
        po::InferenceClient client(route);
        const auto run = [&](std::uint64_t session, const std::string& prompt) {
            return client.generate(session, prompt, tokens).metrics.token_ids;
        };

        // A's conversation is always prompt_a then follow_a (a reset or a new session puts
        // it back to the start). B's runs straight through: prompt_b then follow_b three
        // times. The baseline records exactly those sequences, alone on this same route.
        std::printf("1. baseline: each conversation alone\n");
        const std::uint64_t first = client.create_session();
        const Ids a1 = run(first, prompt_a);
        const Ids a2 = run(first, follow_a);
        client.destroy_session(first);
        const std::uint64_t second = client.create_session();
        std::vector<Ids> b;
        b.push_back(run(second, prompt_b));
        for (int index = 0; index < 3; ++index) b.push_back(run(second, follow_b));
        client.destroy_session(second);
        std::printf("  A: %zu + %zu tokens; B: %zu, %zu, %zu, %zu tokens\n", a1.size(), a2.size(),
            b[0].size(), b[1].size(), b[2].size(), b[3].size());
        if (a1.empty() || b[0].empty()) { std::printf("  FAIL empty baseline\n"); return 1; }

        std::printf("2. both sessions live, requests interleaved\n");
        const std::uint64_t alpha = client.create_session();
        const std::uint64_t beta = client.create_session();
        same("A request 1 while B exists", run(alpha, prompt_a), a1);
        same("B request 1 while A is mid-conversation", run(beta, prompt_b), b[0]);
        same("A request 2 after B's request", run(alpha, follow_a), a2);
        same("B request 2 after A's request", run(beta, follow_b), b[1]);

        std::printf("3. reset A; B must not notice\n");
        client.reset_session(alpha);
        same("A starts over after the reset", run(alpha, prompt_a), a1);
        same("B request 3 spans A's reset", run(beta, follow_b), b[2]);
        same("A follow-up after the reset", run(alpha, follow_a), a2);

        std::printf("4. destroy and recreate A; B must not notice\n");
        client.destroy_session(alpha);
        const std::uint64_t alpha_again = client.create_session();
        same("A in a fresh session", run(alpha_again, prompt_a), a1);
        same("B request 4 spans A's destroy/recreate", run(beta, follow_b), b[3]);
        same("A follow-up in the fresh session", run(alpha_again, follow_a), a2);

        client.destroy_session(alpha_again);
        client.destroy_session(beta);
    } catch (const std::exception& error) {
        std::printf("  FAIL exception: %s\n", error.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "sessions: all comparisons matched"
        : "sessions: MISMATCHES FOUND");
    return failures == 0 ? 0 : 1;
}
