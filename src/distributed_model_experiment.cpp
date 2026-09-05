#include "distributed_runtime.hpp"
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char* argv[])
{
    if (argc != 5 && argc != 7) {
        std::fprintf(stderr, "Usage: %s <rpc-llama-completion> <model.gguf> <group-id> <endpoints> [--tensor-split weights]\n", argv[0]);
        return 1;
    }
    dan::DistributedConfig config{argv[3], argv[1], argv[2], argv[4], "auto"};
    if (argc == 7) {
        if (std::string_view(argv[5]) != "--tensor-split") return 1;
        config.tensor_split = argv[6];
    }
    const auto participants = dan::split_rpc_endpoints(config.endpoints);
    if (participants.size() < 2 || access(config.model.c_str(), R_OK) == -1) return 1;
    std::cout << "Distributed model experiment\nGroup: " << config.id << "\nParticipants:\n";
    for (const auto& endpoint : participants) std::cout << "  - " << endpoint << '\n';
    std::cout << "Prompt: " << std::flush;
    std::string prompt;
    if (!std::getline(std::cin, prompt) || prompt.empty()) return 1;
    std::string response;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = dan::run_distributed_inference(config, prompt, response);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (!ok) {
        std::fprintf(stderr, "Distributed inference failed after %.1f ms\n", ms);
        return 1;
    }
    std::cout << "Response:\n" << response << "\nDistributed latency: " << ms << " ms\n";
}
