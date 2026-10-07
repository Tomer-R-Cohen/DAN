// Runs DAN's real planner against a hardware inventory, for capacity planning before any
// weights are downloaded.
//
//   provider_owned_plan_check --index FILE --offered 24576,24576,... [--context 512]
//                             [--sessions 1] [--min-stages 1]
//   provider_owned_plan_check --layers 94 --hidden 4096 --heads 64 --kv-heads 4
//                             --experts 128 --experts-used 8 --expert-ffn 1536 --head-dim 128
//                             --layer-bytes N --global-bytes N --offered ...
//
// `--index` uses a model index saved by dan-client (the real header of a real artifact). The
// explicit form is for an artifact whose index cannot be read whole, such as a split GGUF:
// state where each number came from when reporting it.
//
// Offered MiB are what a worker advertises, i.e. card VRAM minus the launcher's
// reserve_vram_mib. The planner then keeps back its own max(1 GiB, 15%) on top.
#include "provider_owned/planner.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace po = dan::provider_owned;

namespace {

std::vector<std::uint64_t> parse_list(const std::string& text) {
    std::vector<std::uint64_t> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string piece = text.substr(start, comma - start);
        if (!piece.empty()) values.push_back(std::strtoull(piece.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return values;
}

} // namespace

int main(int argc, char** argv) {
    po::ModelIndex model;
    std::string index_path;
    std::vector<std::uint64_t> offered;
    std::uint32_t context = 512, sessions = 1;
    std::size_t min_stages = 1;
    std::uint64_t layer_bytes = 0, global_bytes = 0, head_dim = 0;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--index") index_path = value;
        else if (option == "--offered") offered = parse_list(value);
        else if (option == "--context") context = std::stoul(value);
        else if (option == "--sessions") sessions = std::stoul(value);
        else if (option == "--min-stages") min_stages = std::stoul(value);
        else if (option == "--layers") model.layers = std::stoul(value);
        else if (option == "--hidden") model.hidden = std::stoul(value);
        else if (option == "--heads") model.heads = std::stoul(value);
        else if (option == "--kv-heads") model.kv_heads = std::stoul(value);
        else if (option == "--experts") model.experts = std::stoul(value);
        else if (option == "--experts-used") model.experts_used = std::stoul(value);
        else if (option == "--expert-ffn") model.expert_ffn_length = std::stoul(value);
        else if (option == "--head-dim") head_dim = std::stoull(value);
        else if (option == "--layer-bytes") layer_bytes = std::strtoull(value.c_str(), nullptr, 10);
        else if (option == "--global-bytes") global_bytes = std::strtoull(value.c_str(), nullptr, 10);
        else { std::fprintf(stderr, "unknown option %s\n", option.c_str()); return 2; }
    }
    if (!index_path.empty()) {
        if (!po::load_model_index(index_path, model)) {
            std::fprintf(stderr, "could not load model index %s\n", index_path.c_str());
            return 2;
        }
    } else {
        // Build the tensor list the planner needs: one entry per layer plus the head/tail
        // globals, with the measured byte spans.
        model.architecture = "qwen3moe";
        model.head_dim_k = model.head_dim_v = static_cast<std::uint32_t>(head_dim);
        model.tensors.push_back({"token_embd.weight", global_bytes / 2, {}, 0});
        model.tensors.push_back({"output.weight", global_bytes - global_bytes / 2, {}, 0});
        model.tensors.push_back({"output_norm.weight", 0, {}, 0});
        for (std::uint32_t layer = 0; layer < model.layers; ++layer) {
            model.tensors.push_back({"blk." + std::to_string(layer) + ".weight", layer_bytes, {}, 0});
        }
    }
    if (offered.empty() || model.layers == 0) {
        std::fprintf(stderr, "need --offered and a model (--index or explicit geometry)\n");
        return 2;
    }

    constexpr double gib = 1024.0 * 1024 * 1024;
    std::printf("model=%s layers=%u hidden=%u heads=%u kv_heads=%u experts=%u used=%u\n",
        model.architecture.c_str(), model.layers, model.hidden, model.heads, model.kv_heads,
        model.experts, model.experts_used);
    std::printf("head width k/v = %u/%u (hidden/heads would be %u)\n",
        po::head_width_k(model), po::head_width_v(model),
        model.heads ? model.hidden / model.heads : 0);
    std::printf("whole-model bytes = %.2f GiB; context=%u sessions=%u -> %u allocated positions\n",
        po::stage_model_bytes(model, 0, static_cast<int>(model.layers)) / gib, context, sessions,
        po::allocated_positions(context, sessions));
    std::printf("workers offering:");
    for (const std::uint64_t mib : offered) std::printf(" %llu MiB", (unsigned long long) mib);
    std::printf("  (min stages %zu)\n", min_stages);

    const auto plan = po::plan_stages(model, offered, context, sessions, min_stages, std::nullopt);
    if (!plan) {
        std::printf("RESULT: no plan fits this inventory\n");
        return 1;
    }
    double weights = 0, kv = 0;
    for (const po::StageAssignment& stage : *plan) {
        std::printf("  stage worker=%zu layers=[%d,%d) weights=%.2f GiB kv=%.2f GiB of %.0f MiB offered\n",
            stage.provider, stage.begin, stage.end, stage.model_bytes / gib, stage.kv_bytes / gib,
            (double) offered[stage.provider]);
        weights += stage.model_bytes / gib;
        kv += stage.kv_bytes / gib;
    }
    std::printf("RESULT: %zu stages, %.2f GiB weights + %.2f GiB KV placed\n",
        plan->size(), weights, kv);
    return 0;
}
