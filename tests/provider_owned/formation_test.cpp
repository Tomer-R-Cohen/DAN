#include "provider_owned/formation.hpp"

#include <cassert>

int main() {
    namespace po = dan::provider_owned;
    constexpr std::uint64_t gib = 1024ull * 1024 * 1024;
    po::ModelIndex model;
    model.architecture = "qwen2";
    model.layers = 6; model.hidden = 1024; model.heads = 16; model.kv_heads = 4;
    model.header_bytes = 4096;
    model.tensors.push_back({"token_embd.weight", gib / 2});
    model.tensors.push_back({"output_norm.weight", 4096});
    for (int layer = 0; layer < 6; ++layer) {
        model.tensors.push_back({"blk." + std::to_string(layer) + ".weight", gib});
    }
    model.tensors.push_back({"output.weight", gib / 2});

    std::vector<po::ProviderCapability> three{
        {"a", "gpu-a", 4096}, {"b", "gpu-b", 4096}, {"c", "gpu-c", 4096}};
    const auto plan = po::plan_replica(model, three, 128, 1);
    assert(plan && plan->size() == 3);
    assert(plan->front().begin == 0 && plan->back().end == 6);
    for (std::size_t index = 1; index < plan->size(); ++index) {
        assert((*plan)[index - 1].end == (*plan)[index].begin);
    }
    std::vector<po::ProviderCapability> two{
        {"large-a", "gpu-a", 8192}, {"large-b", "gpu-b", 8192},
        {"unused", "gpu-c", 4096}};
    const auto smaller = po::plan_replica(model, two, 128, 1);
    assert(smaller && smaller->size() == 2);
    std::vector<po::ProviderCapability> one{{"single", "gpu", 10240}};
    const auto local = po::plan_replica(model, one, 128, 1);
    assert(local && local->size() == 1 && local->front().begin == 0
        && local->front().end == 6);

    assert(po::compatible_dense_qwen2(model));
    po::ModelIndex incompatible = model;
    incompatible.architecture = "qwen2moe";
    assert(!po::compatible_dense_qwen2(incompatible));
    incompatible = model;
    incompatible.tensors.erase(std::remove_if(incompatible.tensors.begin(), incompatible.tensors.end(),
        [](const po::ModelTensor& tensor) { return tensor.name.starts_with("blk.5."); }),
        incompatible.tensors.end());
    assert(!po::compatible_dense_qwen2(incompatible));

    const po::ProviderCapability original{"provider-1", "RTX", 6656};
    po::ProviderCapability parsed;
    assert(po::parse_available(po::available_message(original), parsed));
    assert(parsed.id == original.id && parsed.offered_vram_mib == original.offered_vram_mib);

    po::ModelAssignment assignment{"qwen", "https://example.test/model.gguf",
        std::string(40, 'a'), std::string(64, 'b'), 0, 3, 128, 4}, decoded;
    assert(po::parse_assignment(po::assignment_message(assignment), decoded));
    assert(decoded.end == 3 && decoded.sessions == 4);
}
