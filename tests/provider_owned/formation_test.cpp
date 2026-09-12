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
    assert(decoded == assignment);
    assert(decoded.end == 3 && decoded.sessions == 4);
    decoded.end = 4;
    assert(decoded != assignment);
    assignment.next_endpoint = "127.0.0.1:5001";
    assert(po::parse_assignment(po::assignment_message(assignment), decoded));
    assert(decoded == assignment);
    decoded.next_endpoint = "127.0.0.1:5002";
    assert(decoded != assignment && decoded.same_stage(assignment));

    const auto two_stage = po::plan_replica(model, two, 128, 1, 2);
    assert(two_stage && two_stage->size() == 2);
    assert((*two_stage)[0].begin == 0 && (*two_stage)[0].end == 3
        && (*two_stage)[1].begin == 3 && (*two_stage)[1].end == 6);

    po::ProviderCapability capability{"a", "GPU", 4096, "127.0.0.1:5001"}, parsed_ring;
    assert(po::parse_available(po::available_message(capability), parsed_ring));
    assert(parsed_ring.ring_endpoint == capability.ring_endpoint);
    assert(!po::parse_available("id=a\ngpu=GPU\nvram_mib=4096\nring=host:not-a-port", parsed_ring));
    assert(!po::parse_available("id=a\ngpu=GPU\nvram_mib=4096\nring=host:70000", parsed_ring));
    const std::string peer = "12D3KooWS2gYS5PaxaUy1mU5eY3CnBjUVfYuNbuRxqqeSz9v8KCr";
    capability.ring_endpoint = "/ip4/127.0.0.1/tcp/50202/p2p/" + peer;
    assert(po::parse_available(po::available_message(capability), parsed_ring));
    assignment.next_endpoint = capability.ring_endpoint;
    assignment.previous_peer_id = peer;
    assert(po::parse_assignment(po::assignment_message(assignment), decoded));
    assert(decoded == assignment);
    assignment.next_endpoint += ',';
    assert(!po::parse_assignment(po::assignment_message(assignment), decoded));
    assignment.next_endpoint = "host:0";
    assert(!po::parse_assignment(po::assignment_message(assignment), decoded));
}
