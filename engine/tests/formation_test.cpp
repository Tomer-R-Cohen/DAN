#include "provider_owned/formation.hpp"
#include "provider_owned/planner.hpp"

#include "model_fixtures.hpp"
#include "provider_owned/speculation.hpp"

#include <cassert>
#include <chrono>

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


    // --- OLMoE: the second supported stage architecture -----------------------------
    {
        po::ModelIndex olmoe = dan::test::olmoe_index(6);
        std::string reason;
        assert(po::compatible_stage_model(olmoe, &reason));

        // Unsupported architectures stay rejected, MoE or not.
        for (const char* architecture : {"qwen2moe", "qwen3moe", "deepseek2", "mixtral",
                "granitemoe", "olmo2", "", "OLMOE"}) {
            po::ModelIndex other = olmoe;
            other.architecture = architecture;
            assert(!po::compatible_stage_model(other, &reason));
        }

        // Expert metadata must be sane, and expert_used_count is never the resident count.
        po::ModelIndex broken = olmoe;
        broken.experts = 0;
        assert(!po::compatible_stage_model(broken, &reason));
        broken = olmoe;
        broken.experts_used = 0;
        assert(!po::compatible_stage_model(broken, &reason));
        broken = olmoe;
        broken.experts_used = broken.experts + 1;
        assert(!po::compatible_stage_model(broken, &reason));
        broken = olmoe;
        broken.ffn_length = 0;
        assert(!po::compatible_stage_model(broken, &reason));

        // A missing router or expert tensor in any owned layer is refused.
        for (const char* missing : {"ffn_gate_inp.weight", "ffn_gate_exps.weight",
                "ffn_up_exps.weight", "ffn_down_exps.weight", "attn_q_norm.weight",
                "attn_k_norm.weight", "ffn_norm.weight", "attn_v.weight"}) {
            po::ModelIndex gap = olmoe;
            dan::test::erase_tensor(gap, std::string("blk.4.") + missing);
            assert(!po::compatible_stage_model(gap, &reason));
        }

        // OLMoE's head is explicit: no tied-embedding fallback.
        po::ModelIndex headless = olmoe;
        dan::test::erase_tensor(headless, "output.weight");
        assert(!po::compatible_stage_model(headless, &reason));
        po::ModelIndex unnormed = olmoe;
        dan::test::erase_tensor(unnormed, "output_norm.weight");
        assert(!po::compatible_stage_model(unnormed, &reason));

        // Expert banks must match the declared geometry, in every axis.
        po::ModelIndex reshaped = olmoe;
        dan::test::tensor_named(reshaped, "blk.2.ffn_gate_exps.weight")->dimensions
            = {reshaped.hidden, reshaped.ffn_length, reshaped.experts_used};  // 8 of 64
        assert(!po::compatible_stage_model(reshaped, &reason));
        reshaped = olmoe;
        dan::test::tensor_named(reshaped, "blk.2.ffn_down_exps.weight")->dimensions
            = {reshaped.hidden, reshaped.ffn_length, reshaped.experts};  // gate order, not down
        assert(!po::compatible_stage_model(reshaped, &reason));
        reshaped = olmoe;
        dan::test::tensor_named(reshaped, "blk.0.ffn_up_exps.weight")->dimensions
            = {reshaped.hidden, reshaped.ffn_length};  // two dimensions, not three
        assert(!po::compatible_stage_model(reshaped, &reason));
        reshaped = olmoe;
        dan::test::tensor_named(reshaped, "blk.0.ffn_gate_inp.weight")->dimensions
            = {reshaped.hidden, reshaped.experts + 1};
        assert(!po::compatible_stage_model(reshaped, &reason));

        // A dense model that claims experts is refused rather than planned as dense.
        po::ModelIndex confused = dan::test::qwen2_index(6);
        confused.experts = 8;
        confused.experts_used = 2;
        assert(!po::compatible_stage_model(confused, &reason));

        // Planning: OLMoE splits into contiguous stages like any other supported model.
        std::vector<po::ProviderCapability> moe_three{
            {"a", "gpu-a", 4096}, {"b", "gpu-b", 4096}, {"c", "gpu-c", 4096}};
        const auto moe_plan = po::plan_replica(olmoe, moe_three, 512, 1);
        assert(moe_plan && moe_plan->size() >= 1 && moe_plan->front().begin == 0
            && moe_plan->back().end == 6);
        for (std::size_t index = 1; index < moe_plan->size(); ++index) {
            assert((*moe_plan)[index - 1].end == (*moe_plan)[index].begin);
        }
        // One layer is indivisible: a worker that cannot hold a whole layer gets no plan.
        std::vector<po::ProviderCapability> tiny{{"small", "gpu", 1100}};
        assert(!po::plan_replica(olmoe, tiny, 512, 1));
    }

    // --- Qwen3-MoE: the second MoE family, with an explicit head width ---------------
    {
        po::ModelIndex qwen3 = dan::test::qwen3moe_index(6);
        std::string reason;
        assert(po::compatible_stage_model(qwen3, &reason));
        // A tied output head is legal here (llama.cpp falls back to the token embedding),
        // unlike OLMoE which requires an explicit one.
        po::ModelIndex tied = dan::test::qwen3moe_index(6, 128, 8, 1536, true);
        assert(po::compatible_stage_model(tied, &reason));
        assert(dan::test::tensor_named(tied, "output.weight") == nullptr);

        // The head width is declared, not derived: 128 while hidden/heads is 64. Everything
        // downstream must use the declared value.
        assert(qwen3.hidden / qwen3.heads == 64);
        assert(po::head_width_k(qwen3) == 128 && po::head_width_v(qwen3) == 128);
        // A model that leaves the widths out falls back to the quotient.
        po::ModelIndex derived = qwen3;
        derived.head_dim_k = 0;
        derived.head_dim_v = 0;
        assert(po::head_width_k(derived) == 64);
        // ...and the KV estimate differs by exactly the ratio of the widths.
        assert(po::kv_bytes(qwen3, 0, 6, 512, 1) == 2 * po::kv_bytes(derived, 0, 6, 512, 1));
        // Explicit widths are used for Qwen2/OLMoE too, where they equal the quotient.
        po::ModelIndex olmoe = dan::test::olmoe_index(6);
        assert(po::head_width_k(olmoe) == olmoe.hidden / olmoe.heads);

        // llama.cpp's Qwen3-MoE graph asserts equal K and V widths.
        po::ModelIndex lopsided = qwen3;
        lopsided.head_dim_v = 64;
        assert(!po::compatible_stage_model(lopsided, &reason));

        // Expert geometry, including the per-expert width that is not feed_forward_length.
        po::ModelIndex broken = qwen3;
        broken.expert_ffn_length = 0;   // and no feed_forward_length to fall back on
        assert(!po::compatible_stage_model(broken, &reason));
        broken = qwen3;
        broken.experts_used = broken.experts + 1;
        assert(!po::compatible_stage_model(broken, &reason));
        broken = qwen3;
        broken.experts = 1;
        assert(!po::compatible_stage_model(broken, &reason));

        // Missing router/expert/attention tensors in any owned layer.
        for (const char* missing : {"ffn_gate_inp.weight", "ffn_gate_exps.weight",
                "ffn_up_exps.weight", "ffn_down_exps.weight", "attn_q_norm.weight",
                "attn_k_norm.weight", "attn_v.weight", "ffn_norm.weight"}) {
            po::ModelIndex gap = qwen3;
            dan::test::erase_tensor(gap, std::string("blk.3.") + missing);
            assert(!po::compatible_stage_model(gap, &reason));
        }
        // Expert banks must match the declared expert geometry.
        po::ModelIndex reshaped = qwen3;
        dan::test::tensor_named(reshaped, "blk.2.ffn_up_exps.weight")->dimensions
            = {reshaped.hidden, reshaped.expert_ffn_length, reshaped.experts_used};
        assert(!po::compatible_stage_model(reshaped, &reason));
        reshaped = qwen3;
        dan::test::tensor_named(reshaped, "blk.2.ffn_down_exps.weight")->dimensions
            = {reshaped.hidden, reshaped.expert_ffn_length, reshaped.experts};
        assert(!po::compatible_stage_model(reshaped, &reason));
        // A per-head norm that contradicts the declared head width is caught.
        reshaped = qwen3;
        dan::test::tensor_named(reshaped, "blk.0.attn_q_norm.weight")->dimensions = {64};
        assert(!po::compatible_stage_model(reshaped, &reason));

        // An OLMoE index must not pass as Qwen3-MoE, or the reverse.
        po::ModelIndex mislabelled = dan::test::olmoe_index(6);
        mislabelled.architecture = "qwen3moe";
        assert(!po::compatible_stage_model(mislabelled, &reason));
        mislabelled = qwen3;
        mislabelled.architecture = "olmoe";
        assert(!po::compatible_stage_model(mislabelled, &reason));
        mislabelled = qwen3;
        mislabelled.architecture = "qwen2";
        assert(!po::compatible_stage_model(mislabelled, &reason));

        // A deep model must not make the planner search forever before it reports that a
        // small stage count cannot work. 94 layers against workers that together cannot hold
        // the model used to run for minutes; the capacity bound answers immediately, and the
        // answers themselves are unchanged.
        {
            po::ModelIndex deep = dan::test::qwen3moe_index(94);
            const auto started = std::chrono::steady_clock::now();
            std::vector<po::ProviderCapability> too_small;
            for (int index = 0; index < 8; ++index) {
                too_small.push_back({"w" + std::to_string(index), "gpu", 14848});
            }
            assert(!po::plan_replica(deep, too_small, 512, 1));
            std::vector<po::ProviderCapability> roomy;
            for (int index = 0; index < 8; ++index) {
                roomy.push_back({"w" + std::to_string(index), "gpu", 81920});
            }
            const auto plan_deep = po::plan_replica(deep, roomy, 512, 1);
            assert(plan_deep && plan_deep->front().begin == 0 && plan_deep->back().end == 94);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            assert(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() < 20);
        }

        // Planning works on the ordinary path: contiguous cuts, no special rules.
        std::vector<po::ProviderCapability> four{
            {"a", "gpu-a", 24576}, {"b", "gpu-b", 24576},
            {"c", "gpu-c", 24576}, {"d", "gpu-d", 24576}};
        const auto plan = po::plan_replica(qwen3, four, 512, 1);
        assert(plan && plan->front().begin == 0 && plan->back().end == 6);
        for (std::size_t index = 1; index < plan->size(); ++index) {
            assert((*plan)[index - 1].end == (*plan)[index].begin);
        }
    }

    // --- KV: llama.cpp pads the context, so the planner must too --------------------
    assert(po::allocated_positions(512, 1) == 512);
    assert(po::allocated_positions(128, 1) == 256);   // padded up to 256
    assert(po::allocated_positions(1, 1) == 256);
    assert(po::allocated_positions(512, 2) == 512);   // 1024 total, 512 per sequence
    assert(po::allocated_positions(300, 2) == 512);   // 600 -> 768 total -> 384 -> 512 each
    assert(po::allocated_positions(0, 1) == 0 && po::allocated_positions(512, 0) == 0);
    {
        const po::ModelIndex olmoe = dan::test::olmoe_index(6);
        // 512 positions x 1 session x 6 layers x 128 head width x 16 kv heads x 4 bytes.
        assert(po::kv_bytes(olmoe, 0, 6, 512, 1) == 512ull * 6 * 128 * 16 * 4);
        // A 128-position request really allocates 256 positions.
        assert(po::kv_bytes(olmoe, 0, 6, 128, 1) == po::kv_bytes(olmoe, 0, 6, 256, 1));
        assert(po::kv_bytes(olmoe, 0, 6, 512, 2) == 2 * po::kv_bytes(olmoe, 0, 6, 512, 1));
        assert(po::kv_bytes(olmoe, 2, 4, 512, 1) * 3 == po::kv_bytes(olmoe, 0, 6, 512, 1));
        assert(po::kv_bytes(olmoe, 0, 6, 0, 1) == 0 && po::kv_bytes(olmoe, 4, 4, 512, 1) == 0);
        // Checked arithmetic: a context that would overflow the byte count is refused.
        assert(po::kv_bytes(olmoe, 0, 6, 0xFFFFFFFFu, 0xFFFFFFFFu) == 0);
    }

    assert(po::compatible_stage_model(model));
    po::ModelIndex incompatible = model;
    incompatible.architecture = "qwen2moe";
    assert(!po::compatible_stage_model(incompatible));
    incompatible = model;
    incompatible.tensors.erase(std::remove_if(incompatible.tensors.begin(), incompatible.tensors.end(),
        [](const po::ModelTensor& tensor) { return tensor.name.starts_with("blk.5."); }),
        incompatible.tensors.end());
    assert(!po::compatible_stage_model(incompatible));

    po::ProviderCapability original{"provider-1", "RTX", 6656};
    po::ProviderCapability parsed;
    assert(po::parse_available(po::available_message(original), parsed));
    assert(parsed.id == original.id && parsed.offered_vram_mib == original.offered_vram_mib);

    // Cached ranges ride along as a planning hint.
    original.cached.push_back({std::string(64, 'c'), 0, 12});
    original.cached.push_back({std::string(64, 'c'), 12, 24});
    assert(po::parse_available(po::available_message(original), parsed));
    assert(parsed.cached == original.cached);
    assert(!po::parse_available("id=a\ngpu=G\nvram_mib=1\ncached=zz:0-1", parsed));
    assert(!po::parse_available("id=a\ngpu=G\nvram_mib=1\ncached="
        + std::string(64, 'c') + ":5-5", parsed));

    // Measured speed and the replica-owner flag ride along too; unknown keys are skipped.
    original.speed_us_per_gib = 1234;
    original.replica_owner = true;
    assert(po::parse_available(po::available_message(original), parsed));
    assert(parsed.speed_us_per_gib == 1234 && parsed.replica_owner && !parsed.f16_activations);
    original.f16_activations = true;
    assert(po::parse_available(po::available_message(original), parsed) && parsed.f16_activations
        && !parsed.fp8_activations);
    original.fp8_activations = true;
    assert(po::parse_available(po::available_message(original), parsed) && parsed.f16_activations
        && parsed.fp8_activations);
    assert(po::parse_available("id=a\ngpu=G\nvram_mib=1\nsomething_new=5", parsed));
    assert(parsed.speed_us_per_gib == 0 && !parsed.replica_owner);

    // Guesses that still fit: the batch [current, guesses...] decoded at `position` must end
    // inside the context, so there is always room for the plain step and no more than that.
    assert(po::speculation_room(4096, 0) == 4095);
    assert(po::speculation_room(4096, 4092) == 3);
    assert(po::speculation_room(4096, 4094) == 1);
    assert(po::speculation_room(4096, 4095) == 0);  // the last slot: plain step only
    assert(po::speculation_room(4096, 4096) == 0 && po::speculation_room(4096, 9000) == 0);
    assert(po::speculation_room(0, 0) == 0);
    assert(po::speculation_room(4096, 0xFFFFFFFFu - 1) == 0);
    for (std::uint32_t position = 4080; position < 4096; ++position) {  // batch always fits
        assert(position + 1 + po::speculation_room(4096, position) <= 4096);
    }

    // Speculative decoding: commit the first sample, then one per correct guess.
    assert((po::accept_speculation({7, 8, 9}, {7, 8, 9, 10}) == std::vector<std::uint32_t>{7, 8, 9, 10}));
    assert((po::accept_speculation({7, 99, 9}, {7, 8, 9, 10}) == std::vector<std::uint32_t>{7, 8}));
    assert((po::accept_speculation({99}, {7, 8}) == std::vector<std::uint32_t>{7}));
    assert((po::accept_speculation({}, {7}) == std::vector<std::uint32_t>{7}));
    assert(po::accept_speculation({7}, {}).empty());
    // A batch that came back short cannot commit more than it verified.
    assert((po::accept_speculation({7, 8}, {7, 8}) == std::vector<std::uint32_t>{7, 8}));

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
