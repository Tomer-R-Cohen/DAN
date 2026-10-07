// These are executable correctness checks, including in a Release build.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "provider_owned/formation.hpp"
#include "provider_owned/planner.hpp"

#include "model_fixtures.hpp"
#include "provider_owned/speculation.hpp"

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
    // Configurable verify widths: any rejected guess must stop the committed path.
    for (std::size_t width = 1; width <= 32; ++width) {
        std::vector<std::uint32_t> samples(width), guesses;
        for (std::size_t i = 0; i < width; ++i) samples[i] = static_cast<std::uint32_t>(100 + i);
        guesses.assign(samples.begin(), samples.end() - 1);
        assert(po::accept_speculation(guesses, samples) == samples);
        for (std::size_t reject = 0; reject < guesses.size(); ++reject) {
            auto wrong = guesses;
            wrong[reject] = 999;
            assert(po::accept_speculation(wrong, samples).size() == reject + 1);
        }
    }

    // Synthetic round costs exercise the controller; these are not GPU/WAN benchmarks.
    const auto block = [](po::AdaptiveDraftWidth& policy, std::uint32_t accepted,
                           std::uint64_t ns) {
        const auto width = policy.width();
        for (int round = 0; round < 4; ++round) policy.observe(width, accepted, ns);
    };
    for (std::uint32_t cap = 1; cap <= 32; ++cap) {
        po::AdaptiveDraftWidth policy(cap);
        assert(policy.width() == 1);
        for (int round = 0; round < 24 && policy.width() != cap; ++round) {
            const auto width = policy.width();
            // A large fixed transit cost, cheap drafting, perfect acceptance.
            policy.observe(width, width, 1000 + 10 * width);
            assert(policy.width() >= 1 && policy.width() <= cap);
        }
        // 4 plain + at most 5 candidate blocks: the search reaches any cap.
        assert(policy.width() == cap);
        block(policy, cap, 1000 + 10 * cap);
        assert(policy.width() == cap);
    }
    {
        po::AdaptiveDraftWidth policy(32), other_session(32);
        // Invalid, clipped and incomplete observations cannot advance the search.
        for (int round = 0; round < 10; ++round) {
            policy.observe(2, 1, 1000);
            policy.observe(1, 0, 1000);
            policy.observe(1, 2, 1000);
            policy.observe(1, 1, 0);
        }
        assert(policy.width() == 1);
        block(policy, 1, 1000);
        assert(policy.width() == 2 && other_session.width() == 1);
        // All guesses rejected: extra work wins nothing, fall back to plain decode.
        block(policy, 1, 1200);
        assert(policy.width() == 1);
        for (int round = 0; round < 16; ++round) policy.observe(1, 1, 1000);
        block(policy, 1, 1000);
        assert(policy.width() == 2);  // periodic probes recover from a poor draft period
        block(policy, 2, 1200);
        assert(policy.width() == 4);
        // Acceptance is perfect, but verification is costly: do not increase width.
        block(policy, 4, 3000);
        assert(policy.width() == 2);
        // A workload change makes even the held width slower than the plain baseline.
        block(policy, 1, 1600);
        assert(policy.width() == 1);
        policy = po::AdaptiveDraftWidth(32); // session reset discards learned state
        assert(policy.width() == 1 && other_session.width() == 1);
    }
    {
        po::AdaptiveDraftWidth policy(4);
        block(policy, 1, 1000);
        block(policy, 2, 1910); // less than 5% improvement: avoid noisy width changes
        assert(policy.width() == 1);
    }

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
