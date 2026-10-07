// Dynamic placement against fake serve-mode workers on loopback (no model needed).

#include "provider_owned/placement.hpp"
#include "provider_owned/lease.hpp"

#include "model_fixtures.hpp"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace po = dan::provider_owned;

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "failed line %d\n", __LINE__); return __LINE__; } } while (false)

namespace {

constexpr std::uint64_t gib = 1024ull * 1024 * 1024;
const std::string model_sha = std::string(64, 'e');
const std::string peer = "12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C";

class FakeWorker {
public:
    // refuse: reject every reservation with this reason; after a "busy" refusal the worker
    // greets later connections as reserved, like a worker another client just took.
    FakeWorker(std::string id, std::uint64_t offered_mib, std::string refuse = {},
        std::string abi = DAN_RUNTIME_ABI, std::string ring = {})
        : refuse_(std::move(refuse)) {
        listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t length = sizeof(address);
        if (bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || listen(listener_, 8) != 0
            || getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
            throw std::runtime_error("fake worker listen failed");
        }
        const int port = ntohs(address.sin_port);
        endpoint_ = "127.0.0.1:" + std::to_string(port);
        hello_.id = std::move(id);
        hello_.gpu = "fake";
        hello_.offered_vram_mib = offered_mib;
        hello_.ring_endpoint = ring.empty() ? "127.0.0.1:" + std::to_string(port + 1) : ring;
        hello_.state = "available";
        hello_.runtime_abi = std::move(abi);
        hello_.max_context = 4096;
        hello_.max_sessions = 8;
        hello_.models = {model_sha};
        acceptor_ = std::thread([this] {
            for (;;) {
                const po::socket_t client = accept(listener_, nullptr, nullptr);
                if (client == po::invalid_socket) return;
                std::lock_guard lock(mutex_);
                connections_.emplace_back([this, client] { serve(client); });
            }
        });
    }

    ~FakeWorker() {
#ifdef _WIN32
        ::shutdown(listener_, SD_BOTH);
#else
        ::shutdown(listener_, SHUT_RDWR);
#endif
        po::close_socket(listener_);
        acceptor_.join();
        for (std::thread& connection : connections_) connection.join();
    }

    const std::string& endpoint() const { return endpoint_; }
    const std::string& id() const { return hello_.id; }
    // Greets as a replica owner whose GPU runs one GiB of weights per token in `speed` µs.
    void owner(std::uint64_t speed) {
        std::lock_guard lock(mutex_);
        hello_.replica_owner = true;
        hello_.speed_us_per_gib = speed;
    }
    int refusals() {
        std::lock_guard lock(mutex_);
        return refusals_;
    }
    std::vector<std::string> events() {
        std::lock_guard lock(mutex_);
        return events_;
    }

private:
    void record(std::string event) {
        std::lock_guard lock(mutex_);
        events_.push_back(std::move(event));
    }

    void serve(po::socket_t client) {
        std::string error;
        po::Frame hello;
        hello.type = po::Type::provider_available;
        std::string text;
        {
            std::lock_guard lock(mutex_);
            text = po::available_message(hello_);
        }
        hello.payload.assign(text.begin(), text.end());
        po::send_frame(client, hello, error);
        po::Frame input;
        while (po::recv_frame(client, input, error)) {
            const std::string payload(input.payload.begin(), input.payload.end());
            po::StageRequest request;
            po::Frame output;
            output.type = po::Type::ack;
            if (input.type == po::Type::reserve) {
                if (!po::parse_stage_request(payload, request) || request.lease_ms == 0) {
                    output = po::error_frame(input, "invalid_reservation");
                } else if (!refuse_.empty()) {
                    output = po::error_frame(input, refuse_);
                    std::lock_guard lock(mutex_);
                    ++refusals_;
                    if (refuse_ == "busy") hello_.state = "reserved";
                } else {
                    record("reserve " + request.route_id);
                }
            } else if (input.type == po::Type::release_route) {
                record("release " + payload);
            } else if (input.type == po::Type::assign_stage && po::parse_stage_request(payload, request)
                && request.model_sha256 == model_sha) {
                record("assign " + request.route_id + " " + std::to_string(request.begin)
                    + " " + std::to_string(request.end));
                output.type = po::Type::stage_ready;
                output.payload.assign(request.route_id.begin(), request.route_id.end());
            } else {
                output = po::error_frame(input, "unexpected");
            }
            if (!po::send_frame(client, output, error)) break;
        }
        po::close_socket(client);
    }

    std::string refuse_;
    int refusals_ = 0;
    po::socket_t listener_ = po::invalid_socket;
    std::string endpoint_;
    po::ProviderCapability hello_;
    std::thread acceptor_;
    std::mutex mutex_;
    std::vector<std::thread> connections_;
    std::vector<std::string> events_;
};

po::PlacementRequest make_request() {
    po::PlacementRequest request;
    request.models.emplace_back();
    request.models.front().manifest.sha256 = model_sha;
    request.models.front().manifest.hidden = 1024;
    request.models.front().manifest.model_id = "test-model";
    po::ModelIndex& model = request.models.front().model;
    model.architecture = "qwen2";
    model.layers = 6; model.hidden = 1024; model.heads = 16; model.kv_heads = 4;
    model.header_bytes = 4096;
    model.tensors.push_back({"token_embd.weight", gib / 2});
    model.tensors.push_back({"output_norm.weight", 4096});
    for (int layer = 0; layer < 6; ++layer) {
        model.tensors.push_back({"blk." + std::to_string(layer) + ".weight", gib});
    }
    model.tensors.push_back({"output.weight", gib / 2});
    request.context = 128;
    request.sessions = 1;
    request.minimum_stages = 3;
    request.runtime_abi = DAN_RUNTIME_ABI;
    request.lease_ms = 5000;
    return request;
}

std::size_t count_prefix(const std::vector<std::string>& events, const std::string& prefix) {
    return static_cast<std::size_t>(std::count_if(events.begin(), events.end(),
        [&](const std::string& event) { return event.starts_with(prefix); }));
}

int check_cached_planning() {
    const po::PlacementRequest request = make_request();
    const std::vector<std::uint64_t> offered{4096, 4096, 4096};
    // Each worker holds two layers: that exact tiling is the plan.
    const std::vector<std::vector<std::pair<int, int>>> cached{{{0, 2}}, {{2, 4}}, {{4, 6}}};
    auto plan = po::plan_from_cache(request.models.front().model, offered, cached, request.context,
        request.sessions, 2, 3);
    CHECK(plan && plan->size() == 3);
    for (std::size_t index = 0; index < plan->size(); ++index) {
        CHECK((*plan)[index].provider == index);
        CHECK((*plan)[index].begin == static_cast<int>(index) * 2);
    }
    // A bigger worker holding a longer range wins: fewer stages, fewer hops per token.
    const auto fewer = po::plan_from_cache(request.models.front().model, {12288, 4096, 4096},
        {{{0, 4}, {0, 2}}, {{4, 6}}, {{2, 4}}}, request.context, request.sessions, 1, 3);
    CHECK(fewer && fewer->size() == 2 && (*fewer)[0].end == 4 && (*fewer)[0].provider == 0);
    // A gap in the cached coverage means no cached plan at all.
    CHECK(!po::plan_from_cache(request.models.front().model, offered, {{{0, 2}}, {{2, 4}}, {{5, 6}}},
        request.context, request.sessions, 2, 3));
    // A cached range that does not fit its worker's memory is not used.
    CHECK(!po::plan_from_cache(request.models.front().model, {256, 256, 256}, cached, request.context,
        request.sessions, 2, 3));
    // More stages than the limit allows is refused.
    CHECK(!po::plan_from_cache(request.models.front().model, offered, cached, request.context,
        request.sessions, 2, 2));
    return 0;
}

// Several models, biggest first: the first one the workers can actually run is placed.
int check_model_choice() {
    po::PlacementRequest request = make_request();
    request.minimum_stages = 1;
    // A model no worker lists in its catalog, preferred over the usable one.
    po::ModelOption bigger = request.models.front();
    bigger.manifest.sha256 = std::string(64, 'a');
    bigger.manifest.model_id = "too-big";
    request.models.insert(request.models.begin(), bigger);
    FakeWorker a("a", 4096), b("b", 4096), c("c", 4096);
    po::PlacedRoute placed = po::place_route(
        {{a.endpoint(), {}}, {b.endpoint(), {}}, {c.endpoint(), {}}}, request);
    CHECK(placed.manifest.sha256 == model_sha && placed.manifest.model_id == "test-model");
    CHECK(placed.stages.front().begin == 0 && placed.stages.back().end == 6);
    return 0;
}

// Equal workers, unequal links: the closest, directly reachable one runs the first stage.
int check_link_preference() {
    po::PlacementRequest request = make_request();
    FakeWorker slow("slow", 4096), fast("fast", 4096), middle("middle", 4096);
    po::PlacedRoute placed = po::place_route({
        {slow.endpoint(), {}, 300, true},
        {fast.endpoint(), {}, 5, false},
        {middle.endpoint(), {}, 120, false}}, request);
    CHECK(placed.stages.size() == 3);
    CHECK(placed.stages[0].worker_id == "fast" && placed.stages[0].rtt_ms == 5);
    CHECK(placed.stages[1].worker_id == "middle");
    CHECK(placed.stages[2].worker_id == "slow" && placed.stages[2].relayed);
    return 0;
}

// Replica formation: the owner's worker must lead, in both planners.
int check_required_head() {
    const po::PlacementRequest request = make_request();
    const po::ModelIndex& model = request.models.front().model;
    const auto plan = po::plan_stages(model, {4096, 8192, 4096}, request.context,
        request.sessions, 3, std::size_t{2});
    CHECK(plan && plan->size() == 3 && (*plan)[0].provider == 2 && (*plan)[0].begin == 0);
    // Without the constraint the same candidates plan with candidate 0 first.
    const auto free = po::plan_stages(model, {4096, 8192, 4096}, request.context, request.sessions, 3);
    CHECK(free && (*free)[0].provider == 0);
    const std::vector<std::vector<std::pair<int, int>>> cached{{{0, 2}}, {{2, 4}}, {{4, 6}}};
    CHECK(po::plan_from_cache(model, {4096, 4096, 4096}, cached, request.context,
        request.sessions, 2, 3, std::size_t{0}));
    // The head holds layers 2-4 only: no cached plan can start with it.
    CHECK(!po::plan_from_cache(model, {4096, 4096, 4096}, cached, request.context,
        request.sessions, 2, 3, std::size_t{1}));
    CHECK(!po::plan_stages(model, {4096}, request.context, request.sessions, 1, std::size_t{3}));

    FakeWorker a("a", 4096), b("b", 4096), c("c", 4096);
    po::PlacementRequest placed_request = request;
    placed_request.head = c.endpoint();
    po::PlacedRoute placed = po::place_route(
        {{a.endpoint(), {}}, {b.endpoint(), {}}, {c.endpoint(), {}}}, placed_request);
    CHECK(placed.stages.size() == 3 && placed.stages[0].worker_id == "c");
    return 0;
}

// Replica formation: a planned route whose link is too slow is replanned without that
// worker, before anyone is reserved.
int check_route_rejection() {
    po::PlacementRequest request = make_request();
    FakeWorker a("a", 4096), b("b", 4096), c("c", 4096), d("d", 4096);
    int checks = 0;
    std::vector<std::string> seen;
    request.check_route = [&](const std::vector<po::PlacementCandidate>& route) {
        ++checks;
        for (const po::PlacementCandidate& stage : route) {
            if (stage.control == b.endpoint()) return stage.control;
        }
        return std::string{};
    };
    po::PlacementTimings timings;
    request.report = &timings;
    po::PlacedRoute placed = po::place_route({{a.endpoint(), {}}, {b.endpoint(), {}},
        {c.endpoint(), {}}, {d.endpoint(), {}}}, request);
    CHECK(checks == 2 && timings.rejected_links == 1);
    CHECK(std::none_of(placed.stages.begin(), placed.stages.end(),
        [](const po::PlacedStage& stage) { return stage.worker_id == "b"; }));
    CHECK(b.events().empty());  // never reserved
    return 0;
}

// Speed-aware formation: a slow, far head stands aside when another owner can lead a clearly
// faster replica without it; it goes ahead when the other one is not an owner, or not faster.

// Admission for a MoE stage: every expert of every owned layer is charged, the KV uses the
// padded position count, and a draft is still measured on top.
int check_olmoe_admission() {
    const po::ModelIndex olmoe = dan::test::olmoe_index(6);
    po::StageAssignment assignment;
    // Whole model: weights are the sum of every owned tensor, never a routed fraction.
    CHECK(po::stage_fits(olmoe, 8192, 0, 6, 512, 1, assignment));
    CHECK(assignment.model_bytes == po::stage_model_bytes(olmoe, 0, 6));
    CHECK(assignment.kv_bytes == po::kv_bytes(olmoe, 0, 6, 512, 1));
    const std::uint64_t routed = assignment.model_bytes * olmoe.experts_used / olmoe.experts;
    CHECK(assignment.model_bytes > routed * 4);  // nowhere near the active-parameter price
    // A worker that could hold only the routed fraction must not be given the stage.
    const std::uint64_t routed_mib = routed / (1024 * 1024) + 1024 + 256;
    CHECK(!po::stage_fits(olmoe, routed_mib, 0, 6, 512, 1, assignment));
    // Splitting reduces what each worker must hold.
    po::StageAssignment head, middle, tail;
    CHECK(po::stage_fits(olmoe, 4096, 0, 2, 512, 1, head));
    CHECK(po::stage_fits(olmoe, 4096, 2, 4, 512, 1, middle));
    CHECK(po::stage_fits(olmoe, 4096, 4, 6, 512, 1, tail));
    CHECK(head.model_bytes > middle.model_bytes && tail.model_bytes > middle.model_bytes);
    CHECK(head.kv_bytes == middle.kv_bytes && middle.kv_bytes == tail.kv_bytes);
    // Padded KV: asking for 128 positions costs what 256 cost.
    po::StageAssignment padded;
    CHECK(po::stage_fits(olmoe, 8192, 0, 6, 128, 1, padded));
    CHECK(padded.kv_bytes == po::kv_bytes(olmoe, 0, 6, 256, 1));
    // Two sessions cost twice the KV, and can push a tight worker over the edge.
    po::StageAssignment pair;
    const bool one_session = po::stage_fits(olmoe, 3000, 0, 2, 512, 1, pair);
    const std::uint64_t single_kv = pair.kv_bytes;
    const bool two_sessions = po::stage_fits(olmoe, 3000, 0, 2, 512, 2, pair);
    CHECK(one_session && pair.kv_bytes == 2 * single_kv);
    CHECK(!two_sessions || pair.model_bytes + pair.kv_bytes
        <= 3000ull * 1024 * 1024 - 1024ull * 1024 * 1024);
    // A draft on top of an OLMoE head is measured the same way (placement does not pick one
    // for OLMoE today): a small draft fits beside the head, a large one is refused.
    CHECK(po::stage_with_draft_fits(olmoe, dan::test::qwen2_index(6), 3000, 2, 512, 1));
    CHECK(!po::stage_with_draft_fits(olmoe, dan::test::qwen2_index(400), 3000, 2, 512, 1));
    return 0;
}

int check_draft_memory() {
    po::PlacementRequest request = make_request();
    const po::ModelIndex& model = request.models.front().model;
    po::ModelIndex draft = model;  // a draft as big as the model never fits beside it
    po::ModelIndex small;
    small.architecture = "qwen2";
    small.layers = 1; small.hidden = 1024; small.heads = 16; small.kv_heads = 4;
    small.header_bytes = 4096;
    small.tensors.push_back({"token_embd.weight", gib / 8});
    small.tensors.push_back({"blk.0.weight", gib / 4});
    small.tensors.push_back({"output_norm.weight", 4096});
    small.tensors.push_back({"output.weight", gib / 8});
    po::StageAssignment fit;
    CHECK(po::stage_fits(model, 12288, 0, 6, 128, 1, fit));
    CHECK(po::stage_with_draft_fits(model, small, 12288, 6, 128, 1));
    CHECK(!po::stage_with_draft_fits(model, draft, 12288, 6, 128, 1));
    CHECK(!po::stage_with_draft_fits(model, small, 8192, 6, 128, 1));  // stage alone does not fit
    // Both KV caches count: a context large enough for the stage but not for the draft too.
    po::StageAssignment stage_only;
    CHECK(po::stage_fits(model, 12288, 0, 6, 8192, 1, stage_only));
    const std::uint64_t spare = 12288ull * 1024 * 1024 * 85 / 100
        - stage_only.model_bytes - stage_only.kv_bytes;
    const std::uint64_t draft_need = po::stage_model_bytes(small, 0, 1) + po::kv_bytes(small, 0, 1, 8192, 1);
    CHECK(po::stage_with_draft_fits(model, small, 12288, 6, 8192, 1) == (draft_need <= spare));
    CHECK(!po::stage_with_draft_fits(model, small, 12288, 6, 8192, 64));  // 64 sessions of KV
    po::ModelIndex unknown = small;
    unknown.heads = 0;  // KV size cannot be computed: refused, not treated as free
    CHECK(!po::stage_with_draft_fits(model, unknown, 12288, 6, 128, 1));
    return 0;
}

int check_speed_aware_formation() {
    po::PlacementRequest request = make_request();
    request.minimum_stages = 1;
    const po::ModelIndex& model = request.models.front().model;
    // decode_bytes: the whole stage except the token embedding (a token reads one row of it).
    CHECK(po::decode_bytes(model, 0, 6) == po::stage_model_bytes(model, 0, 6) - gib / 2);
    CHECK(po::decode_bytes(model, 3, 6) == po::stage_model_bytes(model, 3, 6));
    const double one_stage = po::estimate_token_ms(model, {{0, 0, 6}}, {1000}, {});
    CHECK(one_stage > 6.49 && one_stage < 6.51);
    const double two_stage = po::estimate_token_ms(model, {{0, 0, 3}, {1, 3, 6}}, {1000, 1000}, {60, 40});
    CHECK(two_stage > one_stage + 49.9 && two_stage < one_stage + 50.1);  // half of each round trip

    // Self (slow GPU, 8 GiB) must lead; the peer (fast, 16 GiB, 70 ms away) fits the model alone.
    FakeWorker self("self", 8192), fast("fast", 16384);
    fast.owner(1000);
    request.head = self.endpoint();
    request.yield_margin = 0.25;
    bool yielded = false;
    try {
        po::place_route({{self.endpoint(), {}, 0}, {fast.endpoint(), {}, 70}}, request);
    } catch (const po::FasterReplicaElsewhere&) { yielded = true; }
    CHECK(yielded && self.events().empty() && fast.events().empty());  // nobody reserved
    // Without the margin rule, or when the fast GPU runs no owner, self forms as before.
    request.yield_margin = 0;
    CHECK(po::place_route({{self.endpoint(), {}, 0}, {fast.endpoint(), {}, 70}}, request).stages.size() >= 1);
    return 0;
}

// Replica formation without a fixed context: the largest context that needs no extra stage.
int check_largest_context() {
    po::PlacementRequest request = make_request();
    request.context = 0;
    request.largest_context = true;
    request.minimum_stages = 1;
    request.models.front().manifest.context = 512;
    request.models.front().model.trained_context = 32768;
    request.models.front().model.kv_heads = 16;  // 24 KiB of KV per position
    // 8504 MiB leaves about 60 MiB beside the weights on one worker: 2048 positions (48 MiB)
    // fit, 4096 (96 MiB) would need a second stage, and 8192 exceeds the workers' limit.
    // Each route is dropped before its fake workers, which wait for its connections to close.
    {
        FakeWorker a("a", 8504), b("b", 8504);
        const po::PlacedRoute placed = po::place_route({{a.endpoint(), {}}, {b.endpoint(), {}}}, request);
        CHECK(placed.stages.size() == 1 && placed.context == 2048);
    }
    {
        // A fixed context is used as given.
        request.context = 1024;
        FakeWorker c("c", 8504);
        const po::PlacedRoute placed = po::place_route({{c.endpoint(), {}}}, request);
        CHECK(placed.context == 1024);
    }
    {
        // Without largest_context the manifest's context is used.
        request.context = 0;
        request.largest_context = false;
        FakeWorker d("d", 8504);
        const po::PlacedRoute placed = po::place_route({{d.endpoint(), {}}}, request);
        CHECK(placed.context == 512);
    }
    return 0;
}

int run() {
    if (const int failure = check_largest_context()) return failure;
    const po::PlacementRequest request = make_request();
    if (const int failure = check_cached_planning()) return failure;
    if (const int failure = check_olmoe_admission()) return failure;
    if (const int failure = check_draft_memory()) return failure;
    if (const int failure = check_speed_aware_formation()) return failure;
    if (const int failure = check_required_head()) return failure;
    if (const int failure = check_route_rejection()) return failure;
    if (const int failure = check_link_preference()) return failure;
    if (const int failure = check_model_choice()) return failure;
    {
        // Three equal workers, three stages covering all layers.
        FakeWorker a("a", 4096), b("b", 4096), c("c", 4096);
        po::PlacedRoute placed = po::place_route(
            {{a.endpoint(), {}}, {b.endpoint(), {}}, {c.endpoint(), {}}}, request);
        CHECK(placed.stages.size() == 3 && placed.connections.size() == 3);
        CHECK(placed.stages.front().begin == 0 && placed.stages.back().end == 6);
        for (std::size_t index = 1; index < placed.stages.size(); ++index) {
            CHECK(placed.stages[index - 1].end == placed.stages[index].begin);
        }
        CHECK(placed.route.ring_targets.size() == 3 && placed.route.ring_targets[0].empty());
        CHECK(placed.route.peer_ids.empty() && placed.route.hidden == 1024);
        for (FakeWorker* worker : {&a, &b, &c}) {
            const auto events = worker->events();
            CHECK(events.size() == 2 && events[0] == "reserve " + placed.route_id);
            CHECK(events[1].starts_with("assign " + placed.route_id));
        }
    }
    {
        // The largest worker refuses: the others are released, then the route is replanned
        // without it.
        FakeWorker big("big", 8192, "busy"), a("a", 4096), b("b", 4096), c("c", 4096);
        po::PlacementTimings timings;
        po::PlacementRequest counted = request;
        counted.report = &timings;
        po::PlacedRoute placed = po::place_route({{big.endpoint(), {}}, {a.endpoint(), {}},
            {b.endpoint(), {}}, {c.endpoint(), {}}}, counted);
        CHECK(timings.refusals == 1 && timings.attempts == 2);
        CHECK(placed.stages.size() == 3);
        CHECK(std::none_of(placed.stages.begin(), placed.stages.end(),
            [](const po::PlacedStage& stage) { return stage.worker_id == "big"; }));
        CHECK(big.events().empty());
        std::size_t released = 0;
        for (FakeWorker* worker : {&a, &b, &c}) {
            const auto events = worker->events();
            released += count_prefix(events, "release ");
            CHECK(count_prefix(events, "assign " + placed.route_id) == 1);
        }
        CHECK(released == 2);  // the two reserved alongside "big" in the first plan
        CHECK(big.refusals() == 1);  // its fresh greeting said reserved, so it was not retried
    }
    {
        // A refusal other than busy drops the worker for this request without a retry.
        FakeWorker big("big", 8192, "insufficient_memory"), a("a", 4096), b("b", 4096),
            c("c", 4096);
        po::PlacedRoute placed = po::place_route({{big.endpoint(), {}}, {a.endpoint(), {}},
            {b.endpoint(), {}}, {c.endpoint(), {}}}, request);
        CHECK(placed.stages.size() == 3 && big.refusals() == 1);
    }
    {
        // Wrong ABI is filtered out; two usable workers cannot make three stages.
        FakeWorker a("a", 4096), b("b", 4096), c("c", 4096, {}, "other-abi");
        bool failed = false;
        try {
            po::place_route({{a.endpoint(), {}}, {b.endpoint(), {}}, {c.endpoint(), {}}}, request);
        } catch (const std::exception&) { failed = true; }
        CHECK(failed);
        CHECK(a.events().empty() && b.events().empty() && c.events().empty());
    }
    {
        // libp2p: a worker whose ring address names another peer is not used.
        const std::string ring = "/ip4/127.0.0.1/tcp/1/p2p/" + peer;
        FakeWorker a("a", 16384, {}, DAN_RUNTIME_ABI, ring);
        bool failed = false;
        po::PlacementRequest single = request;
        single.minimum_stages = 1;
        try {
            po::place_route({{a.endpoint(), "12D3KooWS2gYS5PaxaUy1mU5eY3CnBjUVfYuNbuRxqqeSz9v8KCr"}},
                single);
        } catch (const std::exception&) { failed = true; }
        CHECK(failed && a.events().empty());
        po::PlacedRoute placed = po::place_route({{a.endpoint(), peer}}, single);
        CHECK(placed.route.peer_ids.size() == 1 && placed.route.peer_ids[0] == peer);
    }
    return 0;
}

} // namespace

int main() {
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    int result = 1;
    try { result = run(); }
    catch (const std::exception& error) { std::fprintf(stderr, "placement_test: %s\n", error.what()); }
#ifdef _WIN32
    WSACleanup();
#endif
    return result;
}
