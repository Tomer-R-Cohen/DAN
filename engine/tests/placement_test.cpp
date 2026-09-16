// Dynamic placement against fake serve-mode workers on loopback (no model needed).

#include "provider_owned/placement.hpp"
#include "provider_owned/lease.hpp"

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
    request.manifest.sha256 = model_sha;
    request.manifest.hidden = 1024;
    po::ModelIndex& model = request.model;
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

int run() {
    const po::PlacementRequest request = make_request();
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
        po::PlacedRoute placed = po::place_route({{big.endpoint(), {}}, {a.endpoint(), {}},
            {b.endpoint(), {}}, {c.endpoint(), {}}}, request);
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
