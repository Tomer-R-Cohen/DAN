// Drives InferenceClient against two fake stages on loopback (no model needed).
// Stage A turns each prompt byte into one activation row; stage B returns one
// letter per step. Both track per-session positions and request IDs the way
// dan-stage-worker does, so the client's token loop is checked end to end.

#include "provider_owned/client.hpp"

#include <atomic>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>
#include <unordered_map>

namespace po = dan::provider_owned;

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "failed line %d\n", __LINE__); return __LINE__; } } while (false)

namespace {

constexpr std::uint32_t hidden = 4;

struct FakeSession {
    std::uint32_t position = 0;
    std::uint64_t last_request = 0;
};

po::Frame ack(const po::Frame& input, std::uint32_t position, bool timing = false) {
    po::Frame output;
    output.type = po::Type::ack;
    output.session = input.session;
    output.request = input.request;
    output.position = position;
    if (timing) output.payload.resize(8);
    return output;
}

po::Frame activation(const po::Frame& input, po::Type type, std::uint32_t start,
    std::uint32_t rows) {
    po::Frame output;
    output.type = type;
    output.session = input.session;
    output.request = input.request;
    output.position = start;
    output.rows = rows;
    output.cols = hidden;
    output.dtype = po::DType::f32le;
    output.payload.resize(8 + std::size_t(rows) * hidden * 4);
    return output;
}

po::Frame handle(bool first, std::unordered_map<std::uint64_t, FakeSession>& sessions,
    const po::Frame& input) {
    if (input.type == po::Type::create_session) {
        if (input.session == 0 || sessions.contains(input.session)) throw std::runtime_error("bad create");
        sessions[input.session] = {};
        return ack(input, 0);
    }
    const auto found = sessions.find(input.session);
    if (found == sessions.end()) throw std::runtime_error("unknown session ID");
    FakeSession& session = found->second;
    switch (input.type) {
    case po::Type::destroy_session:
        sessions.erase(found);
        return ack(input, 0);
    case po::Type::end_request:
        session.last_request = input.request;
        return ack(input, session.position);
    case po::Type::prompt:
        if (input.request <= session.last_request) throw std::runtime_error("request ID is not increasing");
        [[fallthrough]];
    case po::Type::token:
    case po::Type::commit_token: {
        if (!first || input.position != session.position) throw std::runtime_error("bad first-stage frame");
        const std::uint32_t rows = input.type == po::Type::prompt
            ? static_cast<std::uint32_t>(input.payload.size()) : 1;
        const std::uint32_t start = session.position;
        session.position += rows;
        return activation(input, input.type == po::Type::commit_token
            ? po::Type::commit_activation : po::Type::activation, start, rows);
    }
    case po::Type::activation:
    case po::Type::commit_activation: {
        if (first || input.position != session.position) throw std::runtime_error("bad last-stage frame");
        session.position += input.rows;
        if (input.type == po::Type::commit_activation) return ack(input, session.position, true);
        po::Frame output;
        output.type = po::Type::result;
        output.session = input.session;
        output.request = input.request;
        output.position = session.position;
        output.payload.resize(14);
        po::put32(output.payload.data(), session.position);
        output.payload[12] = 0;
        output.payload[13] = static_cast<std::uint8_t>('a' + session.position % 26);
        return output;
    }
    default:
        throw std::runtime_error("unexpected frame");
    }
}

class FakeStage {
public:
    explicit FakeStage(bool first) {
        listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t length = sizeof(address);
        if (bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || listen(listener_, 1) != 0
            || getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
            throw std::runtime_error("fake stage listen failed");
        }
        endpoint_ = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        thread_ = std::thread([this, first] {
            const po::socket_t client = accept(listener_, nullptr, nullptr);
            std::unordered_map<std::uint64_t, FakeSession> sessions;
            std::string error;
            po::Frame input;
            while (po::recv_frame(client, input, error)) {
                po::Frame output;
                try { output = handle(first, sessions, input); }
                catch (const std::exception& failure) { output = po::error_frame(input, failure.what()); }
                if (!po::send_frame(client, output, error)) break;
            }
            po::close_socket(client);
        });
    }

    ~FakeStage() {
        po::close_socket(listener_);
        thread_.join();
    }

    const std::string& endpoint() const { return endpoint_; }

private:
    po::socket_t listener_ = po::invalid_socket;
    std::string endpoint_;
    std::thread thread_;
};

int run() {
    FakeStage a(true), b(false);
    {
        po::InferenceClient client({{a.endpoint(), b.endpoint()}, hidden});

        // One-off request: prompt of 3 bytes, then 2 decode steps.
        po::RequestResult once = client.generate_once("abc", 3);
        CHECK(once.metrics.token_ids.size() == 3);
        CHECK(once.output == "def");
        CHECK(once.position == 5);

        // Persistent session: the second prompt continues after the committed token.
        const std::uint64_t session = client.create_session();
        CHECK(session != 0);
        po::RequestResult first = client.generate(session, "xy", 2);
        CHECK(first.output == "cd");
        CHECK(first.position == 4);
        po::RequestResult second = client.generate(session, "z", 1);
        CHECK(second.output == "f");
        CHECK(second.position == 6);

        bool rejected = false;
        try { client.generate(session + 1, "q", 1); } catch (const std::exception&) { rejected = true; }
        CHECK(rejected);

        client.destroy_session(session);
        CHECK(client.create_session() != session);
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
    catch (const std::exception& error) { std::fprintf(stderr, "client_test: %s\n", error.what()); }
#ifdef _WIN32
    WSACleanup();
#endif
    return result;
}
