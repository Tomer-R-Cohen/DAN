#include "provider_owned/lease.hpp"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

namespace po = dan::provider_owned;
using Lease = po::WorkerLease;

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "failed line %d\n", __LINE__); return __LINE__; } } while (false)

int main() {
    po::StageRequest request{std::string(32, 'a'), std::string(64, 'b'), 8, 17, 512, 1, 30000};
    po::StageRequest parsed;
    CHECK(po::parse_stage_request(po::stage_request_message(request), parsed));
    CHECK(parsed.same_stage(request) && parsed.lease_ms == 30000);
    request.lease_ms = 0;
    CHECK(po::parse_stage_request(po::stage_request_message(request), parsed));
    CHECK(!po::parse_stage_request("route_id=short\nmodel_sha256=" + std::string(64, 'b')
        + "\nbegin=0\nend=1\ncontext=1\nsessions=1", parsed));
    CHECK(!po::parse_stage_request(po::stage_request_message(request) + "\nurl=https://x", parsed));
    CHECK(!po::parse_stage_request(po::stage_request_message(request) + "\nlease_ms=60001", parsed));
    request.begin = 17;
    CHECK(!po::parse_stage_request(po::stage_request_message(request), parsed));
    request = {std::string(32, 'a'), std::string(64, 'b'), 8, 17, 512, 1, 1000};

    const auto start = Lease::Clock::now();
    Lease lease;
    CHECK(lease.state(start) == Lease::State::available);
    po::StageRequest no_lease = request;
    no_lease.lease_ms = 0;
    CHECK(!lease.reserve(no_lease, start));
    CHECK(lease.reserve(request, start));
    po::StageRequest other = request;
    other.route_id = std::string(32, 'c');
    CHECK(!lease.reserve(other, start));                       // first lease wins
    CHECK(lease.state(start) == Lease::State::reserved);
    CHECK(!lease.release(other.route_id));                      // only the holder releases
    CHECK(lease.release(request.route_id));
    CHECK(lease.state(start) == Lease::State::available);

    // An unassigned reservation expires; the expired route cannot start loading.
    CHECK(lease.reserve(request, start));
    const auto later = start + std::chrono::milliseconds(1001);
    CHECK(lease.state(later) == Lease::State::available);
    CHECK(!lease.begin_loading(request, later));
    CHECK(lease.reserve(other, later));
    CHECK(lease.release(other.route_id));

    // Assignment must match the reservation; loading and serving never expire.
    CHECK(lease.reserve(request, start));
    po::StageRequest different = request;
    different.end = 18;
    CHECK(!lease.begin_loading(different, start));
    CHECK(lease.begin_loading(request, start));
    CHECK(lease.state(later + std::chrono::hours(1)) == Lease::State::loading);
    CHECK(!lease.serving(other.route_id));
    CHECK(lease.serving(request.route_id));
    CHECK(lease.state(later + std::chrono::hours(1)) == Lease::State::serving);
    CHECK(!lease.reserve(other, later + std::chrono::hours(1)));
    CHECK(lease.release(request.route_id));
    CHECK(!lease.release(request.route_id));

    // Many clients at once: exactly one reservation is accepted.
    std::atomic<int> accepted{0};
    std::vector<std::thread> clients;
    for (int index = 0; index < 16; ++index) {
        clients.emplace_back([&, index] {
            po::StageRequest attempt = request;
            attempt.route_id = std::string(31, 'd') + "0123456789abcdef"[index];
            if (lease.reserve(attempt, Lease::Clock::now())) ++accepted;
        });
    }
    for (std::thread& client : clients) client.join();
    CHECK(accepted.load() == 1);
    return 0;
}
