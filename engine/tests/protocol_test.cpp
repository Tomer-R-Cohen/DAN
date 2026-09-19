#include "provider_owned/protocol.hpp"
#include "provider_owned/activations.hpp"
#include "provider_owned/fair_queue.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace po = dan::provider_owned;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

// fp8 (e4m3) activations: every code decodes and re-encodes to itself, rounding picks the
// nearest code, a row keeps its values within e4m3's precision, and bad values stay contained.
int check_fp8() {
    std::vector<float> finite;
    for (int code = 0; code < 256; ++code) {
        const float value = po::fp8_e4m3_to_float(static_cast<std::uint8_t>(code));
        if ((code & 0x7F) == 0x7F) { CHECK(std::isnan(value)); continue; }
        CHECK(po::float_to_fp8_e4m3(value) == code || (value == 0 && (code & 0x7F) == 0));
        finite.push_back(value);
    }
    CHECK(po::fp8_e4m3_to_float(0x7E) == 448.0f && po::fp8_e4m3_to_float(0x01) == std::ldexp(1.0f, -9));
    CHECK(po::float_to_fp8_e4m3(1e9f) == 0x7E && po::float_to_fp8_e4m3(-1e9f) == 0xFE);
    CHECK(po::float_to_fp8_e4m3(std::numeric_limits<float>::quiet_NaN()) == 0);
    std::mt19937 random(7);
    std::uniform_real_distribution<float> range(-500.0f, 500.0f);
    for (int sample = 0; sample < 20000; ++sample) {
        const float value = range(random) * std::ldexp(1.0f, static_cast<int>(random() % 20) - 12);
        const float chosen = po::fp8_e4m3_to_float(po::float_to_fp8_e4m3(value));
        float best = std::numeric_limits<float>::infinity();
        for (const float candidate : finite) best = std::min(best, std::fabs(candidate - value));
        CHECK(std::fabs(chosen - value) <= best * 1.0001f + 1e-12f);  // nearest code
    }
    // A hidden-state-like row: small values with a few huge channels.
    std::vector<float> row(896), back(896);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : row) value = normal(random);
    row[7] = 3000.0f;
    row[100] = -1200.0f;
    std::vector<std::uint8_t> packed(po::activation_bytes(po::DType::fp8e4m3, 1, row.size()));
    CHECK(packed.size() == 4 + row.size());
    po::pack_fp8_row(row.data(), row.size(), packed.data());
    po::unpack_fp8_row(packed.data(), row.size(), back.data());
    const float scale = 3000.0f / 448.0f;
    for (std::size_t index = 0; index < row.size(); ++index) {
        // e4m3 keeps 3 mantissa bits (relative step 1/8, so rounding error <= 1/16), and
        // values below its smallest normal share an absolute step of 2^-9 * scale.
        CHECK(std::fabs(back[index] - row[index])
            <= std::fabs(row[index]) / 16 + std::ldexp(1.0f, -10) * scale + 1e-6f);
    }
    CHECK(back[7] == 3000.0f);
    // Zeros stay zeros; one infinity saturates alone.
    std::vector<float> zeros(16, 0.0f);
    po::pack_fp8_row(zeros.data(), zeros.size(), packed.data());
    po::unpack_fp8_row(packed.data(), zeros.size(), back.data());
    for (std::size_t index = 0; index < zeros.size(); ++index) CHECK(back[index] == 0.0f);
    std::vector<float> spike{1.0f, -2.0f, std::numeric_limits<float>::infinity(), 0.5f};
    po::pack_fp8_row(spike.data(), spike.size(), packed.data());
    po::unpack_fp8_row(packed.data(), spike.size(), back.data());
    CHECK(back[0] == 1.0f && back[1] == -2.0f && back[3] == 0.5f && std::isfinite(back[2]));
    return 0;
}

int main() {
    if (const int failure = check_fp8()) return failure;
    po::Frame sent;
    sent.type = po::Type::activation;
    sent.session = 11;
    sent.request = 22;
    sent.position = 33;
    sent.rows = 5;
    sent.cols = 896;
    sent.dtype = po::DType::f32le;
    sent.payload.resize(8 + 5 * 896 * 4);

    std::array<std::uint8_t, po::header_size> header{};
    std::string error;
    CHECK(po::encode_header(sent, header, error));
    po::Frame received;
    std::uint64_t payload_size = 0;
    CHECK(po::decode_header(header, received, payload_size, error));
    CHECK(received.type == sent.type);
    CHECK(received.session == sent.session);
    CHECK(received.request == sent.request);
    CHECK(received.position == sent.position);
    CHECK(received.rows == sent.rows && received.cols == sent.cols);
    CHECK(received.dtype == sent.dtype);
    CHECK(payload_size == sent.payload.size());

    sent.type = po::Type::speculative_activation;
    CHECK(po::encode_header(sent, header, error));
    CHECK(po::decode_header(header, received, payload_size, error));
    CHECK(received.type == po::Type::speculative_activation);

    sent.type = po::Type::rollback;
    sent.rows = 0;
    sent.cols = 0;
    sent.dtype = po::DType::none;
    sent.payload.clear();
    CHECK(po::encode_header(sent, header, error));
    CHECK(po::decode_header(header, received, payload_size, error));
    CHECK(received.type == po::Type::rollback);

    sent.type = po::Type::client_chunk;
    CHECK(po::encode_header(sent, header, error));
    CHECK(po::decode_header(header, received, payload_size, error));
    CHECK(received.type == po::Type::client_chunk);

    po::put64(header.data() + 40, po::max_payload + 1);
    CHECK(!po::decode_header(header, received, payload_size, error));
    po::put64(header.data() + 40, 0);
    po::put16(header.data() + 6, 99);
    CHECK(!po::decode_header(header, received, payload_size, error));
    po::put16(header.data() + 6, static_cast<std::uint16_t>(po::Type::ack));
    po::put16(header.data() + 38, 1);
    CHECK(!po::decode_header(header, received, payload_size, error));

    po::Frame control;
    control.type = po::Type::create_session;
    control.session = 1;
    CHECK(po::empty_control(control));
    control.payload.push_back(0);
    CHECK(!po::empty_control(control));

    std::string peer_id;
    CHECK(po::parse_peer_id(
        "DAN-P2P/1 12D3KooWEyRoFjjXiJoUBQz9VARtjiUJHwfPtetgVfcG3DXDWLk2", peer_id));
    CHECK(peer_id == "12D3KooWEyRoFjjXiJoUBQz9VARtjiUJHwfPtetgVfcG3DXDWLk2");
    CHECK(!po::parse_peer_id("DAN-P2P/1 not-a-peer", peer_id));
    CHECK(!po::parse_peer_id(
        "DAN-P2P/1 12D3KooWEyRoFjjXiJoUBQz9VARtjiUJHwfPtetgVfcG3DXDWLk0", peer_id));

    po::FairQueue<int> queue(3);
    CHECK(queue.push(1, 10));
    CHECK(queue.push(1, 11));
    CHECK(queue.push(2, 20));
    CHECK(!queue.push(3, 30));
    CHECK(queue.pop() == 10);
    CHECK(queue.pop() == 20);
    CHECK(queue.remove_if([](int value) { return value == 11; }) == 11);
    CHECK(queue.size() == 0);
    CHECK(queue.push(4, 40));
    CHECK(queue.remove_if([](int value) { return value == 40; }) == 40);
    CHECK(queue.push(4, 40));
    CHECK(queue.drain().size() == 1);
    CHECK(queue.push(4, 41));
    CHECK(queue.pop() == 41);
    CHECK(queue.push(5, 50));
    CHECK(queue.pop() == 50);
    CHECK(!queue.pop_for(std::chrono::milliseconds(1)));
    CHECK(queue.push(5, 51));
    CHECK(queue.pop_for(std::chrono::milliseconds(1)) == 51);
    CHECK(queue.push(6, 60));
    CHECK(queue.close().size() == 1);
    CHECK(!queue.push(4, 41));
    CHECK(!queue.pop().has_value());
}
