#include "provider_owned/protocol.hpp"

#include <array>
#include <string>

namespace po = dan::provider_owned;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
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
}
