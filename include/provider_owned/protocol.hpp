#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace dan::provider_owned {

#ifdef _WIN32
using socket_t = SOCKET;
inline constexpr socket_t invalid_socket = INVALID_SOCKET;
#else
using socket_t = int;
inline constexpr socket_t invalid_socket = -1;
#endif

inline constexpr std::uint32_t magic = 0x44414e31; // DAN1
inline constexpr std::uint16_t version = 2;
inline constexpr std::size_t header_size = 48;
inline constexpr std::uint64_t max_payload = 64ull * 1024 * 1024;

enum class Type : std::uint16_t {
    error = 0,
    create_session = 1,
    reset_session = 2,
    destroy_session = 3,
    prompt = 4,
    token = 5,
    activation = 6,
    result = 7,
    end_request = 8,
    ack = 9,
    metrics = 10,
    shutdown = 11,
    commit_token = 12,
    commit_activation = 13,
    cancel_request = 14,
    client_result = 15,
};

enum class DType : std::uint16_t { none = 0, f32le = 1 };

struct Frame {
    Type type = Type::error;
    std::uint64_t session = 0;
    std::uint64_t request = 0;
    std::uint32_t position = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    DType dtype = DType::none;
    std::vector<std::uint8_t> payload;
};

inline void close_socket(socket_t socket) {
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

inline bool send_all(socket_t socket, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    while (size != 0) {
        const int count = send(socket, bytes, static_cast<int>(std::min<std::size_t>(size, INT_MAX)),
#ifdef _WIN32
            0
#else
            MSG_NOSIGNAL
#endif
        );
        if (count <= 0) return false;
        bytes += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

inline bool recv_all(socket_t socket, void* data, std::size_t size) {
    auto* bytes = static_cast<char*>(data);
    while (size != 0) {
        const int count = recv(socket, bytes,
            static_cast<int>(std::min<std::size_t>(size, INT_MAX)), 0);
        if (count <= 0) return false;
        bytes += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

inline void put16(std::uint8_t* data, std::uint16_t value) {
    data[0] = static_cast<std::uint8_t>(value >> 8);
    data[1] = static_cast<std::uint8_t>(value);
}

inline void put32(std::uint8_t* data, std::uint32_t value) {
    data[0] = static_cast<std::uint8_t>(value >> 24);
    data[1] = static_cast<std::uint8_t>(value >> 16);
    data[2] = static_cast<std::uint8_t>(value >> 8);
    data[3] = static_cast<std::uint8_t>(value);
}

inline void put64(std::uint8_t* data, std::uint64_t value) {
    for (int index = 7; index >= 0; --index) {
        data[index] = static_cast<std::uint8_t>(value);
        value >>= 8;
    }
}

inline std::uint16_t get16(const std::uint8_t* data) {
    return (std::uint16_t(data[0]) << 8) | data[1];
}

inline std::uint32_t get32(const std::uint8_t* data) {
    return (std::uint32_t(data[0]) << 24) | (std::uint32_t(data[1]) << 16)
        | (std::uint32_t(data[2]) << 8) | data[3];
}

inline std::uint64_t get64(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) value = (value << 8) | data[index];
    return value;
}

inline bool encode_header(const Frame& frame, std::array<std::uint8_t, header_size>& header,
    std::string& error) {
    if (frame.payload.size() > max_payload) {
        error = "payload exceeds 64 MiB";
        return false;
    }
    header.fill(0);
    put32(header.data(), magic);
    put16(header.data() + 4, version);
    put16(header.data() + 6, static_cast<std::uint16_t>(frame.type));
    put64(header.data() + 8, frame.session);
    put64(header.data() + 16, frame.request);
    put32(header.data() + 24, frame.position);
    put32(header.data() + 28, frame.rows);
    put32(header.data() + 32, frame.cols);
    put16(header.data() + 36, static_cast<std::uint16_t>(frame.dtype));
    put64(header.data() + 40, frame.payload.size());
    return true;
}

inline bool decode_header(const std::array<std::uint8_t, header_size>& header, Frame& frame,
    std::uint64_t& payload_size, std::string& error) {
    if (get32(header.data()) != magic || get16(header.data() + 4) != version
        || get16(header.data() + 38) != 0) {
        error = "bad frame header";
        return false;
    }
    const auto raw_type = get16(header.data() + 6);
    if (raw_type > static_cast<std::uint16_t>(Type::client_result)) {
        error = "unknown frame type";
        return false;
    }
    payload_size = get64(header.data() + 40);
    if (payload_size > max_payload) {
        error = "payload exceeds 64 MiB";
        return false;
    }
    frame.type = static_cast<Type>(raw_type);
    frame.session = get64(header.data() + 8);
    frame.request = get64(header.data() + 16);
    frame.position = get32(header.data() + 24);
    frame.rows = get32(header.data() + 28);
    frame.cols = get32(header.data() + 32);
    frame.dtype = static_cast<DType>(get16(header.data() + 36));
    return true;
}

inline bool send_frame(socket_t socket, const Frame& frame, std::string& error) {
    std::array<std::uint8_t, header_size> header{};
    if (!encode_header(frame, header, error)) return false;
    if (!send_all(socket, header.data(), header.size())
        || !send_all(socket, frame.payload.data(), frame.payload.size())) {
        error = "peer disconnected while sending frame";
        return false;
    }
    return true;
}

inline bool recv_frame(socket_t socket, Frame& frame, std::string& error) {
    std::array<std::uint8_t, header_size> header{};
    if (!recv_all(socket, header.data(), header.size())) {
        error = "peer disconnected while receiving frame";
        return false;
    }
    std::uint64_t payload_size = 0;
    if (!decode_header(header, frame, payload_size, error)) return false;
    frame.payload.resize(static_cast<std::size_t>(payload_size));
    if (!recv_all(socket, frame.payload.data(), frame.payload.size())) {
        error = "peer disconnected during payload";
        return false;
    }
    return true;
}

inline Frame error_frame(const Frame& source, const std::string& message) {
    Frame frame;
    frame.type = Type::error;
    frame.session = source.session;
    frame.request = source.request;
    frame.payload.assign(message.begin(), message.end());
    return frame;
}

inline bool empty_control(const Frame& frame) {
    return frame.position == 0 && frame.rows == 0 && frame.cols == 0
        && frame.dtype == DType::none && frame.payload.empty();
}

} // namespace dan::provider_owned
