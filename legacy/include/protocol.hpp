#pragma once

#include "platform.hpp"

#ifndef _WIN32
#include <arpa/inet.h>
#endif
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace dan {

constexpr std::uint32_t max_message_size = 16 * 1024 * 1024;

inline bool send_all(platform::Socket socket_fd, const void* data, std::size_t size,
    bool report_errors = true)
{
    const auto* bytes = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const int result = send(socket_fd, bytes + sent,
            static_cast<int>(size - sent),
#ifdef _WIN32
            0
#else
            MSG_NOSIGNAL
#endif
        );
        if (result == -1 && platform::interrupted_socket_error(platform::socket_error())) continue;
        if (result <= 0) {
            if (result == -1 && report_errors) platform::report_socket_error("send");
            return false;
        }
        sent += static_cast<std::size_t>(result);
    }
    return true;
}

inline bool receive_all(platform::Socket socket_fd, void* data, std::size_t size,
    bool report_errors = true)
{
    auto* bytes = static_cast<char*>(data);
    std::size_t received = 0;
    while (received < size) {
        const int result = recv(socket_fd, bytes + received, static_cast<int>(size - received), 0);
        if (result == -1 && platform::interrupted_socket_error(platform::socket_error())) continue;
        if (result <= 0) {
            if (result == -1 && report_errors) platform::report_socket_error("recv");
            else if (result == 0 && report_errors) std::fprintf(stderr, "Peer disconnected during a message\n");
            return false;
        }
        received += static_cast<std::size_t>(result);
    }
    return true;
}

inline bool send_message(platform::Socket socket_fd, std::string_view message,
    bool report_errors = true)
{
    if (message.size() > max_message_size) {
        std::fprintf(stderr, "Message exceeds the 16 MiB limit\n");
        return false;
    }
    const std::uint32_t network_size = htonl(static_cast<std::uint32_t>(message.size()));
    return send_all(socket_fd, &network_size, sizeof(network_size), report_errors)
        && send_all(socket_fd, message.data(), message.size(), report_errors);
}

inline bool receive_message(platform::Socket socket_fd, std::string& message,
    bool report_errors = true)
{
    std::uint32_t network_size = 0;
    if (!receive_all(socket_fd, &network_size, sizeof(network_size), report_errors)) return false;

    const std::uint32_t message_size = ntohl(network_size);
    if (message_size > max_message_size) {
        std::fprintf(stderr, "Received message exceeds the 16 MiB limit\n");
        return false;
    }
    message.resize(message_size);
    return receive_all(socket_fd, message.data(), message.size(), report_errors);
}

} // namespace dan
