#pragma once

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <sys/socket.h>

namespace dan {

constexpr std::uint32_t max_message_size = 16 * 1024 * 1024;

inline bool send_all(int socket_fd, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t result = send(
            socket_fd, bytes + sent, size - sent, MSG_NOSIGNAL);
        if (result == -1 && errno == EINTR) continue;
        if (result <= 0) {
            if (result == -1) perror("send");
            return false;
        }
        sent += static_cast<std::size_t>(result);
    }
    return true;
}

inline bool receive_all(int socket_fd, void* data, std::size_t size)
{
    auto* bytes = static_cast<char*>(data);
    std::size_t received = 0;
    while (received < size) {
        const ssize_t result = recv(socket_fd, bytes + received, size - received, 0);
        if (result == -1 && errno == EINTR) continue;
        if (result <= 0) {
            if (result == -1) perror("recv");
            else std::fprintf(stderr, "Peer disconnected during a message\n");
            return false;
        }
        received += static_cast<std::size_t>(result);
    }
    return true;
}

inline bool send_message(int socket_fd, std::string_view message)
{
    if (message.size() > max_message_size) {
        std::fprintf(stderr, "Message exceeds the 16 MiB limit\n");
        return false;
    }
    const std::uint32_t network_size = htonl(static_cast<std::uint32_t>(message.size()));
    return send_all(socket_fd, &network_size, sizeof(network_size))
        && send_all(socket_fd, message.data(), message.size());
}

inline bool receive_message(int socket_fd, std::string& message)
{
    std::uint32_t network_size = 0;
    if (!receive_all(socket_fd, &network_size, sizeof(network_size))) return false;

    const std::uint32_t message_size = ntohl(network_size);
    if (message_size > max_message_size) {
        std::fprintf(stderr, "Received message exceeds the 16 MiB limit\n");
        return false;
    }
    message.resize(message_size);
    return receive_all(socket_fd, message.data(), message.size());
}

} // namespace dan
