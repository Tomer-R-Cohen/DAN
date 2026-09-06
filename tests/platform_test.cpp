#include "platform.hpp"
#include "protocol.hpp"

#ifndef _WIN32
#include <netinet/in.h>
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <thread>

int main(int argc, char* argv[])
{
    if (argc > 1 && std::string_view(argv[1]).starts_with("--query-gpu=")) {
        std::puts("0, NVIDIA GeForce RTX Test, 8192, GPU-test");
        return 0;
    }
    if (argc > 1 && std::string_view(argv[1]) == "ip") {
        std::puts("100.64.0.2");
        return 0;
    }
    if (argc > 1 && std::string_view(argv[1]) == "--child") {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        return 0;
    }
    std::string error;
    assert(dan::platform::initialize(error));
    if (argc > 1 && std::string_view(argv[1]) == "--nested") {
        dan::platform::Process grandchild;
        assert(grandchild.start({argv[0], "--child"}, error));
        grandchild.stop();
        dan::platform::cleanup();
        return 0;
    }
    assert(dan::platform::private_ipv4("127.0.0.1"));
    assert(dan::platform::private_ipv4("100.64.0.1"));
    assert(!dan::platform::private_ipv4("8.8.8.8"));
    const auto listener = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(listener != dan::platform::invalid_socket);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    assert(listen(listener, 1) == 0);
#ifdef _WIN32
    int address_size = sizeof(address);
#else
    socklen_t address_size = sizeof(address);
#endif
    assert(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_size) == 0);
    std::jthread peer([&] {
        const auto socket = accept(listener, nullptr, nullptr);
        std::string message;
        assert(socket != dan::platform::invalid_socket);
        assert(dan::receive_message(socket, message) && message == "hello");
        assert(dan::send_message(socket, "world"));
        dan::platform::close_socket(socket);
    });
    const auto client = dan::platform::connect_tcp("127.0.0.1",
        std::to_string(ntohs(address.sin_port)));
    assert(client != dan::platform::invalid_socket);
    assert(dan::send_message(client, "hello"));
    std::string reply;
    assert(dan::receive_message(client, reply) && reply == "world");
    dan::platform::close_socket(client);
    dan::platform::close_socket(listener);
    peer.join();
#ifdef _WIN32
    assert(dan::platform::data_directory().filename() == "DAN");
    const std::vector<std::string> arguments{
        R"(C:\Users\Test User\DAN Provider\rpc-server.exe)", "plain", "has space",
        R"(quote"inside)", R"(trailing\)"};
    assert(dan::platform::windows_command_line(arguments)
        == R"("C:\Users\Test User\DAN Provider\rpc-server.exe" plain "has space" "quote\"inside" trailing\)"
    );
#endif
    const auto hash_file = std::filesystem::temp_directory_path() / "dan-platform-sha256-test";
    { std::ofstream output(hash_file, std::ios::binary); output << "abc"; }
    std::string digest;
    assert(dan::platform::sha256_file(hash_file, digest, error));
    assert(digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::filesystem::remove(hash_file);
    dan::platform::Process child;
    assert(child.start({argv[0], "--child"}, error));
    assert(child.running());
    assert(!child.start({argv[0], "--child"}, error));
    child.stop();
    assert(!child.running());
    assert(dan::platform::run({argv[0], "--nested"}, error));
    dan::platform::cleanup();
}
