#include "platform.hpp"
#include "protocol.hpp"
#include "control_plane.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <string>
#include <thread>

int main(int argc, char* argv[])
{
    assert(argc == 2);
    std::string error;
    assert(dan::platform::initialize(error));
    const std::string port = dan::platform::free_tcp_port("127.0.0.1");
    assert(!port.empty());
    dan::platform::Process coordinator;
    assert(coordinator.start({argv[1], port}, error, true));
    dan::platform::Socket socket = dan::platform::invalid_socket;
    for (int attempt = 0; attempt < 50 && socket == dan::platform::invalid_socket; ++attempt) {
        socket = dan::platform::connect_tcp("127.0.0.1", port);
        if (socket == dan::platform::invalid_socket)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    assert(socket != dan::platform::invalid_socket);
    assert(dan::send_message(socket, "HELLO"));
    assert(dan::send_message(socket,
        "CAPABILITIES\nprovider_id=incompatible\nprotocol_version=99\n"
        "device_type=GPU\ngpu_name=RTX-Test\nvram=8192-MiB\nvram_mib=6656\n"
        "model_name=dan-main\nbackend=llama-rpc\ncontrol_plane=1"));
    std::string rejection;
    assert(dan::receive_message(socket, rejection) && rejection.starts_with("INCOMPATIBLE\n"));
    dan::platform::close_socket(socket);
    socket = dan::platform::connect_tcp("127.0.0.1", port);
    assert(socket != dan::platform::invalid_socket);
    assert(dan::send_message(socket, "HELLO"));
    assert(dan::send_message(socket,
        "CAPABILITIES\nprovider_id=windows-test\nprovider_name=Windows-Test\n"
        "protocol_version=" + std::to_string(dan::protocol_version) + "\n"
        "device_type=GPU\ngpu_name=RTX-Test\nvram=8192-MiB\nvram_mib=6656\n"
        "model_name=dan-main\nbackend=llama-rpc\ncontrol_plane=1\n"
        "worker_endpoint=127.0.0.1:50052"));
    assert(dan::send_message(socket, "HEARTBEAT"));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    dan::platform::close_socket(socket);
    coordinator.stop();
    assert(coordinator.read_stdout().find("Provider connected") != std::string::npos);
    dan::platform::cleanup();
}
