#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#endif

namespace dan::platform {

#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket invalid_socket = INVALID_SOCKET;
#else
using Socket = int;
constexpr Socket invalid_socket = -1;
#endif

bool initialize(std::string& error);
bool is_windows();
void cleanup();
void install_stop_handlers();
void configure_output();
void clear_console();
bool interactive_stdout();
bool color_stdout();
std::size_t terminal_width();
bool stop_requested();
bool acquire_single_instance(std::string_view name);

void close_socket(Socket socket);
void shutdown_socket(Socket socket);
Socket connect_tcp(std::string_view host, std::string_view port);
bool tcp_healthy(std::string_view host, std::string_view port);
bool wait_readable(Socket socket, int timeout_ms, bool& failed);
std::string free_tcp_port(std::string_view host);
bool private_ipv4(std::string_view text);
int socket_error();
bool interrupted_socket_error(int error);
void report_socket_error(const char* operation);

std::filesystem::path data_directory();
std::filesystem::path current_executable(std::string& error);
std::filesystem::path managed_provider_executable(std::string& error);
std::filesystem::path network_client_executable();
bool start_network_client(const std::filesystem::path& client, std::string& error);
bool executable_file(const std::filesystem::path& path);
bool install_msi(const std::filesystem::path& path, std::string& error);
bool open_url(std::string_view url, std::string& error);
std::uint64_t process_id();

std::string windows_command_line(const std::vector<std::string>& arguments);

class Process {
public:
    Process() = default;
    ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    bool start(const std::vector<std::string>& arguments, std::string& error,
        bool capture_stdout = false, bool discard_output = false);
    bool running();
    int wait();
    void stop();
    std::string read_stdout();
    std::uint64_t pid() const;

private:
#ifdef _WIN32
    void* process_ = nullptr;
    void* job_ = nullptr;
    void* output_ = nullptr;
    unsigned long pid_ = 0;
#else
    pid_t pid_ = -1;
    int output_ = -1;
#endif
};

bool run(const std::vector<std::string>& arguments, std::string& error,
    std::string* output = nullptr);
int replace_with_provider(const std::vector<std::string>& arguments, std::string& error);
bool sha256_file(const std::filesystem::path& path, std::string& digest,
    std::string& error);

} // namespace dan::platform
