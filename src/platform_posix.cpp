#include "platform.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <sys/prctl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <thread>

namespace dan::platform {
namespace {
volatile sig_atomic_t stopping = 0;
void request_stop(int) { stopping = 1; }
}

bool initialize(std::string&) { return true; }
bool is_windows() { return false; }
void cleanup() { }
void install_stop_handlers()
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);
}
void configure_output() { setvbuf(stdout, nullptr, _IOLBF, 0); }
void clear_console() { std::fputs("\x1b[2J\x1b[H", stdout); }
bool interactive_stdout() { return isatty(STDOUT_FILENO) == 1; }
bool color_stdout() { return interactive_stdout(); }
std::size_t terminal_width()
{
    winsize size{};
    return ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0 ? size.ws_col : 80;
}
bool stop_requested() { return stopping != 0; }
bool acquire_single_instance(std::string_view) { return true; }
void close_socket(Socket socket) { if (socket != invalid_socket) close(socket); }
void shutdown_socket(Socket socket) { if (socket != invalid_socket) shutdown(socket, SHUT_RDWR); }
int socket_error() { return errno; }
bool interrupted_socket_error(int error) { return error == EINTR; }
void report_socket_error(const char* operation) { perror(operation); }

Socket connect_tcp(std::string_view host, std::string_view port)
{
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(std::string(host).c_str(), std::string(port).c_str(), &hints, &addresses) != 0)
        return invalid_socket;
    Socket connected = invalid_socket;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        connected = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (connected != invalid_socket
            && connect(connected, address->ai_addr, address->ai_addrlen) == 0) break;
        close_socket(connected); connected = invalid_socket;
    }
    freeaddrinfo(addresses);
    return connected;
}

bool tcp_healthy(std::string_view host, std::string_view port)
{
    const Socket socket = connect_tcp(host, port);
    close_socket(socket);
    return socket != invalid_socket;
}

bool wait_readable(Socket socket, int timeout_ms, bool& failed)
{
    pollfd watched{socket, POLLIN, 0};
    const int result = poll(&watched, 1, timeout_ms);
    failed = result < 0 && errno != EINTR;
    if (result > 0 && (watched.revents & (POLLHUP | POLLERR | POLLNVAL))) failed = true;
    return result > 0 && (watched.revents & POLLIN);
}

std::string free_tcp_port(std::string_view host)
{
    const Socket socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == invalid_socket) return {};
    sockaddr_in address{}; address.sin_family = AF_INET;
    if (inet_pton(AF_INET, std::string(host).c_str(), &address.sin_addr) != 1
        || bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == -1) {
        close_socket(socket_fd); return {};
    }
    socklen_t size = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &size) == -1) {
        close_socket(socket_fd); return {};
    }
    close_socket(socket_fd);
    return std::to_string(ntohs(address.sin_port));
}

bool private_ipv4(std::string_view text)
{
    in_addr address{};
    if (inet_pton(AF_INET, std::string(text).c_str(), &address) != 1) return false;
    const std::uint32_t host = ntohl(address.s_addr);
    const unsigned first = host >> 24, second = (host >> 16) & 0xff;
    return first == 10 || first == 127 || (first == 172 && second >= 16 && second <= 31)
        || (first == 192 && second == 168) || (first == 100 && second >= 64 && second <= 127);
}

std::filesystem::path data_directory()
{
    const char* home = std::getenv("HOME");
    return home ? std::filesystem::path(home) / ".dan" : std::filesystem::path(".dan");
}

std::filesystem::path current_executable(std::string& error)
{
    std::error_code ec;
    auto result = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) error = "could not locate current executable: " + ec.message();
    return result;
}

std::filesystem::path managed_provider_executable(std::string& error)
{
    return current_executable(error).parent_path() / "managed_provider";
}
std::filesystem::path network_client_executable() { return "tailscale"; }
bool start_network_client(const std::filesystem::path&, std::string&) { return true; }

bool executable_file(const std::filesystem::path& path) { return access(path.c_str(), X_OK) == 0; }
bool install_msi(const std::filesystem::path&, std::string& error)
{
    error = "MSI installation is only available on Windows"; return false;
}
bool open_url(std::string_view, std::string& error)
{
    error = "opening a browser is only available on Windows"; return false;
}
std::uint64_t process_id() { return static_cast<std::uint64_t>(getpid()); }
std::string windows_command_line(const std::vector<std::string>&) { return {}; }

Process::~Process() { stop(); if (output_ != -1) close(output_); }

bool Process::start(const std::vector<std::string>& arguments, std::string& error,
    bool capture_stdout, bool discard_output)
{
    if (arguments.empty() || running() || (capture_stdout && discard_output)) {
        error = "invalid or duplicate process launch"; return false;
    }
    int output[2] = {-1, -1};
    if (capture_stdout && pipe(output) == -1) { error = "could not create output pipe"; return false; }
    pid_ = fork();
    if (pid_ == -1) {
        if (capture_stdout) { close(output[0]); close(output[1]); }
        error = "could not fork process"; return false;
    }
    if (pid_ == 0) {
        if (capture_stdout) {
            close(output[0]);
            if (dup2(output[1], STDOUT_FILENO) == -1
                || dup2(output[1], STDERR_FILENO) == -1) _exit(127);
            close(output[1]);
        } else if (discard_output) {
            const int null_output = open("/dev/null", O_WRONLY);
            if (null_output == -1 || dup2(null_output, STDOUT_FILENO) == -1
                || dup2(null_output, STDERR_FILENO) == -1) _exit(127);
            close(null_output);
        }
        const pid_t parent = getppid();
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1 || getppid() != parent) _exit(127);
        std::vector<char*> argv;
        for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    if (capture_stdout) { close(output[1]); output_ = output[0]; }
    return true;
}

bool Process::running()
{
    if (pid_ == -1) return false;
    int status = 0;
    const pid_t result = waitpid(pid_, &status, WNOHANG);
    if (result == 0) return true;
    if (result == pid_) pid_ = -1;
    return false;
}

int Process::wait()
{
    if (pid_ == -1) return -1;
    int status = 0;
    while (waitpid(pid_, &status, 0) == -1 && errno == EINTR) { }
    pid_ = -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

void Process::stop()
{
    if (pid_ == -1) return;
    kill(pid_, SIGTERM);
    for (int attempt = 0; attempt < 20; ++attempt) {
        int status = 0;
        if (waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    kill(pid_, SIGKILL);
    while (waitpid(pid_, nullptr, 0) == -1 && errno == EINTR) { }
    pid_ = -1;
}

std::string Process::read_stdout()
{
    std::string result;
    if (output_ == -1) return result;
    char buffer[4096];
    while (result.size() < 1024 * 1024) {
        const ssize_t count = read(output_, buffer, sizeof(buffer));
        if (count == -1 && errno == EINTR) continue;
        if (count <= 0) break;
        result.append(buffer, static_cast<std::size_t>(count));
    }
    close(output_); output_ = -1;
    return result;
}

std::uint64_t Process::pid() const { return static_cast<std::uint64_t>(pid_); }

bool run(const std::vector<std::string>& arguments, std::string& error, std::string* output)
{
    Process child;
    if (!child.start(arguments, error, output != nullptr)) return false;
    if (output) *output = child.read_stdout();
    if (child.wait() != 0) { error = arguments.front() + " exited unsuccessfully"; return false; }
    return true;
}

int replace_with_provider(const std::vector<std::string>& arguments, std::string& error)
{
    std::vector<char*> argv;
    for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    execv(argv[0], argv.data());
    error = "could not start managed_provider";
    return 1;
}

bool sha256_file(const std::filesystem::path& path, std::string& digest, std::string& error)
{
    std::string result;
    if (!run({"sha256sum", "--", path.string()}, error, &result) || result.size() < 65) {
        error = "could not calculate SHA-256"; return false;
    }
    digest = result.substr(0, 64);
    for (const unsigned char byte : digest) {
        if (!std::isxdigit(byte)) { error = "could not calculate SHA-256"; return false; }
    }
    for (char& byte : digest) byte = static_cast<char>(std::tolower(static_cast<unsigned char>(byte)));
    return true;
}

} // namespace dan::platform
