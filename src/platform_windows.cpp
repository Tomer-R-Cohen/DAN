#include "platform.hpp"

#include <windows.h>
#include <ws2tcpip.h>
#include <bcrypt.h>
#include <shellapi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

namespace dan::platform {
namespace {
volatile LONG stopping = 0;
HANDLE instance_mutex = nullptr;
bool vt_output = false;
BOOL WINAPI control_handler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT
        || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) {
        InterlockedExchange(&stopping, 1); return TRUE;
    }
    return FALSE;
}

std::wstring wide(std::string_view text)
{
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), nullptr, 0);
    if (size <= 0) return {};
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), result.data(), size);
    return result;
}

std::string quote_windows(std::string_view argument)
{
    if (!argument.empty() && argument.find_first_of(" \t\n\v\"") == std::string_view::npos)
        return std::string(argument);
    std::string result = "\"";
    std::size_t slashes = 0;
    for (const char ch : argument) {
        if (ch == '\\') { ++slashes; continue; }
        if (ch == '"') result.append(slashes * 2 + 1, '\\');
        else result.append(slashes, '\\');
        slashes = 0; result.push_back(ch);
    }
    result.append(slashes * 2, '\\');
    return result + '"';
}
}

bool initialize(std::string& error)
{
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) { error = "Winsock initialization failed"; return false; }
    return true;
}
bool is_windows() { return true; }
void cleanup() { if (instance_mutex) { CloseHandle(instance_mutex); instance_mutex=nullptr; } WSACleanup(); }
void install_stop_handlers() { SetConsoleCtrlHandler(control_handler, TRUE); }
void configure_output()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (output != INVALID_HANDLE_VALUE && GetConsoleMode(output, &mode)) {
        vt_output = SetConsoleMode(output, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
    }
}
void clear_console()
{
    if (vt_output) {
        std::fputs("\x1b[2J\x1b[3J\x1b[H", stdout);
        std::fflush(stdout);
        return;
    }
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (output == INVALID_HANDLE_VALUE || !GetConsoleScreenBufferInfo(output, &info)) return;
    const DWORD cells = static_cast<DWORD>(info.dwSize.X) * static_cast<DWORD>(info.dwSize.Y);
    DWORD written = 0; const COORD origin{0, 0};
    FillConsoleOutputCharacterW(output, L' ', cells, origin, &written);
    FillConsoleOutputAttribute(output, info.wAttributes, cells, origin, &written);
    SetConsoleCursorPosition(output, origin);
}
bool interactive_stdout()
{
    DWORD mode = 0;
    return GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode) != 0;
}
bool color_stdout() { return vt_output; }
std::size_t terminal_width()
{
    CONSOLE_SCREEN_BUFFER_INFO info{};
    return GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)
        ? static_cast<std::size_t>(info.srWindow.Right - info.srWindow.Left + 1) : 80;
}
bool stop_requested() { return InterlockedCompareExchange(&stopping, 0, 0) != 0; }
bool acquire_single_instance(std::string_view name)
{
    if (instance_mutex) return true;
    const std::wstring mutex_name = L"Local\\" + wide(name);
    instance_mutex = CreateMutexW(nullptr, FALSE, mutex_name.c_str());
    return instance_mutex && GetLastError() != ERROR_ALREADY_EXISTS;
}
void close_socket(Socket socket) { if (socket != invalid_socket) closesocket(socket); }
void shutdown_socket(Socket socket) { if (socket != invalid_socket) shutdown(socket, SD_BOTH); }
int socket_error() { return WSAGetLastError(); }
bool interrupted_socket_error(int error) { return error == WSAEINTR; }
void report_socket_error(const char* operation)
{
    std::fprintf(stderr, "%s failed (Winsock error %d)\n", operation, WSAGetLastError());
}

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
            && connect(connected, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) break;
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
    fd_set sockets; FD_ZERO(&sockets); FD_SET(socket, &sockets);
    timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    const int result = select(0, &sockets, nullptr, nullptr, &timeout);
    failed = result == SOCKET_ERROR;
    return result > 0 && FD_ISSET(socket, &sockets);
}

std::string free_tcp_port(std::string_view host)
{
    const Socket socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == invalid_socket) return {};
    sockaddr_in address{}; address.sin_family = AF_INET;
    if (InetPtonA(AF_INET, std::string(host).c_str(), &address.sin_addr) != 1
        || bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) {
        close_socket(socket_fd); return {};
    }
    int size = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &size) == SOCKET_ERROR) {
        close_socket(socket_fd); return {};
    }
    close_socket(socket_fd);
    return std::to_string(ntohs(address.sin_port));
}

bool private_ipv4(std::string_view text)
{
    in_addr address{};
    if (InetPtonA(AF_INET, std::string(text).c_str(), &address) != 1) return false;
    const std::uint32_t host = ntohl(address.s_addr);
    const unsigned first = host >> 24, second = (host >> 16) & 0xff;
    return first == 10 || first == 127 || (first == 172 && second >= 16 && second <= 31)
        || (first == 192 && second == 168) || (first == 100 && second >= 64 && second <= 127);
}

std::filesystem::path data_directory()
{
    char* local = nullptr;
    std::size_t size = 0;
    _dupenv_s(&local, &size, "LOCALAPPDATA");
    const std::filesystem::path result = local
        ? std::filesystem::path(local) / "DAN" : std::filesystem::path("DAN");
    std::free(local);
    return result;
}

std::filesystem::path current_executable(std::string& error)
{
    std::wstring path(32768, L'\0');
    const DWORD size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (size == 0 || size == path.size()) { error = "could not locate current executable"; return {}; }
    path.resize(size);
    return std::filesystem::path(path);
}

std::filesystem::path managed_provider_executable(std::string& error)
{
    error.clear();
    const auto directory = current_executable(error).parent_path();
    const auto packaged = directory / "runtime" / "managed_provider.exe";
    return std::filesystem::is_regular_file(packaged)
        ? packaged : directory / "managed_provider.exe";
}

std::filesystem::path network_client_executable()
{
    wchar_t override_path[32768];
    const DWORD override_size = GetEnvironmentVariableW(L"DAN_TAILSCALE", override_path,
        static_cast<DWORD>(std::size(override_path)));
    if (override_size > 0 && override_size < std::size(override_path)) return override_path;
    std::wstring directory(32768, L'\0');
    const DWORD size = GetEnvironmentVariableW(L"ProgramFiles", directory.data(),
        static_cast<DWORD>(directory.size()));
    if (size > 0 && size < directory.size()) {
        directory.resize(size);
        const std::filesystem::path installed = std::filesystem::path(directory)
            / "Tailscale" / "tailscale.exe";
        if (std::filesystem::is_regular_file(installed)) return installed;
    }
    return "tailscale.exe";
}

bool start_network_client(const std::filesystem::path& client, std::string& error)
{
    const std::filesystem::path tray = client.parent_path() / L"tailscale-ipn.exe";
    if (!executable_file(tray)) { error = "Tailscale Windows client was not found"; return false; }
    if (reinterpret_cast<std::intptr_t>(ShellExecuteW(nullptr, L"open", tray.c_str(),
            nullptr, tray.parent_path().c_str(), SW_HIDE)) <= 32) {
        error = "could not start the Tailscale Windows client"; return false;
    }
    return true;
}

bool executable_file(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec)
        && (path.extension() == ".exe" || path.extension() == ".EXE");
}
bool install_msi(const std::filesystem::path& path, std::string& error)
{
    if (!std::filesystem::is_regular_file(path)) { error = "bundled network installer is missing"; return false; }
    const std::wstring parameters = L"/i \"" + path.wstring() + L"\" /passive /norestart";
    SHELLEXECUTEINFOW launch{};
    launch.cbSize = sizeof(launch);
    launch.fMask = SEE_MASK_NOCLOSEPROCESS;
    launch.lpVerb = L"runas";
    launch.lpFile = L"msiexec.exe";
    launch.lpParameters = parameters.c_str();
    launch.nShow = SW_SHOWNORMAL;
    if (!ShellExecuteExW(&launch)) {
        error = GetLastError() == ERROR_CANCELLED ? "network installation was cancelled"
            : "could not start the official Tailscale installer";
        return false;
    }
    WaitForSingleObject(launch.hProcess, INFINITE);
    DWORD status = 1;
    GetExitCodeProcess(launch.hProcess, &status);
    CloseHandle(launch.hProcess);
    if (status != 0 && status != 3010) {
        error = "official Tailscale installer failed (" + std::to_string(status) + ')'; return false;
    }
    return true;
}
bool open_url(std::string_view url, std::string& error)
{
    const std::wstring target = wide(url);
    if (reinterpret_cast<std::intptr_t>(ShellExecuteW(nullptr, L"open", target.c_str(),
            nullptr, nullptr, SW_SHOWNORMAL)) <= 32) {
        error = "could not open the authentication page"; return false;
    }
    return true;
}
std::uint64_t process_id() { return GetCurrentProcessId(); }

std::string windows_command_line(const std::vector<std::string>& arguments)
{
    std::string result;
    for (const auto& argument : arguments) {
        if (!result.empty()) result.push_back(' ');
        result += quote_windows(argument);
    }
    return result;
}

Process::~Process()
{
    stop();
    if (output_) CloseHandle(output_);
    if (job_) CloseHandle(job_);
}

bool Process::start(const std::vector<std::string>& arguments, std::string& error,
    bool capture_stdout, bool discard_output)
{
    if (arguments.empty() || running() || (capture_stdout && discard_output)) {
        error = "invalid or duplicate process launch"; return false;
    }
    if (job_) { CloseHandle(job_); job_ = nullptr; }
    SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
    HANDLE read_pipe = nullptr, write_pipe = nullptr;
    if (capture_stdout && (!CreatePipe(&read_pipe, &write_pipe, &security, 0)
        || !SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0))) {
        if (read_pipe) CloseHandle(read_pipe);
        if (write_pipe) CloseHandle(write_pipe);
        error = "could not create output pipe"; return false;
    }
    HANDLE null_output = discard_output
        ? CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)
        : nullptr;
    if (discard_output && null_output == INVALID_HANDLE_VALUE) {
        error = "could not suppress child process output"; return false;
    }
    const std::string utf8_command = windows_command_line(arguments);
    std::wstring command = wide(utf8_command);
    std::wstring executable = wide(arguments.front());
    const wchar_t* application = std::filesystem::path(arguments.front()).has_parent_path()
        ? executable.c_str() : nullptr;
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    if (capture_stdout || discard_output) {
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdOutput = discard_output ? null_output : write_pipe;
        startup.hStdError = discard_output ? null_output : write_pipe;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    PROCESS_INFORMATION information{};
    const BOOL created = CreateProcessW(application, command.data(), nullptr, nullptr,
        capture_stdout || discard_output, CREATE_SUSPENDED, nullptr, nullptr, &startup, &information);
    if (write_pipe) CloseHandle(write_pipe);
    if (null_output && null_output != INVALID_HANDLE_VALUE) CloseHandle(null_output);
    if (!created) {
        if (read_pipe) CloseHandle(read_pipe);
        error = "CreateProcessW failed (" + std::to_string(GetLastError()) + ')'; return false;
    }
    job_ = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job_ || !SetInformationJobObject(job_, JobObjectExtendedLimitInformation,
            &limits, sizeof(limits)) || !AssignProcessToJobObject(job_, information.hProcess)) {
        TerminateProcess(information.hProcess, 1);
        CloseHandle(information.hThread); CloseHandle(information.hProcess);
        if (read_pipe) CloseHandle(read_pipe);
        if (job_) { CloseHandle(job_); job_ = nullptr; }
        error = "could not place child process in a Job Object"; return false;
    }
    process_ = information.hProcess; pid_ = information.dwProcessId; output_ = read_pipe;
    ResumeThread(information.hThread); CloseHandle(information.hThread);
    return true;
}

bool Process::running()
{
    if (!process_) return false;
    if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) return true;
    CloseHandle(process_); process_ = nullptr; pid_ = 0; return false;
}

int Process::wait()
{
    if (!process_) return -1;
    WaitForSingleObject(process_, INFINITE);
    DWORD status = 1; GetExitCodeProcess(process_, &status);
    CloseHandle(process_); process_ = nullptr; pid_ = 0;
    return status <= static_cast<DWORD>(std::numeric_limits<int>::max())
        ? static_cast<int>(status) : -1;
}

void Process::stop()
{
    if (!process_) return;
    TerminateJobObject(job_, 1);
    WaitForSingleObject(process_, 5000);
    CloseHandle(process_); process_ = nullptr; pid_ = 0;
}

std::string Process::read_stdout()
{
    std::string result;
    if (!output_) return result;
    char buffer[4096]; DWORD count = 0;
    while (result.size() < 1024 * 1024
        && ReadFile(output_, buffer, sizeof(buffer), &count, nullptr) && count != 0)
        result.append(buffer, count);
    CloseHandle(output_); output_ = nullptr;
    return result;
}

std::uint64_t Process::pid() const { return pid_; }

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
    Process child;
    if (!child.start(arguments, error)) return 1;
    while (child.running() && !stop_requested()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (stop_requested()) child.stop();
    return 0;
}

bool sha256_file(const std::filesystem::path& path, std::string& digest, std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open artifact for SHA-256"; return false; }
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_size = 0, bytes = 0;
    std::vector<unsigned char> object;
    std::vector<unsigned char> value(32);
    bool ok = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0
        && BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &bytes, 0) >= 0;
    if (ok) {
        object.resize(object_size);
        ok = BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) >= 0;
    }
    char buffer[64 * 1024];
    while (ok && input) {
        input.read(buffer, sizeof(buffer));
        const auto count = input.gcount();
        if (count > 0) ok = BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer),
            static_cast<ULONG>(count), 0) >= 0;
    }
    if (input.bad()) ok = false;
    if (ok) ok = BCryptFinishHash(hash, value.data(), static_cast<ULONG>(value.size()), 0) >= 0;
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    if (!ok) { error = "could not calculate SHA-256"; return false; }
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (const auto byte : value) encoded << std::setw(2) << static_cast<unsigned>(byte);
    digest = encoded.str();
    return true;
}

} // namespace dan::platform
