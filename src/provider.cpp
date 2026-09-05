#include "protocol.hpp"

#include <netdb.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <csignal>
#include <vector>
#include <string>
#include <string_view>

namespace {
constexpr const char* default_host = "127.0.0.1";
constexpr const char* default_port = "9000";
constexpr std::string_view input_marker = "\n> ";

struct Capabilities {
    std::string provider_id;
    std::string device_type = "CPU";
    std::string gpu_name = "not available";
    std::string vram = "not available";
    std::string model_name;
    std::string backend = "CPU";
};

std::string file_name(std::string_view path)
{
    const std::size_t slash = path.find_last_of('/');
    return std::string(path.substr(slash == std::string_view::npos ? 0 : slash + 1));
}

void replace_newlines(std::string& value)
{
    for (char& character : value) {
        if (character == '\n' || character == '\r') character = ' ';
    }
}

std::string capability_message(Capabilities capabilities)
{
    replace_newlines(capabilities.provider_id);
    replace_newlines(capabilities.device_type);
    replace_newlines(capabilities.gpu_name);
    replace_newlines(capabilities.vram);
    replace_newlines(capabilities.model_name);
    replace_newlines(capabilities.backend);
    return "CAPABILITIES\nprovider_id=" + capabilities.provider_id
        + "\ndevice_type=" + capabilities.device_type
        + "\ngpu_name=" + capabilities.gpu_name
        + "\nvram=" + capabilities.vram
        + "\nmodel_name=" + capabilities.model_name
        + "\nbackend=" + capabilities.backend;
}

bool set_capability_option(const char* option, const char* value,
    Capabilities& capabilities)
{
    if (std::strcmp(option, "--id") == 0) capabilities.provider_id = value;
    else if (std::strcmp(option, "--device") == 0) capabilities.device_type = value;
    else if (std::strcmp(option, "--gpu") == 0) capabilities.gpu_name = value;
    else if (std::strcmp(option, "--vram") == 0) capabilities.vram = value;
    else if (std::strcmp(option, "--model-name") == 0) capabilities.model_name = value;
    else if (std::strcmp(option, "--backend") == 0) capabilities.backend = value;
    else return false;
    return true;
}

void trim(std::string& text)
{
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
    std::size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) ++first;
    text.erase(0, first);
}

class LlamaRuntime {
public:
    ~LlamaRuntime()
    {
        if (input_fd_ != -1) close(input_fd_);
        if (output_fd_ != -1) close(output_fd_);
        if (pid_ != -1) {
            int status = 0;
            while (waitpid(pid_, &status, 0) == -1 && errno == EINTR) { }
        }
    }

    bool start(const char* executable, const char* model, const std::vector<std::string>& options, std::string& error)
    {
        int input_pipe[2];
        int output_pipe[2];
        if (pipe(input_pipe) == -1) {
            perror("pipe");
            error = "Could not create llama.cpp input pipe";
            return false;
        }
        if (pipe(output_pipe) == -1) {
            perror("pipe");
            close(input_pipe[0]);
            close(input_pipe[1]);
            error = "Could not create llama.cpp output pipe";
            return false;
        }

        pid_ = fork();
        if (pid_ == -1) {
            perror("fork");
            close(input_pipe[0]); close(input_pipe[1]);
            close(output_pipe[0]); close(output_pipe[1]);
            error = "Could not start llama.cpp";
            return false;
        }
        if (pid_ == 0) {
            close(input_pipe[1]);
            close(output_pipe[0]);
            if (dup2(input_pipe[0], STDIN_FILENO) == -1
                || dup2(output_pipe[1], STDOUT_FILENO) == -1) {
                perror("dup2");
                _exit(127);
            }
            close(input_pipe[0]);
            close(output_pipe[1]);
            std::vector<char*> arguments = {
                const_cast<char*>(executable), const_cast<char*>("--model"),
                const_cast<char*>(model), const_cast<char*>("--n-predict"),
                // This revision's interactive token budget spans turns and can
                // yield empty answers when a new input exhausts it. Use EOS.
                const_cast<char*>("-1"), const_cast<char*>("--conversation"),
                const_cast<char*>("--interactive-first"),
                const_cast<char*>("--simple-io"),
                const_cast<char*>("--no-display-prompt"),
                const_cast<char*>("--color"), const_cast<char*>("off")
            };
            for (const auto& option : options) arguments.push_back(const_cast<char*>(option.c_str()));
            arguments.push_back(nullptr);
            execvp(executable, arguments.data());
            perror("execvp llama-completion");
            _exit(127);
        }

        close(input_pipe[0]);
        close(output_pipe[1]);
        input_fd_ = input_pipe[1];
        output_fd_ = output_pipe[0];
        std::string startup_output;
        if (!read_until_input_marker(startup_output)) {
            error = "llama.cpp exited before becoming ready";
            return false;
        }
        return true;
    }

    bool infer(const std::string& prompt, std::string& response, std::string& error)
    {
        // Simple-I/O treats trailing slash/backslash as UI commands. A trailing
        // space keeps user text from changing the subprocess input protocol.
        if (!write_all(prompt + " \n")) {
            error = "Could not write prompt to llama.cpp";
            return false;
        }
        if (!read_until_input_marker(response)) {
            error = "llama.cpp exited during inference";
            return false;
        }
        trim(response);
        if (response.empty()) {
            error = "llama.cpp returned an empty response";
            return false;
        }
        return true;
    }

private:
    bool write_all(std::string_view text)
    {
        std::size_t written = 0;
        while (written < text.size()) {
            const ssize_t result = write(input_fd_, text.data() + written, text.size() - written);
            if (result == -1 && errno == EINTR) continue;
            if (result <= 0) {
                if (result == -1) perror("write");
                return false;
            }
            written += static_cast<std::size_t>(result);
        }
        return true;
    }

    bool read_until_input_marker(std::string& output)
    {
        output.clear();
        char byte = '\0';
        while (true) {
            const ssize_t result = read(output_fd_, &byte, 1);
            if (result == -1 && errno == EINTR) continue;
            if (result <= 0) {
                if (result == -1) perror("read");
                return false;
            }
            output.push_back(byte);
            if (output.size() > dan::max_message_size - 128) return false;
            if (output.ends_with(input_marker)) {
                output.resize(output.size() - input_marker.size());
                return true;
            }
        }
    }

    pid_t pid_ = -1;
    int input_fd_ = -1;
    int output_fd_ = -1;
};
}

int main(int argc, char* argv[])
{
    std::signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <llama-completion> <model.gguf> [host] [port] "
            "[--id value] [--device value] [--gpu value] [--vram value] "
            "[--model-name value] [--backend value] "
            "[--gpu-layers N] [--ctx-size N] [--n-predict N] [--runtime-device CUDA0]\n", argv[0]);
        return 1;
    }
    const char* executable = argv[1];
    const char* model = argv[2];
    const char* host = default_host;
    const char* port = default_port;
    int argument = 3;
    if (argument < argc && !std::string_view(argv[argument]).starts_with("--")) {
        host = argv[argument++];
    }
    if (argument < argc && !std::string_view(argv[argument]).starts_with("--")) {
        port = argv[argument++];
    }

    char hostname[256] = "unknown-host";
    if (gethostname(hostname, sizeof(hostname)) == -1) {
        std::strcpy(hostname, "unknown-host");
    }
    hostname[sizeof(hostname) - 1] = '\0';
    Capabilities capabilities;
    capabilities.provider_id = std::string(hostname) + '-' + std::to_string(getpid());
    capabilities.model_name = file_name(model);
    std::vector<std::string> runtime_options;
    while (argument < argc) {
        const std::string option = argv[argument];
        if (argument + 1 < argc && (option == "--gpu-layers" || option == "--ctx-size"
                || option == "--n-predict" || option == "--runtime-device")) {
            const std::string value = argv[argument + 1];
            if (option != "--runtime-device" && !(option == "--n-predict" && value == "-1")) {
                int number = 0;
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
                    || number < 0 || (option != "--gpu-layers" && number == 0)) {
                    std::fprintf(stderr, "Invalid runtime numeric option: %s\n", option.c_str());
                    return 1;
                }
            }
            runtime_options.push_back(option == "--runtime-device" ? "--device" : option);
            runtime_options.push_back(value);
            argument += 2;
            continue;
        }
        if (argument + 1 >= argc
            || !set_capability_option(argv[argument], argv[argument + 1], capabilities)) {
            std::fprintf(stderr, "Unknown or incomplete metadata option: %s\n", argv[argument]);
            return 1;
        }
        argument += 2;
    }
    if (access(model, R_OK) == -1) {
        perror("model file");
        return 1;
    }

    LlamaRuntime runtime;
    std::string error;
    std::printf("Loading model...\n");
    for (const auto& option : runtime_options) std::printf("Runtime argument: %s\n", option.c_str());
    const auto load_start = std::chrono::steady_clock::now();
    if (!runtime.start(executable, model, runtime_options, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("Model loaded; runtime ready in %.1f ms (includes warmup)\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count());

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    const int address_status = getaddrinfo(host, port, &hints, &addresses);
    if (address_status != 0) {
        std::fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(address_status));
        return 1;
    }
    int coordinator_socket = -1;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        coordinator_socket = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (coordinator_socket == -1) continue;
        if (connect(coordinator_socket, address->ai_addr, address->ai_addrlen) == 0) break;
        close(coordinator_socket);
        coordinator_socket = -1;
    }
    freeaddrinfo(addresses);
    if (coordinator_socket == -1) {
        perror("connect");
        return 1;
    }
    if (!dan::send_message(coordinator_socket, "HELLO")) {
        close(coordinator_socket);
        return 1;
    }
    if (!dan::send_message(coordinator_socket, capability_message(capabilities))) {
        close(coordinator_socket);
        return 1;
    }

    constexpr std::string_view prompt_prefix = "PROMPT\n";
    while (true) {
        std::string message;
        if (!dan::receive_message(coordinator_socket, message)) {
            close(coordinator_socket);
            return 1;
        }
        if (message == "BYE") break;
        if (!message.starts_with(prompt_prefix)) {
            std::fprintf(stderr, "Expected PROMPT or BYE from coordinator\n");
            close(coordinator_socket);
            return 1;
        }

        const std::size_t id_end = message.find('\n', prompt_prefix.size());
        if (id_end == std::string::npos || id_end == prompt_prefix.size()) {
            std::fprintf(stderr, "PROMPT is missing a request ID\n");
            close(coordinator_socket);
            return 1;
        }
        const std::string request_id = message.substr(
            prompt_prefix.size(), id_end - prompt_prefix.size());
        const std::string prompt = message.substr(id_end + 1);
        std::printf("Running request %s: %s\n", request_id.c_str(), prompt.c_str());
        std::string response;
        error.clear();
        const bool succeeded = runtime.infer(prompt, response, error);
        const std::string result = succeeded
            ? "RESPONSE\n" + request_id + '\n' + response
            : "ERROR\n" + request_id + '\n' + error;
        if (!dan::send_message(coordinator_socket, result)) {
            close(coordinator_socket);
            return 1;
        }
        if (!succeeded) {
            close(coordinator_socket);
            return 1;
        }
    }
    close(coordinator_socket);
    return 0;
}
