#include "llama.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
constexpr socket_t invalid_socket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t invalid_socket = -1;
#endif

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t magic = 0x44414e30; // DAN0
constexpr std::uint16_t version = 1;
constexpr std::size_t header_size = 40;
constexpr std::uint64_t max_payload = 64ull * 1024 * 1024;
enum Type : std::uint16_t { error = 0, prompt = 1, token = 2, activation = 3, result = 4 };
enum DType : std::uint16_t { none = 0, f32le = 1 };

struct Frame {
    std::uint16_t type = 0;
    std::uint64_t request = 0;
    std::uint32_t position = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint16_t dtype = none;
    std::vector<std::uint8_t> payload;
};

void close_socket(socket_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

bool send_all(socket_t s, const void * data, std::size_t size) {
    const auto * p = static_cast<const char *>(data);
    while (size) {
        const int n = send(s, p, static_cast<int>(std::min<std::size_t>(size, INT_MAX)),
#ifdef _WIN32
            0
#else
            MSG_NOSIGNAL
#endif
        );
        if (n <= 0) return false;
        p += n;
        size -= static_cast<std::size_t>(n);
    }
    return true;
}

bool recv_all(socket_t s, void * data, std::size_t size) {
    auto * p = static_cast<char *>(data);
    while (size) {
        const int n = recv(s, p, static_cast<int>(std::min<std::size_t>(size, INT_MAX)), 0);
        if (n <= 0) return false;
        p += n;
        size -= static_cast<std::size_t>(n);
    }
    return true;
}

void put16(std::uint8_t * p, std::uint16_t v) { p[0] = v >> 8; p[1] = v; }
void put32(std::uint8_t * p, std::uint32_t v) {
    p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v;
}
void put64(std::uint8_t * p, std::uint64_t v) {
    for (int i = 7; i >= 0; --i) { p[i] = static_cast<std::uint8_t>(v); v >>= 8; }
}
std::uint16_t get16(const std::uint8_t * p) { return (std::uint16_t(p[0]) << 8) | p[1]; }
std::uint32_t get32(const std::uint8_t * p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16)
        | (std::uint32_t(p[2]) << 8) | p[3];
}
std::uint64_t get64(const std::uint8_t * p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

bool send_frame(socket_t s, const Frame & f) {
    if (f.payload.size() > max_payload) return false;
    std::uint8_t h[header_size]{};
    put32(h, magic); put16(h + 4, version); put16(h + 6, f.type);
    put64(h + 8, f.request); put32(h + 16, f.position); put32(h + 20, f.rows);
    put32(h + 24, f.cols); put16(h + 28, f.dtype); put64(h + 32, f.payload.size());
    return send_all(s, h, sizeof(h)) && send_all(s, f.payload.data(), f.payload.size());
}

bool recv_frame(socket_t s, Frame & f, std::string & why) {
    std::uint8_t h[header_size];
    if (!recv_all(s, h, sizeof(h))) return false;
    if (get32(h) != magic || get16(h + 4) != version || get16(h + 30) != 0) {
        why = "bad frame header"; return false;
    }
    const std::uint64_t bytes = get64(h + 32);
    if (bytes > max_payload) { why = "payload exceeds 64 MiB"; return false; }
    f.type = get16(h + 6); f.request = get64(h + 8); f.position = get32(h + 16);
    f.rows = get32(h + 20); f.cols = get32(h + 24); f.dtype = get16(h + 28);
    f.payload.resize(static_cast<std::size_t>(bytes));
    return recv_all(s, f.payload.data(), f.payload.size());
}

Frame error_frame(std::uint64_t request, const std::string & message) {
    Frame f; f.type = error; f.request = request;
    f.payload.assign(message.begin(), message.end());
    return f;
}

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::vector<std::uint8_t> & text) {
    const int n = -llama_tokenize(vocab, reinterpret_cast<const char *>(text.data()),
        static_cast<int>(text.size()), nullptr, 0, true, true);
    if (n <= 0) throw std::runtime_error("prompt tokenization failed");
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, reinterpret_cast<const char *>(text.data()),
            static_cast<int>(text.size()), tokens.data(), n, true, true) != n)
        throw std::runtime_error("prompt tokenization changed size");
    return tokens;
}

std::string piece(const llama_vocab * vocab, llama_token id) {
    int n = llama_token_to_piece(vocab, id, nullptr, 0, 0, false);
    if (n == 0) return {};
    if (n > 0) throw std::runtime_error("unexpected token piece probe result");
    std::string out(static_cast<std::size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, id, out.data(), static_cast<int>(out.size()), 0, false);
    if (n < 0) throw std::runtime_error("token conversion failed");
    out.resize(static_cast<std::size_t>(n));
    return out;
}

class Stage {
public:
    Stage(const std::string & path, int begin, int end, int ctx_size, int gpu_layers)
        : begin_(begin), end_(end) {
        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        mp.dan_stage_start = begin;
        mp.dan_stage_end = end;
        model_ = llama_model_load_from_file(path.c_str(), mp);
        if (!model_) throw std::runtime_error("model load failed");
        layers_ = llama_model_n_layer(model_);
        hidden_ = llama_model_n_embd(model_);
        if (begin < 0 || begin >= end || end > layers_) throw std::runtime_error("bad stage range");
        first_ = begin == 0;
        last_ = end == layers_;
        if (first_ == last_) throw std::runtime_error("v0 worker must be exactly stage A or B");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ctx_size; cp.n_batch = ctx_size; cp.n_ubatch = ctx_size;
        cp.embeddings = first_; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) throw std::runtime_error("context creation failed");
        if (last_) {
            llama_sampler_chain_params sp = llama_sampler_chain_default_params();
            sampler_ = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
        }
        std::fprintf(stderr, "DAN stage ready: layers %d..%d, hidden %d, role %s\n",
            begin_, end_ - 1, hidden_, first_ ? "A" : "B");
    }

    ~Stage() { if (sampler_) llama_sampler_free(sampler_); if (ctx_) llama_free(ctx_); if (model_) llama_model_free(model_); }

    Frame run(const Frame & in) {
        if (in.request == 0) throw std::runtime_error("request ID must be nonzero");
        if (request_ == 0) request_ = in.request;
        if (in.request != request_) throw std::runtime_error("request ID mismatch");
        if (first_) return run_first(in);
        return run_last(in);
    }

private:
    Frame run_first(const Frame & in) {
        std::vector<llama_token> tokens;
        if (in.type == prompt) {
            if (position_ != 0 || in.position != 0 || in.payload.empty()) throw std::runtime_error("bad prefill request");
            tokens = tokenize(llama_model_get_vocab(model_), in.payload);
        } else if (in.type == token) {
            if (in.position != position_ || in.payload.size() != 4) throw std::runtime_error("bad decode token frame");
            tokens.push_back(static_cast<llama_token>(get32(in.payload.data())));
        } else throw std::runtime_error("stage A expected PROMPT or TOKEN");
        if (tokens.size() > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("too many tokens");

        const auto start = std::chrono::steady_clock::now();
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(ctx_, batch) != 0) throw std::runtime_error("stage A decode failed");
        llama_synchronize(ctx_);
        const std::uint64_t compute = elapsed_ns(start);

        const std::size_t values = tokens.size() * static_cast<std::size_t>(hidden_);
        Frame out; out.type = activation; out.request = in.request; out.position = position_;
        out.rows = static_cast<std::uint32_t>(tokens.size()); out.cols = hidden_; out.dtype = f32le;
        out.payload.resize(8 + values * sizeof(float)); put64(out.payload.data(), compute);
        float * dst = reinterpret_cast<float *>(out.payload.data() + 8);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const float * src = llama_get_embeddings_ith(ctx_, static_cast<int32_t>(i));
            if (!src) throw std::runtime_error("stage A returned no hidden state");
            std::memcpy(dst + i * hidden_, src, static_cast<std::size_t>(hidden_) * sizeof(float));
        }
        position_ += static_cast<std::uint32_t>(tokens.size());
        std::fprintf(stderr, "request=%llu stage=A phase=%s shape=%ux%u compute_ms=%.3f bytes=%zu\n",
            static_cast<unsigned long long>(in.request), in.type == prompt ? "prefill" : "decode",
            out.rows, out.cols, compute / 1e6, out.payload.size() - 8);
        return out;
    }

    Frame run_last(const Frame & in) {
        if (in.type != activation || in.dtype != f32le || in.rows == 0
                || in.cols != static_cast<std::uint32_t>(hidden_) || in.position != position_)
            throw std::runtime_error("bad activation metadata");
        const std::uint64_t values = std::uint64_t(in.rows) * in.cols;
        if (values > (max_payload - 8) / sizeof(float)
                || in.payload.size() != 8 + values * sizeof(float))
            throw std::runtime_error("activation payload/shape mismatch");

        llama_batch batch = llama_batch_init(static_cast<int32_t>(in.rows), hidden_, 1);
        batch.n_tokens = static_cast<int32_t>(in.rows);
        std::memcpy(batch.embd, in.payload.data() + 8, static_cast<std::size_t>(values) * sizeof(float));
        for (std::uint32_t i = 0; i < in.rows; ++i) {
            batch.pos[i] = static_cast<llama_pos>(in.position + i);
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i + 1 == in.rows;
        }
        const auto start = std::chrono::steady_clock::now();
        const int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("stage B decode failed");
        const llama_token next = llama_sampler_sample(sampler_, ctx_, -1);
        llama_synchronize(ctx_);
        const std::uint64_t compute = elapsed_ns(start);
        const std::string text = piece(llama_model_get_vocab(model_), next);
        const bool eog = llama_vocab_is_eog(llama_model_get_vocab(model_), next);

        Frame out; out.type = result; out.request = in.request; out.position = in.position + in.rows;
        out.payload.resize(13 + text.size()); put32(out.payload.data(), static_cast<std::uint32_t>(next));
        put64(out.payload.data() + 4, compute); out.payload[12] = eog;
        std::memcpy(out.payload.data() + 13, text.data(), text.size());
        position_ += in.rows;
        std::fprintf(stderr, "request=%llu stage=B phase=%s shape=%ux%u compute_ms=%.3f token=%d\n",
            static_cast<unsigned long long>(in.request), in.rows > 1 ? "prefill" : "decode",
            in.rows, in.cols, compute / 1e6, next);
        return out;
    }

    int begin_, end_, layers_ = 0, hidden_ = 0;
    bool first_ = false, last_ = false;
    std::uint32_t position_ = 0;
    std::uint64_t request_ = 0;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    llama_sampler * sampler_ = nullptr;
};

socket_t listen_on(const std::string & host, int port) {
    socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == invalid_socket) throw std::runtime_error("socket failed");
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1
            || bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(s, 1) != 0) {
        close_socket(s); throw std::runtime_error("bind/listen failed");
    }
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model, host = "127.0.0.1";
    int begin = -1, end = -1, port = 0, ctx = 512, gpu_layers = 999;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", arg.c_str()); return 2; }
        const std::string value = argv[++i];
        if (arg == "--model") model = value; else if (arg == "--host") host = value;
        else if (arg == "--port") port = std::stoi(value); else if (arg == "--stage-start") begin = std::stoi(value);
        else if (arg == "--stage-end") end = std::stoi(value); else if (arg == "--ctx") ctx = std::stoi(value);
        else if (arg == "--gpu-layers") gpu_layers = std::stoi(value); else { std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); return 2; }
    }
    if (model.empty() || port < 1 || port > 65535 || ctx < 1) {
        std::fprintf(stderr, "usage: dan-stage-worker --model FILE --stage-start N --stage-end N --host IP --port N [--ctx N] [--gpu-layers N]\n");
        return 2;
    }
#ifdef _WIN32
    WSADATA data{}; if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    try {
        Stage stage(model, begin, end, ctx, gpu_layers);
        socket_t listener = listen_on(host, port);
        std::fprintf(stderr, "listening on %s:%d\n", host.c_str(), port);
        socket_t client = accept(listener, nullptr, nullptr); close_socket(listener);
        if (client == invalid_socket) throw std::runtime_error("accept failed");
        for (;;) {
            Frame in; std::string why;
            if (!recv_frame(client, in, why)) {
                if (!why.empty()) std::fprintf(stderr, "rejected frame: %s\n", why.c_str());
                break;
            }
            try { if (!send_frame(client, stage.run(in))) break; }
            catch (const std::exception & e) { send_frame(client, error_frame(in.request, e.what())); break; }
        }
        close_socket(client);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "dan-stage-worker: %s\n", e.what());
        return 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
