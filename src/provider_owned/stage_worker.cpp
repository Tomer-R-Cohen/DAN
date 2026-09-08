#include "llama.h"
#include "provider_owned/protocol.hpp"
#include "provider_owned/range_model.hpp"

#include <bit>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace po = dan::provider_owned;

namespace {

static_assert(std::endian::native == std::endian::little,
    "provider-owned v1 requires little-endian hosts for FP32 activations");

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
    const std::vector<std::uint8_t>& text) {
    const int count = -llama_tokenize(vocab,
        reinterpret_cast<const char*>(text.data()), static_cast<int>(text.size()),
        nullptr, 0, true, true);
    if (count <= 0) throw std::runtime_error("prompt tokenization failed");
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    if (llama_tokenize(vocab, reinterpret_cast<const char*>(text.data()),
        static_cast<int>(text.size()), tokens.data(), count, true, true) != count) {
        throw std::runtime_error("prompt tokenization changed size");
    }
    return tokens;
}

std::string piece(const llama_vocab* vocab, llama_token token) {
    int size = llama_token_to_piece(vocab, token, nullptr, 0, 0, false);
    if (size == 0) return {};
    if (size > 0) throw std::runtime_error("unexpected token piece probe result");
    std::string output(static_cast<std::size_t>(-size), '\0');
    size = llama_token_to_piece(vocab, token, output.data(),
        static_cast<int>(output.size()), 0, false);
    if (size < 0) throw std::runtime_error("token conversion failed");
    output.resize(static_cast<std::size_t>(size));
    return output;
}

class Session {
public:
    Session(llama_seq_id sequence, bool last) : sequence(sequence) {
        if (last) {
            sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (!sampler) throw std::runtime_error("session sampler creation failed");
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }
    }

    ~Session() { if (sampler) llama_sampler_free(sampler); }

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    llama_seq_id sequence = 0;
    llama_sampler* sampler = nullptr;
    std::uint32_t position = 0;
    std::uint64_t active_request = 0;
    std::uint64_t last_request = 0;
    bool has_prompt = false;
};

class Stage {
public:
    Stage(const std::string& path, int begin, int end, int context_size,
        int gpu_layers, std::size_t max_sessions)
        : begin_(begin), end_(end), context_size_(context_size),
          max_sessions_(max_sessions), sequence_used_(max_sessions, false),
          started_(std::chrono::steady_clock::now()) {
        ggml_backend_load_all();
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        params.dan_stage_start = begin;
        params.dan_stage_end = end;
        model_ = llama_model_load_from_file(path.c_str(), params);
        if (!model_) throw std::runtime_error("model load failed");
        layers_ = llama_model_n_layer(model_);
        hidden_ = llama_model_n_embd(model_);
        if (begin < 0 || begin >= end || end > layers_) {
            throw std::runtime_error("bad stage range");
        }
        first_ = begin == 0;
        last_ = end == layers_;
        if (first_ == last_) throw std::runtime_error("worker must be exactly stage A or B");
        if (max_sessions_ > std::numeric_limits<std::uint32_t>::max()
            || static_cast<std::uint64_t>(context_size_) * max_sessions_
                > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("session context pool is too large");
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(context_size_) * max_sessions_);
        context_params.n_batch = static_cast<std::uint32_t>(context_size_);
        context_params.n_ubatch = static_cast<std::uint32_t>(context_size_);
        context_params.n_seq_max = static_cast<std::uint32_t>(max_sessions_);
        context_params.embeddings = first_;
        context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        context_ = llama_init_from_model(model_, context_params);
        if (!context_) throw std::runtime_error("shared session context creation failed");
        const std::uint64_t head_size = static_cast<std::uint64_t>(hidden_)
            / static_cast<std::uint64_t>(llama_model_n_head(model_));
        kv_bytes_per_session_ = static_cast<std::uint64_t>(llama_n_ctx_seq(context_))
            * static_cast<std::uint64_t>(end_ - begin_)
            * head_size * static_cast<std::uint64_t>(llama_model_n_head_kv(model_))
            * 2 * sizeof(std::uint16_t);
        startup_ns_ = elapsed_ns(started_);
        std::fprintf(stderr,
            "DAN stage READY: layers %d..%d, hidden %d, role %s, startup_ms=%.3f\n",
            begin_, end_ - 1, hidden_, first_ ? "A" : "B", startup_ns_ / 1e6);
    }

    ~Stage() {
        sessions_.clear();
        if (context_) llama_free(context_);
        if (model_) llama_model_free(model_);
    }

    po::Frame handle(const po::Frame& input) {
        switch (input.type) {
        case po::Type::create_session: return create(input);
        case po::Type::reset_session: return reset(input);
        case po::Type::destroy_session: return destroy(input);
        case po::Type::end_request: return end_request(input);
        case po::Type::metrics: return metrics(input);
        case po::Type::shutdown: return shutdown(input);
        case po::Type::prompt:
        case po::Type::token:
        case po::Type::commit_token:
        case po::Type::activation:
        case po::Type::commit_activation:
            return execute(input);
        default: throw std::runtime_error("unexpected frame type for worker");
        }
    }

    bool shutting_down() const { return shutting_down_; }

    void coordinator_disconnected() {
        if (!sessions_.empty()) {
            std::fprintf(stderr, "coordinator disconnected; discarding %zu sessions\n",
                sessions_.size());
            sessions_.clear();
            std::fill(sequence_used_.begin(), sequence_used_.end(), false);
            llama_memory_clear(llama_get_memory(context_), true);
        }
        active_session_ = 0;
    }

private:
    Session& require_session(const po::Frame& input) {
        if (input.session == 0) throw std::runtime_error("session ID must be nonzero");
        const auto found = sessions_.find(input.session);
        if (found == sessions_.end()) throw std::runtime_error("unknown session ID");
        return *found->second;
    }

    static void require_control(const po::Frame& input, bool request_required = false) {
        if (!po::empty_control(input) || input.session == 0
            || (request_required ? input.request == 0 : input.request != 0)) {
            throw std::runtime_error("invalid control frame");
        }
    }

    static po::Frame ack(const po::Frame& input, std::uint32_t position = 0,
        std::uint64_t compute_ns = 0) {
        po::Frame output;
        output.type = po::Type::ack;
        output.session = input.session;
        output.request = input.request;
        output.position = position;
        if (compute_ns != 0) {
            output.payload.resize(8);
            po::put64(output.payload.data(), compute_ns);
        }
        return output;
    }

    po::Frame create(const po::Frame& input) {
        require_control(input);
        if (sessions_.contains(input.session)) throw std::runtime_error("session already exists");
        if (sessions_.size() >= max_sessions_) throw std::runtime_error("session limit reached");
        const auto free = std::find(sequence_used_.begin(), sequence_used_.end(), false);
        if (free == sequence_used_.end()) throw std::runtime_error("session limit reached");
        const llama_seq_id sequence = static_cast<llama_seq_id>(
            std::distance(sequence_used_.begin(), free));
        sessions_.emplace(input.session, std::make_unique<Session>(sequence, last_));
        sequence_used_[static_cast<std::size_t>(sequence)] = true;
        std::fprintf(stderr, "session=%llu created resident=%zu\n",
            static_cast<unsigned long long>(input.session), sessions_.size());
        return ack(input);
    }

    po::Frame reset(const po::Frame& input) {
        require_control(input);
        Session& old = require_session(input);
        if (old.active_request != 0) throw std::runtime_error("cannot reset active session");
        if (!llama_memory_seq_rm(llama_get_memory(context_), old.sequence, -1, -1)) {
            throw std::runtime_error("could not clear session KV");
        }
        if (old.sampler) llama_sampler_reset(old.sampler);
        old.position = 0;
        old.active_request = 0;
        old.last_request = 0;
        old.has_prompt = false;
        std::fprintf(stderr, "session=%llu reset\n",
            static_cast<unsigned long long>(input.session));
        return ack(input);
    }

    po::Frame destroy(const po::Frame& input) {
        require_control(input);
        Session& session = require_session(input);
        if (session.active_request != 0) throw std::runtime_error("cannot destroy active session");
        if (!llama_memory_seq_rm(llama_get_memory(context_), session.sequence, -1, -1)) {
            throw std::runtime_error("could not clear session KV");
        }
        sequence_used_[static_cast<std::size_t>(session.sequence)] = false;
        sessions_.erase(input.session);
        std::fprintf(stderr, "session=%llu destroyed resident=%zu\n",
            static_cast<unsigned long long>(input.session), sessions_.size());
        return ack(input);
    }

    po::Frame end_request(const po::Frame& input) {
        require_control(input, true);
        Session& session = require_session(input);
        if (session.active_request != input.request) {
            throw std::runtime_error("request ID mismatch at end");
        }
        if (!session.has_prompt) throw std::runtime_error("request ended before prompt");
        session.last_request = input.request;
        session.active_request = 0;
        session.has_prompt = false;
        active_session_ = 0;
        ++requests_served_;
        return ack(input, session.position);
    }

    po::Frame metrics(const po::Frame& input) const {
        if (!po::empty_control(input) || input.session != 0 || input.request != 0) {
            throw std::runtime_error("invalid metrics frame");
        }
        po::Frame output;
        output.type = po::Type::metrics;
        const std::string json = "{\"role\":\"" + std::string(first_ ? "A" : "B")
            + "\",\"requests_served\":" + std::to_string(requests_served_)
            + ",\"tokens_processed\":" + std::to_string(tokens_processed_)
            + ",\"tokens_generated\":" + std::to_string(tokens_generated_)
            + ",\"sessions_resident\":" + std::to_string(sessions_.size())
            + ",\"kv_memory_bytes\":"
                + std::to_string(kv_bytes_per_session_ * max_sessions_)
            + ",\"kv_memory_bytes_resident_estimate\":"
                + std::to_string(kv_bytes_per_session_ * sessions_.size())
            + ",\"kv_bytes_per_session\":" + std::to_string(kv_bytes_per_session_)
            + ",\"startup_ms\":" + std::to_string(startup_ns_ / 1e6)
            + ",\"model_reload_count\":1}";
        output.payload.assign(json.begin(), json.end());
        return output;
    }

    po::Frame shutdown(const po::Frame& input) {
        if (!po::empty_control(input) || input.session != 0 || input.request != 0) {
            throw std::runtime_error("invalid shutdown frame");
        }
        for (const auto& [id, session] : sessions_) {
            (void) id;
            if (session->active_request != 0) {
                throw std::runtime_error("cannot shut down with an active request");
            }
        }
        shutting_down_ = true;
        std::fprintf(stderr,
            "graceful shutdown requests=%llu tokens=%llu sessions=%zu model_reloads=1\n",
            static_cast<unsigned long long>(requests_served_),
            static_cast<unsigned long long>(tokens_generated_), sessions_.size());
        po::Frame output;
        output.type = po::Type::ack;
        return output;
    }

    po::Frame execute(const po::Frame& input) {
        if (input.request == 0) throw std::runtime_error("request ID must be nonzero");
        Session& session = require_session(input);
        // ponytail: one global active session; replace with a bounded executor when concurrency starts.
        if (active_session_ != 0 && active_session_ != input.session) {
            throw std::runtime_error("another session is actively executing");
        }
        if (session.active_request == 0) {
            if (input.request <= session.last_request) {
                throw std::runtime_error("request ID is not increasing");
            }
            session.active_request = input.request;
            active_session_ = input.session;
        }
        if (session.active_request != input.request) throw std::runtime_error("request ID mismatch");
        return first_ ? run_first(input, session) : run_last(input, session);
    }

    po::Frame run_first(const po::Frame& input, Session& session) {
        std::vector<llama_token> tokens;
        if (input.type == po::Type::prompt) {
            if (input.position != session.position || input.payload.empty()
                || input.rows != 0 || input.cols != 0 || input.dtype != po::DType::none) {
                throw std::runtime_error("bad prompt frame");
            }
            if (session.has_prompt) throw std::runtime_error("duplicate prompt in request");
            session.has_prompt = true;
            tokens = tokenize(llama_model_get_vocab(model_), input.payload);
        } else if (input.type == po::Type::token || input.type == po::Type::commit_token) {
            if (input.position != session.position || input.payload.size() != 4
                || input.rows != 0 || input.cols != 0 || input.dtype != po::DType::none) {
                throw std::runtime_error("bad token frame");
            }
            if (!session.has_prompt) throw std::runtime_error("token received before prompt");
            tokens.push_back(static_cast<llama_token>(po::get32(input.payload.data())));
        } else {
            throw std::runtime_error("stage A expected prompt or token");
        }
        if (tokens.size() > std::numeric_limits<std::uint32_t>::max()
            || tokens.size() > static_cast<std::size_t>(context_size_ - session.position)) {
            throw std::runtime_error("session context exhausted");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            batch.token[index] = tokens[index];
            batch.pos[index] = static_cast<llama_pos>(session.position + index);
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = session.sequence;
            batch.logits[index] = true;
        }
        const auto start = std::chrono::steady_clock::now();
        if (llama_decode(context_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("stage A decode failed");
        }
        llama_synchronize(context_);
        const std::uint64_t compute = elapsed_ns(start);

        const std::size_t values = tokens.size() * static_cast<std::size_t>(hidden_);
        po::Frame output;
        output.type = input.type == po::Type::commit_token
            ? po::Type::commit_activation : po::Type::activation;
        output.session = input.session;
        output.request = input.request;
        output.position = session.position;
        output.rows = static_cast<std::uint32_t>(tokens.size());
        output.cols = static_cast<std::uint32_t>(hidden_);
        output.dtype = po::DType::f32le;
        output.payload.resize(8 + values * sizeof(float));
        po::put64(output.payload.data(), compute);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            const float* source = llama_get_embeddings_ith(context_,
                static_cast<int32_t>(index));
            if (!source) throw std::runtime_error("stage A returned no hidden state");
            std::memcpy(output.payload.data() + 8
                    + index * static_cast<std::size_t>(hidden_) * sizeof(float), source,
                static_cast<std::size_t>(hidden_) * sizeof(float));
        }
        llama_batch_free(batch);
        session.position += static_cast<std::uint32_t>(tokens.size());
        tokens_processed_ += tokens.size();
        std::fprintf(stderr,
            "session=%llu request=%llu stage=A phase=%s position=%u shape=%ux%u compute_ms=%.3f bytes=%zu\n",
            static_cast<unsigned long long>(input.session),
            static_cast<unsigned long long>(input.request),
            input.type == po::Type::prompt ? "prefill" :
                (input.type == po::Type::commit_token ? "commit" : "decode"),
            input.position, output.rows, output.cols, compute / 1e6,
            output.payload.size() - 8);
        return output;
    }

    po::Frame run_last(const po::Frame& input, Session& session) {
        if ((input.type != po::Type::activation
                && input.type != po::Type::commit_activation)
            || input.dtype != po::DType::f32le || input.rows == 0
            || input.cols != static_cast<std::uint32_t>(hidden_)
            || input.position != session.position) {
            throw std::runtime_error("bad activation metadata");
        }
        if (input.type == po::Type::commit_activation && !session.has_prompt) {
            throw std::runtime_error("commit received before activation");
        }
        if (input.type == po::Type::activation) session.has_prompt = true;
        const std::uint64_t values = std::uint64_t(input.rows) * input.cols;
        if (values > (po::max_payload - 8) / sizeof(float)
            || input.payload.size() != 8 + values * sizeof(float)
            || input.rows > static_cast<std::uint32_t>(context_size_ - session.position)) {
            throw std::runtime_error("activation payload/shape mismatch");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(input.rows), hidden_, 1);
        batch.n_tokens = static_cast<int32_t>(input.rows);
        std::memcpy(batch.embd, input.payload.data() + 8,
            static_cast<std::size_t>(values) * sizeof(float));
        for (std::uint32_t index = 0; index < input.rows; ++index) {
            batch.pos[index] = static_cast<llama_pos>(input.position + index);
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = session.sequence;
            batch.logits[index] = input.type == po::Type::activation && index + 1 == input.rows;
        }
        const auto start = std::chrono::steady_clock::now();
        const int result = llama_decode(context_, batch);
        llama_batch_free(batch);
        if (result != 0) throw std::runtime_error("stage B decode failed");
        session.position += input.rows;
        tokens_processed_ += input.rows;

        if (input.type == po::Type::commit_activation) {
            llama_synchronize(context_);
            const std::uint64_t compute = elapsed_ns(start);
            std::fprintf(stderr,
                "session=%llu request=%llu stage=B phase=commit position=%u compute_ms=%.3f\n",
                static_cast<unsigned long long>(input.session),
                static_cast<unsigned long long>(input.request), input.position, compute / 1e6);
            return ack(input, session.position, compute);
        }

        const llama_token next = llama_sampler_sample(session.sampler, context_, -1);
        llama_synchronize(context_);
        const std::uint64_t compute = elapsed_ns(start);
        const std::string text = piece(llama_model_get_vocab(model_), next);
        const bool eog = llama_vocab_is_eog(llama_model_get_vocab(model_), next);
        ++tokens_generated_;

        po::Frame output;
        output.type = po::Type::result;
        output.session = input.session;
        output.request = input.request;
        output.position = session.position;
        output.payload.resize(13 + text.size());
        po::put32(output.payload.data(), static_cast<std::uint32_t>(next));
        po::put64(output.payload.data() + 4, compute);
        output.payload[12] = eog ? 1 : 0;
        std::memcpy(output.payload.data() + 13, text.data(), text.size());
        std::fprintf(stderr,
            "session=%llu request=%llu stage=B phase=%s position=%u shape=%ux%u compute_ms=%.3f token=%d\n",
            static_cast<unsigned long long>(input.session),
            static_cast<unsigned long long>(input.request),
            input.rows > 1 ? "prefill" : "decode", input.position,
            input.rows, input.cols, compute / 1e6, next);
        return output;
    }

    int begin_ = 0;
    int end_ = 0;
    int layers_ = 0;
    int hidden_ = 0;
    int context_size_ = 0;
    std::size_t max_sessions_ = 0;
    bool first_ = false;
    bool last_ = false;
    bool shutting_down_ = false;
    std::chrono::steady_clock::time_point started_;
    std::uint64_t startup_ns_ = 0;
    std::uint64_t requests_served_ = 0;
    std::uint64_t tokens_processed_ = 0;
    std::uint64_t tokens_generated_ = 0;
    std::uint64_t kv_bytes_per_session_ = 0;
    std::uint64_t active_session_ = 0;
    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    std::vector<bool> sequence_used_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Session>> sessions_;
};

po::socket_t listen_on(const std::string& host, int port) {
    const po::socket_t listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == po::invalid_socket) throw std::runtime_error("socket failed");
    const int enabled = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
        reinterpret_cast<const char*>(&enabled), sizeof(enabled));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1
        || bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
        || listen(listener, 4) != 0) {
        po::close_socket(listener);
        throw std::runtime_error("bind/listen failed");
    }
    return listener;
}

} // namespace

int main(int argc, char** argv) {
    std::string model;
    std::string model_url;
    std::string model_revision;
    std::string model_sha256;
    std::string host = "127.0.0.1";
    int begin = -1;
    int end = -1;
    int port = 0;
    int context = 512;
    int gpu_layers = 999;
    int max_sessions = 8;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
            const std::string value = argv[++index];
            if (option == "--model") model = value;
            else if (option == "--model-url") model_url = value;
            else if (option == "--model-revision") model_revision = value;
            else if (option == "--model-sha256") model_sha256 = value;
            else if (option == "--host") host = value;
            else if (option == "--port") port = std::stoi(value);
            else if (option == "--stage-start") begin = std::stoi(value);
            else if (option == "--stage-end") end = std::stoi(value);
            else if (option == "--ctx") context = std::stoi(value);
            else if (option == "--gpu-layers") gpu_layers = std::stoi(value);
            else if (option == "--max-sessions") max_sessions = std::stoi(value);
            else throw std::runtime_error("unknown option: " + option);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dan-stage-worker: %s\n", error.what());
        return 2;
    }
    if (model.empty() || port < 1 || port > 65535 || context < 1 || max_sessions < 1) {
        std::fprintf(stderr,
            "usage: dan-stage-worker --model FILE --stage-start N --stage-end N --host IP --port N [--model-url URL --model-revision SHA --model-sha256 SHA] [--ctx N] [--gpu-layers N] [--max-sessions N]\n");
        return 2;
    }
    const bool range_model = !model_url.empty() || !model_revision.empty() || !model_sha256.empty();
    if (range_model && (model_url.empty() || model_revision.empty() || model_sha256.empty())) {
        std::fprintf(stderr, "dan-stage-worker: range-backed model options must be supplied together\n");
        return 2;
    }
#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 1;
#endif
    int exit_code = 0;
    try {
        if (range_model) {
            po::RangeModelStats stats;
            std::string error;
            if (!po::prepare_range_model({model_url, model_revision, model_sha256,
                    model, begin, end}, stats, error)) {
                throw std::runtime_error("range-backed model: " + error);
            }
            std::fprintf(stderr,
                "range model %s logical=%llu physical=%llu downloaded=%llu tensors=%llu shared=%llu cache=%s\n",
                model.c_str(), static_cast<unsigned long long>(stats.logical_bytes),
                static_cast<unsigned long long>(stats.physical_bytes),
                static_cast<unsigned long long>(stats.downloaded_bytes),
                static_cast<unsigned long long>(stats.tensors_present),
                static_cast<unsigned long long>(stats.shared_bytes),
                stats.cache_reused ? "reused" : "downloaded");
        }
        Stage stage(model, begin, end, context, gpu_layers,
            static_cast<std::size_t>(max_sessions));
        const po::socket_t listener = listen_on(host, port);
        std::fprintf(stderr, "listening on %s:%d\n", host.c_str(), port);
        while (!stage.shutting_down()) {
            const po::socket_t client = accept(listener, nullptr, nullptr);
            if (client == po::invalid_socket) throw std::runtime_error("accept failed");
            while (!stage.shutting_down()) {
                po::Frame input;
                std::string error;
                if (!po::recv_frame(client, input, error)) {
                    std::fprintf(stderr, "coordinator connection closed: %s\n", error.c_str());
                    stage.coordinator_disconnected();
                    break;
                }
                try {
                    const po::Frame output = stage.handle(input);
                    if (!po::send_frame(client, output, error)) {
                        std::fprintf(stderr, "coordinator connection closed: %s\n", error.c_str());
                        stage.coordinator_disconnected();
                        break;
                    }
                } catch (const std::exception& exception) {
                    const po::Frame output = po::error_frame(input, exception.what());
                    po::send_frame(client, output, error);
                    stage.coordinator_disconnected();
                    std::fprintf(stderr, "rejected frame: %s\n", exception.what());
                    break;
                }
            }
            po::close_socket(client);
        }
        po::close_socket(listener);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dan-stage-worker: %s\n", error.what());
        exit_code = 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return exit_code;
}
