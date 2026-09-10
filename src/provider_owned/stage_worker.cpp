#include "llama.h"
#include "ggml-backend.h"
#include "provider_owned/protocol.hpp"
#include "provider_owned/formation.hpp"
#include "provider_owned/range_model.hpp"
#include "provider_ui.hpp"
#include "platform.hpp"

#include <bit>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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
        // Every non-final stage must expose its boundary hidden state.
        context_params.embeddings = !last_;
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
            begin_, end_ - 1, hidden_, first_ && last_ ? "single"
                : (first_ ? "first" : (last_ ? "last" : "middle")),
            startup_ns_ / 1e6);
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
        case po::Type::rollback: return rollback(input);
        case po::Type::metrics: return metrics(input);
        case po::Type::shutdown: return shutdown(input);
        case po::Type::prompt:
        case po::Type::token:
        case po::Type::commit_token:
        case po::Type::activation:
        case po::Type::speculative_activation:
        case po::Type::commit_activation:
            return execute(input);
        default: throw std::runtime_error("unexpected frame type for worker");
        }
    }

    bool shutting_down() const { return shutting_down_; }
    std::uint64_t tokens_processed() const { return tokens_processed_; }
    std::uint64_t requests_served() const { return requests_served_; }

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

    po::Frame rollback(const po::Frame& input) {
        Session& session = require_session(input);
        if (input.request == 0 || session.active_request != input.request
            || input.position > session.position || input.rows != 0 || input.cols != 0
            || input.dtype != po::DType::none || !input.payload.empty()) {
            throw std::runtime_error("invalid rollback frame");
        }
        if (!llama_memory_seq_rm(llama_get_memory(context_), session.sequence,
                input.position, -1)) {
            throw std::runtime_error("could not roll back session KV");
        }
        session.position = input.position;
        return ack(input, session.position);
    }

    po::Frame metrics(const po::Frame& input) const {
        if (!po::empty_control(input) || input.session != 0 || input.request != 0) {
            throw std::runtime_error("invalid metrics frame");
        }
        po::Frame output;
        output.type = po::Type::metrics;
        const std::string role = first_ && last_ ? "single"
            : (first_ ? "first" : (last_ ? "last" : "middle"));
        const std::string json = "{\"role\":\"" + role
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
        if (first_) return run_first(input, session);
        if (last_) return run_last(input, session);
        return run_middle(input, session);
    }

    po::Frame run_first(const po::Frame& input, Session& session) {
        std::vector<llama_token> tokens;
        const bool speculative = input.type == po::Type::token && input.rows != 0;
        if (input.type == po::Type::prompt) {
            if (input.position != session.position || input.payload.empty()
                || input.rows != 0 || input.cols != 0 || input.dtype != po::DType::none) {
                throw std::runtime_error("bad prompt frame");
            }
            if (session.has_prompt) throw std::runtime_error("duplicate prompt in request");
            session.has_prompt = true;
            tokens = tokenize(llama_model_get_vocab(model_), input.payload);
        } else if (input.type == po::Type::token || input.type == po::Type::commit_token) {
            const std::size_t count = speculative ? input.rows : 1;
            if (input.position != session.position || input.payload.size() != count * 4
                || (!speculative && input.rows != 0) || input.cols != 0
                || input.dtype != po::DType::none) {
                throw std::runtime_error("bad token frame");
            }
            if (!session.has_prompt) throw std::runtime_error("token received before prompt");
            tokens.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                tokens.push_back(static_cast<llama_token>(
                    po::get32(input.payload.data() + index * 4)));
            }
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
            batch.logits[index] = speculative || !last_ || (input.type != po::Type::commit_token
                && index + 1 == tokens.size());
        }
        const auto start = std::chrono::steady_clock::now();
        if (llama_decode(context_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("stage A decode failed");
        }
        llama_synchronize(context_);
        const std::uint64_t compute = elapsed_ns(start);

        if (last_) {
            llama_batch_free(batch);
            session.position += static_cast<std::uint32_t>(tokens.size());
            tokens_processed_ += tokens.size();
            if (input.type == po::Type::commit_token) {
                return ack(input, session.position, compute);
            }
            if (speculative) {
                po::Frame output;
                output.type = po::Type::result;
                output.session = input.session;
                output.request = input.request;
                output.position = session.position;
                output.rows = static_cast<std::uint32_t>(tokens.size());
                output.payload.resize(8 + tokens.size() * 4);
                po::put64(output.payload.data(), compute);
                for (std::size_t index = 0; index < tokens.size(); ++index) {
                    po::put32(output.payload.data() + 8 + index * 4,
                        static_cast<std::uint32_t>(llama_sampler_sample(
                            session.sampler, context_, static_cast<int32_t>(index))));
                }
                tokens_generated_ += tokens.size();
                return output;
            }
            const llama_token next = llama_sampler_sample(session.sampler, context_, -1);
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
                "session=%llu request=%llu stage=single phase=%s position=%u compute_ms=%.3f token=%d\n",
                static_cast<unsigned long long>(input.session),
                static_cast<unsigned long long>(input.request),
                tokens.size() > 1 ? "prefill" : "decode", input.position,
                compute / 1e6, next);
            return output;
        }

        const std::size_t values = tokens.size() * static_cast<std::size_t>(hidden_);
        po::Frame output;
        output.type = input.type == po::Type::commit_token
            ? po::Type::commit_activation : (speculative
                ? po::Type::speculative_activation : po::Type::activation);
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
                && input.type != po::Type::speculative_activation
                && input.type != po::Type::commit_activation)
            || input.dtype != po::DType::f32le || input.rows == 0
            || input.cols != static_cast<std::uint32_t>(hidden_)
            || input.position != session.position) {
            throw std::runtime_error("bad activation metadata");
        }
        if (input.type == po::Type::commit_activation && !session.has_prompt) {
            throw std::runtime_error("commit received before activation");
        }
        if (input.type == po::Type::activation
            || input.type == po::Type::speculative_activation) session.has_prompt = true;
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
            batch.logits[index] = input.type == po::Type::speculative_activation
                || (input.type == po::Type::activation && index + 1 == input.rows);
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

        if (input.type == po::Type::speculative_activation) {
            llama_synchronize(context_);
            const std::uint64_t compute = elapsed_ns(start);
            po::Frame output;
            output.type = po::Type::result;
            output.session = input.session;
            output.request = input.request;
            output.position = session.position;
            output.rows = input.rows;
            output.payload.resize(8 + static_cast<std::size_t>(input.rows) * 4);
            po::put64(output.payload.data(), compute);
            for (std::uint32_t index = 0; index < input.rows; ++index) {
                po::put32(output.payload.data() + 8 + static_cast<std::size_t>(index) * 4,
                    static_cast<std::uint32_t>(llama_sampler_sample(
                        session.sampler, context_, static_cast<int32_t>(index))));
            }
            tokens_generated_ += input.rows;
            return output;
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

    po::Frame run_middle(const po::Frame& input, Session& session) {
        if ((input.type != po::Type::activation
                && input.type != po::Type::speculative_activation
                && input.type != po::Type::commit_activation)
            || input.dtype != po::DType::f32le || input.rows == 0
            || input.cols != static_cast<std::uint32_t>(hidden_)
            || input.position != session.position) {
            throw std::runtime_error("bad middle-stage activation metadata");
        }
        if (input.type == po::Type::commit_activation && !session.has_prompt) {
            throw std::runtime_error("commit received before activation");
        }
        if (input.type == po::Type::activation
            || input.type == po::Type::speculative_activation) session.has_prompt = true;
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
            batch.logits[index] = true;
        }
        const auto start = std::chrono::steady_clock::now();
        const int result = llama_decode(context_, batch);
        if (result != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("middle-stage decode failed");
        }
        llama_synchronize(context_);
        const std::uint64_t compute = elapsed_ns(start);

        po::Frame output = input;
        output.payload.resize(8 + static_cast<std::size_t>(values) * sizeof(float));
        po::put64(output.payload.data(), compute);
        for (std::uint32_t index = 0; index < input.rows; ++index) {
            const float* source = llama_get_embeddings_ith(context_,
                static_cast<int32_t>(index));
            if (!source) {
                llama_batch_free(batch);
                throw std::runtime_error("middle stage returned no hidden state");
            }
            std::memcpy(output.payload.data() + 8
                    + static_cast<std::size_t>(index) * hidden_ * sizeof(float), source,
                static_cast<std::size_t>(hidden_) * sizeof(float));
        }
        llama_batch_free(batch);
        session.position += input.rows;
        tokens_processed_ += input.rows;
        std::fprintf(stderr,
            "session=%llu request=%llu stage=middle layers=%d..%d phase=%s position=%u shape=%ux%u compute_ms=%.3f bytes=%zu\n",
            static_cast<unsigned long long>(input.session),
            static_cast<unsigned long long>(input.request), begin_, end_ - 1,
            input.type == po::Type::commit_activation ? "commit" :
                (input.rows > 1 ? "prefill" : "decode"), input.position,
            output.rows, output.cols, compute / 1e6, output.payload.size() - 8);
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

po::socket_t connect_to(std::string_view endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == endpoint.size()) {
        throw std::runtime_error("invalid coordinator endpoint");
    }
    const std::string host(endpoint.substr(0, colon));
    const std::string port(endpoint.substr(colon + 1));
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        throw std::runtime_error("could not resolve coordinator");
    }
    po::socket_t result = po::invalid_socket;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        result = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (result != po::invalid_socket
            && connect(result, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) break;
        if (result != po::invalid_socket) po::close_socket(result);
        result = po::invalid_socket;
    }
    freeaddrinfo(addresses);
    if (result == po::invalid_socket) throw std::runtime_error("could not connect to coordinator");
    return result;
}

po::socket_t wait_for_coordinator(std::string_view endpoint, dan::ProviderTerminalUi* ui) {
    bool announced = false;
    while (!dan::platform::stop_requested()) {
        try {
            return connect_to(endpoint);
        } catch (const std::exception&) {
            if (!announced) {
                std::fprintf(stderr, "coordinator unavailable; waiting for %.*s\n",
                    static_cast<int>(endpoint.size()), endpoint.data());
                announced = true;
            }
            if (ui) ui->update([&](auto& state) {
                state.network_connected = false;
                state.status = announced ? dan::ProviderUiStatus::reconnecting
                    : dan::ProviderUiStatus::connecting;
                state.message = "Waiting for the DAN coordinator at " + std::string(endpoint);
            });
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    return po::invalid_socket;
}

std::size_t gpu_free_mib() {
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        std::size_t free = 0, total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        return free / (1024 * 1024);
    }
    return 0;
}

bool redirect_diagnostics(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
#ifdef _WIN32
    FILE* redirected = nullptr;
    return _wfreopen_s(&redirected, path.c_str(), L"a", stderr) == 0;
#else
    return std::freopen(path.c_str(), "a", stderr) != nullptr;
#endif
}

void print_range_stats(const std::filesystem::path& path, const po::RangeModelStats& stats) {
    std::fprintf(stderr,
        "range model %s logical=%llu physical=%llu downloaded=%llu tensors=%llu shared=%llu cache=%s\n",
        path.string().c_str(), static_cast<unsigned long long>(stats.logical_bytes),
        static_cast<unsigned long long>(stats.physical_bytes),
        static_cast<unsigned long long>(stats.downloaded_bytes),
        static_cast<unsigned long long>(stats.tensors_present),
        static_cast<unsigned long long>(stats.shared_bytes),
        stats.cache_reused ? "reused" : "downloaded");
}

} // namespace

int main(int argc, char** argv) {
    std::string model;
    std::string model_url;
    std::string model_revision;
    std::string model_sha256;
    std::string host = "127.0.0.1";
    std::string coordinator;
    std::string provider_id;
    std::string gpu_name;
    std::filesystem::path cache_dir;
    std::uint64_t offered_vram_mib = 0;
    int begin = -1;
    int end = -1;
    int port = 0;
    int context = 512;
    int gpu_layers = 999;
    int max_sessions = 8;
    bool tui = false;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--tui") { tui = true; continue; }
            if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
            const std::string value = argv[++index];
            if (option == "--model") model = value;
            else if (option == "--model-url") model_url = value;
            else if (option == "--model-revision") model_revision = value;
            else if (option == "--model-sha256") model_sha256 = value;
            else if (option == "--coordinator") coordinator = value;
            else if (option == "--provider-id") provider_id = value;
            else if (option == "--gpu") gpu_name = value;
            else if (option == "--vram-mib") offered_vram_mib = std::stoull(value);
            else if (option == "--cache-dir") cache_dir = value;
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
    const bool generic = !coordinator.empty();
    if (generic && (gpu_name.empty() || offered_vram_mib == 0)) {
        ggml_backend_load_all();
        for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            std::size_t free = 0, total = 0;
            ggml_backend_dev_memory(device, &free, &total);
            if (gpu_name.empty()) gpu_name = ggml_backend_dev_description(device);
            if (offered_vram_mib == 0) offered_vram_mib = free / (1024 * 1024);
            break;
        }
    }
    if ((!generic && (model.empty() || port < 1 || port > 65535))
        || (generic && (provider_id.empty() || gpu_name.empty() || cache_dir.empty()
            || offered_vram_mib == 0)) || context < 1 || max_sessions < 1) {
        std::fprintf(stderr,
            "usage: dan-stage-worker (--coordinator HOST:PORT --provider-id ID --gpu NAME --vram-mib N --cache-dir DIR | --model FILE --stage-start N --stage-end N --host IP --port N) [range/model options]\n");
        return 2;
    }
    const bool range_model = !model_url.empty() || !model_revision.empty() || !model_sha256.empty();
    if (range_model && (model_url.empty() || model_revision.empty() || model_sha256.empty())) {
        std::fprintf(stderr, "dan-stage-worker: range-backed model options must be supplied together\n");
        return 2;
    }
    std::string platform_error;
    if (!dan::platform::initialize(platform_error)) {
        std::fprintf(stderr, "dan-stage-worker: %s\n", platform_error.c_str());
        return 1;
    }
    struct PlatformCleanup { ~PlatformCleanup() { dan::platform::cleanup(); } } cleanup;
    dan::platform::install_stop_handlers();
    dan::platform::configure_output();

    const auto diagnostics = dan::platform::data_directory()
        / "logs" / "provider-owned.log";
    if (tui && generic && !redirect_diagnostics(diagnostics)) {
        std::fprintf(stderr, "dan-stage-worker: could not open diagnostics log\n");
        tui = false;
    }
    dan::ProviderUiState initial_ui;
    initial_ui.gpu_name = gpu_name;
    initial_ui.offered_vram_mib = static_cast<std::size_t>(offered_vram_mib);
    initial_ui.status = dan::ProviderUiStatus::connecting;
    initial_ui.message = "Connecting to DAN automatically...";
    initial_ui.diagnostics = diagnostics.string();
    dan::ProviderTerminalUi terminal_ui(std::move(initial_ui), tui && generic);
    dan::ProviderTerminalUi* ui = tui && generic ? &terminal_ui : nullptr;
    int exit_code = 0;
    try {
        if (generic) {
            bool shutdown = false;
            while (!shutdown && !dan::platform::stop_requested()) {
                const po::socket_t coordinator_socket = wait_for_coordinator(coordinator, ui);
                if (coordinator_socket == po::invalid_socket) break;
                std::jthread stop_watcher([coordinator_socket](std::stop_token stop) {
                    while (!stop.stop_requested() && !dan::platform::stop_requested()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    if (dan::platform::stop_requested()) {
                        dan::platform::shutdown_socket(coordinator_socket);
                    }
                });
                std::string error;
                try {
                    po::Frame available;
                    available.type = po::Type::provider_available;
                    const std::string capabilities = po::available_message(
                        {provider_id, gpu_name, offered_vram_mib});
                    available.payload.assign(capabilities.begin(), capabilities.end());
                    if (!po::send_frame(coordinator_socket, available, error)) {
                        throw std::runtime_error(error);
                    }
                    if (ui) ui->update([](auto& state) {
                        state.network_connected = true;
                        state.status = dan::ProviderUiStatus::available;
                        state.message = "Connected. Waiting for useful work...";
                    });
                    std::unique_ptr<Stage> stage;
                    const std::size_t free_before_load = gpu_free_mib();
                    while (!dan::platform::stop_requested()) {
                        po::Frame input;
                        if (!po::recv_frame(coordinator_socket, input, error)) {
                            throw std::runtime_error(error);
                        }
                        if (input.type == po::Type::assign_stage) {
                            if (stage) throw std::runtime_error("stage already loaded");
                            po::ModelAssignment assignment;
                            const std::string text(input.payload.begin(), input.payload.end());
                            if (!po::parse_assignment(text, assignment)) {
                                throw std::runtime_error("invalid stage assignment");
                            }
                            const auto path = cache_dir / (assignment.model_id + "-"
                                + std::to_string(assignment.begin) + "-"
                                + std::to_string(assignment.end) + ".gguf");
                            po::RangeModelStats stats;
                            if (ui) ui->update([&](auto& state) {
                                state.status = dan::ProviderUiStatus::downloading;
                                state.model_name = assignment.model_id;
                                state.stage = "Assigned layers " + std::to_string(assignment.begin)
                                    + "-" + std::to_string(assignment.end - 1);
                                state.message = "Downloading and verifying required model data...";
                                state.cache_status.clear();
                                state.download_percent = 0;
                            });
                            const auto progress = [&](std::uint64_t downloaded,
                                std::uint64_t total, std::uint64_t speed) {
                                if (ui) ui->update([&](auto& state) {
                                    state.downloaded_bytes = static_cast<std::size_t>(downloaded);
                                    state.download_total_bytes = static_cast<std::size_t>(total);
                                    state.download_bytes_per_second = static_cast<std::size_t>(speed);
                                    state.download_percent = total == 0 ? 0
                                        : static_cast<int>(downloaded * 100 / total);
                                });
                            };
                            std::fprintf(stderr, "Downloading required model data...\n");
                            if (!po::prepare_range_model({assignment.url, assignment.revision,
                                    assignment.sha256, path, assignment.begin, assignment.end,
                                    progress}, stats, error)) {
                                throw std::runtime_error("range-backed model: " + error);
                            }
                            print_range_stats(path, stats);
                            if (ui) ui->update([&](auto& state) {
                                state.status = dan::ProviderUiStatus::loading;
                                state.download_percent = -1;
                                state.cache_status = stats.cache_reused ? "Reused and verified"
                                    : "Downloaded and verified";
                                state.message = "Loading assigned layers onto the GPU...";
                            });
                            stage = std::make_unique<Stage>(path.string(), assignment.begin,
                                assignment.end, static_cast<int>(assignment.context), gpu_layers,
                                assignment.sessions);
                            po::Frame ready;
                            ready.type = po::Type::stage_ready;
                            ready.payload.assign(provider_id.begin(), provider_id.end());
                            if (!po::send_frame(coordinator_socket, ready, error)) {
                                throw std::runtime_error(error);
                            }
                            if (ui) ui->update([&](auto& state) {
                                state.status = dan::ProviderUiStatus::contributing;
                                state.message = "Ready. Waiting for inference requests...";
                                const std::size_t free_now = gpu_free_mib();
                                state.used_vram_mib = free_before_load > free_now
                                    ? free_before_load - free_now : 0;
                            });
                            continue;
                        }
                        if (input.type == po::Type::unload_stage) {
                            stage.reset();
                            if (ui) ui->update([](auto& state) {
                                state.status = dan::ProviderUiStatus::available;
                                state.model_name.clear(); state.stage.clear();
                                state.used_vram_mib = 0;
                                state.message = "Stage unloaded. Waiting for useful work...";
                            });
                            po::Frame output; output.type = po::Type::ack;
                            if (!po::send_frame(coordinator_socket, output, error)) {
                                throw std::runtime_error(error);
                            }
                            continue;
                        }
                        if (!stage) throw std::runtime_error("provider has no stage assignment");
                        po::Frame output;
                        try { output = stage->handle(input); }
                        catch (const std::exception& exception) {
                            output = po::error_frame(input, exception.what());
                        }
                        if (!po::send_frame(coordinator_socket, output, error)) {
                            throw std::runtime_error(error);
                        }
                        if (ui) ui->update([&](auto& state) {
                            state.tokens_participated = static_cast<std::size_t>(
                                stage->tokens_processed());
                            state.requests_participated = static_cast<std::size_t>(
                                stage->requests_served());
                            state.message = "Contributing to DAN inference...";
                        });
                        if (stage->shutting_down()) { shutdown = true; break; }
                    }
                } catch (const std::exception& failure) {
                    std::fprintf(stderr, "coordinator connection closed: %s\n", failure.what());
                    if (ui) ui->update([&](auto& state) {
                        state.network_connected = false;
                        state.status = dan::ProviderUiStatus::reconnecting;
                        state.used_vram_mib = 0;
                        state.model_name.clear();
                        state.stage.clear();
                        state.download_percent = -1;
                        state.message = std::string(failure.what())
                            + ". Reconnecting automatically...";
                    });
                }
                stop_watcher.request_stop();
                stop_watcher.join();
                po::close_socket(coordinator_socket);
                if (!shutdown && !dan::platform::stop_requested()) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
            return 0;
        }
        if (range_model) {
            po::RangeModelStats stats;
            std::string error;
            if (!po::prepare_range_model({model_url, model_revision, model_sha256,
                    model, begin, end}, stats, error)) {
                throw std::runtime_error("range-backed model: " + error);
            }
            print_range_stats(model, stats);
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
        if (ui) ui->update([&](auto& state) {
            state.status = dan::ProviderUiStatus::error;
            state.message = error.what();
        });
        exit_code = 1;
    }
    return exit_code;
}
