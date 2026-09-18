#include "llama.h"
#include "ggml-backend.h"
#include "provider_owned/protocol.hpp"
#include "provider_owned/formation.hpp"
#include "provider_owned/lease.hpp"
#include "provider_owned/manifest.hpp"
#include "provider_owned/planner.hpp"
#include "provider_owned/range_model.hpp"
#include "provider_owned/route.hpp"
#include "provider_owned/speculation.hpp"
#include "provider_ui.hpp"
#include "platform.hpp"

#include <bit>
#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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

// Speculative batches crossing the ring carry the draft's guesses after the activations:
// one token id per guessed position (rows - 1 of them), so the last stage can check them.
// Every other frame carries none.
std::size_t guess_bytes(const po::Frame& frame, std::uint64_t values) {
    if (frame.type != po::Type::speculative_activation || frame.rows < 2) return 0;
    const std::size_t tail = static_cast<std::size_t>(frame.rows - 1) * 4;
    return frame.payload.size() == 8 + values * sizeof(float) + tail ? tail : 0;
}

// The guesses a speculative batch carried, or none.
std::vector<std::uint32_t> carried_guesses(const po::Frame& frame) {
    std::vector<std::uint32_t> guesses;
    const std::uint64_t values = std::uint64_t(frame.rows) * frame.cols;
    const std::size_t bytes = guess_bytes(frame, values);
    for (std::size_t offset = frame.payload.size() - bytes; offset < frame.payload.size();
            offset += 4) {
        guesses.push_back(po::get32(frame.payload.data() + offset));
    }
    return guesses;
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

    // emit_chunk is called synchronously, zero or more times, only by run_first splitting an
    // incoming prompt into chunks (docs/PIPELINED_SPECULATION_V1.md phase 5); every other frame
    // type ignores it. The return value is always the LAST thing produced for this call -- the
    // final chunk's own activation/result -- so callers that don't chunk see no difference at
    // all from calling handle(input) alone.
    using ChunkSink = std::function<void(const po::Frame&)>;

    po::Frame handle(const po::Frame& input, std::size_t prefill_chunk = 0,
        const ChunkSink& emit_chunk = {}) {
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
        case po::Type::prompt_chunk:
            return execute(input, prefill_chunk, emit_chunk);
        default: throw std::runtime_error("unexpected frame type for worker");
        }
    }

    bool last() const { return last_; }
    int begin() const { return begin_; }
    int layers() const { return layers_; }
    std::string text_of(std::uint32_t token) const {
        return piece(llama_model_get_vocab(model_), static_cast<llama_token>(token));
    }
    bool ends_text(std::uint32_t token) const {
        return llama_vocab_is_eog(llama_model_get_vocab(model_), static_cast<llama_token>(token));
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

    po::Frame execute(const po::Frame& input, std::size_t prefill_chunk = 0,
        const ChunkSink& emit_chunk = {}) {
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
        // Implicit rollback: a frame at a lower position than the session has already
        // reached is a correction after a rejected speculative round, not an error. Truncate
        // the KV to that position so the frame's own position check below passes normally;
        // truncation by absolute position is exact for plain (non-windowed, non-recurrent) KV.
        if (input.position < session.position) {
            if (!llama_memory_seq_rm(llama_get_memory(context_), session.sequence,
                    input.position, -1)) {
                throw std::runtime_error("could not roll back session KV to a lower-position frame");
            }
            session.position = input.position;
        }
        if (first_) return run_first(input, session, prefill_chunk, emit_chunk);
        if (last_) return run_last(input, session);
        return run_middle(input, session);
    }

    // Decodes one prefill chunk (a sub-range of the tokenized prompt) and returns its activation,
    // with embeddings for every position -- exactly the shape run_first already builds for a
    // whole (unchunked) !last_ prompt, just parameterized so both the chunked and final-chunk
    // paths can share it. Never called when last_ (chunking exists only to hand a chunk to the
    // next stage; the last stage's own final decode still goes through the ordinary path below,
    // since only it may sample).
    po::Frame decode_prefill_chunk(const std::vector<llama_token>& tokens, Session& session,
        po::Type type, std::uint64_t frame_session, std::uint64_t frame_request) {
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
            throw std::runtime_error("stage A chunk decode failed");
        }
        llama_synchronize(context_);
        const std::uint64_t compute = elapsed_ns(start);

        const std::size_t values = tokens.size() * static_cast<std::size_t>(hidden_);
        po::Frame output;
        output.type = type;
        output.session = frame_session;
        output.request = frame_request;
        output.position = session.position;
        output.rows = static_cast<std::uint32_t>(tokens.size());
        output.cols = static_cast<std::uint32_t>(hidden_);
        output.dtype = po::DType::f32le;
        output.payload.resize(8 + values * sizeof(float));
        po::put64(output.payload.data(), compute);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            const float* source = llama_get_embeddings_ith(context_, static_cast<int32_t>(index));
            if (!source) {
                llama_batch_free(batch);
                throw std::runtime_error("stage A chunk returned no hidden state");
            }
            std::memcpy(output.payload.data() + 8
                    + index * static_cast<std::size_t>(hidden_) * sizeof(float), source,
                static_cast<std::size_t>(hidden_) * sizeof(float));
        }
        llama_batch_free(batch);
        session.position += static_cast<std::uint32_t>(tokens.size());
        tokens_processed_ += tokens.size();
        std::fprintf(stderr,
            "session=%llu request=%llu stage=A phase=prefill-chunk position=%u shape=%ux%u "
            "compute_ms=%.3f\n",
            static_cast<unsigned long long>(frame_session),
            static_cast<unsigned long long>(frame_request), output.position, output.rows,
            output.cols, compute / 1e6);
        return output;
    }

    po::Frame run_first(const po::Frame& input, Session& session, std::size_t prefill_chunk = 0,
        const ChunkSink& emit_chunk = {}) {
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

        // Chunked pipelined prefill (docs/PIPELINED_SPECULATION_V1.md phase 5): only for a whole
        // incoming prompt with somewhere to forward chunks to (ring mode) and more tokens than
        // fit in one chunk. Send every chunk but the last as prompt_chunk, immediately, via
        // emit_chunk -- so the next stage can start on chunk 1 while this stage is still on
        // chunk 2 -- then let `tokens` fall through holding only the final chunk, so the rest of
        // this function (unchanged below) handles it exactly like an unchunked prompt would.
        if (input.type == po::Type::prompt && !last_ && emit_chunk && prefill_chunk > 0
            && tokens.size() > prefill_chunk) {
            std::size_t offset = 0;
            while (tokens.size() - offset > prefill_chunk) {
                std::vector<llama_token> chunk(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                    tokens.begin() + static_cast<std::ptrdiff_t>(offset + prefill_chunk));
                emit_chunk(decode_prefill_chunk(chunk, session, po::Type::prompt_chunk,
                    input.session, input.request));
                offset += prefill_chunk;
            }
            tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset), tokens.end());
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
                && input.type != po::Type::commit_activation
                && input.type != po::Type::prompt_chunk)
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
        // Guesses after the activations are read by the ring thread (continue the decode loop),
        // not here.
        if (values > (po::max_payload - 8) / sizeof(float)
            || input.payload.size() != 8 + values * sizeof(float) + guess_bytes(input, values)
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

        if (input.type == po::Type::prompt_chunk) {
            // Extends KV like any other chunk; never samples, since more of the prompt is
            // still coming. The reply is a plain ack with nowhere useful to go -- there is no
            // next stage, and the coordinator is waiting for the eventual result, not a
            // per-chunk acknowledgement -- so the caller (main()'s ring thread) drops it rather
            // than forwarding it into the coordinator's single expected result read.
            llama_synchronize(context_);
            const std::uint64_t compute = elapsed_ns(start);
            std::fprintf(stderr,
                "session=%llu request=%llu stage=B phase=prefill-chunk position=%u compute_ms=%.3f\n",
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
                && input.type != po::Type::commit_activation
                && input.type != po::Type::prompt_chunk)
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
        const std::size_t guesses = guess_bytes(input, values);
        if (values > (po::max_payload - 8) / sizeof(float)
            || input.payload.size() != 8 + values * sizeof(float) + guesses
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
        // The draft's guesses ride along to the last stage, which checks them.
        output.payload.insert(output.payload.end(), input.payload.end() - guesses,
            input.payload.end());
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

po::socket_t wait_for_coordinator(std::string_view endpoint, dan::ProviderTerminalUi* ui,
    dan::platform::Process* hosted_coordinator = nullptr) {
    bool announced = false;
    while (!dan::platform::stop_requested()) {
        try {
            return connect_to(endpoint);
        } catch (const std::exception&) {
            if (hosted_coordinator && !hosted_coordinator->running()) return po::invalid_socket;
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

void set_socket_receive_timeout(po::socket_t socket, std::uint32_t milliseconds) {
#ifdef _WIN32
    const DWORD timeout = milliseconds;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
        reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    const timeval timeout{static_cast<long>(milliseconds / 1000),
        static_cast<long>((milliseconds % 1000) * 1000)};
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
}

// Ring connections (both directions: a stage's --next to the next hop's --ring-listen, and the
// tail's --next to the coordinator's --ring-return) validate themselves with one send/receive
// round trip immediately after connect()/accept(), before either side trusts the socket for
// real traffic. A bare successful connect() is not enough: a TCP tunnel (SSH -R and similar) can
// complete a LOCAL accept on the connecting side before its forwarded channel to the real remote
// listener exists yet, leaving that side believing it "connected" over a socket that silently
// goes nowhere -- and ring connections otherwise have no retry once connect() returns
// successfully (see docs/PIPELINED_RING_PHYSICAL_TEST.md, which hit exactly this over SSH port
// forwarding). A real round trip, bounded by a short timeout, turns that silent dead socket into
// an ordinary retryable failure.
bool ring_handshake_connect(po::socket_t socket) {
    set_socket_receive_timeout(socket, 5000);
    po::Frame hello; hello.type = po::Type::ack;
    std::string error;
    if (!po::send_frame(socket, hello, error)) {
        std::fprintf(stderr, "ring handshake send failed: %s\n", error.c_str()); return false;
    }
    po::Frame reply;
    if (!po::recv_frame(socket, reply, error) || reply.type != po::Type::ack) {
        std::fprintf(stderr, "ring handshake receive failed: %s\n", error.c_str()); return false;
    }
    po::Frame confirmed; confirmed.type = po::Type::ack;
    if (!po::send_frame(socket, confirmed, error)) {
        std::fprintf(stderr, "ring handshake confirmation failed: %s\n", error.c_str()); return false;
    }
    set_socket_receive_timeout(socket, 0);
    return true;
}

bool connect_ring_proxy(po::socket_t socket, std::string_view target) {
    const std::string request = "DAN-RING/1 " + std::string(target) + '\n';
    if (!po::send_all(socket, request.data(), request.size())) return false;
    char response[3]{};
    return po::recv_all(socket, response, sizeof(response))
        && std::string_view(response, sizeof(response)) == "OK\n";
}

bool ring_handshake_accept(po::socket_t socket) {
    set_socket_receive_timeout(socket, 5000);
    po::Frame hello;
    std::string error;
    if (!po::recv_frame(socket, hello, error) || hello.type != po::Type::ack) {
        std::fprintf(stderr, "ring handshake hello failed: %s\n", error.c_str()); return false;
    }
    po::Frame ack; ack.type = po::Type::ack;
    if (!po::send_frame(socket, ack, error)) {
        std::fprintf(stderr, "ring handshake reply failed: %s\n", error.c_str()); return false;
    }
    po::Frame confirmed;
    if (!po::recv_frame(socket, confirmed, error) || confirmed.type != po::Type::ack) {
        std::fprintf(stderr, "ring handshake confirmation failed: %s\n", error.c_str()); return false;
    }
    set_socket_receive_timeout(socket, 0);
    return true;
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

// commit_token/commit_activation are deliberately excluded: they are used only by
// commit_final_token (client.cpp), which always uses the direct per-stage
// exchange() -- send and receive on the same control connection -- and was not
// changed to use the ring. A commit reply must go back to that same connection, not
// forward into the ring, or commit_final_token's exchange() never sees it and blocks.
bool is_hot_path(po::Type type) {
    return type == po::Type::prompt || type == po::Type::token
        || type == po::Type::activation || type == po::Type::speculative_activation;
}

// Ring and control state shared by static mode and serve mode. `stage_mutex` guards every
// Stage::handle call and the `stage` pointer; both threads take it around the call, never
// around socket I/O, so a slow send/recv on one connection cannot block the other.
// Loop mode: the last stage keeps decoding without the client. The client sends one
// stream_prompt (its token budget) to this stage, then the prompt to the first stage; every
// sampled token is streamed to the client as client_chunk and fed back to the first stage
// (or straight back into this worker, when it is the whole route) until the budget runs out,
// the model ends the text, or the client cancels. The final token rides the usual result
// frame, so the client sees exactly one result per request either way.
struct LoopState {
    std::mutex mutex;
    bool active = false;
    std::uint64_t session = 0;
    std::uint64_t request = 0;
    std::uint32_t remaining = 0;
    bool cancelled = false;

    bool owns(const po::Frame& frame) {
        return active && frame.session == session && frame.request == request;
    }
    void clear() { active = false; remaining = 0; cancelled = false; }
};

struct RingState {
    bool p2p = false;                 // next hops go through the sidecar ring proxy
    std::string ring_proxy;
    po::socket_t ring_listener = po::invalid_socket;
    std::atomic<po::socket_t> next{po::invalid_socket};
    std::atomic<po::socket_t> loop{po::invalid_socket};  // last stage -> first stage
    std::string loop_target;          // dialed on the first token, not at route setup
    bool loop_self = false;           // this worker is the whole route: loop without a socket
    LoopState decode;
    std::mutex route_mutex;           // guards expected_previous (read by the ring thread)
    std::string expected_previous;
    std::mutex stage_mutex;
    Stage* stage = nullptr;           // null while serve mode has nothing loaded
    // Speculative decoding (first stage only): a small model that proposes the next few
    // tokens, so one pass through the route can commit several of them.
    Stage* draft = nullptr;
    std::uint64_t draft_session = 0;  // the session its KV currently follows
    // First stage of a multi-stage route: the last round's start and guesses. The next
    // token's position tells how many the last stage accepted.
    std::uint32_t guessed_at = 0;
    std::vector<std::uint32_t> guessed;
    std::atomic<bool> shutdown{false};
    // Total time to establish a route's next hop (lookup, relay, hole punch, handshake).
    std::chrono::milliseconds connect_budget{20000};
    // Called once a client route's next hop is connected (serve-mode dashboard).
    std::function<void(const po::RingRoute&)> on_route;

    void disconnected() {
        std::lock_guard lock(stage_mutex);
        if (stage) stage->coordinator_disconnected();
    }
};

void release_route(RingState& ring);

// Dials one ring target (through the sidecar proxy in libp2p mode) until the budget runs
// out; invalid_socket when it never answered.
po::socket_t dial_ring_target(RingState& ring, const std::string& target,
    std::chrono::steady_clock::time_point deadline) {
    po::socket_t socket = po::invalid_socket;
    const auto remaining_ms = [&] {
        return static_cast<std::uint32_t>(std::max<std::int64_t>(1,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count()));
    };
    for (bool first = true; socket == po::invalid_socket
            && std::chrono::steady_clock::now() < deadline; first = false) {
        if (!first) {
            std::this_thread::sleep_for(std::min<std::chrono::milliseconds>(
                std::chrono::seconds(1), std::chrono::milliseconds(remaining_ms())));
        }
        try {
            socket = connect_to(ring.p2p ? ring.ring_proxy : target);
        } catch (const std::exception&) {
            continue;
        }
        // The sidecar answers once it has a stream to the peer (or gave up); do not wait
        // for it past this route's budget.
        set_socket_receive_timeout(socket, remaining_ms());
        if ((ring.p2p && !connect_ring_proxy(socket, target))
            || !ring_handshake_connect(socket)) {
            po::close_socket(socket);
            socket = po::invalid_socket;
        }
    }
    return socket;
}

// Connects the next hop named by a client route (create_session payload). Throws when the
// route is invalid, conflicts with the bound one, or the next hop cannot be reached.
void bind_route(RingState& ring, const po::RingRoute& route, int stage_begin, std::string& bound) {
    if (ring.p2p ? !po::valid_p2p_target(route.next) : !po::valid_private_endpoint(route.next)) {
        throw std::runtime_error(ring.p2p ? "ring target must be a libp2p peer address"
            : "ring target must be a private IPv4 host:port");
    }
    if (!ring.p2p && !route.previous_peer.empty()) {
        throw std::runtime_error("previous_peer needs --peer-header to be verified");
    }
    if (ring.p2p && stage_begin != 0 && route.previous_peer.empty()) {
        throw std::runtime_error("a non-first stage needs previous_peer");
    }
    if (stage_begin != 0 && ring.ring_listener == po::invalid_socket) {
        throw std::runtime_error("a non-first stage needs --ring-listen");
    }
    {
        std::lock_guard lock(ring.route_mutex);
        if (!bound.empty()) {
            if (bound != route.next || ring.expected_previous != route.previous_peer) {
                throw std::runtime_error("worker is bound to a different route");
            }
            return;
        }
        if (ring.next.load() != po::invalid_socket) {
            throw std::runtime_error("worker already serves another route");
        }
        ring.expected_previous = route.previous_peer;
    }
    std::fprintf(stderr, "ring: connecting to route next hop %s\n", route.next.c_str());
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + ring.connect_budget;
    po::socket_t socket = dial_ring_target(ring, route.next, deadline);
    if (socket == po::invalid_socket) {
        std::lock_guard lock(ring.route_mutex);
        ring.expected_previous.clear();
        throw std::runtime_error("could not reach the route's next hop within "
            + std::to_string(ring.connect_budget.count()) + " ms");
    }
    std::fprintf(stderr, "ring: next hop ready after %lld ms\n", static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count()));
    ring.next.store(socket);
    // Loop mode: this stage also needs a way back to the first stage (or, when it is the
    // whole route, no socket at all). The connection itself waits until the first token:
    // the first stage only learns to expect this one once the client sets up its own
    // session, which happens after this call returns.
    ring.loop_self = route.loop == po::loop_self;
    ring.loop_target = ring.loop_self ? std::string{} : route.loop;
    bound = route.next;
    if (ring.on_route) ring.on_route(route);
    std::fprintf(stderr, "ring: route next hop connected\n");
}

void release_route(RingState& ring) {
    for (std::atomic<po::socket_t>* held : {&ring.next, &ring.loop}) {
        if (const auto socket = held->exchange(po::invalid_socket);
            socket != po::invalid_socket) {
            dan::platform::shutdown_socket(socket);
            po::close_socket(socket);
        }
    }
    ring.loop_self = false;
    ring.loop_target.clear();
    {
        std::lock_guard lock(ring.decode.mutex);
        ring.decode.clear();
    }
    std::lock_guard lock(ring.route_mutex);
    ring.expected_previous.clear();
}

// ---- Speculative decoding (draft model on the first stage) ----
//
// One round: the draft model proposes `draft_width - 1` tokens after the current one, the
// real stage verifies all of them in a single batch, and every correct guess is a token the
// route commits without another pass. A wrong guess costs nothing but the draft's own time:
// the batch still yields the correct next token at that position, and the stage's KV is
// truncated by the next frame's lower position (implicit rollback).
inline constexpr std::uint32_t draft_width = 4;

po::Frame token_frame(const po::Frame& like, std::uint32_t position,
    const std::vector<std::uint32_t>& tokens) {
    po::Frame frame;
    frame.type = po::Type::token;
    frame.session = like.session;
    frame.request = like.request;
    frame.position = position;
    frame.rows = tokens.size() > 1 ? static_cast<std::uint32_t>(tokens.size()) : 0;
    frame.payload.resize(tokens.size() * 4);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        po::put32(frame.payload.data() + index * 4, tokens[index]);
    }
    return frame;
}

// Feeds `tokens` to the draft model one at a time from `position`, collecting what it would
// say next. Returns fewer proposals if anything goes wrong; the round then just verifies
// what it has.
std::vector<std::uint32_t> draft_proposals(Stage& draft, const po::Frame& like,
    std::uint32_t position, std::uint32_t current, std::uint32_t count) {
    std::vector<std::uint32_t> proposals;
    std::uint32_t token = current;
    for (std::uint32_t index = 0; index < count; ++index) {
        const po::Frame reply = draft.handle(token_frame(like, position + index, {token}));
        if (reply.type != po::Type::result || reply.payload.size() < 13) break;
        token = po::get32(reply.payload.data());
        proposals.push_back(token);
        if (draft.ends_text(token)) break;
    }
    return proposals;
}

// The tokens a verified batch commits (see provider_owned/speculation.hpp for the rule).
std::vector<std::uint32_t> accepted_tokens(const std::vector<std::uint32_t>& proposals,
    const po::Frame& verified) {
    if (verified.type != po::Type::result || verified.rows == 0
        || verified.payload.size() < 8 + static_cast<std::size_t>(verified.rows) * 4) {
        return {};
    }
    std::vector<std::uint32_t> samples;
    for (std::uint32_t index = 0; index < verified.rows; ++index) {
        samples.push_back(po::get32(verified.payload.data() + 8 + static_cast<std::size_t>(index) * 4));
    }
    return po::accept_speculation(proposals, samples);
}

po::Frame ack_frame(const po::Frame& input) {
    po::Frame output;
    output.type = po::Type::ack;
    output.session = input.session;
    output.request = input.request;
    return output;
}

// What one local decode step committed: usually a single token, or several when the draft
// model guessed right. `position` is where the next input token goes.
struct LoopStep {
    std::vector<std::uint32_t> tokens;
    std::vector<std::string> text;
    std::vector<char> ends;
    std::uint32_t position = 0;
    std::uint64_t compute_ns = 0;
    std::string error;
};

using LoopDecoder = std::function<LoopStep(std::uint32_t token, std::uint32_t position,
    const po::Frame& like)>;

// One streamed token, shaped exactly like an ordinary result frame.
po::Frame streamed_frame(const po::Frame& like, std::uint32_t position, std::uint32_t token,
    const std::string& text, bool ends, std::uint64_t compute_ns) {
    po::Frame frame;
    frame.type = po::Type::result;
    frame.session = like.session;
    frame.request = like.request;
    frame.position = position;
    frame.payload.resize(13 + text.size());
    po::put32(frame.payload.data(), token);
    po::put64(frame.payload.data() + 4, compute_ns);
    frame.payload[12] = ends ? 1 : 0;
    std::memcpy(frame.payload.data() + 13, text.data(), text.size());
    return frame;
}

// Handles results the last stage produced while loop mode owns the request: stream each one
// to the client and, unless it was the final token, keep decoding. Returns false when the
// connection to the client broke. `decode` runs one step locally and is used only when this
// worker is the whole route.
bool continue_loop(RingState& ring, std::deque<po::Frame> pending, std::string& error,
    const LoopDecoder& decode) {
    while (!pending.empty()) {
        po::Frame frame = std::move(pending.front());
        pending.pop_front();
        bool final_token = frame.payload.size() < 13 || frame.payload[12] != 0;
        {
            std::lock_guard lock(ring.decode.mutex);
            if (ring.decode.cancelled || ring.decode.remaining <= 1) final_token = true;
            else --ring.decode.remaining;
            if (final_token) ring.decode.clear();
        }
        po::Frame to_client = frame;
        if (!final_token) to_client.type = po::Type::client_chunk;
        const po::socket_t client = ring.next.load();
        if (client == po::invalid_socket || !po::send_frame(client, to_client, error)) {
            std::lock_guard lock(ring.decode.mutex);
            ring.decode.clear();
            return false;
        }
        if (final_token) return true;
        if (!pending.empty()) continue;  // this round committed more than one token
        if (!ring.loop_self) {
            // The token goes back to the first stage and comes around the ring again.
            const po::Frame token = token_frame(frame, frame.position,
                {po::get32(frame.payload.data())});
            po::socket_t loop = ring.loop.load();
            if (loop == po::invalid_socket && !ring.loop_target.empty()) {
                loop = dial_ring_target(ring, ring.loop_target,
                    std::chrono::steady_clock::now() + ring.connect_budget);
                if (loop != po::invalid_socket) {
                    ring.loop.store(loop);
                    std::fprintf(stderr, "ring: decode loop connected to %s\n",
                        ring.loop_target.c_str());
                }
            }
            if (loop == po::invalid_socket || !po::send_frame(loop, token, error)) {
                if (error.empty()) error = "could not reach the route's first stage";
                std::lock_guard lock(ring.decode.mutex);
                ring.decode.clear();
                return false;
            }
            return true;
        }
        const LoopStep step = decode(po::get32(frame.payload.data()), frame.position, frame);
        if (!step.error.empty() || step.tokens.empty()) {
            std::lock_guard lock(ring.decode.mutex);
            ring.decode.clear();
            po::Frame failure = po::error_frame(frame,
                step.error.empty() ? "decode produced no token" : step.error);
            return po::send_frame(ring.next.load(), failure, error);
        }
        for (std::size_t index = 0; index < step.tokens.size(); ++index) {
            pending.push_back(streamed_frame(frame,
                frame.position + static_cast<std::uint32_t>(index) + 1, step.tokens[index],
                step.text[index], step.ends[index] != 0,
                index == 0 ? step.compute_ns : 0));
        }
    }
    return true;
}

// One decode step on this worker: the draft model (when loaded) proposes the next few
// tokens and the stage verifies them all in one batch, so a round can commit several.
LoopStep local_decode(RingState& ring, std::uint32_t token, std::uint32_t position,
    const po::Frame& like) {
    LoopStep step;
    std::lock_guard<std::mutex> lock(ring.stage_mutex);
    if (!ring.stage) {
        step.error = "no stage is loaded";
        return step;
    }
    try {
        std::vector<std::uint32_t> proposals;
        if (ring.draft) {
            proposals = draft_proposals(*ring.draft, like, position, token, draft_width - 1);
        }
        std::vector<std::uint32_t> batch{token};
        batch.insert(batch.end(), proposals.begin(), proposals.end());
        const po::Frame verified = ring.stage->handle(token_frame(like, position, batch));
        std::vector<std::uint32_t> accepted;
        if (batch.size() == 1) {
            if (verified.type != po::Type::result || verified.payload.size() < 13) {
                step.error = "unexpected decode reply";
                return step;
            }
            accepted.push_back(po::get32(verified.payload.data()));
            step.compute_ns = po::get64(verified.payload.data() + 4);
        } else {
            accepted = accepted_tokens(proposals, verified);
            if (verified.payload.size() >= 8) step.compute_ns = po::get64(verified.payload.data());
            // The draft fed itself every proposal but the last one. When the stage accepted
            // them all, that last proposal is now committed too, so the draft has to catch up
            // or the next round starts a token behind. (Fewer acceptances need nothing: the
            // next frame's lower position truncates its KV.)
            if (!proposals.empty() && accepted.size() == proposals.size() + 1) {
                ring.draft->handle(token_frame(like,
                    position + static_cast<std::uint32_t>(proposals.size()), {proposals.back()}));
            }
            std::fprintf(stderr, "speculation: proposed=%zu accepted=%zu\n",
                proposals.size(), accepted.empty() ? 0 : accepted.size() - 1);
        }
        for (const std::uint32_t next : accepted) {
            step.tokens.push_back(next);
            step.text.push_back(ring.stage->text_of(next));
            step.ends.push_back(ring.stage->ends_text(next) ? 1 : 0);
        }
        step.position = position + static_cast<std::uint32_t>(step.tokens.size());
    } catch (const std::exception& failure) {
        step.error = failure.what();
    }
    return step;
}

// First stage of a multi-stage route: the token coming around the ring starts a round.
// The draft guesses the next few tokens, the stage runs all positions as one batch, and the
// guesses ride along to the last stage. Called with ring.stage_mutex held.
po::Frame draft_round(RingState& ring, const po::Frame& input) {
    const std::uint32_t token = po::get32(input.payload.data());
    // Last round's guesses all accepted: the draft fed itself every guess but the last one,
    // which is now committed text too.
    if (!ring.guessed.empty()
        && input.position == ring.guessed_at + ring.guessed.size() + 1) {
        ring.draft->handle(token_frame(input,
            ring.guessed_at + static_cast<std::uint32_t>(ring.guessed.size()),
            {ring.guessed.back()}));
    }
    std::vector<std::uint32_t> guesses;
    try {
        guesses = draft_proposals(*ring.draft, input, input.position, token, draft_width - 1);
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "speculation off for this route: %s\n", failure.what());
        ring.draft = nullptr;
    }
    ring.guessed_at = input.position;
    ring.guessed = guesses;
    std::vector<std::uint32_t> batch{token};
    batch.insert(batch.end(), guesses.begin(), guesses.end());
    po::Frame output = ring.stage->handle(token_frame(input, input.position, batch));
    if (output.type == po::Type::speculative_activation) {
        for (const std::uint32_t guess : guesses) {
            const std::size_t at = output.payload.size();
            output.payload.resize(at + 4);
            po::put32(output.payload.data() + at, guess);
        }
    }
    return output;
}

// Last stage: turn a verified speculative batch into the tokens it commits, each shaped as
// an ordinary streamed result. Called with ring.stage_mutex held.
std::deque<po::Frame> verify_round(Stage& stage, const po::Frame& input,
    const po::Frame& verified) {
    std::deque<po::Frame> committed;
    const std::vector<std::uint32_t> accepted = accepted_tokens(carried_guesses(input), verified);
    std::fprintf(stderr, "speculation: proposed=%u accepted=%zu\n",
        verified.rows > 0 ? verified.rows - 1 : 0, accepted.empty() ? 0 : accepted.size() - 1);
    for (std::size_t index = 0; index < accepted.size(); ++index) {
        committed.push_back(streamed_frame(verified,
            input.position + static_cast<std::uint32_t>(index) + 1, accepted[index],
            stage.text_of(accepted[index]), stage.ends_text(accepted[index]),
            index == 0 && verified.payload.size() >= 8 ? po::get64(verified.payload.data()) : 0));
    }
    return committed;
}

// Accepts predecessors one at a time and feeds their frames through the stage, forwarding
// every outcome to the next hop.
void run_ring(RingState& ring, std::stop_token stop) {
    while (!ring.shutdown.load() && !stop.stop_requested()) {
        const po::socket_t predecessor = accept(ring.ring_listener, nullptr, nullptr);
        if (predecessor == po::invalid_socket) break;
        if (ring.p2p) {
            // The sidecar writes the predecessor's authenticated PeerID first.
            set_socket_receive_timeout(predecessor, 5000);
            std::string peer_id;
            const bool received = po::recv_peer_id(predecessor, peer_id);
            std::string expected;
            {
                std::lock_guard lock(ring.route_mutex);
                expected = ring.expected_previous;
            }
            if (!received || expected.empty() || peer_id != expected) {
                std::fprintf(stderr, "ring: rejected unexpected predecessor peer\n");
                po::close_socket(predecessor);
                continue;
            }
        }
        if (!ring_handshake_accept(predecessor)) {
            std::fprintf(stderr,
                "ring: predecessor connection failed the handshake (stale tunnel?) "
                "-- waiting for a new one\n");
            po::close_socket(predecessor);
            continue;
        }
        std::fprintf(stderr, "ring: predecessor connected\n");
        while (!ring.shutdown.load()) {
            po::Frame input;
            std::string error;
            if (!po::recv_frame(predecessor, input, error)) {
                std::fprintf(stderr, "ring predecessor connection closed: %s\n", error.c_str());
                ring.disconnected();
                break;
            }
            po::Frame output;
            bool errored = false;
            bool last = false;
            std::deque<po::Frame> committed;  // last stage: tokens this round commits
            {
                std::lock_guard<std::mutex> lock(ring.stage_mutex);
                try {
                    if (!ring.stage) throw std::runtime_error("no stage is loaded");
                    last = ring.stage->last();
                    if (ring.draft && !last && ring.stage->begin() == 0
                        && input.type == po::Type::token && input.rows == 0
                        && input.payload.size() == 4) {
                        output = draft_round(ring, input);
                    } else {
                        output = ring.stage->handle(input);
                    }
                    if (last && output.type == po::Type::result && output.rows > 0) {
                        committed = verify_round(*ring.stage, input, output);
                    }
                } catch (const std::exception& exception) {
                    output = po::error_frame(input, exception.what());
                    errored = true;
                    std::fprintf(stderr, "ring: rejected frame: %s\n", exception.what());
                }
            }
            // Both success and error outcomes ride the ring forward: this connection is
            // forward-only (the predecessor never reads a reply on it), so there is nowhere
            // else to put either one. See the option comment above main() for why this
            // generalizes to any stage count.
            //
            // One exception: the last stage's reply to a successfully-handled prefill chunk
            // (phase 5) is a plain ack nobody is waiting for -- there is no next stage, and the
            // caller's ring return expects exactly one reply per route_step call, the eventual
            // real result, not a per-chunk acknowledgement. Forwarding it would be read as that
            // result and fail validation. Errors on a chunk still ride forward as usual, since a
            // stuck caller waiting forever on a silently dropped error would be worse.
            bool looping = false;
            if (!errored && last && output.type == po::Type::result) {
                std::lock_guard lock(ring.decode.mutex);
                looping = ring.decode.owns(output);
            }
            if (looping) {
                const auto decode = [&](std::uint32_t current, std::uint32_t position,
                    const po::Frame& like) {
                    return local_decode(ring, current, position, like);
                };
                if (committed.empty()) committed.push_back(output);
                if (!continue_loop(ring, std::move(committed), error, decode)) {
                    std::fprintf(stderr, "ring: decode loop ended: %s\n", error.c_str());
                    ring.disconnected();
                    break;
                }
                continue;
            }
            const bool drop = !errored && last && input.type == po::Type::prompt_chunk;
            const po::socket_t next = ring.next.load();
            if (!drop && (next == po::invalid_socket || !po::send_frame(next, output, error))) {
                std::fprintf(stderr, "ring next connection closed: %s\n", error.c_str());
                ring.disconnected();
                break;
            }
        }
        po::close_socket(predecessor);
    }
}

// Serves one client's control connection until it closes. In routed mode a create_session
// payload chooses this stage's next hop; the route is released when the client leaves.
void serve_control(RingState& ring, po::socket_t client, std::size_t prefill_chunk, bool routed) {
    std::string bound;  // this client's route, once set
    while (!ring.shutdown.load()) {
        bool last_stage = false;
        po::Frame input;
        std::string error;
        if (!po::recv_frame(client, input, error)) {
            std::fprintf(stderr, "coordinator connection closed: %s\n", error.c_str());
            ring.disconnected();
            break;
        }
        if (input.type == po::Type::stream_prompt || input.type == po::Type::cancel_request) {
            // Loop mode bookkeeping: a token budget for one request, or its cancellation.
            std::string problem;
            {
                std::lock_guard lock(ring.decode.mutex);
                if (input.type == po::Type::cancel_request) {
                    if (ring.decode.owns(input)) ring.decode.cancelled = true;
                } else if (input.rows == 0 || !input.payload.empty()) {
                    problem = "bad stream_prompt frame";
                } else if (ring.decode.active) {
                    problem = "another request is already streaming";
                } else {
                    ring.decode.active = true;
                    ring.decode.session = input.session;
                    ring.decode.request = input.request;
                    ring.decode.remaining = input.rows;
                    ring.decode.cancelled = false;
                }
            }
            po::Frame reply = problem.empty() ? ack_frame(input) : po::error_frame(input, problem);
            if (!po::send_frame(client, reply, error) || !problem.empty()) {
                if (!problem.empty()) std::fprintf(stderr, "rejected frame: %s\n", problem.c_str());
                ring.disconnected();
                break;
            }
            continue;
        }
        po::Frame output;
        bool handled = true;
        try {
            if (input.type == po::Type::create_session && !input.payload.empty()) {
                if (!routed) throw std::runtime_error("this worker uses a fixed --next");
                po::RingRoute route;
                const std::string text(input.payload.begin(), input.payload.end());
                if (!po::parse_route(text, route)) throw std::runtime_error("invalid ring route");
                int stage_begin = 0;
                {
                    std::lock_guard<std::mutex> lock(ring.stage_mutex);
                    if (!ring.stage) throw std::runtime_error("no stage is loaded");
                    stage_begin = ring.stage->begin();
                }
                bind_route(ring, route, stage_begin, bound);
                input.payload.clear();
            }
            std::lock_guard<std::mutex> lock(ring.stage_mutex);
            if (!ring.stage) throw std::runtime_error("no stage is loaded");
            // Only run_first's chunked-prefill path (phase 5) ever calls this, and only when it
            // and a next hop are both configured; every other frame type ignores it entirely.
            const auto emit_chunk = [&](const po::Frame& chunk) {
                std::string chunk_error;
                const po::socket_t next = ring.next.load();
                if (next == po::invalid_socket || !po::send_frame(next, chunk, chunk_error)) {
                    throw std::runtime_error(chunk_error.empty()
                        ? "ring next hop unavailable" : chunk_error);
                }
            };
            output = ring.stage->handle(input, prefill_chunk, emit_chunk);
            if (ring.stage->shutting_down()) ring.shutdown.store(true);
            last_stage = ring.stage->last();
            // The draft model follows the same session: same sessions, same prompts, so its
            // proposals continue the same text.
            if (ring.draft && (input.type == po::Type::create_session
                || input.type == po::Type::reset_session
                || input.type == po::Type::destroy_session
                || input.type == po::Type::prompt)) {
                po::Frame mirrored = input;
                mirrored.payload = input.type == po::Type::prompt ? input.payload
                    : std::vector<std::uint8_t>{};
                try {
                    ring.draft->handle(mirrored);
                } catch (const std::exception& failure) {
                    std::fprintf(stderr, "speculation off for this route: %s\n", failure.what());
                    ring.draft = nullptr;
                }
            }
        } catch (const std::exception& exception) {
            output = po::error_frame(input, exception.what());
            handled = false;
            std::fprintf(stderr, "rejected frame: %s\n", exception.what());
        }
        bool looping = false;
        if (handled && last_stage && output.type == po::Type::result) {
            std::lock_guard lock(ring.decode.mutex);
            looping = ring.decode.owns(output);
        }
        if (looping) {
            const auto decode = [&](std::uint32_t current, std::uint32_t position,
                const po::Frame& like) {
                return local_decode(ring, current, position, like);
            };
            if (!continue_loop(ring, std::deque<po::Frame>{output}, error, decode)) {
                std::fprintf(stderr, "decode loop ended: %s\n", error.c_str());
                ring.disconnected();
                break;
            }
            continue;
        }
        // Hot-path outcomes (success or error) go to the ring's next hop when ring mode is
        // active, matching the ring thread; everything else (session lifecycle, metrics,
        // shutdown) always replies here, on the connection it arrived on.
        const po::socket_t next = ring.next.load();
        const po::socket_t reply_socket = (next != po::invalid_socket
            && is_hot_path(input.type)) ? next : client;
        if (!po::send_frame(reply_socket, output, error)) {
            std::fprintf(stderr, "coordinator connection closed: %s\n", error.c_str());
            ring.disconnected();
            break;
        }
        if (!handled) {
            ring.disconnected();
            break;
        }
    }
    if (routed) release_route(ring);
}

// Reads the sidecar's "DAN-P2P/1 <PeerID>" line; false (connection unusable) otherwise.
bool read_peer_header(po::socket_t client, std::string& peer_id) {
    set_socket_receive_timeout(client, 5000);
    const bool received = po::recv_peer_id(client, peer_id);
    set_socket_receive_timeout(client, 0);
    return received;
}

// ---- Serve mode (decentralized placement) ----
//
// The worker accepts client control connections (through its sidecar), greets each with its
// capabilities, and lets exactly one client at a time reserve, assign and use a stage
// (lease.hpp). The worker checks every reservation against its own catalog and memory.

inline constexpr const char* runtime_abi = DAN_RUNTIME_ABI;
inline constexpr std::uint32_t reserve_idle_ms = 60000;         // before a reservation
inline constexpr std::uint32_t serving_idle_ms = 10 * 60 * 1000;  // after stage_ready

struct CatalogModel {
    po::Manifest manifest;
    po::ModelIndex index;
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
    return value;
}

struct ServeContext {
    RingState ring;
    po::WorkerLease lease;
    std::unordered_map<std::string, CatalogModel> catalog;  // by lowercase SHA-256
    po::ProviderCapability hello;                           // state filled per greeting
    std::filesystem::path cache_dir;
    int gpu_layers = 999;
    std::size_t prefill_chunk = 0;
    std::mutex load_mutex;
    std::unique_ptr<Stage> stage;                           // swapped under ring.stage_mutex
    std::unique_ptr<Stage> draft;                           // speculative decoding, may be null
    std::optional<po::StageRequest> loaded;                 // guarded by load_mutex
    std::atomic<int> connections{0};
    std::mutex status_mutex;
    std::optional<po::StageRequest> cached_hint;            // guarded by status_mutex
    // Dashboard (dan-provider network=dht shows it; null otherwise).
    dan::ProviderTerminalUi* ui = nullptr;
    std::filesystem::path net_status_file;
    std::atomic<std::uint64_t> finished_tokens{0};          // from stages already unloaded
    std::atomic<std::uint64_t> finished_requests{0};
    std::atomic<std::size_t> routes_served{0};
};

void show(ServeContext& context, const std::function<void(dan::ProviderUiState&)>& change) {
    if (context.ui) context.ui->update(change);
}

std::string peer_of_target(std::string_view target) {
    const std::size_t found = target.rfind("/p2p/");
    return std::string(found == std::string_view::npos ? target : target.substr(found + 5));
}

std::string route_label(const std::string& route_id) {
    return route_id.substr(0, 8);
}

void show_idle(ServeContext& context, std::string activity) {
    show(context, [&](dan::ProviderUiState& state) {
        state.status = dan::ProviderUiStatus::available;
        state.message = "Waiting for a client to reserve this GPU";
        state.route_id.clear();
        state.layers.clear();
        state.previous_peer.clear();
        state.next_peer.clear();
        state.link_in.clear();
        state.link_out.clear();
        state.download_percent = -1;
        dan::add_activity(state, std::move(activity));
    });
}

// Why a reservation cannot be accepted, or empty if it fits this worker.
std::string reservation_problem(const ServeContext& context, const po::StageRequest& request) {
    const auto found = context.catalog.find(lowercase(request.model_sha256));
    if (found == context.catalog.end()) return "unknown_model";
    const po::ModelIndex& index = found->second.index;
    if (request.end > static_cast<int>(index.layers)) return "invalid_range";
    if (request.lease_ms == 0) return "invalid_lease";
    if (request.context > context.hello.max_context
        || request.sessions > context.hello.max_sessions) return "limits_exceeded";
    if (std::uint64_t(request.context) * index.hidden * sizeof(float) > po::max_payload - 8) {
        return "context_too_large";
    }
    po::StageAssignment fit;
    if (!po::stage_fits(index, context.hello.offered_vram_mib, request.begin, request.end,
            request.context, request.sessions, fit)) return "insufficient_memory";
    return {};
}

// Layer ranges already downloaded for one model: cache files named <model_id>-<begin>-<end>
// .gguf with a complete .ranges index beside them. Scanned instead of remembered, so a
// restart still knows what this worker has.
std::vector<po::CachedRange> cached_ranges(const std::filesystem::path& cache_dir,
    const std::string& model_id, const std::string& model_sha256) {
    std::vector<po::CachedRange> ranges;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(cache_dir, error)) {
        const std::string name = entry.path().filename().string();
        if (!name.starts_with(model_id + "-") || !name.ends_with(".gguf")) continue;
        if (!std::filesystem::exists(entry.path().string() + ".ranges", error)) continue;
        const std::string span = name.substr(model_id.size() + 1,
            name.size() - model_id.size() - 1 - 5);
        const std::size_t dash = span.find('-');
        if (dash == std::string::npos) continue;
        po::CachedRange range{model_sha256, 0, 0};
        try {
            range.begin = std::stoi(span.substr(0, dash));
            range.end = std::stoi(span.substr(dash + 1));
        } catch (const std::exception&) { continue; }
        if (range.end > range.begin) ranges.push_back(range);
    }
    return ranges;
}

// Loads the small model the first stage uses to propose tokens (speculative decoding).
// Never fatal: a route without a draft model simply decodes one token at a time.
void load_draft_model(ServeContext& context, const po::StageRequest& request) {
    const bool wanted = !request.draft_sha256.empty() && request.begin == 0
        && context.catalog.count(lowercase(request.draft_sha256)) != 0;
    if (!wanted) {
        std::lock_guard lock(context.ring.stage_mutex);
        context.ring.draft = nullptr;
        context.draft.reset();
        return;
    }
    if (context.draft && context.loaded
        && lowercase(context.loaded->draft_sha256) == lowercase(request.draft_sha256)) {
        return;  // already loaded for the previous route
    }
    {
        std::lock_guard lock(context.ring.stage_mutex);
        context.ring.draft = nullptr;
        context.draft.reset();
    }
    const CatalogModel& model = context.catalog.at(lowercase(request.draft_sha256));
    const auto path = context.cache_dir / (model.manifest.model_id + "-draft.gguf");
    po::RangeModelStats stats;
    std::string error;
    try {
        if (!po::prepare_range_model({model.manifest.url, model.manifest.revision,
                model.manifest.sha256, path, 0, static_cast<int>(model.index.layers)},
                stats, error)) {
            throw std::runtime_error(error);
        }
        auto draft = std::make_unique<Stage>(path.string(), 0,
            static_cast<int>(model.index.layers), static_cast<int>(request.context),
            context.gpu_layers, 1);
        std::lock_guard lock(context.ring.stage_mutex);
        context.draft = std::move(draft);
        context.ring.draft = context.draft.get();
        std::fprintf(stderr, "speculation: draft model %s ready\n", model.manifest.model_id.c_str());
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "speculation off: draft model %s could not be loaded: %s\n",
            model.manifest.model_id.c_str(), failure.what());
    }
}

// Loads (or keeps) the stage a client was assigned. Throws on download/load failure.
void load_assigned_stage(ServeContext& context, const po::StageRequest& request) {
    std::lock_guard load(context.load_mutex);
    if (context.loaded && context.stage
        && lowercase(context.loaded->model_sha256) == lowercase(request.model_sha256)
        && context.loaded->begin == request.begin && context.loaded->end == request.end
        && context.loaded->context == request.context
        && context.loaded->sessions == request.sessions) {
        std::fprintf(stderr, "Reusing loaded stage layers %d..%d.\n", request.begin, request.end - 1);
        show(context, [](dan::ProviderUiState& state) {
            dan::add_activity(state, "layers already loaded on the GPU");
        });
        return;
    }
    {
        std::lock_guard lock(context.ring.stage_mutex);
        if (context.stage) {
            context.finished_tokens += context.stage->tokens_processed();
            context.finished_requests += context.stage->requests_served();
        }
        context.ring.stage = nullptr;
        context.stage.reset();
    }
    context.loaded.reset();
    const CatalogModel& model = context.catalog.at(lowercase(request.model_sha256));
    show(context, [](dan::ProviderUiState& state) {
        state.status = dan::ProviderUiStatus::downloading;
        state.message = "Fetching only the layers this GPU will run";
        state.download_percent = 0;
    });
    const auto progress = [&](std::uint64_t downloaded, std::uint64_t total, std::uint64_t speed) {
        show(context, [&](dan::ProviderUiState& state) {
            state.downloaded_bytes = static_cast<std::size_t>(downloaded);
            state.download_total_bytes = static_cast<std::size_t>(total);
            state.download_bytes_per_second = static_cast<std::size_t>(speed);
            state.download_percent = total == 0 ? 0 : static_cast<int>(downloaded * 100 / total);
        });
    };
    const auto path = context.cache_dir / (model.manifest.model_id + "-"
        + std::to_string(request.begin) + "-" + std::to_string(request.end) + ".gguf");
    po::RangeModelStats stats;
    std::string error;
    std::fprintf(stderr, "Downloading required model data for layers %d..%d...\n",
        request.begin, request.end - 1);
    if (!po::prepare_range_model({model.manifest.url, model.manifest.revision,
            model.manifest.sha256, path, request.begin, request.end, progress}, stats, error)) {
        throw std::runtime_error("range-backed model: " + error);
    }
    print_range_stats(path, stats);
    show(context, [&](dan::ProviderUiState& state) {
        state.status = dan::ProviderUiStatus::loading;
        state.download_percent = -1;
        state.message = stats.cache_reused ? "Layers were already cached" : "Download verified";
        dan::add_activity(state, stats.cache_reused ? "layers found in cache"
            : "downloaded " + std::to_string(stats.downloaded_bytes / 1000000) + " MB");
    });
    auto stage = std::make_unique<Stage>(path.string(), request.begin, request.end,
        static_cast<int>(request.context), context.gpu_layers, request.sessions);
    {
        std::lock_guard lock(context.ring.stage_mutex);
        context.stage = std::move(stage);
        context.ring.stage = context.stage.get();
    }
    load_draft_model(context, request);
    context.loaded = request;
    std::lock_guard status(context.status_mutex);
    context.cached_hint = request;
}

bool reply(po::socket_t client, const po::Frame& frame) {
    std::string error;
    return po::send_frame(client, frame, error);
}

po::Frame lease_ack(const po::Frame& input) {
    po::Frame output;
    output.type = po::Type::ack;
    output.session = input.session;
    output.request = input.request;
    return output;
}

void serve_connection(ServeContext& context, po::socket_t client) {
    std::string peer_id;
    if (context.ring.p2p && !read_peer_header(client, peer_id)) {
        std::fprintf(stderr, "rejected control connection without a PeerID\n");
        po::close_socket(client);
        return;
    }
    po::Frame hello;
    hello.type = po::Type::provider_available;
    {
        po::ProviderCapability capability = context.hello;
        capability.state = po::WorkerLease::name(
            context.lease.state(po::WorkerLease::Clock::now()));
        // What this worker already holds, so a client can plan a split that needs no download.
        for (const auto& [sha, model] : context.catalog) {
            for (const po::CachedRange& range
                    : cached_ranges(context.cache_dir, model.manifest.model_id, sha)) {
                capability.cached.push_back(range);
            }
        }
        const std::string text = po::available_message(capability);
        hello.payload.assign(text.begin(), text.end());
    }
    std::string held;  // route_id this connection holds
    bool served = false;
    if (reply(client, hello)) {
        set_socket_receive_timeout(client, reserve_idle_ms);
        for (;;) {
            po::Frame input;
            std::string error;
            if (!po::recv_frame(client, input, error)) break;
            const std::string text(input.payload.begin(), input.payload.end());
            if (input.type == po::Type::reserve) {
                po::StageRequest request;
                std::string problem = !held.empty() ? "already_reserved"
                    : !po::parse_stage_request(text, request) ? "invalid_reservation"
                    : reservation_problem(context, request);
                if (problem.empty()
                    && !context.lease.reserve(request, po::WorkerLease::Clock::now())) {
                    problem = "busy";
                }
                if (!problem.empty()) {
                    if (!reply(client, po::error_frame(input, problem))) break;
                    continue;
                }
                held = request.route_id;
                show(context, [&](dan::ProviderUiState& state) {
                    const CatalogModel& model = context.catalog.at(lowercase(request.model_sha256));
                    state.status = dan::ProviderUiStatus::preparing;
                    state.message = "A client reserved this GPU";
                    state.route_id = request.route_id;
                    state.model_name = model.manifest.model_id;
                    state.layers = std::to_string(request.begin) + "-" + std::to_string(request.end - 1)
                        + " of " + std::to_string(model.index.layers);
                    dan::add_activity(state, "route " + route_label(held) + " reserved layers "
                        + std::to_string(request.begin) + "-" + std::to_string(request.end - 1));
                });
                std::fprintf(stderr, "route %s reserved layers %d..%d%s%s\n", held.c_str(),
                    request.begin, request.end - 1, peer_id.empty() ? "" : " by ", peer_id.c_str());
                set_socket_receive_timeout(client, request.lease_ms);
                if (!reply(client, lease_ack(input))) break;
            } else if (input.type == po::Type::release_route) {
                if (held.empty() || text != held) {
                    if (!reply(client, po::error_frame(input, "route_not_held"))) break;
                    continue;
                }
                context.lease.release(held);
                show_idle(context, "route " + route_label(held) + " released by the client");
                std::fprintf(stderr, "route %s released before assignment\n", held.c_str());
                held.clear();
                set_socket_receive_timeout(client, reserve_idle_ms);
                if (!reply(client, lease_ack(input))) break;
            } else if (input.type == po::Type::assign_stage) {
                po::StageRequest request;
                if (held.empty() || !po::parse_stage_request(text, request)
                    || request.route_id != held
                    || !context.lease.begin_loading(request, po::WorkerLease::Clock::now())) {
                    reply(client, po::error_frame(input, "reservation_expired_or_mismatched"));
                    break;
                }
                // The connection owns the lease while loading; no expiry during a download.
                set_socket_receive_timeout(client, 0);
                try {
                    load_assigned_stage(context, request);
                } catch (const std::exception& failure) {
                    std::fprintf(stderr, "route %s load failed: %s\n", held.c_str(), failure.what());
                    show(context, [&](dan::ProviderUiState& state) {
                        dan::add_activity(state, std::string("could not load layers: ") + failure.what());
                    });
                    reply(client, po::error_frame(input, failure.what()));
                    break;
                }
                context.lease.serving(held);
                po::Frame ready;
                ready.type = po::Type::stage_ready;
                ready.payload.assign(held.begin(), held.end());
                if (!reply(client, ready)) break;
                std::fprintf(stderr, "route %s serving layers %d..%d\n", held.c_str(),
                    request.begin, request.end - 1);
                show(context, [&](dan::ProviderUiState& state) {
                    state.status = dan::ProviderUiStatus::contributing;
                    state.message.clear();
                    state.download_percent = -1;
                    dan::add_activity(state, "serving route " + route_label(held));
                });
                served = true;
                set_socket_receive_timeout(client, serving_idle_ms);
                serve_control(context.ring, client, context.prefill_chunk, true);
                break;
            } else {
                reply(client, po::error_frame(input, "reserve a stage first"));
                break;
            }
        }
    }
    if (!held.empty()) {
        context.lease.release(held);
        std::fprintf(stderr, "route %s ended\n", held.c_str());
        if (served) ++context.routes_served;
        show_idle(context, "route " + route_label(held)
            + (served ? " finished" : " ended before loading"));
    }
    po::close_socket(client);
}

// Status for the local sidecar (/dan/capabilities/1.0.0). Resources and lease state, not a
// fixed range. `updated_unix_ms` lets the sidecar treat a stale file as a stopped worker.
std::string status_json(ServeContext& context) {
    const auto now = po::WorkerLease::Clock::now();
    const po::WorkerLease::State state = context.lease.state(now);
    const std::optional<po::StageRequest> assignment = context.lease.current(now);
    std::optional<po::StageRequest> cached;
    {
        std::lock_guard lock(context.status_mutex);
        cached = context.cached_hint;
    }
    const po::ProviderCapability& hello = context.hello;
    std::string json = "{\"protocol_version\":1,\"worker_id\":\"" + po::json_escape(hello.id)
        + "\",\"runtime_abi\":\"" + po::json_escape(hello.runtime_abi)
        + "\",\"device\":\"" + po::json_escape(hello.gpu)
        + "\",\"offered_memory_mib\":" + std::to_string(hello.offered_vram_mib)
        + ",\"max_context\":" + std::to_string(hello.max_context)
        + ",\"max_sessions\":" + std::to_string(hello.max_sessions)
        + ",\"state\":\"" + po::WorkerLease::name(state) + "\",\"models\":[";
    bool first = true;
    for (const auto& [sha, model] : context.catalog) {
        json += std::string(first ? "" : ",") + "{\"sha256\":\"" + sha
            + "\",\"layers\":" + std::to_string(model.index.layers)
            + ",\"hidden\":" + std::to_string(model.index.hidden) + ",\"cached\":[";
        if (cached && lowercase(cached->model_sha256) == sha) {
            json += "{\"begin\":" + std::to_string(cached->begin)
                + ",\"end\":" + std::to_string(cached->end) + "}";
        }
        json += "]}";
        first = false;
    }
    json += "]";
    if (assignment) {
        json += ",\"assignment\":{\"route_id\":\"" + assignment->route_id
            + "\",\"model_sha256\":\"" + lowercase(assignment->model_sha256)
            + "\",\"begin\":" + std::to_string(assignment->begin)
            + ",\"end\":" + std::to_string(assignment->end) + "}";
    }
    return json;
}

bool write_status(const std::filesystem::path& path, const std::string& body) {
    const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string text = body + ",\"updated_unix_ms\":" + std::to_string(unix_ms) + "}\n";
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output || !(output << text)) return false;
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    return !error;
}

// Rewrites the status file when it changes, and every few seconds as a liveness signal.
void run_status_writer(std::shared_ptr<ServeContext> context, std::filesystem::path path,
    std::stop_token stop) {
    std::error_code ignored;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ignored);
    std::string written;
    auto last_write = std::chrono::steady_clock::time_point{};
    while (!stop.stop_requested()) {
        const std::string body = status_json(*context);
        const auto now = std::chrono::steady_clock::now();
        if ((body != written || now - last_write >= std::chrono::seconds(3))
            && write_status(path, body)) {
            written = body;
            last_write = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
}

// What the sidecar reports in its -net-status-file.
struct NetView {
    bool fresh = false;
    std::string peer;
    std::size_t relays = 0;
    std::size_t peers = 0;
    bool ipv6 = false;
    struct Stream { std::string peer, protocol, path, transport; };
    std::vector<Stream> streams;
};

std::string json_text(std::string_view object, std::string_view key) {
    const std::string marker = "\"" + std::string(key) + "\":\"";
    const std::size_t start = object.find(marker);
    if (start == std::string_view::npos) return {};
    const std::size_t begin = start + marker.size();
    const std::size_t end = object.find('"', begin);
    return end == std::string_view::npos ? std::string{} : std::string(object.substr(begin, end - begin));
}

std::uint64_t json_count(std::string_view object, std::string_view key) {
    const std::string marker = "\"" + std::string(key) + "\":";
    const std::size_t start = object.find(marker);
    std::uint64_t value = 0;
    for (std::size_t index = start == std::string_view::npos ? object.size() : start + marker.size();
            index < object.size() && object[index] >= '0' && object[index] <= '9'; ++index) {
        value = value * 10 + static_cast<unsigned>(object[index] - '0');
    }
    return value;
}

NetView read_net_status(const std::filesystem::path& path) {
    NetView view;
    std::ifstream input(path, std::ios::binary);
    if (!input) return view;
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::size_t streams = text.find("\"streams\":[");
    const std::string_view head = std::string_view(text).substr(0, streams);
    // updated_unix_ms follows the stream list; stream objects never carry that key.
    view.fresh = now_ms - static_cast<std::int64_t>(json_count(text, "updated_unix_ms")) < 10000;
    view.peer = json_text(head, "peer_id");
    view.relays = json_count(head, "relay_addresses");
    view.peers = json_count(head, "connected_peers");
    view.ipv6 = head.find("\"public_ipv6\":true") != std::string_view::npos;
    for (std::size_t open = text.find('{', streams); streams != std::string::npos
            && open != std::string::npos; open = text.find('{', open + 1)) {
        const std::size_t close = text.find('}', open);
        if (close == std::string::npos) break;
        const std::string_view object = std::string_view(text).substr(open, close - open);
        view.streams.push_back({json_text(object, "peer"), json_text(object, "protocol"),
            json_text(object, "path"), json_text(object, "transport")});
    }
    return view;
}

void run_dashboard(std::shared_ptr<ServeContext> context, std::stop_token stop) {
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t last_tokens = 0;
    bool first = true;
    bool joined = false;
    while (!stop.stop_requested()) {
        std::uint64_t tokens = context->finished_tokens.load();
        std::uint64_t requests = context->finished_requests.load();
        {
            std::lock_guard lock(context->ring.stage_mutex);
            if (context->stage) {
                tokens += context->stage->tokens_processed();
                requests += context->stage->requests_served();
            }
        }
        const double rate = first || tokens < last_tokens ? 0.0 : static_cast<double>(tokens - last_tokens);
        last_tokens = tokens;
        first = false;
        const NetView net = read_net_status(context->net_status_file);
        const bool connected = net.fresh && net.peers > 0;
        show(*context, [&](dan::ProviderUiState& state) {
            state.tokens_participated = static_cast<std::size_t>(tokens);
            state.requests_participated = static_cast<std::size_t>(requests);
            state.routes_served = context->routes_served.load();
            state.throughput.push_back(rate);
            if (state.throughput.size() > 24) state.throughput.erase(state.throughput.begin());
            state.uptime_seconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - started).count());
            if (!net.peer.empty()) state.peer_id = net.peer;
            state.relay_addresses = net.relays;
            state.public_ipv6 = net.ipv6;
            state.peers = net.peers;
            state.network_connected = connected;
            const auto link = [&](const std::string& peer) {
                for (const NetView::Stream& stream : net.streams) {
                    if (!peer.empty() && stream.peer == peer && stream.protocol.ends_with("/ring/1.0.0")) {
                        return stream.path == "direct" ? "direct " + stream.transport : stream.path;
                    }
                }
                return std::string{};
            };
            state.link_in = link(state.previous_peer);
            state.link_out = link(state.next_peer);
            if (!joined && connected) {
                joined = true;
                if (state.status == dan::ProviderUiStatus::connecting) {
                    state.status = dan::ProviderUiStatus::available;
                    state.message = "Waiting for a client to reserve this GPU";
                }
                dan::add_activity(state, "joined the DAN network"
                    + std::string(net.relays ? " (relay ready)" : ""));
            }
        });
        for (int tick = 0; tick < 10 && !stop.stop_requested(); ++tick) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

int run_serve_mode(std::shared_ptr<ServeContext> context, const std::string& status_file,
    const std::string& control_host,
    int control_port, const std::string& ring_host, int ring_port) {
    context->ring.ring_listener = listen_on(ring_host, ring_port);
    std::fprintf(stderr, "ring: listening on %s:%d\n", ring_host.c_str(), ring_port);
    const po::socket_t listener = listen_on(control_host, control_port);
    std::fprintf(stderr, "serving placement requests on %s:%d abi=%s\n",
        control_host.c_str(), control_port, runtime_abi);
    std::jthread ring_thread([context](std::stop_token stop) { run_ring(context->ring, stop); });
    std::jthread dashboard_thread;
    if (context->ui) {
        context->ring.on_route = [context](const po::RingRoute& route) {
            show(*context, [&](dan::ProviderUiState& state) {
                state.previous_peer = route.previous_peer;
                state.next_peer = peer_of_target(route.next);
                dan::add_activity(state, "linked " + (route.previous_peer.empty() ? std::string("client")
                    : dan::short_peer(route.previous_peer)) + " -> this node -> "
                    + dan::short_peer(state.next_peer));
            });
        };
        dashboard_thread = std::jthread([context](std::stop_token stop) { run_dashboard(context, stop); });
    }
    std::jthread status_thread;
    if (!status_file.empty()) {
        status_thread = std::jthread([context, status_file](std::stop_token stop) {
            run_status_writer(context, status_file, stop);
        });
    }
    while (!context->ring.shutdown.load() && !dan::platform::stop_requested()) {
        const po::socket_t client = accept(listener, nullptr, nullptr);
        if (client == po::invalid_socket) break;
        if (context->connections.load() >= 16) {
            po::close_socket(client);
            continue;
        }
        ++context->connections;
        std::thread([context, client] {
            serve_connection(*context, client);
            --context->connections;
        }).detach();
    }
    po::close_socket(listener);
    dan::platform::shutdown_socket(context->ring.ring_listener);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string model;
    std::string model_url;
    std::string model_revision;
    std::string model_sha256;
    std::string host = "127.0.0.1";
    std::string coordinator;
    std::string host_manifest;
    std::string serve_endpoint;
    std::string provider_listen;
    std::filesystem::path metadata_cache;
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
    // Ring topology (docs/PIPELINED_SPECULATION_V1.md phase 4): opt-in, off by default.
    // --next is where this stage forwards hot-path outcomes (data frames only -- prompt, token,
    // activation, speculative_activation, prompt_chunk -- both success and error) instead of
    // replying on the connection the frame arrived on. commit_token/commit_activation are
    // deliberately excluded: they belong to commit_final_token's direct per-stage exchange, not
    // the ring (see is_hot_path's own comment). For the first stage the ring-input connection is
    // the coordinator's; for every other stage it is --next of the stage before it. --ring-listen
    // is where a non-first stage accepts that forwarded connection; the first stage has none.
    // The last stage's --next points at the coordinator's return listener, so hot-path outcomes
    // -- including errors -- surface there without any stage needing to know it is last. Control
    // frames (session lifecycle, metrics, shutdown) are unaffected: they always reply on the
    // connection they arrived on, exactly as without --next.
    //
    // --prefill-chunk N (phase 5, ring mode only): the first stage splits a prompt longer than N
    // tokens into chunks, forwarding each one immediately via --next instead of waiting to
    // process the whole prompt before forwarding anything -- so the next stage can start on
    // chunk 1 while this stage is still on chunk 2. 0 (default) disables chunking; the option is
    // silently a no-op without --next, since there would be nowhere to stream chunks to ahead of
    // the final one.
    std::string next_endpoint;
    std::string ring_proxy;
    std::string ring_target;
    std::string ring_host = "0.0.0.0";
    int ring_port = 0;
    int prefill_chunk = 0;
    int connect_timeout_ms = 20000;
    // Static routed mode behind a dan-sidecar: control and ring connections start with the
    // sidecar's authenticated "DAN-P2P/1 <PeerID>" line, and route next hops go through
    // --ring-proxy. Both listeners must then be loopback-only so the line cannot be forged.
    bool peer_header = false;
    // Serve mode (decentralized placement, lease.hpp): accept client control connections
    // and load whatever stage a client reserves, from models in the worker's own catalog.
    std::string control_listen;
    std::vector<std::string> catalog_paths;
    std::string status_file;
    std::string net_status_file;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--tui") { tui = true; continue; }
            if (option == "--peer-header") { peer_header = true; continue; }
            if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
            const std::string value = argv[++index];
            if (option == "--model") model = value;
            else if (option == "--model-url") model_url = value;
            else if (option == "--model-revision") model_revision = value;
            else if (option == "--model-sha256") model_sha256 = value;
            else if (option == "--coordinator") coordinator = value;
            else if (option == "--host-coordinator") host_manifest = value;
            else if (option == "--serve") serve_endpoint = value;
            else if (option == "--provider-listen") provider_listen = value;
            else if (option == "--metadata-cache") metadata_cache = value;
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
            else if (option == "--next") next_endpoint = value;
            else if (option == "--ring-proxy") ring_proxy = value;
            else if (option == "--ring-target") ring_target = value;
            else if (option == "--ring-listen") {
                const std::size_t colon = value.rfind(':');
                if (colon == std::string::npos || colon == 0 || colon + 1 == value.size()) {
                    throw std::runtime_error("invalid --ring-listen address");
                }
                ring_host = value.substr(0, colon);
                ring_port = std::stoi(value.substr(colon + 1));
            }
            else if (option == "--prefill-chunk") prefill_chunk = std::stoi(value);
            else if (option == "--connect-timeout-ms") connect_timeout_ms = std::stoi(value);
            else if (option == "--control-listen") control_listen = value;
            else if (option == "--catalog") catalog_paths.push_back(value);
            else if (option == "--status-file") status_file = value;
            else if (option == "--net-status-file") net_status_file = value;
            else throw std::runtime_error("unknown option: " + option);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dan-stage-worker: %s\n", error.what());
        return 2;
    }
    const bool hosted = !host_manifest.empty();
    const bool remote_coordinator = !coordinator.empty();
    if (hosted && provider_listen.empty()) provider_listen = "0.0.0.0:50201";
    if (hosted && metadata_cache.empty()) metadata_cache = cache_dir / "coordinator-model-index.gguf";
    if (hosted) {
        const std::size_t colon = provider_listen.rfind(':');
        if (colon != std::string::npos) {
            const std::string_view listen_host(provider_listen.data(), colon);
            coordinator = (listen_host == "0.0.0.0" || listen_host == "::" ? "127.0.0.1"
                : std::string(listen_host)) + provider_listen.substr(colon);
        }
    }
    const bool generic = !coordinator.empty();
    const bool serve = !control_listen.empty();
    std::string control_host;
    int control_port = 0;
    if (serve) {
        const std::size_t colon = control_listen.rfind(':');
        if (colon != std::string::npos && colon != 0 && colon + 1 != control_listen.size()
            && po::valid_endpoint(control_listen)) {
            control_host = control_listen.substr(0, colon);
            control_port = std::stoi(control_listen.substr(colon + 1));
        }
    }
    if ((generic || serve) && (gpu_name.empty() || offered_vram_mib == 0)) {
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
    if ((!generic && !serve && (model.empty() || port < 1 || port > 65535))
        || (serve && (generic || control_port == 0 || !model.empty() || begin != -1
            || end != -1 || port != 0 || !next_endpoint.empty() || catalog_paths.empty()
            || provider_id.empty() || gpu_name.empty() || cache_dir.empty()
            || offered_vram_mib == 0 || ring_port == 0
            || ring_host == "0.0.0.0" || ring_host == "::"
            || (peer_header ? !po::valid_p2p_target(ring_target) || control_host != "127.0.0.1"
                : !ring_target.empty())))
        || (generic && (provider_id.empty() || gpu_name.empty() || cache_dir.empty()
            || offered_vram_mib == 0)) || context < 1 || max_sessions < 1
        || (hosted && (serve_endpoint.empty() || provider_listen.empty()
            || metadata_cache.empty() || remote_coordinator))
        || (!hosted && (!serve_endpoint.empty() || !provider_listen.empty()
            || !metadata_cache.empty()))
        || (generic && ring_port != 0 && (ring_host == "0.0.0.0" || ring_host == "::"))
        || (!generic && !serve && !ring_target.empty())
        || (!serve && (!status_file.empty() || !net_status_file.empty()))
        || (!generic && (!ring_proxy.empty() != peer_header
            || (!ring_proxy.empty() && !po::valid_endpoint(ring_proxy))))
        || (!generic && !serve && peer_header && (!next_endpoint.empty() || host != "127.0.0.1"
            || (ring_port != 0 && ring_host != "127.0.0.1")))
        || (generic && peer_header)
        || (generic && ((!ring_proxy.empty() || !ring_target.empty())
            && (ring_proxy.empty() || ring_target.empty() || ring_port == 0
                || !po::valid_endpoint(ring_proxy) || !po::valid_ring_target(ring_target))))
        || (ring_port != 0 && (ring_port < 1 || ring_port > 65535))
        || prefill_chunk < 0 || connect_timeout_ms < 1000) {
        std::fprintf(stderr,
            "usage: dan-stage-worker (--coordinator HOST:PORT | --host-coordinator MANIFEST --serve HOST:PORT [--provider-listen HOST:PORT] [--metadata-cache FILE]) --provider-id ID --gpu NAME --vram-mib N --cache-dir DIR | --model FILE --stage-start N --stage-end N --host IP --port N [--next HOST:PORT | [--peer-header --ring-proxy HOST:PORT]] [--ring-listen HOST:PORT] [ring/model options] | --control-listen HOST:PORT --catalog MANIFEST [...] --ring-listen HOST:PORT [--peer-header --ring-proxy HOST:PORT --ring-target MULTIADDR] --provider-id ID --cache-dir DIR [--gpu NAME --vram-mib N] [--ctx MAX] [--max-sessions MAX] [--status-file FILE] [--net-status-file FILE] [--connect-timeout-ms 20000] [--tui]\n");
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
    // A stop signal (Ctrl+C, or the launcher exiting) sets a flag that loops check between
    // frames -- but a worker blocked in accept()/recv() would sit there holding its GPU
    // memory forever. Leave a short grace period for a clean stop, then exit anyway.
    // No parent-process check here: on POSIX dan-provider execs into this worker, so the
    // worker *is* the launcher and its parent is only the shell that started it. On Windows
    // the launcher keeps its children in a job object that dies with it.
    std::thread([] {
        while (!dan::platform::stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
        std::fprintf(stderr, "stopping now\n");
        std::fflush(nullptr);
        std::_Exit(0);
    }).detach();
    dan::platform::configure_output();

    // Serve mode keeps its log next to its status file (the node's own state folder).
    const auto diagnostics = serve && !status_file.empty()
        ? std::filesystem::path(status_file).parent_path() / "logs" / "stage-worker.log"
        : dan::platform::data_directory() / "logs" / "provider-owned.log";
    const bool dashboard = tui && (generic || serve);
    if (dashboard && !redirect_diagnostics(diagnostics)) {
        std::fprintf(stderr, "dan-stage-worker: could not open diagnostics log\n");
        tui = false;
    }
    const bool show_dashboard = tui && (generic || serve);
    dan::ProviderUiState initial_ui;
    initial_ui.gpu_name = gpu_name;
    initial_ui.offered_vram_mib = static_cast<std::size_t>(offered_vram_mib);
    initial_ui.status = dan::ProviderUiStatus::connecting;
    initial_ui.message = "Connecting to DAN automatically...";
    initial_ui.diagnostics = diagnostics.string();
    if (serve) {
        initial_ui.dht_mode = true;
        initial_ui.message = "Joining the DAN network...";
        initial_ui.peer_id = peer_of_target(ring_target);
    }
    dan::ProviderTerminalUi terminal_ui(std::move(initial_ui), show_dashboard);
    dan::ProviderTerminalUi* ui = show_dashboard ? &terminal_ui : nullptr;
    dan::platform::Process hosted_coordinator;
    if (hosted) {
        const auto executable = dan::platform::current_executable(platform_error).parent_path()
            / (dan::platform::is_windows() ? "dan-provider-owned-coordinator.exe"
                : "dan-provider-owned-coordinator");
        if (!platform_error.empty() || !dan::platform::executable_file(executable)
            || !hosted_coordinator.start({executable.string(), "--manifest", host_manifest,
                    "--provider-listen", provider_listen, "--metadata-cache",
                    metadata_cache.string(), "--listen", serve_endpoint,
                    "--shutdown-workers"}, platform_error)) {
            std::fprintf(stderr, "dan-stage-worker: could not host coordinator: %s\n",
                platform_error.empty() ? "coordinator runtime is missing" : platform_error.c_str());
            return 1;
        }
        std::fprintf(stderr, "provider hosting coordinator: providers=%s clients=%s\n",
            provider_listen.c_str(), serve_endpoint.c_str());
    }
    int exit_code = 0;
    try {
        if (serve) {
            auto serve_context = std::make_shared<ServeContext>();
            ServeContext& serving = *serve_context;
            serving.cache_dir = cache_dir;
            serving.gpu_layers = gpu_layers;
            serving.prefill_chunk = static_cast<std::size_t>(prefill_chunk);
            serving.ring.p2p = peer_header;
            serving.ring.ring_proxy = ring_proxy;
            serving.ring.connect_budget = std::chrono::milliseconds(connect_timeout_ms);
            serving.ui = ui;
            serving.net_status_file = net_status_file;
            for (const std::string& catalog_path : catalog_paths) {
                po::Manifest manifest = po::load_manifest(catalog_path);
                const std::string key = lowercase(manifest.sha256);
                po::ModelIndex index;
                std::string error;
                if (!po::inspect_range_model({manifest.url, manifest.revision, manifest.sha256,
                        cache_dir / "metadata" / (key + ".gguf"), 0, 1}, index, error)) {
                    throw std::runtime_error("catalog model " + manifest.model_id + ": " + error);
                }
                std::string incompatibility;
                if (!po::compatible_dense_qwen2(index, &incompatibility)
                    || (manifest.layers != 0 && manifest.layers != index.layers)
                    || (manifest.hidden != 0 && manifest.hidden != index.hidden)) {
                    throw std::runtime_error("catalog model " + manifest.model_id
                        + " does not match its GGUF: " + incompatibility);
                }
                std::fprintf(stderr, "catalog: %s sha256=%s layers=%u hidden=%u\n",
                    manifest.model_id.c_str(), key.c_str(), index.layers, index.hidden);
                serving.hello.models.push_back(key);
                serving.catalog[key] = {std::move(manifest), std::move(index)};
            }
            serving.hello.id = provider_id;
            serving.hello.gpu = gpu_name;
            serving.hello.offered_vram_mib = offered_vram_mib;
            serving.hello.ring_endpoint = peer_header ? ring_target
                : ring_host + ':' + std::to_string(ring_port);
            serving.hello.runtime_abi = runtime_abi;
            serving.hello.max_context = static_cast<std::uint32_t>(context);
            serving.hello.max_sessions = static_cast<std::uint32_t>(max_sessions);
            return run_serve_mode(serve_context, status_file, control_host, control_port,
                ring_host, ring_port);
        }
        if (generic) {
            bool shutdown = false;
            std::unique_ptr<Stage> stage;
            std::optional<po::ModelAssignment> loaded_assignment;
            while (!shutdown && !dan::platform::stop_requested()) {
                std::atomic<po::socket_t> automatic_next{po::invalid_socket};
                std::atomic<po::socket_t> automatic_predecessor{po::invalid_socket};
                const po::socket_t automatic_ring_listener = ring_port == 0
                    ? po::invalid_socket : listen_on(ring_host, ring_port);
                std::mutex stage_mutex;
                std::jthread automatic_ring_thread;
                const po::socket_t coordinator_socket = wait_for_coordinator(
                    coordinator, ui, hosted ? &hosted_coordinator : nullptr);
                if (coordinator_socket == po::invalid_socket) {
                    if (automatic_ring_listener != po::invalid_socket) {
                        po::close_socket(automatic_ring_listener);
                    }
                    if (hosted && !dan::platform::stop_requested()) {
                        std::fprintf(stderr, "hosted coordinator stopped unexpectedly\n");
                        exit_code = 1;
                    }
                    break;
                }
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
                        {provider_id, gpu_name, offered_vram_mib, ring_port == 0 ? ""
                            : (ring_target.empty() ? ring_host + ':'
                                + std::to_string(ring_port) : ring_target)});
                    available.payload.assign(capabilities.begin(), capabilities.end());
                    if (!po::send_frame(coordinator_socket, available, error)) {
                        throw std::runtime_error(error);
                    }
                    if (ui) ui->update([](auto& state) {
                        state.network_connected = true;
                        state.status = dan::ProviderUiStatus::available;
                        state.message = "Connected. Waiting for useful work...";
                    });
                    while (!dan::platform::stop_requested()) {
                        po::Frame input;
                        if (!po::recv_frame(coordinator_socket, input, error)) {
                            throw std::runtime_error(error);
                        }
                        if (input.type == po::Type::assign_stage) {
                            po::ModelAssignment assignment;
                            const std::string text(input.payload.begin(), input.payload.end());
                            if (!po::parse_assignment(text, assignment)) {
                                throw std::runtime_error("invalid stage assignment");
                            }
                            if (!loaded_assignment || !loaded_assignment->same_stage(assignment)) {
                                stage.reset();
                                loaded_assignment.reset();
                                const std::size_t free_before_load = gpu_free_mib();
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
                                loaded_assignment = assignment;
                                if (ui) ui->update([&](auto& state) {
                                    const std::size_t free_now = gpu_free_mib();
                                    state.used_vram_mib = free_before_load > free_now
                                        ? free_before_load - free_now : 0;
                                });
                            } else {
                                loaded_assignment = assignment;
                                std::fprintf(stderr, "Reusing loaded stage after coordinator reconnect.\n");
                            }
                            const auto send_ready = [&] {
                                po::Frame ready;
                                ready.type = po::Type::stage_ready;
                                ready.payload.assign(provider_id.begin(), provider_id.end());
                                if (!po::send_frame(coordinator_socket, ready, error)) {
                                    throw std::runtime_error(error);
                                }
                            };
                            const auto connect_next = [&] {
                                std::fprintf(stderr, "ring: connecting to assigned next hop at %s\n",
                                    assignment.next_endpoint.c_str());
                                for (;;) {
                                    const po::socket_t socket = wait_for_coordinator(
                                        ring_proxy.empty() ? assignment.next_endpoint : ring_proxy,
                                        nullptr);
                                    if (socket == po::invalid_socket) {
                                        throw std::runtime_error("ring connection stopped");
                                    }
                                    if ((!ring_proxy.empty()
                                            && !connect_ring_proxy(socket, assignment.next_endpoint))) {
                                        std::fprintf(stderr, "ring proxy could not reach assigned peer\n");
                                    } else if (ring_handshake_connect(socket)) {
                                        automatic_next.store(socket);
                                        break;
                                    }
                                    po::close_socket(socket);
                                    std::this_thread::sleep_for(std::chrono::seconds(2));
                                }
                                std::fprintf(stderr, "ring: connected to assigned next hop\n");
                            };
                            if (!assignment.next_endpoint.empty()) {
                                if (automatic_ring_listener == po::invalid_socket) {
                                    throw std::runtime_error("ring assignment requires --ring-listen");
                                }
                                if (assignment.begin != 0) {
                                    automatic_ring_thread = std::jthread([&](std::stop_token stop) {
                                        po::socket_t predecessor = po::invalid_socket;
                                        while (!stop.stop_requested()) {
                                            predecessor = accept(automatic_ring_listener, nullptr, nullptr);
                                            if (predecessor == po::invalid_socket) break;
                                            automatic_predecessor.store(predecessor);
                                            if (!assignment.previous_peer_id.empty()) {
                                                std::string peer_id;
                                                if (!po::recv_peer_id(predecessor, peer_id)
                                                    || peer_id != assignment.previous_peer_id) {
                                                    std::fprintf(stderr,
                                                        "ring: rejected unexpected predecessor peer\n");
                                                    automatic_predecessor.store(po::invalid_socket);
                                                    po::close_socket(predecessor);
                                                    predecessor = po::invalid_socket;
                                                    continue;
                                                }
                                            }
                                            if (ring_handshake_accept(predecessor)) break;
                                            automatic_predecessor.store(po::invalid_socket);
                                            po::close_socket(predecessor);
                                            predecessor = po::invalid_socket;
                                        }
                                        if (predecessor == po::invalid_socket) {
                                            if (!stop.stop_requested()) {
                                                dan::platform::shutdown_socket(coordinator_socket);
                                            }
                                            return;
                                        }
                                        std::fprintf(stderr, "ring: assigned predecessor connected\n");
                                        while (!stop.stop_requested()) {
                                            po::Frame ring_input;
                                            std::string ring_error;
                                            if (!po::recv_frame(predecessor, ring_input, ring_error)) break;
                                            po::Frame ring_output;
                                            bool errored = false;
                                            {
                                                std::lock_guard lock(stage_mutex);
                                                try { ring_output = stage->handle(ring_input); }
                                                catch (const std::exception& exception) {
                                                    ring_output = po::error_frame(ring_input, exception.what());
                                                    errored = true;
                                                }
                                            }
                                            const bool drop = !errored && stage->last()
                                                && ring_input.type == po::Type::prompt_chunk;
                                            const po::socket_t next = automatic_next.load();
                                            if (!drop && (next == po::invalid_socket
                                                    || !po::send_frame(next, ring_output, ring_error))) break;
                                        }
                                        automatic_predecessor.store(po::invalid_socket);
                                        po::close_socket(predecessor);
                                        dan::platform::shutdown_socket(coordinator_socket);
                                    });
                                }
                                if (stage->last()) { send_ready(); connect_next(); }
                                else { connect_next(); send_ready(); }
                            } else send_ready();
                            if (ui) ui->update([&](auto& state) {
                                state.status = dan::ProviderUiStatus::contributing;
                                state.message = "Ready. Waiting for inference requests...";
                            });
                            continue;
                        }
                        if (input.type == po::Type::unload_stage) {
                            stage.reset();
                            loaded_assignment.reset();
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
                        {
                            std::lock_guard lock(stage_mutex);
                            try {
                                const auto emit_chunk = [&](const po::Frame& chunk) {
                                    std::string chunk_error;
                                    const po::socket_t next = automatic_next.load();
                                    if (next == po::invalid_socket
                                        || !po::send_frame(next, chunk, chunk_error)) {
                                        throw std::runtime_error(chunk_error.empty()
                                            ? "ring next hop unavailable" : chunk_error);
                                    }
                                };
                                output = stage->handle(input,
                                    static_cast<std::size_t>(prefill_chunk), emit_chunk);
                            } catch (const std::exception& exception) {
                                output = po::error_frame(input, exception.what());
                            }
                        }
                        const bool hot = input.type == po::Type::prompt
                            || input.type == po::Type::token || input.type == po::Type::activation
                            || input.type == po::Type::speculative_activation;
                        const po::socket_t next = automatic_next.load();
                        const po::socket_t reply = hot && next != po::invalid_socket
                            ? next : coordinator_socket;
                        if (!po::send_frame(reply, output, error)) {
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
                    if (stage) {
                        std::lock_guard lock(stage_mutex);
                        stage->coordinator_disconnected();
                    }
                    if (ui) ui->update([&](auto& state) {
                        state.network_connected = false;
                        state.status = dan::ProviderUiStatus::reconnecting;
                        state.download_percent = -1;
                        state.message = std::string(failure.what())
                            + ". Reconnecting automatically...";
                    });
                }
                stop_watcher.request_stop();
                stop_watcher.join();
                if (const auto socket = automatic_next.exchange(po::invalid_socket);
                    socket != po::invalid_socket) {
                    dan::platform::shutdown_socket(socket); po::close_socket(socket);
                }
                if (const auto socket = automatic_predecessor.exchange(po::invalid_socket);
                    socket != po::invalid_socket) {
                    dan::platform::shutdown_socket(socket);
                }
                if (automatic_ring_listener != po::invalid_socket) {
                    dan::platform::shutdown_socket(automatic_ring_listener);
                    po::close_socket(automatic_ring_listener);
                }
                if (automatic_ring_thread.joinable()) {
                    automatic_ring_thread.request_stop();
                    automatic_ring_thread.join();
                }
                po::close_socket(coordinator_socket);
                if (!shutdown && !dan::platform::stop_requested()) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
            return exit_code;
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
        // Ring topology (docs/PIPELINED_SPECULATION_V1.md phase 4). Off by default (no next hop,
        // no ring thread): every code path falls back exactly to the pre-ring behavior.
        // Next hop: fixed for the whole process by --next (legacy), or chosen per route by the
        // client in create_session (route.hpp). In routed mode one client owns the route at a
        // time; it is released when that client's control connection closes.
        RingState ring;
        ring.stage = &stage;
        ring.p2p = !ring_proxy.empty();
        ring.ring_proxy = ring_proxy;
        ring.connect_budget = std::chrono::milliseconds(connect_timeout_ms);
        const bool routed = next_endpoint.empty();
        // Bind every listener this stage owns -- the ring-input listener (if any) and the
        // regular control listener -- before attempting the blocking --next connect. A
        // predecessor (or, for the control listener, the coordinator) only needs the listener
        // bound to connect successfully; the OS accept queue holds the connection until this
        // process calls accept(), so binding early and accepting late is safe. Binding late,
        // as an earlier version of this function did for both listeners, deadlocks any ring:
        // the last stage would block dialing the coordinator's return listener before its own
        // control listener is even open for the coordinator to reach in the first place, and
        // the coordinator won't open its return listener until every stage's control port has
        // answered -- an unbreakable circular wait with no ordering of stage/coordinator
        // startup that resolves it.
        ring.ring_listener = ring_port != 0 ? listen_on(ring_host, ring_port) : po::invalid_socket;
        if (ring.ring_listener != po::invalid_socket) {
            std::fprintf(stderr, "ring: listening on %s:%d\n", ring_host.c_str(), ring_port);
        }
        const po::socket_t listener = listen_on(host, port);
        std::fprintf(stderr, "listening on %s:%d\n", host.c_str(), port);
        if (!next_endpoint.empty()) {
            std::fprintf(stderr, "ring: connecting to next hop at %s\n", next_endpoint.c_str());
            for (;;) {
                const po::socket_t fixed = wait_for_coordinator(next_endpoint, nullptr);
                if (fixed == po::invalid_socket) return 0; // stop requested while connecting
                if (ring_handshake_connect(fixed)) { ring.next.store(fixed); break; }
                std::fprintf(stderr,
                    "ring: next hop accepted the connection but never answered the handshake "
                    "(stale tunnel?) -- retrying\n");
                po::close_socket(fixed);
                if (dan::platform::stop_requested()) return 0;
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            std::fprintf(stderr, "ring: connected to next hop\n");
        }
        std::jthread ring_thread;
        if (ring.ring_listener != po::invalid_socket) {
            ring_thread = std::jthread([&ring](std::stop_token stop) { run_ring(ring, stop); });
        }
        while (!ring.shutdown.load()) {
            const po::socket_t client = accept(listener, nullptr, nullptr);
            if (client == po::invalid_socket) throw std::runtime_error("accept failed");
            if (peer_header) {
                std::string peer_id;
                if (!read_peer_header(client, peer_id)) {
                    std::fprintf(stderr, "rejected control connection without a PeerID\n");
                    po::close_socket(client);
                    continue;
                }
                std::fprintf(stderr, "control connection from peer %s\n", peer_id.c_str());
            }
            serve_control(ring, client, static_cast<std::size_t>(prefill_chunk), routed);
            po::close_socket(client);
        }
        po::close_socket(listener);
        if (ring_thread.joinable()) {
            dan::platform::shutdown_socket(ring.ring_listener);
            po::close_socket(ring.ring_listener);
            ring_thread.request_stop();
            ring_thread.join();
        }
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
