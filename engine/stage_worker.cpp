#include "llama.h"
#include "provider_owned/chat.hpp"
#include "log.h"
#include "ngram-map.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "provider_owned/protocol.hpp"
#include "provider_owned/activations.hpp"
#include "provider_owned/formation.hpp"
#include "provider_owned/lease.hpp"
#include "provider_owned/manifest.hpp"
#include "provider_owned/measurement.hpp"
#include "provider_owned/planner.hpp"
#include "provider_owned/range_model.hpp"
#include "provider_owned/route.hpp"
#include "provider_owned/speculation.hpp"
#include "provider_owned/vocab_compat.hpp"
#include "provider_ui.hpp"
#include "platform.hpp"

#include <bit>
#include <algorithm>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#else
#include <sys/ioctl.h>
#endif
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
#include <map>
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

// Worker-owned setting; placement continues to reserve the conservative F16 size.
ggml_type kv_cache_type = GGML_TYPE_F16;
std::uint32_t prefill_batch = 512;
std::uint32_t draft_width = 4;  // adaptive cap, or fixed verify width; includes current token
bool adaptive_draft = true;
bool ngram_draft = false;
bool continuous_batching = false;

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
    (void) values;
    return frame.payload.size() == 8 + po::activation_bytes(frame.dtype, frame.rows, frame.cols) + tail
        ? tail : 0;
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

struct DraftState {
    po::AdaptiveDraftWidth policy{draft_width};
    // During ordinary decode the draft does no compute. Replay these committed inputs
    // before the next probe or session control; token history never leaves session RAM.
    std::vector<std::uint32_t> pending;
    std::uint32_t at = 0;
    std::uint64_t request = 0;
};

class Session {
public:
    Session(llama_seq_id sequence, bool last) : sequence(sequence) {
        if (last) {
            sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (!sampler) throw std::runtime_error("session sampler creation failed");
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }
    }

    ~Session() {
        if (sampler) llama_sampler_free(sampler);
        if (chat_sampler) common_sampler_free(chat_sampler);
    }

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    llama_seq_id sequence = 0;
    llama_sampler* sampler = nullptr;
    std::uint32_t position = 0;
    std::uint64_t active_request = 0;
    std::uint64_t last_request = 0;
    bool has_prompt = false;
    DraftState draft;
    common_sampler* chat_sampler = nullptr;
    bool chat_request = false;
    bool chat_speculative = false;
    bool chat_pending = false;
    llama_tokens history;
    llama_tokens chat_tokens;
};

// The GPU this worker computes on, chosen with --device. Null: llama.cpp's default (every
// GPU it finds), with memory read from the first one.
ggml_backend_dev_t bound_device = nullptr;

bool gpu_device(ggml_backend_dev_t device) {
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

// nvidia-smi writes "00000000:07:00.0", ggml "0000:07:00.0": the same device.
std::string normalized_pci(std::string_view id) {
    std::string value;
    for (const unsigned char byte : id) value += static_cast<char>(std::tolower(byte));
    const std::size_t colon = value.find(':');
    if (colon == std::string::npos) return value;
    std::string domain = value.substr(0, colon);
    domain.erase(0, std::min(domain.find_first_not_of('0'), domain.size()));
    if (domain.size() < 4) domain.insert(0, 4 - domain.size(), '0');
    return domain + value.substr(colon);
}

// A GPU by PCI bus ID or llama.cpp device name (CUDA0, Vulkan1, MTL0), or null.
ggml_backend_dev_t find_device(const std::string& selector) {
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (!gpu_device(device)) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        if ((props.device_id && normalized_pci(props.device_id) == normalized_pci(selector))
            || selector == ggml_backend_dev_name(device)) return device;
    }
    return nullptr;
}

// Where free memory is read: the bound GPU, else the first GPU llama.cpp found.
ggml_backend_dev_t memory_device() {
    if (bound_device) return bound_device;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        if (gpu_device(ggml_backend_dev_get(index))) return ggml_backend_dev_get(index);
    }
    return nullptr;
}

class Stage {
public:
    // `head_k`/`head_v` are the model's declared attention head widths (0 = derive
    // hidden/heads). They only size the reported KV metric; llama.cpp allocates the real KV.
    Stage(const std::string& path, int begin, int end, int context_size,
        int gpu_layers, std::size_t max_sessions,
        std::uint32_t head_k = 0, std::uint32_t head_v = 0)
        : begin_(begin), end_(end), context_size_(context_size),
          max_sessions_(max_sessions), sequence_used_(max_sessions, false),
          started_(std::chrono::steady_clock::now()) {
        ggml_backend_load_all();
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        // Only the device this worker advertised: never spill onto another local GPU.
        static ggml_backend_dev_t devices[2] = {nullptr, nullptr};
        if (bound_device) {
            devices[0] = bound_device;
            params.devices = devices;
        }
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
        // A long context must not reserve one enormous prefill compute graph.
        context_params.n_ubatch = std::min(static_cast<std::uint32_t>(context_size_), prefill_batch);
        context_params.n_seq_max = static_cast<std::uint32_t>(max_sessions_);
        // The tail samples one prompt output or at most one supported verify batch.
        // Reuse llama.cpp's output bound to avoid reserving a vocabulary-sized output
        // for every prefill token. Intermediate stages still expose every hidden row.
        if (last_) {
            context_params.n_outputs_max = std::max(context_params.n_seq_max,
                std::min(context_params.n_batch, po::max_speculative_width));
        }
        context_params.type_k = kv_cache_type;
        context_params.type_v = kv_cache_type;
        if (kv_cache_type != GGML_TYPE_F16) {
            context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        // Every non-final stage must expose its boundary hidden state.
        context_params.embeddings = !last_;
        context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        context_ = llama_init_from_model(model_, context_params);
        if (!context_) throw std::runtime_error("shared session context creation failed");
        // Not hidden/heads in general: Qwen3-MoE declares a head width of 128 where that
        // quotient is 64, so the derived value would halve this metric.
        const std::uint64_t derived = static_cast<std::uint64_t>(hidden_)
            / static_cast<std::uint64_t>(llama_model_n_head(model_));
        const std::uint64_t width_k = head_k != 0 ? head_k : derived;
        const std::uint64_t width_v = head_v != 0 ? head_v : derived;
        kv_bytes_per_session_ = static_cast<std::uint64_t>(llama_n_ctx_seq(context_))
            * static_cast<std::uint64_t>(end_ - begin_)
            * (ggml_row_size(kv_cache_type, width_k
                    * static_cast<std::uint64_t>(llama_model_n_head_kv(model_)))
                + ggml_row_size(kv_cache_type, width_v
                    * static_cast<std::uint64_t>(llama_model_n_head_kv(model_))));
        startup_ns_ = elapsed_ns(started_);
        std::fprintf(stderr,
            "DAN stage READY: layers %d..%d, hidden %d, role %s, startup_ms=%.3f\n",
            begin_, end_ - 1, hidden_, first_ && last_ ? "single"
                : (first_ ? "first" : (last_ ? "last" : "middle")),
            startup_ns_ / 1e6);
    }

    ~Stage() {
        chat_templates_.reset();
        speculative_.reset();
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
        case po::Type::prepare_chat: return prepare_chat(input);
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
    int context_size() const { return context_size_; }
    std::size_t max_sessions() const { return max_sessions_; }
    const llama_model* model() const { return model_; }
    DraftState& draft_state(const po::Frame& frame) { return require_session(frame).draft; }
    bool can_draft(const po::Frame& frame) {
        const auto found = sessions_.find(frame.session);
        return found == sessions_.end() || !found->second->chat_request || found->second->chat_speculative;
    }
    bool can_model_draft(const po::Frame& frame) {
        const auto found = sessions_.find(frame.session);
        return found == sessions_.end() || !found->second->chat_request;
    }
    bool batchable(const po::Frame& frame, std::uint64_t size) const {
        const auto found = sessions_.find(frame.session);
        if (found == sessions_.end()) return false;
        const auto& session = *found->second;
        if (!frame.request || session.active_request != frame.request || !session.has_prompt
            || frame.position != session.position || frame.position >= static_cast<std::uint32_t>(context_size_)) return false;
        if (first_) return frame.type == po::Type::token && !frame.rows && !frame.cols
            && frame.dtype == po::DType::none && size == 4;
        return frame.type == po::Type::activation && frame.rows == 1
            && frame.cols == static_cast<std::uint32_t>(hidden_) && po::activation_dtype(frame.dtype)
            && size == 8 + po::activation_bytes(frame.dtype, 1, frame.cols);
    }
    // DAN maps ready wire frames to upstream sequence IDs. llama.cpp owns batched
    // attention, KV and execution; there is no batching delay or global scheduler.
    std::vector<po::Frame> decode_batch(const std::vector<po::Frame>& inputs) {
        std::vector<po::Frame> outputs;
        llama_batch batch = llama_batch_init(static_cast<int32_t>(inputs.size()), first_ ? 0 : hidden_, 1);
        struct FreeBatch { llama_batch& batch; ~FreeBatch() { llama_batch_free(batch); } } free{batch};
        batch.n_tokens = static_cast<int32_t>(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (!batchable(input, input.payload.size())) throw std::runtime_error("invalid decode batch");
            auto& session = require_session(input);
            if (first_) {
                const auto token = po::get32(input.payload.data());
                if (token >= static_cast<std::uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model_)))) {
                    throw std::runtime_error("invalid token in decode batch");
                }
                batch.token[i] = static_cast<llama_token>(token);
            } else get_hidden(input, batch.embd + i * hidden_, hidden_);
            batch.pos[i] = input.position;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = session.sequence;
            batch.logits[i] = true;
        }
        const auto started = std::chrono::steady_clock::now();
        if (llama_decode(context_, batch) != 0) throw std::runtime_error("batched decode failed");
        llama_synchronize(context_);
        const auto compute = elapsed_ns(started) / inputs.size();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            auto& session = require_session(input);
            ++session.position;
            if (first_) session.history.push_back(batch.token[i]);
            auto output = po::frame_header(input);
            output.rows = output.cols = 0;
            output.dtype = po::DType::none;
            if (last_) {
                const auto token = sample(session, static_cast<int>(i));
                const auto text = text_of(token);
                output.type = po::Type::result;
                output.position = session.position;
                output.payload.resize(13 + text.size());
                po::put32(output.payload.data(), token);
                po::put64(output.payload.data() + 4, compute);
                output.payload[12] = ends_text(token) ? 1 : 0;
                std::memcpy(output.payload.data() + 13, text.data(), text.size());
                ++tokens_generated_;
            } else {
                output.type = po::Type::activation;
                output.rows = 1;
                output.cols = hidden_;
                if (!put_hidden(output, 1, i)) throw std::runtime_error("batch returned no hidden state");
                po::put64(output.payload.data(), compute);
            }
            outputs.push_back(std::move(output));
        }
        tokens_processed_ += inputs.size();
        ++decode_batches_;
        return outputs;
    }
    std::vector<std::uint32_t> ngram_proposals(const po::Frame& frame, std::uint32_t count) {
        auto& session = require_session(frame);
        count = std::min(count, po::speculation_room(context_size_, frame.position));
        if (count < 3) return {};
        if (frame.position > session.history.size()) throw std::runtime_error("n-gram history mismatch");
        session.history.resize(frame.position);
        const auto tokens = common_ngram_simple_draft({3, static_cast<std::uint16_t>(count)},
            session.history, static_cast<llama_token>(po::get32(frame.payload.data())));
        return {tokens.begin(), tokens.end()};
    }
    std::vector<std::uint32_t> model_proposals(Stage& target, const po::Frame& input, std::uint32_t count) {
        auto& session = require_session(input);
        count = std::min(count, po::speculation_room(context_size_, input.position));
        if (!count) return {};
        if (!llama_memory_seq_rm(llama_get_memory(context_), session.sequence, input.position, -1)) {
            throw std::runtime_error("draft rollback failed");
        }
        if (!speculative_ || speculative_target_ != target.context_) {
            common_params_speculative params;
            params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
            params.draft.ctx_tgt = target.context_;
            params.draft.ctx_dft = context_;
            params.draft.n_max = po::max_speculative_width - 1;
            params.draft.backend_sampling = false; // preserve current memory/readback behavior
            speculative_.reset(common_speculative_init(params, static_cast<std::uint32_t>(max_sessions_)));
            if (!speculative_) throw std::runtime_error("upstream draft initialization failed");
            speculative_target_ = target.context_;
        }
        llama_tokens result;
        auto& params = common_speculative_get_draft_params(speculative_.get(), session.sequence);
        params.drafting = true;
        params.n_max = static_cast<int32_t>(count);
        params.n_past = input.position;
        params.id_last = static_cast<llama_token>(po::get32(input.payload.data()));
        params.prompt = &session.history;
        params.result = &result;
        common_speculative_draft(speculative_.get());
        session.position = static_cast<std::uint32_t>(
            llama_memory_seq_pos_max(llama_get_memory(context_), session.sequence) + 1);
        session.history.resize(input.position);
        session.history.push_back(params.id_last);
        for (auto token : result) {
            if (session.history.size() >= session.position) break;
            session.history.push_back(token);
        }
        return {result.begin(), result.end()};
    }
    bool ends_text(std::uint32_t token) const {
        return llama_vocab_is_eog(llama_model_get_vocab(model_), static_cast<llama_token>(token));
    }

    bool shutting_down() const { return shutting_down_; }
    std::uint64_t tokens_processed() const { return tokens_processed_; }
    // Typical time of one decode step (a few tokens at most; prompts excluded), 0 before the
    // first. Decoding is memory-bound, so this measures how fast this GPU runs its layers.
    std::uint64_t step_ns() const { return step_ns_.load(); }
    // For the greeting (upgrade.hpp): how long since this stage last computed, and how many
    // sessions are open on it. Called with the stage mutex held.
    std::uint32_t idle_seconds() const {
        const auto last = std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(last_compute_.load()));
        const auto since = std::chrono::steady_clock::now() - std::max(last, started_);
        return static_cast<std::uint32_t>(std::min<long long>(
            std::chrono::duration_cast<std::chrono::seconds>(since).count(), UINT32_MAX));
    }
    std::uint32_t open_sessions() const { return static_cast<std::uint32_t>(sessions_.size()); }
    void set_wire(po::DType dtype) { wire_ = dtype; }
    std::uint64_t requests_served() const { return requests_served_; }

    void coordinator_disconnected() {
        speculative_.reset();
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

    po::Frame prepare_chat(const po::Frame& input) {
        Session& session = require_session(input);
        if (session.active_request || input.request || input.rows || input.cols
            || input.dtype != po::DType::none || input.payload.empty()
            || input.payload.size() > 1024 * 1024) throw std::runtime_error("invalid chat preparation");
        try {
            if (!chat_templates_) chat_templates_ = common_chat_templates_init(model_, "");
            auto prepared = po::prepare_chat(model_, chat_templates_.get(), std::string(input.payload.begin(), input.payload.end()));
            const auto tokens = tokenize(llama_model_get_vocab(model_),
                std::vector<std::uint8_t>(prepared.chat.prompt.begin(), prepared.chat.prompt.end()));
            if (tokens.empty() || tokens.size() >= static_cast<std::size_t>(context_size_)) {
                throw std::runtime_error("session context exhausted");
            }
            std::uint32_t reuse = input.position;
            if (first_) {
                reuse = 0;
                const auto limit = std::min(session.history.size(), tokens.size() - 1);
                while (reuse < limit && session.history[reuse] == tokens[reuse]) ++reuse;
            }
            if (reuse > session.position || reuse >= tokens.size()) throw std::runtime_error("invalid cached prefix");
            std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(nullptr, common_sampler_free);
            if (last_) {
                sampler.reset(common_sampler_init(model_, prepared.sampling));
                if (!sampler) throw std::runtime_error("sampler creation failed");
                for (const auto token : tokens) common_sampler_accept(sampler.get(), token, false);
            }
            if (!llama_memory_seq_rm(llama_get_memory(context_), session.sequence, reuse, -1)) {
                throw std::runtime_error("could not trim cached prefix");
            }
            session.position = reuse;
            if (first_) {
                session.history.resize(reuse);
                session.chat_tokens.assign(tokens.begin() + reuse, tokens.end());
                session.chat_pending = true;
            }
            if (session.chat_sampler) common_sampler_free(session.chat_sampler);
            session.chat_sampler = sampler.release();
            session.chat_request = true;
            session.chat_speculative = prepared.sampling.temp == 0 && prepared.sampling.grammar.empty()
                && prepared.sampling.penalty_repeat == 1 && prepared.sampling.penalty_freq == 0
                && prepared.sampling.penalty_present == 0;
            auto reply = ack(input, reuse);
            reply.rows = static_cast<std::uint32_t>(tokens.size());
            return reply;
        } catch (const std::exception& error) {
            if (std::string(error.what()).find("context exhausted") != std::string::npos) throw;
            // Upstream template/schema diagnostics can contain user content.
            throw std::runtime_error("invalid chat request or unsupported template/schema");
        }
    }

    llama_token sample(Session& session, int index) {
        if (!session.chat_sampler) return llama_sampler_sample(session.sampler, context_, index);
        const auto token = common_sampler_sample(session.chat_sampler, context_, index);
        common_sampler_accept(session.chat_sampler, token, true);
        return token;
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
        old.draft = DraftState{};
        old.history.clear();
        old.chat_tokens.clear();
        old.chat_pending = old.chat_request = false;
        if (old.chat_sampler) common_sampler_free(old.chat_sampler);
        old.chat_sampler = nullptr;
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
        if (first_) session.history.resize(input.position);
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
        const std::string json = "{\"kv_cache_type\":\"" + std::string(ggml_type_name(kv_cache_type))
            + "\",\"role\":\"" + role
            + "\",\"requests_served\":" + std::to_string(requests_served_)
            + ",\"tokens_processed\":" + std::to_string(tokens_processed_)
            + ",\"tokens_generated\":" + std::to_string(tokens_generated_)
            + ",\"decode_batches\":" + std::to_string(decode_batches_)
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
        if (session.chat_request && !session.chat_speculative && (input.type == po::Type::speculative_activation
            || (input.type == po::Type::token && input.rows != 0))) {
            throw std::runtime_error("speculation is unavailable for this chat sampler");
        }
        // Sessions may have requests in progress at the same time (a replica serving several
        // chats): each has its own KV sequence and position, and frames run one at a time.
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
            if (first_) session.history.resize(input.position);
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
        (void) values;
        if (!put_hidden(output, tokens.size())) {
            llama_batch_free(batch);
            throw std::runtime_error("stage A chunk returned no hidden state");
        }
        po::put64(output.payload.data(), compute);
        llama_batch_free(batch);
        session.history.insert(session.history.end(), tokens.begin(), tokens.end());
        session.position += static_cast<std::uint32_t>(tokens.size());
        tokens_processed_ += tokens.size();
        note_step(tokens.size(), true, compute);
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
        if (speculative && input.rows > po::max_speculative_width) {
            throw std::runtime_error("speculative batch exceeds supported width");
        }
        if (input.type == po::Type::prompt) {
            if (input.position != session.position || input.payload.empty()
                || input.rows != 0 || input.cols != 0 || input.dtype != po::DType::none) {
                throw std::runtime_error("bad prompt frame");
            }
            if (session.has_prompt) throw std::runtime_error("duplicate prompt in request");
            session.has_prompt = true;
            if (session.chat_pending) {
                tokens = std::move(session.chat_tokens);
                session.chat_pending = false;
            } else tokens = tokenize(llama_model_get_vocab(model_), input.payload);
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
        session.history.insert(session.history.end(), tokens.begin(), tokens.end());

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
            const llama_token next = sample(session, -1);
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
            note_step(tokens.size(), input.type == po::Type::prompt, compute);
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
        (void) values;
        if (!put_hidden(output, tokens.size())) throw std::runtime_error("stage A returned no hidden state");
        po::put64(output.payload.data(), compute);
        llama_batch_free(batch);
        session.position += static_cast<std::uint32_t>(tokens.size());
        tokens_processed_ += tokens.size();
        note_step(tokens.size(), input.type == po::Type::prompt, compute);
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
        if (input.type == po::Type::speculative_activation
            && input.rows > po::max_speculative_width) {
            throw std::runtime_error("speculative batch exceeds supported width");
        }
        if ((input.type != po::Type::activation
                && input.type != po::Type::speculative_activation
                && input.type != po::Type::commit_activation
                && input.type != po::Type::prompt_chunk)
            || !po::activation_dtype(input.dtype) || input.rows == 0
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
        if (values > (po::max_payload - 8) / 4
            || input.payload.size() != 8 + po::activation_bytes(input.dtype, input.rows, input.cols)
                + guess_bytes(input, values)
            || input.rows > static_cast<std::uint32_t>(context_size_ - session.position)) {
            throw std::runtime_error("activation payload/shape mismatch");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(input.rows), hidden_, 1);
        batch.n_tokens = static_cast<int32_t>(input.rows);
        get_hidden(input, batch.embd, values);
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
            note_step(input.rows, true, compute);
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
            note_step(input.rows, false, compute);
            for (std::uint32_t index = 0; index < input.rows; ++index) {
                po::put32(output.payload.data() + 8 + static_cast<std::size_t>(index) * 4,
                    static_cast<std::uint32_t>(llama_sampler_sample(
                        session.sampler, context_, static_cast<int32_t>(index))));
            }
            tokens_generated_ += input.rows;
            return output;
        }

        const llama_token next = sample(session, -1);
        llama_synchronize(context_);
        const std::uint64_t compute = elapsed_ns(start);
        note_step(input.rows, false, compute);
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
            || !po::activation_dtype(input.dtype) || input.rows == 0
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
        if (values > (po::max_payload - 8) / 4
            || input.payload.size() != 8 + po::activation_bytes(input.dtype, input.rows, input.cols)
                + guesses
            || input.rows > static_cast<std::uint32_t>(context_size_ - session.position)) {
            throw std::runtime_error("activation payload/shape mismatch");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(input.rows), hidden_, 1);
        batch.n_tokens = static_cast<int32_t>(input.rows);
        get_hidden(input, batch.embd, values);
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

        // Header only: put_hidden builds a fresh payload, so copying the input's is wasted.
        po::Frame output = po::frame_header(input);
        if (!put_hidden(output, input.rows)) {
            llama_batch_free(batch);
            throw std::runtime_error("middle stage returned no hidden state");
        }
        po::put64(output.payload.data(), compute);
        llama_batch_free(batch);
        // The draft's guesses ride along to the last stage, which checks them.
        output.payload.insert(output.payload.end(), input.payload.end() - guesses,
            input.payload.end());
        session.position += input.rows;
        tokens_processed_ += input.rows;
        note_step(input.rows, input.type == po::Type::prompt_chunk, compute);
        std::fprintf(stderr,
            "session=%llu request=%llu stage=middle layers=%d..%d phase=%s position=%u shape=%ux%u compute_ms=%.3f bytes=%zu\n",
            static_cast<unsigned long long>(input.session),
            static_cast<unsigned long long>(input.request), begin_, end_ - 1,
            input.type == po::Type::commit_activation ? "commit" :
                (input.rows > 1 ? "prefill" : "decode"), input.position,
            output.rows, output.cols, compute / 1e6, output.payload.size() - 8);
        return output;
    }

    // This stage's hidden states for `rows` tokens into output (payload = 8 reserved bytes, then
    // the rows) in the route's wire format. False when llama.cpp returned none.
    bool put_hidden(po::Frame& output, std::size_t rows, std::size_t offset = 0) {
        const std::size_t width = static_cast<std::size_t>(hidden_);
        output.dtype = wire_;
        const std::size_t row_bytes = po::activation_bytes(wire_, 1, width);
        output.payload.assign(8 + rows * row_bytes, 0);
        for (std::size_t index = 0; index < rows; ++index) {
            const float* source = llama_get_embeddings_ith(context_, static_cast<int32_t>(index + offset));
            if (!source) return false;
            std::uint8_t* target = output.payload.data() + 8 + index * row_bytes;
            if (wire_ == po::DType::f16le) {
                ggml_fp32_to_fp16_row(source, reinterpret_cast<ggml_fp16_t*>(target),
                    static_cast<std::int64_t>(width));
            } else if (wire_ == po::DType::fp8e4m3) {
                po::pack_fp8_row(source, width, target);
            } else {
                std::memcpy(target, source, width * sizeof(float));
            }
        }
        return true;
    }

    // An incoming activation's `values` numbers, as f32, whichever format they crossed in.
    static void get_hidden(const po::Frame& input, float* destination, std::uint64_t values) {
        if (input.dtype == po::DType::fp8e4m3) {
            const std::size_t width = input.cols;
            const std::size_t row_bytes = po::activation_bytes(input.dtype, 1, width);
            for (std::size_t row = 0; row < input.rows; ++row) {
                po::unpack_fp8_row(input.payload.data() + 8 + row * row_bytes, width,
                    destination + row * width);
            }
        } else if (input.dtype == po::DType::f16le) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(input.payload.data() + 8),
                destination, static_cast<std::int64_t>(values));
        } else {
            std::memcpy(destination, input.payload.data() + 8,
                static_cast<std::size_t>(values) * sizeof(float));
        }
    }

    void note_step(std::size_t rows, bool prompt, std::uint64_t compute) {
        last_compute_.store(std::chrono::steady_clock::now().time_since_epoch().count());
        if (rows == 0 || compute == 0) return;
        const auto active = static_cast<std::uint32_t>(
            std::count(sequence_used_.begin(), sequence_used_.end(), true));
        // Decode steps (a few rows) and real prompt chunks are timed for the speed table;
        // speculative verify batches in between are neither.
        if (!prompt && rows <= 8) {
            Timing& timing = timings_[{po::Phase::decode, po::sessions_bucket(active)}];
            timing.ns += compute; timing.rows += rows; ++timing.samples;
        } else if (rows >= 64) {
            Timing& timing = timings_[{po::Phase::prefill, po::sessions_bucket(active)}];
            timing.ns += compute; timing.rows += rows; ++timing.samples;
        }
        if (prompt || rows > 8) return;
        const std::uint64_t previous = step_ns_.load();
        step_ns_.store(previous == 0 ? compute : (previous * 9 + compute) / 10);
    }

public:
    // Compute time since the last call, by phase and active-session bucket. Called with the
    // stage mutex held, like every other use of this stage.
    struct Timing { std::uint64_t ns = 0; std::uint64_t rows = 0; std::uint32_t samples = 0; };
    std::map<std::pair<po::Phase, std::uint32_t>, Timing> take_timings() {
        return std::exchange(timings_, {});
    }
private:
    std::map<std::pair<po::Phase, std::uint32_t>, Timing> timings_;
    std::atomic<std::chrono::steady_clock::rep> last_compute_{0};
    std::atomic<std::uint64_t> step_ns_{0};
    po::DType wire_ = po::DType::f32le;  // how this stage sends activations (the route agreed)
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
    std::uint64_t decode_batches_ = 0;
    std::uint64_t kv_bytes_per_session_ = 0;
    std::uint64_t active_session_ = 0;
    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    common_speculative_ptr speculative_;
    common_chat_templates_ptr chat_templates_;
    llama_context* speculative_target_ = nullptr;
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
    ggml_backend_dev_t device = memory_device();
    if (!device) return 0;
    std::size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    return free / (1024 * 1024);
}

bool redirect_diagnostics(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
#ifdef _WIN32
    // Shared, so the node's owner (or a tail command) can read the log while it runs;
    // _wfreopen_s would lock it.
    FILE* log = _wfsopen(path.c_str(), L"a", _SH_DENYNO);
    if (!log) return false;
    const bool redirected = _dup2(_fileno(log), _fileno(stderr)) == 0;
    std::fclose(log);
    if (redirected) std::setvbuf(stderr, nullptr, _IONBF, 0);
    return redirected;
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
//
// Several sessions can stream at once (a replica serving several chats): one entry per session
// that has a request streaming. Each ring frame names its session, so the requests interleave
// token by token without waiting for each other.
struct LoopState {
    struct Stream {
        std::uint64_t request = 0;
        std::uint32_t remaining = 0;
        bool cancelled = false;
    };
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Stream> streams;  // by session

    Stream* find(const po::Frame& frame) {
        const auto found = streams.find(frame.session);
        return found != streams.end() && found->second.request == frame.request ? &found->second : nullptr;
    }
    bool owns(const po::Frame& frame) { return find(frame) != nullptr; }
    void end(const po::Frame& frame) { if (owns(frame)) streams.erase(frame.session); }
    void clear() { streams.clear(); }
};

struct RingState {
    bool p2p = false;                 // next hops go through the sidecar ring proxy
    std::string ring_proxy;
    po::socket_t ring_listener = po::invalid_socket;
    std::atomic<po::socket_t> next{po::invalid_socket};
    std::atomic<po::socket_t> loop{po::invalid_socket};  // last stage -> first stage
    std::string loop_target;          // dialed in the background once the route is bound
    std::atomic<bool> loop_dialing{false};
    std::atomic<std::uint64_t> route_generation{0};  // bumped on release: stale dials close
    bool loop_self = false;           // this worker is the whole route: loop without a socket
    LoopState decode;
    std::mutex route_mutex;           // guards expected_previous (read by the ring thread)
    std::string expected_previous;
    // How fast large frames arrived from each predecessor peer (greeting link_bw= lines).
    std::mutex bandwidth_mutex;
    po::BandwidthTable bandwidth;
    std::mutex stage_mutex;
    Stage* stage = nullptr;           // null while serve mode has nothing loaded
    // Speculative decoding (first stage only): a small model that proposes the next few
    // tokens, so one pass through the route can commit several of them.
    Stage* draft = nullptr;
    // First stage of a multi-stage route: each session's last round start and guesses. The
    // next token's position tells how many the last stage accepted.
    struct Guesses {
        std::uint32_t at = 0;
        std::vector<std::uint32_t> tokens;
        std::uint64_t request = 0;
        std::chrono::steady_clock::time_point started;
    };
    std::unordered_map<std::uint64_t, Guesses> guessed;  // by session
    std::atomic<bool> shutdown{false};
    // Total time to establish a route's next hop (lookup, relay, hole punch, handshake).
    std::chrono::milliseconds connect_budget{20000};
    // Called once a client route's next hop is connected (serve-mode dashboard).
    std::function<void(const po::RingRoute&)> on_route;

    void disconnected() {
        std::lock_guard lock(stage_mutex);
        if (stage) stage->coordinator_disconnected();
        // The draft follows the same sessions, so it forgets them too.
        if (draft) draft->coordinator_disconnected();
        guessed.clear();
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
    if (!ring.loop_target.empty()) {
        // Dial the way back to the first stage now, in the background, so the first token
        // does not wait for it. The first stage may not expect this stage yet (the client
        // sets every stage up at once); the dial retries until it does.
        ring.loop_dialing.store(true);
        const std::uint64_t generation = ring.route_generation.load();
        std::thread([&ring, target = ring.loop_target, generation] {
            const po::socket_t socket = dial_ring_target(ring, target,
                std::chrono::steady_clock::now() + ring.connect_budget);
            if (socket != po::invalid_socket) {
                po::socket_t none = po::invalid_socket;
                if (ring.route_generation.load() == generation
                    && ring.loop.compare_exchange_strong(none, socket)) {
                    std::fprintf(stderr, "ring: decode loop connected to %s\n", target.c_str());
                } else {
                    po::close_socket(socket);  // the route ended meanwhile
                }
            }
            ring.loop_dialing.store(false);
        }).detach();
    }
    bound = route.next;
    if (ring.on_route) ring.on_route(route);
    std::fprintf(stderr, "ring: route next hop connected\n");
}

void release_route(RingState& ring) {
    ring.route_generation.fetch_add(1);
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
// One round: the draft model proposes a bounded number of tokens after the current one, the
// real stage verifies all of them in a single batch, and every correct guess is a token the
// route commits without another pass. Wrong guesses waste draft and verify work, but the
// batch still yields the correct next token at that position, and the stage's KV is
// truncated by the next frame's lower position (implicit rollback).
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
bool peek_ready_frame(po::socket_t socket, po::Frame& frame, std::uint64_t& size);

bool continue_loop(RingState& ring, std::deque<po::Frame> pending, std::string& error,
    const LoopDecoder& decode, po::socket_t control = po::invalid_socket) {
    while (!pending.empty()) {
        // A whole-model loop runs on the control thread itself. Service a ready
        // cancellation between decode steps instead of waiting until generation ends.
        po::Frame cancel;
        std::uint64_t size = 0;
        if (control != po::invalid_socket && peek_ready_frame(control, cancel, size)
            && cancel.type == po::Type::cancel_request && size == 0) {
            if (!po::recv_frame(control, cancel, error)) return false;
            {
                std::lock_guard lock(ring.decode.mutex);
                if (auto* stream = ring.decode.find(cancel)) stream->cancelled = true;
            }
            if (!po::send_frame(control, ack_frame(cancel), error)) return false;
        }
        po::Frame frame = std::move(pending.front());
        pending.pop_front();
        bool final_token = frame.payload.size() < 13 || frame.payload[12] != 0;
        {
            std::lock_guard lock(ring.decode.mutex);
            LoopState::Stream* stream = ring.decode.find(frame);
            if (!stream || stream->cancelled || stream->remaining <= 1) final_token = true;
            else --stream->remaining;
            if (final_token) ring.decode.end(frame);
        }
        po::Frame to_client = frame;
        if (!final_token) to_client.type = po::Type::client_chunk;
        const po::socket_t client = ring.next.load();
        if (client == po::invalid_socket || !po::send_frame(client, to_client, error)) {
            std::lock_guard lock(ring.decode.mutex);
            ring.decode.clear();  // the way back is gone for every session
            return false;
        }
        if (final_token) return true;
        if (!pending.empty()) continue;  // this round committed more than one token
        if (!ring.loop_self) {
            // The token goes back to the first stage and comes around the ring again.
            const po::Frame token = token_frame(frame, frame.position,
                {po::get32(frame.payload.data())});
            po::socket_t loop = ring.loop.load();
            // Usually already connected in the background (bind_route); if that dial is
            // still running, wait for it rather than opening a second link.
            const auto give_up = std::chrono::steady_clock::now() + ring.connect_budget;
            while (loop == po::invalid_socket && ring.loop_dialing.load()
                    && std::chrono::steady_clock::now() < give_up) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                loop = ring.loop.load();
            }
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
                ring.decode.clear();  // the loop is gone for every session
                return false;
            }
            return true;
        }
        const LoopStep step = decode(po::get32(frame.payload.data()), frame.position, frame);
        if (!step.error.empty() || step.tokens.empty()) {
            std::lock_guard lock(ring.decode.mutex);
            ring.decode.end(frame);
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
po::Frame draft_round(RingState& ring, const po::Frame& input);

LoopStep local_decode(RingState& ring, std::uint32_t token, std::uint32_t position,
    const po::Frame& like) {
    LoopStep step;
    std::lock_guard<std::mutex> lock(ring.stage_mutex);
    if (!ring.stage) {
        step.error = "no stage is loaded";
        return step;
    }
    try {
        const po::Frame input = token_frame(like, position, {token});
        const po::Frame verified = ((ring.draft && ring.stage->can_model_draft(input)) || ngram_draft)
            && ring.stage->can_draft(input)
            ? draft_round(ring, input) : ring.stage->handle(input);
        const auto found = ring.guessed.find(like.session);
        const std::vector<std::uint32_t> proposals = found == ring.guessed.end()
            ? std::vector<std::uint32_t>{} : found->second.tokens;
        std::vector<std::uint32_t> accepted;
        if (proposals.empty()) {
            if (verified.type != po::Type::result || verified.payload.size() < 13) {
                step.error = "unexpected decode reply";
                return step;
            }
            accepted.push_back(po::get32(verified.payload.data()));
            step.compute_ns = po::get64(verified.payload.data() + 4);
        } else {
            accepted = accepted_tokens(proposals, verified);
            if (verified.payload.size() >= 8) step.compute_ns = po::get64(verified.payload.data());
            // Catch-up and learning happen on the next input, when its position proves how
            // much was actually streamed. The client's token budget may clip this batch.
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

// When the last stage accepted every guess of the previous round, the draft (which fed
// itself every guess but the last) is one token behind the committed text. `next` is the
// frame about to reach the draft; its position says how much was committed.
void catch_up_draft(RingState& ring, const po::Frame& next) {
    const auto found = ring.guessed.find(next.session);
    if (found == ring.guessed.end()) return;
    const RingState::Guesses last = std::move(found->second);
    ring.guessed.erase(found);
    const std::uint32_t behind_at = last.at + static_cast<std::uint32_t>(last.tokens.size());
    if (next.request != last.request) return;
    if (ring.draft && ring.stage->can_model_draft(next) && !last.tokens.empty() && next.position == behind_at + 1) {
        ring.draft->handle(token_frame(next, behind_at, {last.tokens.back()}));
    }
    // Only another decode token proves this was a complete nonterminal round. Prompt,
    // commit and end frames may clip a batch and must not bias the controller.
    if (adaptive_draft && next.type == po::Type::token && next.position > last.at
        && next.position <= behind_at + 1) {
        auto& policy = ring.stage->draft_state(next).policy;
        const auto before = policy.width();
        policy.observe(static_cast<std::uint32_t>(last.tokens.size()) + 1,
            next.position - last.at, elapsed_ns(last.started));
        if (before != policy.width()) {
            std::fprintf(stderr, "speculation: session=%llu width=%u -> %u\n",
                static_cast<unsigned long long>(next.session), before, policy.width());
        }
    }
}

void flush_draft(RingState& ring, const po::Frame& like) {
    auto& state = ring.stage->draft_state(like);
    if (state.pending.empty()) return;
    po::Frame replay = token_frame(like, state.at, state.pending);
    replay.request = state.request;
    ring.draft->handle(replay);
    state.pending.clear();
}

// First stage, including a whole-model route: the incoming token starts a round.
// The draft guesses the next few tokens, the stage runs all positions as one batch, and the
// guesses ride along to the last stage. Called with ring.stage_mutex held.
po::Frame draft_round(RingState& ring, const po::Frame& input) {
    const std::uint32_t token = po::get32(input.payload.data());
    auto& state = ring.stage->draft_state(input);
    std::uint32_t width = 1;
    auto started = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> guesses;
    try {
        catch_up_draft(ring, input);
        width = adaptive_draft ? state.policy.width() : draft_width;
        started = std::chrono::steady_clock::now();
        if (width > 1) {
            if (ring.draft && ring.stage->can_model_draft(input)) {
                flush_draft(ring, input);
                guesses = ring.draft->model_proposals(*ring.stage, input, width - 1);
            } else if (ngram_draft) guesses = ring.stage->ngram_proposals(input, width - 1);
        }
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "speculation off for this route: %s\n", failure.what());
        ring.draft = nullptr;
        state.pending.clear();
    }
    std::vector<std::uint32_t> batch{token};
    batch.insert(batch.end(), guesses.begin(), guesses.end());
    po::Frame output = ring.stage->handle(token_frame(input, input.position, batch));
    if (ring.draft || ngram_draft) {
        if (ring.draft && ring.stage->can_model_draft(input) && guesses.empty()) {
            if (state.pending.empty()) { state.at = input.position; state.request = input.request; }
            if (state.request != input.request || input.position != state.at + state.pending.size()) {
                throw std::runtime_error("noncontiguous deferred draft inputs");
            }
            state.pending.push_back(token);
        }
        ring.guessed[input.session] = {input.position, guesses, input.request, started};
    } else ring.guessed.erase(input.session);
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
// Bytes already received on `socket` and not yet read; nullopt when unknown.
std::optional<std::uint64_t> buffered_bytes(po::socket_t socket) {
#ifdef _WIN32
    u_long available = 0;
    if (ioctlsocket(socket, FIONREAD, &available) != 0) return std::nullopt;
#else
    int available = 0;
    if (ioctl(socket, FIONREAD, &available) != 0 || available < 0) return std::nullopt;
#endif
    return static_cast<std::uint64_t>(available);
}

bool peek_ready_frame(po::socket_t socket, po::Frame& frame, std::uint64_t& size) {
    const std::optional<std::uint64_t> buffered = buffered_bytes(socket);
    if (!buffered || *buffered < po::header_size) return false;
    const std::uint64_t available = *buffered;
    std::array<std::uint8_t, po::header_size> header;
    if (recv(socket, reinterpret_cast<char*>(header.data()), static_cast<int>(header.size()), MSG_PEEK)
        != static_cast<int>(header.size())) return false;
    std::string error;
    return po::decode_header(header, frame, size, error)
        && static_cast<std::uint64_t>(available) >= po::header_size + size;
}

// recv_frame for the ring, which also times large frames from `peer`: when nothing was
// waiting on the socket as this worker began to read, the payload arrives as fast as the
// link carries it (the sidecars forward as data comes). A frame that was already buffered
// (this stage was busy) would arrive at loopback speed, so it is not timed.
bool recv_ring_frame(RingState& ring, po::socket_t socket, const std::string& peer,
    po::Frame& frame, std::string& error) {
    const std::optional<std::uint64_t> waiting = peer.empty() ? std::nullopt : buffered_bytes(socket);
    std::array<std::uint8_t, po::header_size> header{};
    if (!po::recv_all(socket, header.data(), header.size())) {
        error = "peer disconnected while receiving frame";
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t payload_size = 0;
    if (!po::decode_header(header, frame, payload_size, error)) return false;
    frame.payload.resize(static_cast<std::size_t>(payload_size));
    if (!po::recv_all(socket, frame.payload.data(), frame.payload.size())) {
        error = "peer disconnected during payload";
        return false;
    }
    if (waiting && *waiting == 0 && payload_size >= po::min_bandwidth_frame_bytes) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::lock_guard lock(ring.bandwidth_mutex);
        ring.bandwidth.observe(peer, payload_size, seconds, now);
    }
    return true;
}

void run_ring(RingState& ring, std::stop_token stop) {
    while (!ring.shutdown.load() && !stop.stop_requested()) {
        const po::socket_t predecessor = accept(ring.ring_listener, nullptr, nullptr);
        if (predecessor == po::invalid_socket) break;
        std::string peer_id;  // the predecessor's authenticated PeerID (libp2p only)
        if (ring.p2p) {
            // The sidecar writes the predecessor's authenticated PeerID first.
            set_socket_receive_timeout(predecessor, 5000);
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
        std::deque<std::pair<po::Frame, po::Frame>> batched;
        while (!ring.shutdown.load()) {
            po::Frame input;
            po::Frame output;
            std::string error;
            const bool computed = !batched.empty();
            if (computed) {
                input = std::move(batched.front().first);
                output = std::move(batched.front().second);
                batched.pop_front();
            } else if (!recv_ring_frame(ring, predecessor, peer_id, input, error)) {
                std::fprintf(stderr, "ring predecessor connection closed: %s\n", error.c_str());
                ring.disconnected();
                break;
            }
            bool errored = false;
            bool last = false;
            std::deque<po::Frame> committed;  // last stage: tokens this round commits
            if (computed) {
                std::lock_guard<std::mutex> lock(ring.stage_mutex);
                last = ring.stage && ring.stage->last();
                errored = output.type == po::Type::error;
            } else if (input.type == po::Type::error) {
                // An earlier stage failed this request: pass its reason on unchanged.
                std::lock_guard<std::mutex> lock(ring.stage_mutex);
                last = ring.stage && ring.stage->last();
                output = input;
                errored = true;
            } else {
                std::lock_guard<std::mutex> lock(ring.stage_mutex);
                try {
                    if (!ring.stage) throw std::runtime_error("no stage is loaded");
                    last = ring.stage->last();
                    std::vector<po::Frame> batch{input};
                    if (continuous_batching && !ring.draft && !ngram_draft
                        && ring.stage->batchable(input, input.payload.size())) {
                        while (batch.size() < std::min<std::size_t>(32, ring.stage->max_sessions())) {
                            po::Frame candidate;
                            std::uint64_t size = 0;
                            if (!peek_ready_frame(predecessor, candidate, size)
                                || !ring.stage->batchable(candidate, size)
                                || std::any_of(batch.begin(), batch.end(), [&](const auto& frame) {
                                    return frame.session == candidate.session;
                                })) break;
                            if (!po::recv_frame(predecessor, candidate, error)) throw std::runtime_error("batch connection closed");
                            batch.push_back(std::move(candidate));
                        }
                    }
                    if (batch.size() > 1) {
                        std::vector<po::Frame> replies;
                        try { replies = ring.stage->decode_batch(batch); }
                        catch (const std::exception&) {
                            for (const auto& frame : batch) replies.push_back(po::error_frame(frame, "batched decode failed"));
                        }
                        output = std::move(replies.front());
                        errored = output.type == po::Type::error;
                        for (std::size_t i = 1; i < batch.size(); ++i) {
                            batched.emplace_back(std::move(batch[i]), std::move(replies[i]));
                        }
                    } else if (((ring.draft && ring.stage->can_model_draft(input)) || ngram_draft)
                        && !last && ring.stage->begin() == 0
                        && ring.stage->can_draft(input)
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
            } else if (errored && last) {
                // The request failed: it no longer streams, and the next one may start.
                std::lock_guard lock(ring.decode.mutex);
                ring.decode.end(output);
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
                    if (LoopState::Stream* stream = ring.decode.find(input)) stream->cancelled = true;
                } else if (input.rows == 0 || !input.payload.empty()) {
                    problem = "bad stream_prompt frame";
                } else if (ring.decode.streams.contains(input.session)) {
                    problem = "this session is already streaming a request";
                } else {
                    ring.decode.streams[input.session] = {input.request, input.rows, false};
                }
            }
            po::Frame reply = problem.empty() ? ack_frame(input) : po::error_frame(input, problem);
            if (!problem.empty()) std::fprintf(stderr, "rejected frame: %s\n", problem.c_str());
            // On a linked route one session's bad request is that session's problem only.
            if (!po::send_frame(client, reply, error) || (!problem.empty() && bound.empty())) {
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
            if (input.type == po::Type::reset_session || input.type == po::Type::destroy_session) {
                ring.guessed.erase(input.session);  // its old guesses mean nothing now
            }
            // The draft model follows the same session: same sessions, same prompts, so its
            // proposals continue the same text.
            if (ring.draft && !ring.stage->can_model_draft(input)) {
                // Chat samplers have request-specific state; ordinary greedy drafting is
                // unavailable until its upstream acceptance path is integrated.
                ring.guessed.erase(input.session);
            } else if (ring.draft && (input.type == po::Type::create_session
                || input.type == po::Type::reset_session
                || input.type == po::Type::destroy_session
                || input.type == po::Type::end_request
                || input.type == po::Type::rollback
                || input.type == po::Type::commit_token
                || input.type == po::Type::prompt)) {
                // Prompts and commits carry text the draft must see; the rest are plain
                // bookkeeping. (A commit appends the answer's final token to every stage.)
                po::Frame mirrored = input;
                mirrored.payload = input.type == po::Type::prompt
                    || input.type == po::Type::commit_token ? input.payload
                    : std::vector<std::uint8_t>{};
                try {
                    if (input.type == po::Type::commit_token || input.type == po::Type::prompt
                        || input.type == po::Type::rollback) {
                        catch_up_draft(ring, input);
                    }
                    if (input.type == po::Type::commit_token || input.type == po::Type::prompt
                        || input.type == po::Type::end_request || input.type == po::Type::rollback) {
                        flush_draft(ring, input);
                    }
                    if (input.type == po::Type::end_request || input.type == po::Type::rollback) {
                        ring.guessed.erase(input.session);
                    }
                    ring.draft->handle(mirrored);
                } catch (const std::exception& failure) {
                    // A draft that cannot follow text/rollback stops; ordinary target decode
                    // remains available. Bookkeeping failures are logged.
                    std::fprintf(stderr, "speculation: draft could not follow %s: %s\n",
                        input.type == po::Type::prompt ? "the prompt" : "a session change",
                        failure.what());
                    if (input.type == po::Type::prompt || input.type == po::Type::commit_token
                        || input.type == po::Type::rollback || input.type == po::Type::end_request) {
                        ring.draft = nullptr;
                        ring.guessed.erase(input.session);
                    }
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
            if (!continue_loop(ring, std::deque<po::Frame>{output}, error, decode, client)) {
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
            {
                // A failed request no longer streams: free the loop for the next one.
                std::lock_guard lock(ring.decode.mutex);
                ring.decode.end(input);
            }
            // On a linked route (a persistent replica serves many sessions), one session's
            // failure, e.g. a full context, must not end every other session: the owner
            // ends or resets that session. Anything else still ends the connection.
            if (bound.empty() || input.session == 0) {
                ring.disconnected();
                break;
            }
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
    std::atomic<bool> holds_layers{false};                  // stage loaded: its memory is ours
    std::unique_ptr<Stage> draft;                           // speculative decoding, may be null
    std::optional<po::StageRequest> loaded;                 // guarded by load_mutex
    std::atomic<int> connections{0};
    std::mutex peers_mutex;
    std::unordered_map<std::string, int> peer_connections;  // open control connections by PeerID
    std::mutex status_mutex;
    std::optional<po::StageRequest> cached_hint;            // guarded by status_mutex
    // Dashboard (dan-provider network=dht shows it; null otherwise).
    dan::ProviderTerminalUi* ui = nullptr;
    std::filesystem::path net_status_file;
    std::filesystem::path replica_status_file;              // this node's replica owner, if any
    bool replica_owner = false;                             // told to greet with owner=1
    // Measured decode speed (µs per GiB of weights per token), kept in the cache directory so
    // a restarted worker still knows it. 0 until this GPU has decoded something.
    std::atomic<std::uint64_t> speed_us_per_gib{0};
    // Per-configuration speeds from ordinary traffic (measurement.hpp), kept in speeds.txt.
    std::mutex speeds_mutex;
    po::SpeedTable speeds;                                  // guarded by speeds_mutex
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

// Memory a new stage may use now: the owner's quota, capped by what the GPU has free right
// now when this worker holds nothing (other programs may have taken some since startup). A
// loaded stage is unloaded before the next one loads, so its memory counts as available.
std::uint64_t available_mib(const ServeContext& context) {
    const std::uint64_t quota = context.hello.offered_vram_mib;
    if (context.holds_layers) return quota;
    const std::uint64_t free = gpu_free_mib();
    return free == 0 ? quota : std::min(quota, free);
}

// Round trips this node's sidecar measured to other peers ("links" in its network status
// file, which it rewrites every 2 s), for the greeting. Nothing when the file is missing or
// stale.
std::vector<po::ProviderCapability::PeerLink> measured_links(const std::filesystem::path& path) {
    std::vector<po::ProviderCapability::PeerLink> links;
    if (path.empty()) return links;
    std::ifstream input(path, std::ios::binary);
    if (!input) return links;
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto number_after = [&](std::string_view object, std::string_view key) -> std::optional<std::uint64_t> {
        const std::size_t at = object.find(key);
        if (at == std::string_view::npos) return std::nullopt;
        std::uint64_t value = 0;
        std::size_t index = at + key.size();
        if (index >= object.size() || object[index] < '0' || object[index] > '9') return std::nullopt;
        for (; index < object.size() && object[index] >= '0' && object[index] <= '9'; ++index) {
            value = value * 10 + static_cast<unsigned>(object[index] - '0');
        }
        return value;
    };
    const auto updated = number_after(text, "\"updated_unix_ms\":");
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (!updated || now_ms - static_cast<std::int64_t>(*updated) > 10000) return links;
    const std::size_t begin = text.find("\"links\":[");
    const std::size_t end = begin == std::string::npos ? std::string::npos : text.find(']', begin);
    if (end == std::string::npos) return links;
    const std::string_view list = std::string_view(text).substr(begin, end - begin);
    for (std::size_t open = list.find('{'); open != std::string_view::npos
            && links.size() < po::max_greeting_links; open = list.find('{', open + 1)) {
        const std::size_t close = list.find('}', open);
        if (close == std::string_view::npos) break;
        const std::string_view object = list.substr(open, close - open);
        const std::size_t peer_at = object.find("\"peer\":\"");
        const auto rtt = number_after(object, "\"rtt_ms\":");
        if (peer_at == std::string_view::npos || !rtt || *rtt > 60000) continue;
        const std::size_t peer_begin = peer_at + 8, peer_end = object.find('"', peer_begin);
        const std::string_view peer = object.substr(peer_begin, peer_end - peer_begin);
        if (peer_end == std::string_view::npos || !po::valid_peer_id(peer)) continue;
        links.push_back({std::string(peer), static_cast<std::uint32_t>(*rtt),
            object.find("\"path\":\"relay\"") != std::string_view::npos});
    }
    return links;
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
    // A whole-model worker returns tokens, never a context-sized boundary activation. In a
    // split route only the first stage decides frame sizes: it sends a long prompt as
    // prompt_chunk frames of at most prefill_chunk rows, and every later stage forwards frames
    // with the same rows. So the first stage's largest frame must fit the wire limit.
    const bool whole_model = request.begin == 0 && request.end == static_cast<int>(index.layers);
    const std::uint64_t rows = context.prefill_chunk == 0 ? request.context
        : std::min<std::uint64_t>(request.context,
            std::max<std::uint64_t>(context.prefill_chunk, po::max_speculative_width));
    if (!whole_model && request.begin == 0
        && rows * index.hidden * sizeof(float) > po::max_payload - 8) {
        return "context_too_large";
    }
    po::StageAssignment fit;
    if (!po::stage_fits(index, available_mib(context), request.begin, request.end,
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
// Never fatal: a route without a draft model simply decodes one token at a time. Every check
// fails closed: a draft that does not fit beside the stage, or whose tokenizer cannot be shown
// to match the main model's, is not used.
void load_draft_model(ServeContext& context, const po::StageRequest& request) {
    const bool wanted = !request.draft_sha256.empty() && request.begin == 0
        && context.catalog.count(lowercase(request.draft_sha256)) != 0;
    const auto turn_off = [&] {
        std::lock_guard lock(context.ring.stage_mutex);
        context.ring.draft = nullptr;
        context.draft.reset();
    };
    if (!wanted) { turn_off(); return; }
    const CatalogModel& model = context.catalog.at(lowercase(request.draft_sha256));
    const auto main = context.catalog.find(lowercase(request.model_sha256));
    // Speculation is validated for Qwen2 only; an OLMoE route decodes one token per pass.
    if (main != context.catalog.end()
        && (main->second.index.architecture != "qwen2"
            || model.index.architecture != "qwen2")) {
        std::fprintf(stderr, "speculation off: %s drafts only for qwen2 routes\n",
            model.manifest.model_id.c_str());
        turn_off();
        return;
    }
    if (main == context.catalog.end() || !context.stage
        || !po::stage_with_draft_fits(main->second.index, model.index,
            context.hello.offered_vram_mib, request.end, request.context, request.sessions)) {
        std::fprintf(stderr, "speculation off: draft model %s does not fit beside the stage\n",
            model.manifest.model_id.c_str());
        turn_off();
        return;
    }
    if (context.draft && context.loaded
        && lowercase(context.loaded->draft_sha256) == lowercase(request.draft_sha256)
        && context.draft->context_size() == static_cast<int>(request.context)
        && context.draft->max_sessions() == request.sessions) {
        // Already loaded for the previous route with the same context and session count;
        // make sure the ring uses it again (a route whose prompt the draft could not follow
        // switched it off). The main model may have changed, so the tokenizer is checked again.
        std::string why;
        std::lock_guard lock(context.ring.stage_mutex);
        if (!po::vocabularies_match(context.stage->model(), context.draft->model(), why)) {
            std::fprintf(stderr, "speculation off: draft model %s tokenizer differs: %s\n",
                model.manifest.model_id.c_str(), why.c_str());
            context.ring.draft = nullptr;
            context.draft.reset();
            return;
        }
        context.ring.draft = context.draft.get();
        context.draft->coordinator_disconnected();  // a new route starts from no sessions
        context.ring.guessed.clear();  // a new route starts from no sessions
        return;
    }
    turn_off();
    const auto path = context.cache_dir / (model.manifest.model_id + "-draft.gguf");
    po::RangeModelStats stats;
    std::string error;
    try {
        if (!po::prepare_range_model({model.manifest.url, model.manifest.revision,
                model.manifest.sha256, path, 0, static_cast<int>(model.index.layers)},
                stats, error)) {
            throw std::runtime_error(error);
        }
        // One KV sequence per route session: the draft mirrors every session the stage has.
        auto draft = std::make_unique<Stage>(path.string(), 0,
            static_cast<int>(model.index.layers), static_cast<int>(request.context),
            context.gpu_layers, request.sessions,
            po::head_width_k(model.index), po::head_width_v(model.index));
        std::string why;
        std::lock_guard lock(context.ring.stage_mutex);
        if (!po::vocabularies_match(context.stage->model(), draft->model(), why)) {
            std::fprintf(stderr, "speculation off: draft model %s tokenizer differs: %s\n",
                model.manifest.model_id.c_str(), why.c_str());
            return;
        }
        context.draft = std::move(draft);
        context.ring.draft = context.draft.get();
        std::fprintf(stderr, "speculation: draft model %s ready (tokenizer matches main model)\n",
            model.manifest.model_id.c_str());
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
        {
            std::lock_guard lock(context.ring.stage_mutex);
            context.stage->set_wire(request.activations);
        }
        // The stage is reused, but this route may want a different draft model (or none).
        load_draft_model(context, request);
        context.loaded = request;
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
        context.holds_layers = false;
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
        static_cast<int>(request.context), context.gpu_layers, request.sessions,
        po::head_width_k(model.index), po::head_width_v(model.index));
    stage->set_wire(request.activations);
    {
        std::lock_guard lock(context.ring.stage_mutex);
        context.stage = std::move(stage);
        context.ring.stage = context.stage.get();
        context.holds_layers = true;
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
    // One peer may hold at most a few of this worker's 16 control connections, so a single
    // misbehaving peer cannot lock everyone else out (BETA_SELECTION_PLAN.md M7).
    constexpr int per_peer_connections = 4;
    {
        std::lock_guard lock(context.peers_mutex);
        if (!peer_id.empty() && context.peer_connections[peer_id] >= per_peer_connections) {
            std::fprintf(stderr, "rejected control connection: peer already holds %d\n",
                per_peer_connections);
            po::close_socket(client);
            return;
        }
        ++context.peer_connections[peer_id];
    }
    struct PeerSlot {
        ServeContext& context;
        const std::string& peer;
        ~PeerSlot() {
            std::lock_guard lock(context.peers_mutex);
            if (--context.peer_connections[peer] <= 0) context.peer_connections.erase(peer);
        }
    } slot{context, peer_id};
    po::Frame hello;
    hello.type = po::Type::provider_available;
    {
        po::ProviderCapability capability = context.hello;
        capability.offered_vram_mib = available_mib(context);
        capability.state = po::WorkerLease::name(
            context.lease.state(po::WorkerLease::Clock::now()));
        // What this worker already holds, so a client can plan a split that needs no download.
        for (const auto& [sha, model] : context.catalog) {
            for (const po::CachedRange& range
                    : cached_ranges(context.cache_dir, model.manifest.model_id, sha)) {
                capability.cached.push_back(range);
            }
        }
        capability.speed_us_per_gib = context.speed_us_per_gib.load();
        {
            std::lock_guard lock(context.speeds_mutex);
            capability.speeds = context.speeds.records();
        }
        if (capability.state != "available") {
            // Computing right now (the lock is taken for each step): certainly not idle.
            std::unique_lock lock(context.ring.stage_mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                capability.idle_s = 0;
                capability.open_sessions = 1;
            } else if (context.stage) {
                capability.idle_s = context.stage->idle_seconds();
                capability.open_sessions = context.stage->open_sessions();
            }
        }
        capability.links = measured_links(context.net_status_file);
        {
            const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::lock_guard lock(context.ring.bandwidth_mutex);
            for (po::ProviderCapability::PeerLink& link : capability.links) {
                link.kib_per_s = context.ring.bandwidth.kib_per_s(link.peer, now);
            }
        }
        capability.replica_owner = context.replica_owner;
        capability.f16_activations = true;
        capability.fp8_activations = true;
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

// Turns the loaded stage's typical decode step into this GPU's speed per GiB of weights and
// keeps it (in memory, and on disk when it changes by more than 5%).
void update_speed(ServeContext& context) {
    std::unique_lock load(context.load_mutex, std::try_to_lock);  // held for whole downloads
    if (!load.owns_lock() || !context.loaded) return;
    const po::StageRequest loaded = *context.loaded;
    load.unlock();
    std::uint64_t step = 0;
    std::map<std::pair<po::Phase, std::uint32_t>, Stage::Timing> timings;
    {
        std::lock_guard lock(context.ring.stage_mutex);
        if (context.stage) {
            step = context.stage->step_ns();
            timings = context.stage->take_timings();
        }
    }
    const auto model = context.catalog.find(lowercase(loaded.model_sha256));
    if (model == context.catalog.end()) return;
    const std::uint64_t bytes = po::decode_bytes(model->second.index, loaded.begin, loaded.end);
    if (bytes == 0) return;
    if (!timings.empty()) {
        // Per GiB of this stage's weights: per step for decode (each open session gets one
        // token per step), per prompt row for prefill.
        const double gib = static_cast<double>(bytes) / static_cast<double>(1ull << 30);
        const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string text;
        {
            std::lock_guard lock(context.speeds_mutex);
            for (const auto& [slot, timing] : timings) {
                const double per = slot.first == po::Phase::decode
                    ? static_cast<double>(timing.samples) : static_cast<double>(timing.rows);
                if (per == 0) continue;
                context.speeds.observe({lowercase(loaded.model_sha256), slot.first,
                    po::context_bucket(loaded.context), slot.second},
                    static_cast<double>(timing.ns) / 1000.0 / per / gib, timing.samples, now);
            }
            for (const po::SpeedRecord& record : context.speeds.records()) {
                text += po::speed_text(record, now) + "\n";
            }
        }
        const std::filesystem::path path = context.cache_dir / "speeds.txt";
        std::ofstream(path.string() + ".tmp", std::ios::trunc) << text;
        std::error_code ignored;
        std::filesystem::rename(path.string() + ".tmp", path, ignored);
    }
    if (step == 0) return;
    const auto speed = static_cast<std::uint64_t>(static_cast<double>(step) / 1000.0
        * static_cast<double>(1ull << 30) / static_cast<double>(bytes));
    const std::uint64_t known = context.speed_us_per_gib.load();
    if (speed == 0 || (known != 0 && speed * 20 > known * 19 && speed * 20 < known * 21)) return;
    context.speed_us_per_gib.store(speed);
    std::ofstream(context.cache_dir / "decode-speed.txt", std::ios::trunc) << speed << "\n";
}

// Rewrites the status file when it changes, and every few seconds as a liveness signal.
void run_status_writer(std::shared_ptr<ServeContext> context, std::filesystem::path path,
    std::stop_token stop) {
    std::error_code ignored;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ignored);
    std::string written;
    auto last_write = std::chrono::steady_clock::time_point{};
    while (!stop.stop_requested()) {
        update_speed(*context);
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

// One dashboard line from the replica owner's status file (dan-client --form), or empty.
std::string replica_summary(const std::filesystem::path& path) {
    if (path.empty()) return {};
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::string state = json_text(text, "state");
    const std::uint64_t formed = json_count(text, "formations");
    const std::uint64_t dissolved = json_count(text, "dissolutions");
    const std::string history = formed == 0 ? std::string{}
        : " (formed " + std::to_string(formed) + ", dissolved " + std::to_string(dissolved) + ")";
    if (state == "ready") {
        std::size_t stages = 0;  // one "begin" per member
        for (std::size_t at = text.find("\"begin\":"); at != std::string::npos; at = text.find("\"begin\":", at + 1)) {
            ++stages;
        }
        return "owner of " + json_text(text, "replica_id").substr(0, 8) + ", ready, "
            + std::to_string(stages) + " stage" + (stages == 1 ? "" : "s") + ", sessions "
            + std::to_string(json_count(text, "sessions_in_use")) + "/"
            + std::to_string(json_count(text, "sessions_max")) + history;
    }
    if (state == "forming") return "forming a replica" + history;
    const std::string event = json_text(text, "last_event");
    return (state.empty() ? std::string("starting") : state) + (event.empty() ? "" : ": " + event);
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
            state.replica = replica_summary(context->replica_status_file);
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
    common_log_set_verbosity_thold(-1); // upstream template/schema diagnostics may contain request content
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
    std::string replica_status_file;
    bool replica_owner = false;
    bool list_devices = false;
    std::string device_selector;  // PCI bus ID or llama.cpp device name
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--tui") { tui = true; continue; }
            if (option == "--peer-header") { peer_header = true; continue; }
            if (option == "--replica-owner") { replica_owner = true; continue; }
            if (option == "--list-devices") { list_devices = true; continue; }
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
            else if (option == "--device") device_selector = value;
            else if (option == "--vram-mib") offered_vram_mib = std::stoull(value);
            else if (option == "--cache-dir") cache_dir = value;
            else if (option == "--host") host = value;
            else if (option == "--port") port = std::stoi(value);
            else if (option == "--stage-start") begin = std::stoi(value);
            else if (option == "--stage-end") end = std::stoi(value);
            else if (option == "--ctx") context = std::stoi(value);
            else if (option == "--gpu-layers") gpu_layers = std::stoi(value);
            else if (option == "--kv-cache") {
                if (value == "f16") kv_cache_type = GGML_TYPE_F16;
                else if (value == "q8_0") kv_cache_type = GGML_TYPE_Q8_0;
                else throw std::runtime_error("--kv-cache must be f16 or q8_0");
            }
            else if (option == "--prefill-batch") {
                const int size = std::stoi(value);
                if (size != 256 && size != 512 && size != 1024) {
                    throw std::runtime_error("--prefill-batch must be 256, 512 or 1024");
                }
                prefill_batch = static_cast<std::uint32_t>(size);
            }
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
            else if (option == "--draft-width") {
                const int width = std::stoi(value);
                if (width < 1 || width > static_cast<int>(po::max_speculative_width)) {
                    throw std::runtime_error("--draft-width must be 1..32");
                }
                draft_width = static_cast<std::uint32_t>(width);
            }
            else if (option == "--adaptive-draft") {
                if (value != "true" && value != "false") {
                    throw std::runtime_error("--adaptive-draft must be true or false");
                }
                adaptive_draft = value == "true";
            }
            else if (option == "--ngram-draft") {
                if (value != "true" && value != "false") throw std::runtime_error("--ngram-draft must be true or false");
                ngram_draft = value == "true";
            }
            else if (option == "--continuous-batching") {
                if (value != "true" && value != "false") throw std::runtime_error("--continuous-batching must be true or false");
                continuous_batching = value == "true";
            }
            else if (option == "--connect-timeout-ms") connect_timeout_ms = std::stoi(value);
            else if (option == "--control-listen") control_listen = value;
            else if (option == "--catalog") catalog_paths.push_back(value);
            else if (option == "--status-file") status_file = value;
            else if (option == "--net-status-file") net_status_file = value;
            else if (option == "--replica-status-file") replica_status_file = value;
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
    if (list_devices || !device_selector.empty()) {
        ggml_backend_load_all();
        if (list_devices) {
            // One line per GPU llama.cpp can use: name, backend, PCI ID, free/total MiB, description.
            for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
                ggml_backend_dev_t device = ggml_backend_dev_get(index);
                if (!gpu_device(device)) continue;
                ggml_backend_dev_props props{};
                ggml_backend_dev_get_props(device, &props);
                std::printf("DEVICE %s %s %s %s %zu %zu %s\n", ggml_backend_dev_name(device),
                    ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)),
                    props.type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "igpu" : "gpu",
                    props.device_id ? normalized_pci(props.device_id).c_str() : "-",
                    props.memory_free / (1024 * 1024), props.memory_total / (1024 * 1024),
                    props.description ? props.description : "");
            }
            return 0;
        }
        bound_device = find_device(device_selector);
        if (!bound_device) {
            std::fprintf(stderr, "dan-stage-worker: GPU %s was not found by this runtime's backends "
                "(see --list-devices)\n", device_selector.c_str());
            return 1;
        }
    }
    if ((generic || serve) && (gpu_name.empty() || offered_vram_mib == 0)) {
        ggml_backend_load_all();
        if (ggml_backend_dev_t device = memory_device()) {
            std::size_t free = 0, total = 0;
            ggml_backend_dev_memory(device, &free, &total);
            if (gpu_name.empty()) gpu_name = ggml_backend_dev_description(device);
            if (offered_vram_mib == 0) offered_vram_mib = free / (1024 * 1024);
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
            serving.replica_status_file = replica_status_file;
            serving.replica_owner = replica_owner;
            {
                std::ifstream saved(std::filesystem::path(cache_dir) / "decode-speed.txt");
                std::uint64_t speed = 0;
                if (saved >> speed) serving.speed_us_per_gib.store(speed);
                // Earlier measurements; a malformed or stale line is skipped.
                std::ifstream table(std::filesystem::path(cache_dir) / "speeds.txt");
                const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                for (std::string line; std::getline(table, line);) {
                    po::SpeedRecord record;
                    if (po::parse_speed_text(line, record, now)
                        && now - record.updated_unix_s <= po::speed_max_age_s) {
                        serving.speeds.add(record);
                    }
                }
            }
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
                if (!po::compatible_stage_model(index, &incompatibility)
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
                                // Older coordinator path: no parsed index here, and it serves
                                // dense Qwen2 only, where hidden/heads is the right head width.
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
