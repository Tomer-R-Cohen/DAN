#pragma once

// Synthetic models for tests: a dense Qwen2 index, an OLMoE index with the real tensor names
// and shapes (tiny sizes), and a GGUF header builder. The fixtures prove selection, metadata
// and accounting rules -- they contain no weights and cannot be executed.

#include "provider_owned/range_model.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dan::test {

namespace po = dan::provider_owned;

// --- indexes -------------------------------------------------------------------------

inline po::ModelTensor tensor(std::string name, std::uint64_t bytes,
    std::vector<std::uint64_t> dimensions = {}, std::uint32_t type = 0) {
    return {std::move(name), bytes, std::move(dimensions), type};
}

inline po::ModelIndex qwen2_index(std::uint32_t layers = 6, bool tied_output = false) {
    constexpr std::uint64_t mib = 1024 * 1024;
    po::ModelIndex model;
    model.architecture = "qwen2";
    model.layers = layers;
    model.hidden = 1024;
    model.heads = 16;
    model.kv_heads = 4;
    model.header_bytes = 4096;
    model.logical_bytes = 64 * mib;
    model.tensors.push_back(tensor("token_embd.weight", 8 * mib, {1024, 4096}, 12));
    model.tensors.push_back(tensor("output_norm.weight", 4096, {1024}, 0));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        model.tensors.push_back(tensor("blk." + std::to_string(layer) + ".weight", 4 * mib,
            {1024, 1024}, 12));
    }
    if (!tied_output) {
        model.tensors.push_back(tensor("output.weight", 8 * mib, {1024, 4096}, 14));
    }
    return model;
}

// OLMoE geometry of the pinned artifact (hidden 2048, 64 experts of width 1024, 8 routed),
// with small byte counts. Every layer carries its router and complete expert bank.
inline po::ModelIndex olmoe_index(std::uint32_t layers = 6, std::uint32_t experts = 64,
    std::uint32_t experts_used = 8, std::uint32_t ffn = 1024) {
    constexpr std::uint64_t mib = 1024 * 1024;
    po::ModelIndex model;
    model.architecture = "olmoe";
    model.layers = layers;
    model.hidden = 2048;
    model.heads = 16;
    model.kv_heads = 16;
    model.experts = experts;
    model.experts_used = experts_used;
    model.ffn_length = ffn;
    model.header_bytes = 8192;
    model.logical_bytes = 512 * mib;
    const std::uint64_t hidden = model.hidden;
    model.tensors.push_back(tensor("token_embd.weight", 16 * mib, {hidden, 50304}, 12));
    model.tensors.push_back(tensor("output_norm.weight", 8192, {hidden}, 0));
    model.tensors.push_back(tensor("output.weight", 24 * mib, {hidden, 50304}, 14));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const char* norm : {"attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                "ffn_norm.weight"}) {
            model.tensors.push_back(tensor(prefix + norm, 8192, {hidden}, 0));
        }
        for (const char* projection : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight"}) {
            model.tensors.push_back(tensor(prefix + projection, 2 * mib, {hidden, hidden}, 12));
        }
        model.tensors.push_back(tensor(prefix + "ffn_gate_inp.weight", 512 * 1024,
            {hidden, experts}, 0));
        // The whole bank: 64 experts per tensor, not the 8 a token routes to.
        model.tensors.push_back(tensor(prefix + "ffn_gate_exps.weight", 72 * mib,
            {hidden, ffn, experts}, 12));
        model.tensors.push_back(tensor(prefix + "ffn_up_exps.weight", 72 * mib,
            {hidden, ffn, experts}, 12));
        model.tensors.push_back(tensor(prefix + "ffn_down_exps.weight", 104 * mib,
            {ffn, hidden, experts}, 14));
    }
    return model;
}

inline void erase_tensor(po::ModelIndex& model, const std::string& name) {
    for (auto it = model.tensors.begin(); it != model.tensors.end(); ++it) {
        if (it->name == name) { model.tensors.erase(it); return; }
    }
}

inline po::ModelTensor* tensor_named(po::ModelIndex& model, const std::string& name) {
    for (po::ModelTensor& entry : model.tensors) {
        if (entry.name == name) return &entry;
    }
    return nullptr;
}

// --- GGUF header builder -------------------------------------------------------------

class GgufBuilder {
public:
    // Metadata entries are emitted in the order they are added, so a test can place
    // general.architecture after the architecture-prefixed keys.
    GgufBuilder& string_value(const std::string& key, const std::string& value) {
        entry(key, 8);
        length(value.size());
        bytes_.insert(bytes_.end(), value.begin(), value.end());
        return *this;
    }

    GgufBuilder& uint32_value(const std::string& key, std::uint32_t value) {
        entry(key, 4);
        integer(value);
        return *this;
    }

    // `bytes` is how much tensor data follows this tensor's offset; the header records the
    // offset and the file's logical size is the sum, as in a real GGUF.
    GgufBuilder& tensor(const std::string& name, const std::vector<std::uint64_t>& dimensions,
        std::uint32_t type, std::uint64_t bytes) {
        tensors_.push_back({name, dimensions, type, data_bytes_});
        data_bytes_ += bytes;
        return *this;
    }

    // Returns the header bytes; `logical_size` receives the size of the whole file.
    std::vector<std::uint8_t> build(std::uint64_t& logical_size) const {
        std::vector<std::uint8_t> out;
        const auto put32 = [&](std::uint32_t value) {
            for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(value >> (i * 8)));
        };
        const auto put64 = [&](std::uint64_t value) {
            for (int i = 0; i < 8; ++i) out.push_back(std::uint8_t(value >> (i * 8)));
        };
        const auto put_string = [&](const std::string& value) {
            put64(value.size());
            out.insert(out.end(), value.begin(), value.end());
        };
        put32(0x46554747);                       // "GGUF"
        put32(3);                                // version
        put64(tensors_.size());
        put64(metadata_count_);
        out.insert(out.end(), bytes_.begin(), bytes_.end());
        for (const Tensor& entry : tensors_) {
            put_string(entry.name);
            put32(static_cast<std::uint32_t>(entry.dimensions.size()));
            for (const std::uint64_t dimension : entry.dimensions) put64(dimension);
            put32(entry.type);
            put64(entry.offset);
        }
        constexpr std::uint64_t alignment = 32;
        const std::uint64_t data_offset = (out.size() + alignment - 1) / alignment * alignment;
        out.resize(static_cast<std::size_t>(data_offset), 0);
        logical_size = data_offset + data_bytes_;
        return out;
    }

private:
    struct Tensor {
        std::string name;
        std::vector<std::uint64_t> dimensions;
        std::uint32_t type;
        std::uint64_t offset;
    };

    void entry(const std::string& key, std::uint32_t type) {
        length(key.size());
        bytes_.insert(bytes_.end(), key.begin(), key.end());
        integer(type);
        ++metadata_count_;
    }
    void length(std::uint64_t value) {
        for (int i = 0; i < 8; ++i) bytes_.push_back(std::uint8_t(value >> (i * 8)));
    }
    void integer(std::uint32_t value) {
        for (int i = 0; i < 4; ++i) bytes_.push_back(std::uint8_t(value >> (i * 8)));
    }

    std::vector<std::uint8_t> bytes_;
    std::vector<Tensor> tensors_;
    std::uint64_t metadata_count_ = 0;
    std::uint64_t data_bytes_ = 0;
};

// A complete small OLMoE header: `architecture_last` puts general.architecture after the
// olmoe.* keys, which GGUF permits and DAN must handle.
inline std::vector<std::uint8_t> olmoe_header(std::uint64_t& logical_size,
    bool architecture_last = false, std::uint32_t layers = 2, std::uint32_t experts = 4,
    std::uint32_t experts_used = 2, std::uint32_t hidden = 8, std::uint32_t ffn = 16) {
    GgufBuilder builder;
    if (!architecture_last) builder.string_value("general.architecture", "olmoe");
    builder.uint32_value("olmoe.block_count", layers)
        .uint32_value("olmoe.embedding_length", hidden)
        .uint32_value("olmoe.attention.head_count", 4)
        .uint32_value("olmoe.attention.head_count_kv", 4)
        .uint32_value("olmoe.expert_count", experts)
        .uint32_value("olmoe.expert_used_count", experts_used)
        .uint32_value("olmoe.feed_forward_length", ffn)
        // A second architecture's keys must be ignored, whatever the order.
        .uint32_value("qwen2.block_count", 99)
        .uint32_value("qwen2.embedding_length", 99);
    if (architecture_last) builder.string_value("general.architecture", "olmoe");
    builder.tensor("token_embd.weight", {hidden, 32}, 12, 1024);
    builder.tensor("output_norm.weight", {hidden}, 0, 64);
    builder.tensor("output.weight", {hidden, 32}, 14, 1024);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const char* norm : {"attn_norm.weight", "attn_q_norm.weight",
                "attn_k_norm.weight", "ffn_norm.weight"}) {
            builder.tensor(prefix + norm, {hidden}, 0, 64);
        }
        for (const char* projection : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight"}) {
            builder.tensor(prefix + projection, {hidden, hidden}, 12, 256);
        }
        builder.tensor(prefix + "ffn_gate_inp.weight", {hidden, experts}, 0, 128);
        builder.tensor(prefix + "ffn_gate_exps.weight", {hidden, ffn, experts}, 12, 2048);
        builder.tensor(prefix + "ffn_up_exps.weight", {hidden, ffn, experts}, 12, 2048);
        builder.tensor(prefix + "ffn_down_exps.weight", {ffn, hidden, experts}, 14, 2048);
    }
    return builder.build(logical_size);
}

} // namespace dan::test
