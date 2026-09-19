#pragma once

// Whether a draft model's tokenizer is interchangeable with the main model's.
//
// Speculative decoding feeds the main model's token ids straight into the draft, and the
// draft tokenizes the prompt text itself, so both directions must agree exactly: the same
// id -> text/attribute/score table, the same special tokens and BOS/EOS policy, the same
// tokenizer family and pre-tokenizer, and the same ids for real text. The merge table is not
// exposed by the llama.cpp API, so it is checked by behavior: a fixed corpus must tokenize
// identically. Anything that cannot be established is reported as a mismatch (fail closed).

#include "llama.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace dan::provider_owned {

namespace vocab_detail {

inline std::string metadata(const llama_model* model, const char* key) {
    char buffer[256];
    const int size = llama_model_meta_val_str(model, key, buffer, sizeof(buffer));
    if (size < 0) return "<none>";
    if (static_cast<std::size_t>(size) >= sizeof(buffer)) return "<long>" + std::to_string(size);
    return std::string(buffer, static_cast<std::size_t>(size));
}

inline bool tokenize(const llama_vocab* vocab, const std::string& text, bool special,
    std::vector<llama_token>& tokens) {
    int count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr, 0,
        special, special);
    if (count == std::numeric_limits<int32_t>::min()) return false;
    tokens.assign(static_cast<std::size_t>(count < 0 ? -count : count), 0);
    count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(),
        static_cast<int>(tokens.size()), special, special);
    if (count < 0) return false;
    tokens.resize(static_cast<std::size_t>(count));
    return true;
}

inline const std::vector<std::string>& corpus() {
    static const std::vector<std::string> texts = {
        "",
        "Hello, world!",
        "  leading and trailing spaces  ",
        "tabs\tand\nnewlines\r\n\r\nand    runs of spaces",
        "The quick brown fox jumps over the lazy dog. 0123456789 3.14159 1,000,000",
        "def f(x):\n    return x ** 2  # squares\n\tif x: pass",
        "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}} <html><body>&amp;</body></html>",
        "Caf\xC3\xA9 na\xC3\xAFve \xE2\x80\x9Cquotes\xE2\x80\x9D \xE2\x82\xAC 5 \xE2\x80\x94 dash",
        "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C\xE3\x80\x82 "
        "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 "
        "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7 "
        "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF",
        "emoji \xF0\x9F\x98\x80\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD end \xF0\x9F\xA6\x84",
        "don't we've I'm you'll they're it's O'Neill",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n",
        "<|endoftext|><s></s><unk><pad>[INST] [/INST] <|begin_of_text|><|eot_id|>",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "https://example.com/path?query=1&x=y#frag user@example.org C:\\Users\\Test\\file.txt",
        "\xFF\xFE broken \xC3 utf8",
    };
    return texts;
}

} // namespace vocab_detail

// True only when every check passes; `why` names the first difference otherwise.
inline bool vocabularies_match(const llama_model* main, const llama_model* draft, std::string& why) {
    if (!main || !draft) { why = "model missing"; return false; }
    const llama_vocab* a = llama_model_get_vocab(main);
    const llama_vocab* b = llama_model_get_vocab(draft);
    if (!a || !b) { why = "vocabulary missing"; return false; }
    if (llama_vocab_type(a) != llama_vocab_type(b)) { why = "tokenizer type differs"; return false; }
    const std::int32_t count = llama_vocab_n_tokens(a);
    if (count <= 0 || count != llama_vocab_n_tokens(b)) { why = "vocabulary size differs"; return false; }
    for (const char* key : {"tokenizer.ggml.model", "tokenizer.ggml.pre"}) {
        if (vocab_detail::metadata(main, key) != vocab_detail::metadata(draft, key)) {
            why = std::string(key) + " differs"; return false;
        }
    }
    if (llama_vocab_get_add_bos(a) != llama_vocab_get_add_bos(b)
        || llama_vocab_get_add_eos(a) != llama_vocab_get_add_eos(b)
        || llama_vocab_get_add_sep(a) != llama_vocab_get_add_sep(b)) {
        why = "BOS/EOS/SEP insertion policy differs"; return false;
    }
    if (llama_vocab_bos(a) != llama_vocab_bos(b) || llama_vocab_eos(a) != llama_vocab_eos(b)
        || llama_vocab_eot(a) != llama_vocab_eot(b) || llama_vocab_sep(a) != llama_vocab_sep(b)
        || llama_vocab_pad(a) != llama_vocab_pad(b) || llama_vocab_mask(a) != llama_vocab_mask(b)) {
        why = "special token ids differ"; return false;
    }
    for (llama_token token = 0; token < count; ++token) {
        const char* text_a = llama_vocab_get_text(a, token);
        const char* text_b = llama_vocab_get_text(b, token);
        if (!text_a || !text_b || std::strcmp(text_a, text_b) != 0) {
            why = "token " + std::to_string(token) + " text differs"; return false;
        }
        const float score_a = llama_vocab_get_score(a, token);
        const float score_b = llama_vocab_get_score(b, token);
        if (std::memcmp(&score_a, &score_b, sizeof(float)) != 0
            || llama_vocab_get_attr(a, token) != llama_vocab_get_attr(b, token)
            || llama_vocab_is_eog(a, token) != llama_vocab_is_eog(b, token)
            || llama_vocab_is_control(a, token) != llama_vocab_is_control(b, token)) {
            why = "token " + std::to_string(token) + " attributes differ"; return false;
        }
    }
    for (const std::string& text : vocab_detail::corpus()) {
        for (const bool special : {true, false}) {
            std::vector<llama_token> from_main, from_draft;
            const bool ok_main = vocab_detail::tokenize(a, text, special, from_main);
            const bool ok_draft = vocab_detail::tokenize(b, text, special, from_draft);
            if (ok_main != ok_draft || from_main != from_draft) {
                why = "tokenization differs on \"" + text.substr(0, 24) + "\""; return false;
            }
        }
    }
    return true;
}

} // namespace dan::provider_owned
