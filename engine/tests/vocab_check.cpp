// Manual check of provider_owned/vocab_compat.hpp on real GGUF files (vocabulary only, no
// weights are loaded, so sparse or header-only files work):
//   provider_owned_vocab_check <main.gguf> <draft.gguf> <match|mismatch>
// Exit 0 when the verdict is the expected one. Also proves a model matches itself.
#include "provider_owned/vocab_compat.hpp"

#include <cstdio>
#include <cstring>

static llama_model* load(const char* path) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    return llama_model_load_from_file(path, params);
}

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s main.gguf draft.gguf match|mismatch\n", argv[0]); return 2; }
    llama_model* main_model = load(argv[1]);
    llama_model* draft_model = load(argv[2]);
    if (!main_model || !draft_model) { std::fprintf(stderr, "could not load a vocabulary\n"); return 2; }
    std::string why;
    const bool self = dan::provider_owned::vocabularies_match(main_model, main_model, why);
    const bool match = dan::provider_owned::vocabularies_match(main_model, draft_model, why);
    std::printf("self=%d match=%d%s%s\n", self, match, match ? "" : " why=", match ? "" : why.c_str());
    llama_model_free(main_model);
    llama_model_free(draft_model);
    const bool expected = std::strcmp(argv[3], "match") == 0;
    return self && match == expected ? 0 : 1;
}
