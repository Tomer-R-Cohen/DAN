// A reference decode through the pinned llama.cpp, for comparing against a DAN route.
//
//   provider_owned_stage_reference <model.gguf> <tokens> <prompt> [begin end]
//
// Without a layer range it runs the ordinary full model, which is what a DAN route of any
// shape must reproduce. With one it runs that range only, so a single process can show what
// a stage sees. Greedy sampling and one sequence, like the worker's tail. Prints the prompt
// and generated token ids so a comparison is over ids, not rendered text.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::fprintf(stderr, "usage: %s model.gguf tokens prompt [begin end]\n", argv[0]);
        return 2;
    }
    const int wanted = std::atoi(argv[2]);
    const std::string prompt = argv[3];
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::getenv("DAN_REFERENCE_GPU") ? 999 : 0;
    if (argc == 6) {
        model_params.dan_stage_start = std::atoi(argv[4]);
        model_params.dan_stage_end = std::atoi(argv[5]);
    }
    llama_model* model = llama_model_load_from_file(argv[1], model_params);
    if (!model) { std::fprintf(stderr, "could not load the model\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 1;
    llama_context* context = llama_init_from_model(model, context_params);
    if (!context) { std::fprintf(stderr, "could not create the context\n"); return 1; }

    const int count = -llama_tokenize(vocab, prompt.data(), (int) prompt.size(),
        nullptr, 0, true, true);
    std::vector<llama_token> tokens((std::size_t) (count > 0 ? count : 0));
    if (count <= 0 || llama_tokenize(vocab, prompt.data(), (int) prompt.size(), tokens.data(),
            count, true, true) != count) {
        std::fprintf(stderr, "could not tokenize the prompt\n");
        return 1;
    }
    std::printf("prompt_tokens=%d ids=", count);
    for (const llama_token token : tokens) std::printf("%d,", token);
    std::printf("\ngenerated=");

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = (llama_pos) index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index + 1 == tokens.size();
    }
    std::string text;
    int position = (int) tokens.size();
    for (int produced = 0; produced < wanted; ++produced) {
        if (llama_decode(context, batch) != 0) { std::fprintf(stderr, "\ndecode failed\n"); return 1; }
        const llama_token next = llama_sampler_sample(sampler, context, -1);
        std::printf("%d,", next);
        char piece[256];
        const int size = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, false);
        if (size > 0) text.append(piece, (std::size_t) size);
        if (llama_vocab_is_eog(vocab, next)) break;
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = position++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
    }
    std::printf("\ntext=%s\n", text.c_str());
    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    return 0;
}
