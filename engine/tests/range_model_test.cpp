#include "provider_owned/range_model.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// The saved model index reads back exactly; anything damaged or foreign is ignored.
int check_model_index_cache() {
    namespace fs = std::filesystem;
    namespace po = dan::provider_owned;
    const fs::path path = fs::temp_directory_path() / ("dan-model-index-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "/model.index");
    po::ModelIndex original;
    original.architecture = "qwen2";
    original.layers = 2; original.hidden = 64; original.heads = 4; original.kv_heads = 2;
    original.logical_bytes = 4096; original.header_bytes = 512;
    original.tensors = {{"token_embd.weight", 1024}, {"blk.0.attn_q.weight", 256},
        {"blk.1.attn_q.weight", 256}, {"output_norm.weight", 64}};
    po::ModelIndex loaded;
    if (!po::save_model_index(path, original) || !po::load_model_index(path, loaded)
        || loaded.layers != 2 || loaded.hidden != 64 || loaded.logical_bytes != 4096
        || loaded.header_bytes != 512 || loaded.tensors.size() != 4
        || loaded.tensors[1].name != "blk.0.attn_q.weight" || loaded.tensors[1].bytes != 256) {
        std::cerr << "model index did not round-trip\n";
        return 1;
    }
    { std::ofstream(path, std::ios::trunc) << "dan-model-index 1\nqwen2 2 64\n"; }
    po::ModelIndex untouched;
    if (po::load_model_index(path, untouched) || !untouched.tensors.empty()
        || po::load_model_index(path.parent_path() / "missing.index", untouched)) {
        std::cerr << "a truncated or missing index was accepted\n";
        return 1;
    }
    fs::remove_all(path.parent_path());
    return 0;
}

int main() {
    namespace fs = std::filesystem;
    using dan::provider_owned::RangeModelRequest;
    using dan::provider_owned::RangeModelStats;
    if (check_model_index_cache() != 0) return 1;

    const fs::path path = fs::temp_directory_path() /
        ("dan-range-model-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    {
        std::ofstream file(path, std::ios::binary);
        file << "existing model";
    }

    RangeModelRequest request{
        "https://invalid.example/model.gguf",
        std::string(40, 'a'), std::string(64, 'b'), path, 0, 12};
    RangeModelStats stats;
    std::string error;
    bool accepted = false;
    try {
        accepted = dan::provider_owned::prepare_range_model(request, stats, error);
    } catch (const std::exception& exception) {
        std::cerr << "unexpected exception: " << exception.what() << '\n';
        fs::remove(path);
        return 1;
    }

    std::ifstream file(path, std::ios::binary);
    const std::string contents((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    file.close();
    fs::remove(path);
    if (accepted || contents != "existing model") {
        std::cerr << "unrecognized model file was overwritten\n";
        return 1;
    }

    const fs::path partial_path = path.string() + ".ranges.incomplete";
    {
        std::ofstream model(path, std::ios::binary);
        model << "existing partial model";
        std::ofstream marker(partial_path, std::ios::binary);
        marker << "not a DAN cache\n";
    }
    error.clear();
    accepted = dan::provider_owned::prepare_range_model(request, stats, error);
    std::ifstream partial_model(path, std::ios::binary);
    const std::string partial_contents((std::istreambuf_iterator<char>(partial_model)),
        std::istreambuf_iterator<char>());
    partial_model.close();
    fs::remove(path); fs::remove(partial_path);
    if (accepted || partial_contents != "existing partial model"
        || error != "unrecognized partial range cache") {
        std::cerr << "unrecognized partial cache was not preserved\n";
        return 1;
    }
    return 0;
}
