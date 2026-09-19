#include "provider_owned/range_model.hpp"
#include "provider_owned/planner.hpp"

#include "model_fixtures.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace po = dan::provider_owned;

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

// A cached sparse model is reused only while every stored range still hashes to what was
// recorded. The ranges are hashed in place: empty ranges, ranges at the end of the file and
// damage inside or outside a range are all covered; a truncated file is refused.
int check_cache_reuse_hashing() {
    namespace fs = std::filesystem;
    using dan::provider_owned::RangeModelRequest;
    using dan::provider_owned::RangeModelStats;
    const std::string data = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-";
    const fs::path path = fs::temp_directory_path() / "dan-range-reuse-test.gguf";
    const fs::path sidecar = path.string() + ".ranges";
    // Port 1 refuses at once (curl does not retry a refused connection), so the download
    // that follows a refused cache fails fast instead of waiting on DNS retries.
    RangeModelRequest request{"https://127.0.0.1:1/model.gguf",
        std::string(40, 'c'), std::string(64, 'd'), path, 0, 12};
    const auto write_model = [&](const std::string& bytes) {
        std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
    };
    const auto write_sidecar = [&] {
        std::ofstream out(sidecar, std::ios::binary | std::ios::trunc);
        out << "DAN_RANGE_CACHE_V1\nurl=" << request.url << "\nrevision=" << request.revision
            << "\nfull_sha256=" << request.full_sha256 << "\nstage_start=0\nstage_end=12\n"
            << "logical_bytes=64\nheader_bytes=8\ndownloaded_bytes=32\nshared_bytes=0\n"
            << "tensors_present=1\n"
            << "range=0,8,924592b9b103f14f833faafb67f480691f01988aa457c0061769f58cd47311bc\n"
            << "range=8,32,b0895acfe28861b5ee6387cea2c3caaa93d888497dadb7bcea85ce7095d4e4cd\n"
            << "range=64,0,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n";  // empty, at the end
    };
    const auto attempt = [&](RangeModelStats& stats, std::string& error) {
        stats = {};
        error.clear();
        return dan::provider_owned::prepare_range_model(request, stats, error);
    };
    RangeModelStats stats;
    std::string error;
    int failure = 0;
    const auto check = [&](bool condition, const char* what) {
        if (!condition && failure == 0) { std::cerr << "range reuse: " << what << '\n'; failure = 1; }
    };
    write_model(data);
    write_sidecar();
    check(attempt(stats, error) && stats.cache_reused, "intact cache was not reused");
    std::string damaged = data;
    damaged[63] = 'X';  // outside every range: not verified, still reused
    write_model(damaged);
    write_sidecar();
    check(attempt(stats, error) && stats.cache_reused, "damage outside the ranges refused the cache");
    damaged = data;
    damaged[8] = 'X';   // first byte of a range
    write_model(damaged);
    write_sidecar();
    check(!attempt(stats, error) && !stats.cache_reused, "damage inside a range was accepted");
    damaged = data;
    damaged[39] = 'X';  // last byte of a range
    write_model(damaged);
    write_sidecar();
    check(!attempt(stats, error) && !stats.cache_reused, "damage at a range end was accepted");
    write_model(data.substr(0, 40));  // truncated: shorter than the recorded logical size
    write_sidecar();
    check(!attempt(stats, error) && !stats.cache_reused, "truncated cache was accepted");
    fs::remove(path);
    fs::remove(sidecar);
    return failure;
}


// A synthetic OLMoE header parses into the geometry DAN plans with, whatever order the GGUF
// puts its metadata in, and every tensor keeps its shape and type.
int check_olmoe_header() {
    int failure = 0;
    const auto check = [&](bool condition, const char* what) {
        if (!condition && failure == 0) { std::cerr << "olmoe header: " << what << '\n'; failure = 1; }
    };
    for (const bool architecture_last : {false, true}) {
        std::uint64_t logical = 0;
        const auto header = dan::test::olmoe_header(logical, architecture_last);
        po::ModelIndex index;
        std::string error;
        check(po::parse_model_header(header, logical, index, error), "header did not parse");
        check(index.architecture == "olmoe", "wrong architecture");
        check(index.layers == 2 && index.hidden == 8, "wrong layer/hidden geometry");
        check(index.heads == 4 && index.kv_heads == 4, "wrong attention geometry");
        check(index.experts == 4 && index.experts_used == 2 && index.ffn_length == 16,
            "wrong expert geometry");
        check(index.logical_bytes == logical && index.header_bytes != 0, "wrong byte counts");
        const auto find = [&](const std::string& name) -> const po::ModelTensor* {
            for (const po::ModelTensor& tensor : index.tensors) {
                if (tensor.name == name) return &tensor;
            }
            return nullptr;
        };
        const po::ModelTensor* bank = find("blk.1.ffn_gate_exps.weight");
        const po::ModelTensor* down = find("blk.1.ffn_down_exps.weight");
        const po::ModelTensor* router = find("blk.0.ffn_gate_inp.weight");
        check(bank && bank->dimensions == std::vector<std::uint64_t>{8, 16, 4} && bank->type == 12,
            "expert bank shape or type was lost");
        check(down && down->dimensions == std::vector<std::uint64_t>{16, 8, 4} && down->type == 14,
            "expert down shape or type was lost");
        check(router && router->dimensions == std::vector<std::uint64_t>{8, 4}, "router shape");
        check(po::compatible_stage_model(index), "a valid OLMoE index was rejected");
    }
    // A truncated header is incomplete, not silently accepted.
    std::uint64_t logical = 0;
    auto header = dan::test::olmoe_header(logical);
    header.resize(header.size() / 2);
    po::ModelIndex index;
    std::string error;
    check(!po::parse_model_header(header, logical, index, error), "truncated header accepted");
    return failure;
}

// Ownership: a stage pays for every tensor of its own layers (whole expert banks included)
// and for nothing else. The head owns the embedding, the tail the norm and output.
int check_olmoe_ownership() {
    int failure = 0;
    const auto check = [&](bool condition, const char* what) {
        if (!condition && failure == 0) { std::cerr << "olmoe ownership: " << what << '\n'; failure = 1; }
    };
    const po::ModelIndex model = dan::test::olmoe_index(6);
    const auto sum = [&](int begin, int end) {
        std::uint64_t bytes = model.header_bytes;
        for (const po::ModelTensor& tensor : model.tensors) {
            const std::string prefix = "blk.";
            if (tensor.name.starts_with(prefix)) {
                const int layer = std::stoi(tensor.name.substr(4, tensor.name.find('.', 4) - 4));
                if (layer >= begin && layer < end) bytes += tensor.bytes;
            } else if (tensor.name == "token_embd.weight") {
                if (begin == 0) bytes += tensor.bytes;
            } else if (end == static_cast<int>(model.layers)) {
                bytes += tensor.bytes;  // output_norm.weight and output.weight
            }
        }
        return bytes;
    };
    check(po::stage_model_bytes(model, 0, 6) == sum(0, 6), "whole model bytes");
    check(po::stage_model_bytes(model, 0, 2) == sum(0, 2), "head bytes");
    check(po::stage_model_bytes(model, 2, 4) == sum(2, 4), "middle bytes");
    check(po::stage_model_bytes(model, 4, 6) == sum(4, 6), "tail bytes");
    // The three stages together cost the whole model plus the header counted three times.
    check(po::stage_model_bytes(model, 0, 2) + po::stage_model_bytes(model, 2, 4)
        + po::stage_model_bytes(model, 4, 6)
        == po::stage_model_bytes(model, 0, 6) + 2 * model.header_bytes, "stages do not tile");
    // A middle stage owns no embedding, norm or output.
    const std::uint64_t middle = po::stage_model_bytes(model, 2, 4);
    std::uint64_t layer_bytes = 0;
    for (const po::ModelTensor& tensor : model.tensors) {
        if (tensor.name.starts_with("blk.2.") || tensor.name.starts_with("blk.3.")) {
            layer_bytes += tensor.bytes;
        }
    }
    check(middle == model.header_bytes + layer_bytes, "middle stage owns head/tail tensors");
    // Every expert of an owned layer is charged: one layer holds three whole banks.
    std::uint64_t banks = 0;
    for (const po::ModelTensor& tensor : model.tensors) {
        if (tensor.name.starts_with("blk.2.") && tensor.name.find("_exps.") != std::string::npos) {
            banks += tensor.bytes;
            check(tensor.dimensions.size() == 3 && tensor.dimensions[2] == model.experts,
                "expert bank does not hold every expert");
        }
    }
    check(banks == (72 + 72 + 104) * 1024ull * 1024, "expert banks were not charged in full");
    // Never priced by the routed fraction (8 of 64).
    check(po::stage_model_bytes(model, 2, 4) > banks * model.experts_used / model.experts,
        "memory priced from active experts");

    // Qwen2 regression: a tied-output model gives the tail the token embedding instead.
    const po::ModelIndex tied = dan::test::qwen2_index(6, true);
    const po::ModelIndex untied = dan::test::qwen2_index(6, false);
    std::uint64_t tied_embd = 0;
    for (const po::ModelTensor& tensor : tied.tensors) {
        if (tensor.name == "token_embd.weight") tied_embd = tensor.bytes;
    }
    check(po::stage_model_bytes(tied, 3, 6) >= tied_embd, "tied tail lost the embedding");
    check(po::stage_model_bytes(untied, 3, 6) < po::stage_model_bytes(tied, 3, 6) + tied_embd,
        "untied tail also took the embedding");
    return failure;
}

// The index cache carries the new fields and refuses the version it cannot represent.
int check_index_cache_version() {
    namespace fs = std::filesystem;
    int failure = 0;
    const auto check = [&](bool condition, const char* what) {
        if (!condition && failure == 0) { std::cerr << "index cache: " << what << '\n'; failure = 1; }
    };
    const fs::path path = fs::temp_directory_path() / "dan-olmoe-index-test.index";
    const po::ModelIndex model = dan::test::olmoe_index(3);
    check(po::save_model_index(path, model), "could not save");
    po::ModelIndex loaded;
    check(po::load_model_index(path, loaded), "could not load");
    check(loaded.architecture == model.architecture && loaded.experts == model.experts
        && loaded.experts_used == model.experts_used && loaded.ffn_length == model.ffn_length,
        "expert geometry did not round-trip");
    check(loaded.tensors.size() == model.tensors.size(), "tensor count changed");
    bool same = true;
    for (std::size_t i = 0; i < loaded.tensors.size() && i < model.tensors.size(); ++i) {
        same = same && loaded.tensors[i].name == model.tensors[i].name
            && loaded.tensors[i].bytes == model.tensors[i].bytes
            && loaded.tensors[i].type == model.tensors[i].type
            && loaded.tensors[i].dimensions == model.tensors[i].dimensions;
    }
    check(same, "tensor shapes or types did not round-trip");
    check(po::compatible_stage_model(loaded), "a reloaded OLMoE index stopped being valid");
    // A version 1 file (no expert geometry, no shapes) must be refused so it is rebuilt.
    {
        std::ofstream old(path, std::ios::binary | std::ios::trunc);
        old << "dan-model-index 1" << '\n' << "qwen2 6 1024 16 4 100 4096 1" << '\n'
            << "4096 token_embd.weight" << '\n';
    }
    po::ModelIndex stale;
    check(!po::load_model_index(path, stale), "a version 1 index was accepted");
    // An unsupported architecture in a cache file is refused too.
    po::ModelIndex foreign = model;
    foreign.architecture = "deepseek2";
    check(po::save_model_index(path, foreign), "could not save");
    check(!po::load_model_index(path, stale), "an unsupported architecture was accepted");
    fs::remove(path);
    return failure;
}

int main() {
    namespace fs = std::filesystem;
    using dan::provider_owned::RangeModelRequest;
    using dan::provider_owned::RangeModelStats;
    if (check_model_index_cache() != 0) return 1;
    if (check_cache_reuse_hashing() != 0) return 1;
    if (check_olmoe_header() != 0) return 1;
    if (check_olmoe_ownership() != 0) return 1;
    if (check_index_cache_version() != 0) return 1;

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
