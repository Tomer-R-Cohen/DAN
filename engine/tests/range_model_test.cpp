#include "provider_owned/range_model.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    namespace fs = std::filesystem;
    using dan::provider_owned::RangeModelRequest;
    using dan::provider_owned::RangeModelStats;

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
