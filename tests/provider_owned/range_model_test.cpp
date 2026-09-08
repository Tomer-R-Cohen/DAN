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
    return 0;
}
