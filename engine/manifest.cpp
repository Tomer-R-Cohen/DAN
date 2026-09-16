#include "provider_owned/manifest.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace dan::provider_owned {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open manifest");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t value_start(const std::string& json, std::string_view key) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    const std::size_t found = json.find(quoted);
    if (found == std::string::npos) throw std::runtime_error("manifest missing " + std::string(key));
    std::size_t position = json.find(':', found + quoted.size());
    if (position == std::string::npos) throw std::runtime_error("invalid manifest");
    do { ++position; } while (position < json.size()
        && (json[position] == ' ' || json[position] == '\t'
            || json[position] == '\r' || json[position] == '\n'));
    return position;
}

std::string json_string(const std::string& json, std::string_view key) {
    std::size_t position = value_start(json, key);
    if (position >= json.size() || json[position++] != '"') {
        throw std::runtime_error("manifest field is not a string: " + std::string(key));
    }
    std::string value;
    while (position < json.size() && json[position] != '"') {
        if (json[position] == '\\') throw std::runtime_error("escaped manifest strings unsupported");
        value += json[position++];
    }
    if (position >= json.size() || value.empty()) throw std::runtime_error("invalid manifest string");
    return value;
}

std::string optional_json_string(const std::string& json, std::string_view key) {
    return json.find("\"" + std::string(key) + "\"") == std::string::npos
        ? std::string{} : json_string(json, key);
}

std::uint32_t json_uint(const std::string& json, std::string_view key) {
    std::size_t position = value_start(json, key);
    std::uint64_t value = 0;
    const std::size_t begin = position;
    while (position < json.size() && json[position] >= '0' && json[position] <= '9') {
        value = value * 10 + static_cast<unsigned>(json[position++] - '0');
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("manifest integer overflow");
        }
    }
    if (position == begin) throw std::runtime_error("manifest field is not an integer");
    return static_cast<std::uint32_t>(value);
}

std::uint32_t optional_json_uint(const std::string& json, std::string_view key) {
    return json.find("\"" + std::string(key) + "\"") == std::string::npos
        ? 0 : json_uint(json, key);
}

} // namespace

std::uint64_t json_uint64(const std::string& json, std::string_view key) {
    std::size_t position = value_start(json, key);
    std::uint64_t value = 0;
    const std::size_t begin = position;
    while (position < json.size() && json[position] >= '0' && json[position] <= '9') {
        const unsigned digit = static_cast<unsigned>(json[position++] - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::runtime_error("manifest integer overflow");
        }
        value = value * 10 + digit;
    }
    if (position == begin) throw std::runtime_error("JSON field is not an integer");
    return value;
}

std::string json_escape(std::string_view value) {
    std::string output;
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (byte < 0x20) {
                char escaped[7];
                std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
                output += escaped;
            } else output += static_cast<char>(byte);
        }
    }
    return output;
}

Manifest load_manifest(const std::string& path) {
    const std::string json = read_file(path);
    Manifest manifest;
    manifest.model_id = json_string(json, "model_id");
    manifest.architecture = optional_json_string(json, "architecture");
    manifest.layers = optional_json_uint(json, "layers");
    manifest.hidden = optional_json_uint(json, "hidden_size");
    manifest.context = json_uint(json, "context_size");
    manifest.revision = json_string(json, "artifact_revision");
    manifest.sha256 = json_string(json, "artifact_sha256");
    manifest.url = optional_json_string(json, "artifact_url");
    if (manifest.url.empty()) {
        const std::string repository = json_string(json, "hf_repo");
        const std::string filename = json_string(json, "gguf_filename");
        const auto unsafe = [](std::string_view value, bool slash) {
            const std::string_view allowed = slash
                ? "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./"
                : "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.";
            return value.empty() || value.find_first_not_of(allowed) != std::string_view::npos
                || value.contains("..") || value.front() == '/' || value.back() == '/';
        };
        if (unsafe(repository, true) || unsafe(filename, false)) {
            throw std::runtime_error("invalid Hugging Face repository or GGUF filename");
        }
        manifest.url = "https://huggingface.co/" + repository + "/resolve/"
            + manifest.revision + "/" + filename;
    }
    const auto hex = [](std::string_view value, std::size_t length) {
        return value.size() == length && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')
                || (c >= 'A' && c <= 'F');
        });
    };
    if (manifest.model_id.empty() || manifest.model_id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string::npos
        || (!manifest.architecture.empty() && manifest.architecture != "qwen2")
        || manifest.context == 0 || !manifest.url.starts_with("https://")
        || manifest.url.find_first_of("\r\n") != std::string::npos
        || !hex(manifest.revision, 40) || !hex(manifest.sha256, 64)) {
        throw std::runtime_error("manifest is not a valid pinned Qwen2 GGUF selection");
    }
    return manifest;
}

} // namespace dan::provider_owned
