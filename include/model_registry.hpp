#pragma once
#include <cstddef>
#include <charconv>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace dan {
struct ModelDefinition {
    std::string id, role, family, path, runtime_name, quantization;
    std::size_t memory_mib = 0, context_length = 0;
    std::string requirements;
    bool single_provider = false, distributed = false;
};

class ModelRegistry {
public:
    bool load(const std::string& path, std::string& error) {
        std::ifstream input(path); if (!input) { error = "Could not open model registry: " + path; return false; }
        std::string line; std::size_t number = 0;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            ++number; if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> fields; std::stringstream stream(line); std::string field;
            while (std::getline(stream, field, '|')) fields.push_back(field);
            if (fields.size() != 11) { error = "Invalid model registry line " + std::to_string(number); return false; }
            ModelDefinition model;
            model.id=fields[0]; model.role=fields[1]; model.family=fields[2]; model.path=fields[3];
            model.runtime_name=fields[4]; model.quantization=fields[5];
            const auto parse_size = [](const std::string& text, std::size_t& value) {
                auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
                return ec == std::errc{} && end == text.data() + text.size();
            };
            if (!parse_size(fields[6], model.memory_mib) || !parse_size(fields[7], model.context_length)
                || model.context_length == 0) { error="Invalid numeric model field on line "+std::to_string(number); return false; }
            if ((model.role != "test" && model.role != "main" && model.role != "distributed")
                || model.path.empty() || model.runtime_name.empty()
                || (fields[9] != "yes" && fields[9] != "no")
                || (fields[10] != "yes" && fields[10] != "no")) {
                error="Invalid role, path, runtime name or execution flag on line "+std::to_string(number);
                return false;
            }
            model.requirements=fields[8]; model.single_provider=fields[9]=="yes"; model.distributed=fields[10]=="yes";
            if (model.id.empty() || find(model.id)) { error="Missing or duplicate model ID on line "+std::to_string(number); return false; }
            models_.push_back(std::move(model));
        }
        return true;
    }
    const ModelDefinition* find(const std::string& id) const {
        for (const auto& model : models_) if (model.id == id) return &model;
        return nullptr;
    }
    const std::vector<ModelDefinition>& models() const { return models_; }
private:
    std::vector<ModelDefinition> models_;
};
}
