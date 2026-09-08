#pragma once

#include "provider_owned/range_model.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dan::provider_owned {

struct ProviderCapability {
    std::string id;
    std::string gpu;
    std::uint64_t offered_vram_mib = 0;
};

struct StageAssignment {
    std::size_t provider = 0;
    int begin = 0;
    int end = 0;
    std::uint64_t model_bytes = 0;
    std::uint64_t kv_bytes = 0;
};

struct ModelAssignment {
    std::string model_id;
    std::string url;
    std::string revision;
    std::string sha256;
    int begin = 0;
    int end = 0;
    std::uint32_t context = 0;
    std::uint32_t sessions = 0;
};

inline bool compatible_dense_qwen2(const ModelIndex& model, std::string* reason = nullptr) {
    const auto reject = [&](std::string_view message) {
        if (reason) *reason = message;
        return false;
    };
    if (model.architecture != "qwen2") return reject("architecture is not qwen2");
    if (model.layers < 2 || model.hidden == 0 || model.heads == 0 || model.kv_heads == 0
        || model.hidden % model.heads != 0 || model.heads % model.kv_heads != 0) {
        return reject("invalid Qwen2 layer or attention metadata");
    }
    const auto has = [&](std::string_view name) {
        return std::any_of(model.tensors.begin(), model.tensors.end(), [&](const ModelTensor& tensor) {
            return tensor.name == name;
        });
    };
    if (!has("token_embd.weight") || !has("output_norm.weight")) {
        return reject("missing embedding or output normalization tensor");
    }
    // ponytail: metadata is small; replace this linear scan only if model indexes become huge.
    for (std::uint32_t layer = 0; layer < model.layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        if (!std::any_of(model.tensors.begin(), model.tensors.end(), [&](const ModelTensor& tensor) {
                return tensor.name.starts_with(prefix);
            })) {
            return reject("missing tensors for transformer layer " + std::to_string(layer));
        }
    }
    return true;
}

inline std::string assignment_message(const ModelAssignment& assignment) {
    return "model_id=" + assignment.model_id + "\nurl=" + assignment.url
        + "\nrevision=" + assignment.revision + "\nsha256=" + assignment.sha256
        + "\nbegin=" + std::to_string(assignment.begin)
        + "\nend=" + std::to_string(assignment.end)
        + "\ncontext=" + std::to_string(assignment.context)
        + "\nsessions=" + std::to_string(assignment.sessions);
}

inline bool parse_assignment(std::string_view text, ModelAssignment& assignment) {
    assignment = {};
    auto number = [](std::string_view value, auto& output) {
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), output);
        return ec == std::errc{} && end == value.data() + value.size();
    };
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        const std::string_view line = text.substr(0, newline);
        const std::size_t equal = line.find('=');
        if (equal == std::string_view::npos) return false;
        const auto key = line.substr(0, equal), value = line.substr(equal + 1);
        if (key == "model_id") assignment.model_id = value;
        else if (key == "url") assignment.url = value;
        else if (key == "revision") assignment.revision = value;
        else if (key == "sha256") assignment.sha256 = value;
        else if (key == "begin") { if (!number(value, assignment.begin)) return false; }
        else if (key == "end") { if (!number(value, assignment.end)) return false; }
        else if (key == "context") { if (!number(value, assignment.context)) return false; }
        else if (key == "sessions") { if (!number(value, assignment.sessions)) return false; }
        else return false;
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    const auto hex = [](std::string_view value, std::size_t length) {
        return value.size() == length && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        });
    };
    return !assignment.model_id.empty()
        && assignment.model_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") == std::string::npos
        && assignment.url.starts_with("https://") && assignment.url.find_first_of("\r\n") == std::string::npos
        && hex(assignment.revision, 40) && hex(assignment.sha256, 64)
        && assignment.begin >= 0 && assignment.end > assignment.begin
        && assignment.context != 0 && assignment.sessions != 0;
}

inline std::uint64_t kv_bytes(const ModelIndex& model, int begin, int end,
    std::uint32_t context, std::uint32_t sessions) {
    if (model.heads == 0 || model.hidden % model.heads != 0) return 0;
    std::uint64_t value = context;
    for (const std::uint64_t factor : {std::uint64_t(sessions), std::uint64_t(end - begin),
            std::uint64_t(model.hidden / model.heads), std::uint64_t(model.kv_heads),
            std::uint64_t(2 * sizeof(std::uint16_t))}) {
        if (factor != 0 && value > std::numeric_limits<std::uint64_t>::max() / factor) return 0;
        value *= factor;
    }
    return value;
}

inline std::optional<std::vector<StageAssignment>> plan_replica(const ModelIndex& model,
    const std::vector<ProviderCapability>& providers, std::uint32_t context,
    std::uint32_t sessions) {
    if (!compatible_dense_qwen2(model) || providers.empty() || providers.size() > 8
        || context == 0 || sessions == 0) return std::nullopt;
    auto fits = [&](std::size_t provider, int begin, int end, StageAssignment& assignment) {
        constexpr std::uint64_t mib = 1024 * 1024;
        if (providers[provider].offered_vram_mib
            > std::numeric_limits<std::uint64_t>::max() / mib) return false;
        const std::uint64_t offered = providers[provider].offered_vram_mib * mib;
        const std::uint64_t reserve = std::max<std::uint64_t>(1024ull * mib, offered * 15 / 100);
        assignment = {provider, begin, end, stage_model_bytes(model, begin, end),
            kv_bytes(model, begin, end, context, sessions)};
        return assignment.model_bytes != 0 && assignment.kv_bytes != 0
            && offered > reserve && assignment.model_bytes <= offered - reserve
            && assignment.kv_bytes <= offered - reserve - assignment.model_bytes;
    };

    for (std::size_t count = 1; count <= providers.size()
            && count <= model.layers; ++count) {
        std::vector<std::size_t> order;
        std::vector<bool> used(providers.size(), false);
        std::optional<std::vector<StageAssignment>> result;
        std::function<void()> choose = [&] {
            if (result) return;
            if (order.size() != count) {
                for (std::size_t index = 0; index < providers.size(); ++index) {
                    if (used[index]) continue;
                    used[index] = true; order.push_back(index); choose(); order.pop_back(); used[index] = false;
                    if (result) return;
                }
                return;
            }
            std::vector<StageAssignment> stages;
            std::function<bool(std::size_t, int)> split = [&](std::size_t slot, int begin) {
                if (slot + 1 == count) {
                    StageAssignment assignment;
                    if (!fits(order[slot], begin, static_cast<int>(model.layers), assignment)) return false;
                    stages.push_back(assignment); return true;
                }
                const int last = static_cast<int>(model.layers - (count - slot - 1));
                for (int end = begin + 1; end <= last; ++end) {
                    StageAssignment assignment;
                    if (!fits(order[slot], begin, end, assignment)) continue;
                    stages.push_back(assignment);
                    if (split(slot + 1, end)) return true;
                    stages.pop_back();
                }
                return false;
            };
            if (split(0, 0)) result = stages;
        };
        choose();
        if (result) return result;
    }
    return std::nullopt;
}

inline std::string available_message(const ProviderCapability& provider) {
    return "id=" + provider.id + "\ngpu=" + provider.gpu
        + "\nvram_mib=" + std::to_string(provider.offered_vram_mib);
}

inline bool parse_available(std::string_view text, ProviderCapability& provider) {
    provider = {};
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        const std::string_view line = text.substr(0, newline);
        const std::size_t equal = line.find('=');
        if (equal == std::string_view::npos) return false;
        const auto key = line.substr(0, equal), value = line.substr(equal + 1);
        if (key == "id") provider.id = value;
        else if (key == "gpu") provider.gpu = value;
        else if (key == "vram_mib") {
            const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(),
                provider.offered_vram_mib);
            if (ec != std::errc{} || end != value.data() + value.size()) return false;
        } else return false;
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
    return !provider.id.empty() && provider.id.find_first_of("\r\n") == std::string::npos
        && !provider.gpu.empty() && provider.offered_vram_mib != 0;
}

} // namespace dan::provider_owned
