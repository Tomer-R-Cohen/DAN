#include "provider_owned/planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string_view>

namespace dan::provider_owned {

bool compatible_dense_qwen2(const ModelIndex& model, std::string* reason) {
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

std::uint64_t kv_bytes(const ModelIndex& model, int begin, int end,
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

bool stage_fits(const ModelIndex& model, std::uint64_t offered_mib, int begin, int end,
    std::uint32_t context, std::uint32_t sessions, StageAssignment& assignment) {
    constexpr std::uint64_t mib = 1024 * 1024;
    if (offered_mib > std::numeric_limits<std::uint64_t>::max() / mib) return false;
    const std::uint64_t offered = offered_mib * mib;
    const std::uint64_t reserve = std::max<std::uint64_t>(1024ull * mib, offered * 15 / 100);
    assignment = {0, begin, end, stage_model_bytes(model, begin, end),
        kv_bytes(model, begin, end, context, sessions)};
    return assignment.model_bytes != 0 && assignment.kv_bytes != 0
        && offered > reserve && assignment.model_bytes <= offered - reserve
        && assignment.kv_bytes <= offered - reserve - assignment.model_bytes;
}

std::optional<std::vector<StageAssignment>> plan_stages(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages) {
    if (!compatible_dense_qwen2(model) || offered_mib.empty() || offered_mib.size() > 8
        || context == 0 || sessions == 0 || minimum_stages == 0
        || minimum_stages > offered_mib.size()) return std::nullopt;
    auto fits = [&](std::size_t provider, int begin, int end, StageAssignment& assignment) {
        const bool fit = stage_fits(model, offered_mib[provider], begin, end, context, sessions,
            assignment);
        assignment.provider = provider;
        return fit;
    };

    for (std::size_t count = minimum_stages; count <= offered_mib.size()
            && count <= model.layers; ++count) {
        std::vector<std::size_t> order;
        std::vector<bool> used(offered_mib.size(), false);
        std::optional<std::vector<StageAssignment>> result;
        std::function<void()> choose = [&] {
            if (result) return;
            if (order.size() != count) {
                for (std::size_t index = 0; index < offered_mib.size(); ++index) {
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
                long double remaining_capacity = 0;
                for (std::size_t index = slot; index < count; ++index) {
                    remaining_capacity += offered_mib[order[index]];
                }
                const int wanted = std::clamp(begin + static_cast<int>(
                    (model.layers - begin) * offered_mib[order[slot]]
                    / remaining_capacity + 0.5L), begin + 1, last);
                std::vector<int> ends;
                for (int end = begin + 1; end <= last; ++end) ends.push_back(end);
                std::stable_sort(ends.begin(), ends.end(), [wanted](int left, int right) {
                    return std::abs(left - wanted) < std::abs(right - wanted);
                });
                for (const int end : ends) {
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

} // namespace dan::provider_owned
