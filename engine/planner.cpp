#include "provider_owned/planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <array>
#include <limits>
#include <string_view>

namespace dan::provider_owned {

namespace {

bool has_tensor(const ModelIndex& model, std::string_view name) {
    return std::any_of(model.tensors.begin(), model.tensors.end(),
        [&](const ModelTensor& tensor) { return tensor.name == name; });
}

const ModelTensor* find_tensor(const ModelIndex& model, const std::string& name) {
    const auto found = std::find_if(model.tensors.begin(), model.tensors.end(),
        [&](const ModelTensor& tensor) { return tensor.name == name; });
    return found == model.tensors.end() ? nullptr : &*found;
}

bool valid_attention(const ModelIndex& model) {
    return model.layers >= 2 && model.hidden != 0 && model.heads != 0 && model.kv_heads != 0
        && model.hidden % model.heads == 0 && model.heads % model.kv_heads == 0;
}

// Qwen2: dense, and a tail without `output.weight` reuses the token embedding (tied head).
bool compatible_qwen2(const ModelIndex& model, const std::function<bool(std::string)>& reject) {
    if (!valid_attention(model)) return reject("invalid Qwen2 layer or attention metadata");
    if (model.experts != 0 || model.experts_used != 0) {
        return reject("Qwen2 model declares experts");
    }
    if (!has_tensor(model, "token_embd.weight") || !has_tensor(model, "output_norm.weight")) {
        return reject("missing embedding or output normalization tensor");
    }
    // ponytail: metadata is small; replace this linear scan only if model indexes become huge.
    for (std::uint32_t layer = 0; layer < model.layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        if (!std::any_of(model.tensors.begin(), model.tensors.end(),
                [&](const ModelTensor& tensor) { return tensor.name.starts_with(prefix); })) {
            return reject("missing tensors for transformer layer " + std::to_string(layer));
        }
    }
    return true;
}

// OLMoE: every owned layer keeps its whole expert bank, so each layer must carry the router
// and the three packed expert tensors, and their shapes must agree with the metadata. The
// head is explicit (`output.weight`); OLMoE has no tied-embedding fallback in llama.cpp.
bool compatible_olmoe(const ModelIndex& model, const std::function<bool(std::string)>& reject) {
    if (!valid_attention(model)) return reject("invalid OLMoE layer or attention metadata");
    if (model.experts < 2) return reject("OLMoE expert_count must be at least 2");
    if (model.experts_used == 0) return reject("OLMoE expert_used_count must be at least 1");
    if (model.experts_used > model.experts) {
        return reject("OLMoE routes to more experts than it has");
    }
    if (model.ffn_length == 0) return reject("OLMoE feed_forward_length is missing");
    for (const char* name : {"token_embd.weight", "output_norm.weight", "output.weight"}) {
        if (!has_tensor(model, name)) {
            return reject(std::string("missing ") + name + " (OLMoE has no tied output head)");
        }
    }
    const std::uint64_t hidden = model.hidden;
    const std::uint64_t ffn = model.ffn_length;
    const std::uint64_t experts = model.experts;
    for (std::uint32_t layer = 0; layer < model.layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const char* suffix : {"attn_norm.weight", "attn_q.weight", "attn_k.weight",
                "attn_v.weight", "attn_output.weight", "attn_q_norm.weight",
                "attn_k_norm.weight", "ffn_norm.weight"}) {
            if (!has_tensor(model, prefix + suffix)) {
                return reject("layer " + std::to_string(layer) + " is missing " + suffix);
            }
        }
        // Router and the packed expert banks, with the shapes llama.cpp's OLMoE loader asks
        // for: gate/up are {hidden, ffn, experts} and down is {ffn, hidden, experts}.
        const std::pair<const char*, std::array<std::uint64_t, 3>> expected[] = {
            {"ffn_gate_inp.weight", {hidden, experts, 0}},
            {"ffn_gate_exps.weight", {hidden, ffn, experts}},
            {"ffn_up_exps.weight", {hidden, ffn, experts}},
            {"ffn_down_exps.weight", {ffn, hidden, experts}},
        };
        for (const auto& [suffix, shape] : expected) {
            const ModelTensor* tensor = find_tensor(model, prefix + suffix);
            if (!tensor) {
                return reject("layer " + std::to_string(layer) + " is missing " + suffix);
            }
            const std::size_t wanted = shape[2] == 0 ? 2 : 3;
            if (tensor->dimensions.size() != wanted) {
                return reject(std::string(suffix) + " of layer " + std::to_string(layer)
                    + " has the wrong number of dimensions");
            }
            for (std::size_t axis = 0; axis < wanted; ++axis) {
                if (tensor->dimensions[axis] != shape[axis]) {
                    return reject(std::string(suffix) + " of layer " + std::to_string(layer)
                        + " does not match the model's expert geometry");
                }
            }
        }
    }
    return true;
}

} // namespace

bool compatible_stage_model(const ModelIndex& model, std::string* reason) {
    const auto reject = [&](std::string message) {
        if (reason) *reason = std::move(message);
        return false;
    };
    // Deliberately narrow: each architecture here has an audited stage loader and graph in
    // patches/llama-provider-owned.patch. Never widen this to "whatever llama.cpp can load".
    if (model.architecture == "qwen2") return compatible_qwen2(model, reject);
    if (model.architecture == "olmoe") return compatible_olmoe(model, reject);
    return reject("architecture " + (model.architecture.empty() ? "(none)" : model.architecture)
        + " is not supported for staged execution");
}

std::uint32_t allocated_positions(std::uint32_t context, std::uint32_t sessions) {
    // llama.cpp allocates more KV than asked for: it pads the whole context to 256 positions
    // and then, since DAN leaves kv_unified off, pads the per-sequence context to 256 as well
    // (llama-context.cpp). A planner that multiplies the requested context underestimates a
    // short context badly (512 asked, 512 allocated; 128 asked, 256 allocated per sequence).
    constexpr std::uint64_t pad = 256;
    if (context == 0 || sessions == 0) return 0;
    const std::uint64_t total = (std::uint64_t(context) * sessions + pad - 1) / pad * pad;
    const std::uint64_t per_session = (total / sessions + pad - 1) / pad * pad;
    return per_session > std::numeric_limits<std::uint32_t>::max()
        ? 0 : static_cast<std::uint32_t>(per_session);
}

std::uint64_t kv_bytes(const ModelIndex& model, int begin, int end,
    std::uint32_t context, std::uint32_t sessions) {
    if (model.heads == 0 || model.hidden % model.heads != 0 || end <= begin) return 0;
    const std::uint32_t positions = allocated_positions(context, sessions);
    if (positions == 0) return 0;
    std::uint64_t value = positions;
    for (const std::uint64_t factor : {std::uint64_t(sessions), std::uint64_t(end - begin),
            std::uint64_t(model.hidden / model.heads), std::uint64_t(model.kv_heads),
            std::uint64_t(2 * sizeof(std::uint16_t))}) {
        if (factor == 0 || value > std::numeric_limits<std::uint64_t>::max() / factor) return 0;
        value *= factor;
    }
    return value;
}

std::uint64_t decode_bytes(const ModelIndex& model, int begin, int end) {
    std::uint64_t bytes = stage_model_bytes(model, begin, end);
    if (begin == 0) {
        for (const auto& tensor : model.tensors) {
            if (tensor.name == "token_embd.weight" && tensor.bytes < bytes) bytes -= tensor.bytes;
        }
    }
    return bytes;
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

bool stage_with_draft_fits(const ModelIndex& model, const ModelIndex& draft,
    std::uint64_t offered_mib, int end, std::uint32_t context, std::uint32_t sessions) {
    StageAssignment main;
    if (!stage_fits(model, offered_mib, 0, end, context, sessions, main)) return false;
    const int draft_layers = static_cast<int>(draft.layers);
    const std::uint64_t draft_weights = stage_model_bytes(draft, 0, draft_layers);
    const std::uint64_t draft_kv = kv_bytes(draft, 0, draft_layers, context, sessions);
    // Unknown sizes (0) or an overflowing sum mean "does not fit": fail closed.
    if (draft_weights == 0 || draft_kv == 0
        || draft_weights > std::numeric_limits<std::uint64_t>::max() - draft_kv) return false;
    const std::uint64_t draft_bytes = draft_weights + draft_kv;
    constexpr std::uint64_t mib = 1024 * 1024;
    const std::uint64_t offered = offered_mib * mib;
    const std::uint64_t usable = offered - std::max<std::uint64_t>(1024ull * mib, offered * 15 / 100);
    const std::uint64_t used = main.model_bytes + main.kv_bytes;
    return draft_bytes <= usable - used;
}

std::optional<std::vector<StageAssignment>> plan_from_cache(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib,
    const std::vector<std::vector<std::pair<int, int>>>& cached, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages, std::size_t stage_limit,
    std::optional<std::size_t> head) {
    if (offered_mib.size() != cached.size() || minimum_stages == 0 || stage_limit == 0
        || context == 0 || sessions == 0 || !compatible_stage_model(model)
        || (head && *head >= offered_mib.size())) return std::nullopt;
    const int layers = static_cast<int>(model.layers);
    std::vector<StageAssignment> stages;
    std::vector<bool> used(offered_mib.size(), false);
    std::function<bool(int)> extend = [&](int begin) {
        if (begin == layers) return stages.size() >= minimum_stages;
        if (stages.size() >= stage_limit) return false;
        std::vector<std::pair<int, std::size_t>> options;  // (end, candidate)
        for (std::size_t index = 0; index < cached.size(); ++index) {
            if (used[index] || (head && begin == 0 && index != *head)) continue;
            for (const auto& [range_begin, range_end] : cached[index]) {
                if (range_begin == begin && range_end > begin && range_end <= layers) {
                    options.emplace_back(range_end, index);
                }
            }
        }
        std::sort(options.begin(), options.end(), std::greater<>());
        for (const auto& [end, index] : options) {
            StageAssignment assignment;
            if (!stage_fits(model, offered_mib[index], begin, end, context, sessions,
                    assignment)) continue;
            assignment.provider = index;
            used[index] = true;
            stages.push_back(assignment);
            if (extend(end)) return true;
            stages.pop_back();
            used[index] = false;
        }
        return false;
    };
    return extend(0) ? std::optional(stages) : std::nullopt;
}

std::optional<std::vector<StageAssignment>> plan_stages(const ModelIndex& model,
    const std::vector<std::uint64_t>& offered_mib, std::uint32_t context,
    std::uint32_t sessions, std::size_t minimum_stages, std::optional<std::size_t> head) {
    if (!compatible_stage_model(model) || offered_mib.empty() || offered_mib.size() > 8
        || context == 0 || sessions == 0 || minimum_stages == 0
        || minimum_stages > offered_mib.size()
        || (head && *head >= offered_mib.size())) return std::nullopt;
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
                    if (used[index] || (head && order.empty() && index != *head)) continue;
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
