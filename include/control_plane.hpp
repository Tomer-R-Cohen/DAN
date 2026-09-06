#pragma once

#include <charconv>
#include <cstddef>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace dan {

enum class ShardState { unassigned, assigned, downloading, cached, loading, ready, error };

inline std::string_view shard_state_name(ShardState state)
{
    switch (state) {
    case ShardState::unassigned: return "UNASSIGNED";
    case ShardState::assigned: return "ASSIGNED";
    case ShardState::downloading: return "DOWNLOADING";
    case ShardState::cached: return "CACHED";
    case ShardState::loading: return "LOADING";
    case ShardState::ready: return "READY";
    case ShardState::error: return "ERROR";
    }
    return "ERROR";
}

inline bool parse_shard_state(std::string_view text, ShardState& state)
{
    constexpr ShardState states[] = {ShardState::unassigned, ShardState::assigned,
        ShardState::downloading, ShardState::cached, ShardState::loading,
        ShardState::ready, ShardState::error};
    for (const ShardState candidate : states) {
        if (text == shard_state_name(candidate)) { state = candidate; return true; }
    }
    return false;
}

struct ShardMetadata {
    std::string id;
    std::size_t size_bytes = 0;
    std::string hash;
    std::string source;
    std::size_t min_vram_mib = 0;
};

struct ManagedModel {
    std::string id;
    std::string version;
    std::vector<ShardMetadata> shards;
};

struct CachedShard {
    std::string model_id;
    std::string version;
    std::string shard_id;
    std::string hash;
};

inline std::vector<std::string> split_fields(std::string_view text, char delimiter)
{
    std::vector<std::string> fields;
    while (true) {
        const std::size_t next = text.find(delimiter);
        fields.emplace_back(text.substr(0, next));
        if (next == std::string_view::npos) return fields;
        text.remove_prefix(next + 1);
    }
}

inline bool parse_size(std::string_view text, std::size_t& value)
{
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return !text.empty() && error == std::errc{} && end == text.data() + text.size();
}

inline bool parse_cached_shard(std::string_view text, CachedShard& shard)
{
    const auto fields = split_fields(text, '|');
    if (fields.size() != 4) return false;
    shard = {fields[0], fields[1], fields[2], fields[3]};
    return !shard.model_id.empty() && !shard.version.empty()
        && !shard.shard_id.empty() && !shard.hash.empty();
}

inline std::string cached_shard_value(const CachedShard& shard)
{
    return shard.model_id + '|' + shard.version + '|' + shard.shard_id + '|' + shard.hash;
}

inline bool load_managed_model(const std::string& path, ManagedModel& model,
    std::string& error)
{
    std::ifstream input(path);
    if (!input) { error = "Could not open managed-model manifest: " + path; return false; }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split_fields(line, '|');
        if (fields.size() == 3 && fields[0] == "model" && model.id.empty()) {
            model.id = fields[1];
            model.version = fields[2];
            continue;
        }
        ShardMetadata shard;
        if (fields.size() != 6 || fields[0] != "shard"
            || !parse_size(fields[2], shard.size_bytes)
            || !parse_size(fields[5], shard.min_vram_mib)) {
            error = "Invalid managed-model manifest line " + std::to_string(line_number);
            return false;
        }
        shard.id = fields[1]; shard.hash = fields[3]; shard.source = fields[4];
        if (shard.id.empty() || shard.hash.empty() || shard.source.empty()) {
            error = "Missing shard metadata on manifest line " + std::to_string(line_number);
            return false;
        }
        for (const auto& existing : model.shards) {
            if (existing.id == shard.id) {
                error = "Duplicate shard ID on manifest line " + std::to_string(line_number);
                return false;
            }
        }
        model.shards.push_back(std::move(shard));
    }
    if (model.id != "dan-main" || model.version.empty() || model.shards.empty()) {
        error = "Managed-model manifest must define dan-main, one version, and at least one shard";
        return false;
    }
    return true;
}

inline std::string shard_assignment_message(const ManagedModel& model,
    const ShardMetadata& shard)
{
    return "ASSIGN_SHARD\nmodel_id=" + model.id + "\nversion=" + model.version
        + "\nshard_id=" + shard.id + "\nsize_bytes=" + std::to_string(shard.size_bytes)
        + "\nhash=" + shard.hash + "\nsource=" + shard.source;
}

template <typename FieldHandler>
inline bool parse_control_fields(std::string_view message, std::string_view prefix,
    FieldHandler handle)
{
    if (!message.starts_with(prefix)) return false;
    message.remove_prefix(prefix.size());
    if (message.empty()) return false;
    while (!message.empty()) {
        const std::size_t newline = message.find('\n');
        const std::string_view line = message.substr(0, newline);
        const std::size_t equals = line.find('=');
        if (equals == std::string_view::npos || equals == 0
            || !handle(line.substr(0, equals), line.substr(equals + 1))) return false;
        if (newline == std::string_view::npos) return true;
        message.remove_prefix(newline + 1);
    }
    return false;
}

inline bool parse_shard_assignment(std::string_view message, CachedShard& assignment,
    ShardMetadata& metadata)
{
    bool model = false, version = false, id = false, size = false, hash = false, source = false;
    const bool parsed = parse_control_fields(message, "ASSIGN_SHARD\n",
        [&](std::string_view key, std::string_view value) {
            if (key == "model_id" && !model) { assignment.model_id = value; return model = true; }
            if (key == "version" && !version) { assignment.version = value; return version = true; }
            if (key == "shard_id" && !id) {
                assignment.shard_id = value; metadata.id = value; return id = true;
            }
            if (key == "size_bytes" && !size) return size = parse_size(value, metadata.size_bytes);
            if (key == "hash" && !hash) {
                assignment.hash = value; metadata.hash = value; return hash = true;
            }
            if (key == "source" && !source) { metadata.source = value; return source = true; }
            return false;
        });
    return parsed && model && version && id && size && hash && source
        && !assignment.model_id.empty() && !assignment.version.empty()
        && !assignment.shard_id.empty() && !assignment.hash.empty() && !metadata.source.empty();
}

inline std::string shard_state_message(const CachedShard& shard, ShardState state)
{
    return "SHARD_STATE\nmodel_id=" + shard.model_id + "\nversion=" + shard.version
        + "\nshard_id=" + shard.shard_id + "\nhash=" + shard.hash
        + "\nstate=" + std::string(shard_state_name(state));
}

inline bool parse_shard_state_message(std::string_view message, CachedShard& shard,
    ShardState& state)
{
    bool model = false, version = false, id = false, hash = false, state_seen = false;
    const bool parsed = parse_control_fields(message, "SHARD_STATE\n",
        [&](std::string_view key, std::string_view value) {
            if (key == "model_id" && !model) { shard.model_id = value; return model = true; }
            if (key == "version" && !version) { shard.version = value; return version = true; }
            if (key == "shard_id" && !id) { shard.shard_id = value; return id = true; }
            if (key == "hash" && !hash) { shard.hash = value; return hash = true; }
            if (key == "state" && !state_seen) {
                state_seen = parse_shard_state(value, state); return state_seen;
            }
            return false;
        });
    return parsed && model && version && id && hash && state_seen
        && !shard.model_id.empty() && !shard.version.empty()
        && !shard.shard_id.empty() && !shard.hash.empty();
}

} // namespace dan
