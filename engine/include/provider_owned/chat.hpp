#pragma once

// Request translation only. Templates, JSON-schema grammar and sampling are upstream.
#include "chat.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include <cmath>
#include <stdexcept>

namespace dan::provider_owned {
struct ChatPreparation {
    common_chat_params chat;
    common_params_sampling sampling;
};

inline ChatPreparation prepare_chat(const llama_model* model, const common_chat_templates* templates, const std::string& body) {
    if (body.size() > 1024 * 1024) throw std::runtime_error("chat request too large");
    const auto request = common_json::parse(body);
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(request.at("messages"));
    if (inputs.messages.empty()) throw std::runtime_error("messages must not be empty");
    if (request.contains("tools") && !request.at("tools").is_null()) inputs.tools = common_chat_tools_parse_oaicompat(request.at("tools"));
    const auto choice = request.contains("tool_choice") && !request.at("tool_choice").is_null()
        ? request.at("tool_choice").get<std::string>() : std::string("auto");
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(choice);
    inputs.parallel_tool_calls = request.value("parallel_tool_calls", false);
    inputs.enable_thinking = false;
    bool structured = false;
    if (request.contains("response_format") && !request.at("response_format").is_null()) {
        const auto& format = request.at("response_format");
        const auto type = format.value("type", std::string("text"));
        if (type == "json_object") { inputs.json_schema = "{\"type\":\"object\"}"; structured = true; }
        else if (type == "json_schema") {
            inputs.json_schema = format.at("json_schema").at("schema").dump(); structured = true;
        } else if (type != "text") throw std::runtime_error("unsupported response_format");
    }
    if (structured && !inputs.tools.empty() && choice != "none") {
        throw std::runtime_error("structured response and tools cannot be combined");
    }
    ChatPreparation prepared;
    prepared.chat = common_chat_templates_apply(templates, inputs);
    auto& params = prepared.sampling;
    const auto number = [&](const char* name, float fallback, float low, float high) {
        const float value = request.value(name, fallback);
        if (!std::isfinite(value) || value < low || value > high) {
            throw std::runtime_error("invalid sampling parameter");
        }
        return value;
    };
    params.temp = number("temperature", 0, 0, 2);
    params.top_p = number("top_p", 1, 0, 1);
    params.min_p = number("min_p", 0, 0, 1);
    params.penalty_freq = number("frequency_penalty", 0, -2, 2);
    params.penalty_present = number("presence_penalty", 0, -2, 2);
    params.penalty_repeat = number("repeat_penalty", 1, 0.01f, 10);
    params.top_k = request.value("top_k", 0);
    if (params.top_k < 0 || params.top_k > 1000000) throw std::runtime_error("invalid top_k");
    const auto seed = request.value("seed", int64_t(-1));
    if (seed < -1 || seed > UINT32_MAX) throw std::runtime_error("invalid seed");
    params.seed = seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed);
    params.no_perf = true;
    if (!prepared.chat.grammar.empty()) {
        params.grammar = {structured ? COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT : COMMON_GRAMMAR_TYPE_TOOL_CALLS,
            prepared.chat.grammar};
        params.grammar_lazy = prepared.chat.grammar_lazy;
        params.grammar_triggers = prepared.chat.grammar_triggers;
        params.generation_prompt = prepared.chat.generation_prompt;
        for (const auto& text : prepared.chat.preserved_tokens) {
            for (auto token : common_tokenize(llama_model_get_vocab(model), text, false, true)) params.preserved_tokens.insert(token);
        }
    }
    if (structured) {
        // The API promises a JSON value, without template-specific fences or prose.
        params.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
            json_schema_to_grammar(common_json::parse(inputs.json_schema))};
        params.grammar_lazy = false;
        params.grammar_triggers.clear();
        params.generation_prompt.clear();
    }
    return prepared;
}
} // namespace dan::provider_owned
