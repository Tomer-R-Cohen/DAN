package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}
type toolDefinition struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}
type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}
type toolMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []toolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

func textContent(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return "", errors.New("message content must be text")
	}
	var out strings.Builder
	for _, p := range parts {
		if p.Type != "text" {
			return "", errors.New("only text content parts are supported")
		}
		out.WriteString(p.Text)
	}
	return out.String(), nil
}

func jsonArguments(arguments string) (json.RawMessage, error) {
	var object map[string]json.RawMessage
	if json.Unmarshal([]byte(arguments), &object) != nil || object == nil {
		return nil, errors.New("tool arguments must be a JSON object")
	}
	return json.RawMessage(arguments), nil
}

// Validate the HTTP boundary; worker-side llama.cpp renders the model's template.
func validateChat(input localRequest) (map[string]bool, error) {
	names := map[string]bool{}
	for _, tool := range input.Tools {
		if tool.Type != "function" || tool.Function.Name == "" || names[tool.Function.Name] {
			return nil, errors.New("tools must have unique function names")
		}
		if len(tool.Function.Parameters) > 0 {
			var schema map[string]json.RawMessage
			if json.Unmarshal(tool.Function.Parameters, &schema) != nil || schema == nil {
				return nil, errors.New("tool parameters must be an object")
			}
		}
		names[tool.Function.Name] = true
	}
	if len(input.ToolChoice) > 0 && string(input.ToolChoice) != "null" {
		var choice string
		if json.Unmarshal(input.ToolChoice, &choice) != nil || (choice != "auto" && choice != "none" && choice != "required") {
			return nil, errors.New("tool_choice supports auto, none or required")
		}
		if choice == "required" && len(names) == 0 {
			return nil, errors.New("required tool_choice needs tools")
		}
		if choice == "none" {
			names = map[string]bool{}
		}
	}
	if len(input.Messages) == 0 {
		return nil, errors.New("messages must not be empty")
	}
	pending := map[string]bool{}
	for _, m := range input.Messages {
		content, err := textContent(m.Content)
		if err != nil {
			return nil, err
		}
		if m.Role != "tool" && len(pending) > 0 {
			return nil, errors.New("tool results are missing from history")
		}
		switch m.Role {
		case "system", "user", "assistant":
			if len(m.ToolCalls) > 0 && m.Role != "assistant" {
				return nil, errors.New("only assistant messages may call tools")
			}
			if content == "" && len(m.ToolCalls) == 0 {
				return nil, errors.New("message content must not be empty")
			}
			for _, call := range m.ToolCalls {
				if call.ID == "" || pending[call.ID] || call.Type != "function" || call.Function.Name == "" {
					return nil, errors.New("invalid assistant tool call")
				}
				_, err := jsonArguments(call.Function.Arguments)
				if err != nil {
					return nil, err
				}
				pending[call.ID] = true
			}
		case "tool":
			if !pending[m.ToolCallID] || len(m.ToolCalls) > 0 {
				return nil, errors.New("tool result must reference a pending tool_call_id")
			}
			delete(pending, m.ToolCallID)
		default:
			return nil, errors.New("unsupported message role")
		}
	}
	if len(pending) > 0 {
		return nil, errors.New("tool results are missing from history")
	}
	return names, nil
}

// Hold split markers and tool JSON while letting ordinary text through.
type toolTextStream struct {
	pending string
	inCall  bool
}

func (s *toolTextStream) push(text string, final bool) string {
	s.pending += text
	var visible strings.Builder
	for {
		marker := "<tool_call>"
		if s.inCall {
			marker = "</tool_call>"
		}
		if index := strings.Index(s.pending, marker); index >= 0 {
			if !s.inCall {
				visible.WriteString(s.pending[:index])
			}
			s.pending = s.pending[index+len(marker):]
			s.inCall = !s.inCall
			continue
		}
		if s.inCall {
			break
		}
		keep := 0
		if !final {
			for n := 1; n < len(marker) && n <= len(s.pending); n++ {
				if strings.HasSuffix(s.pending, marker[:n]) {
					keep = n
				}
			}
		}
		visible.WriteString(s.pending[:len(s.pending)-keep])
		s.pending = s.pending[len(s.pending)-keep:]
		break
	}
	return visible.String()
}

func parseToolOutput(output string, names map[string]bool, id string) (string, []toolCall, error) {
	var content strings.Builder
	var calls []toolCall
	for {
		before, after, found := strings.Cut(output, "<tool_call>")
		if !found {
			content.WriteString(output)
			break
		}
		content.WriteString(before)
		var payload struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		decoder := json.NewDecoder(strings.NewReader(after))
		if decoder.Decode(&payload) != nil || !names[payload.Name] {
			return "", nil, errors.New("invalid model tool call")
		}
		rest := strings.TrimLeft(after[decoder.InputOffset():], " \r\n\t")
		if !strings.HasPrefix(rest, "</tool_call>") {
			return "", nil, errors.New("incomplete model tool call")
		}
		args, err := jsonArguments(string(payload.Arguments))
		if err != nil {
			return "", nil, err
		}
		call := toolCall{ID: fmt.Sprintf("call_%s_%d", id, len(calls)), Type: "function"}
		call.Function.Name = payload.Name
		call.Function.Arguments = string(args)
		calls = append(calls, call)
		output = strings.TrimPrefix(rest, "</tool_call>")
	}
	if strings.Contains(content.String(), "</tool_call>") {
		return "", nil, errors.New("invalid model tool call")
	}
	if len(calls) > 0 {
		return strings.TrimSpace(content.String()), calls, nil
	}
	return content.String(), nil, nil
}
