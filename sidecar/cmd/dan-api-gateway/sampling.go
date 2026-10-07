package main

import (
	"encoding/json"
	"fmt"
	"math"
)

func validateSampling(input localRequest) error {
	for _, field := range []struct {
		name      string
		value     *float64
		low, high float64
	}{
		{"temperature", input.Temperature, 0, 2}, {"top_p", input.TopP, 0, 1},
		{"min_p", input.MinP, 0, 1}, {"frequency_penalty", input.FrequencyPenalty, -2, 2},
		{"presence_penalty", input.PresencePenalty, -2, 2}, {"repeat_penalty", input.RepeatPenalty, 0.01, 10},
	} {
		if field.value != nil && (math.IsNaN(*field.value) || math.IsInf(*field.value, 0) || *field.value < field.low || *field.value > field.high) {
			return fmt.Errorf("invalid %s", field.name)
		}
	}
	if input.TopK != nil && (*input.TopK < 0 || *input.TopK > 1000000) {
		return fmt.Errorf("invalid top_k")
	}
	if input.Seed != nil && (*input.Seed < -1 || *input.Seed > math.MaxUint32) {
		return fmt.Errorf("invalid seed")
	}
	if len(input.ResponseFormat) == 0 || string(input.ResponseFormat) == "null" {
		return nil
	}
	var format struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Schema map[string]json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if json.Unmarshal(input.ResponseFormat, &format) != nil {
		return fmt.Errorf("invalid response_format")
	}
	switch format.Type {
	case "text":
		return nil
	case "json_object":
	case "json_schema":
		if format.JSONSchema.Schema == nil {
			return fmt.Errorf("json_schema.schema must be an object")
		}
	default:
		return fmt.Errorf("unsupported response_format")
	}
	if len(input.Tools) > 0 && string(input.ToolChoice) != `"none"` {
		return fmt.Errorf("structured response and tools cannot be combined")
	}
	return nil
}
