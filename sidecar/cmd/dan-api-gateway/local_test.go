package main

import (
	"bufio"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// Exercise the real subprocess/stdin/SSE boundary without requiring a GPU.
func init() {
	if os.Getenv("DAN_TEST_CLIENT") != "1" || len(os.Args) < 2 || os.Args[1] != "--api-chat" {
		return
	}
	reader := bufio.NewReader(os.Stdin)
	for {
		header, err := reader.ReadString('\n')
		if err != nil {
			os.Exit(0)
		}
		var limit, size int
		if _, err = fmt.Sscanf(header, "%d %d", &limit, &size); err != nil || size < 1 || size > 1024*1024 {
			os.Exit(2)
		}
		prompt := make([]byte, size)
		if _, err = io.ReadFull(reader, prompt); err != nil {
			os.Exit(2)
		}
		var input localRequest
		if json.Unmarshal(prompt, &input) != nil {
			os.Exit(2)
		}
		if strings.Contains(string(prompt), "CONTEXT_FULL") {
			fmt.Println(`{"error":"context_length_exceeded"}`)
			os.Exit(1)
		}
		if strings.Contains(string(prompt), "WAIT") {
			time.Sleep(30 * time.Second)
			os.Exit(0)
		}
		if strings.Contains(string(prompt), "SLOW") {
			time.Sleep(200 * time.Millisecond)
		}
		if strings.Contains(string(prompt), "TOOL_TEST") {
			fmt.Println(`{"model":"qwen-test"}`)
			output := `<tool_call>{"name":"lookup","arguments":{"query":"hello €"}}</tool_call>`
			hasResult := false
			for _, m := range input.Messages {
				if m.Role == "tool" {
					hasResult = true
				}
			}
			if hasResult {
				if !strings.Contains(string(prompt), "Result from lookup") {
					os.Exit(2)
				}
				output = "Found the answer."
			} else if len(input.Tools) != 1 || input.Tools[0].Function.Name != "lookup" {
				os.Exit(2)
			}
			for _, b := range []byte(output) {
				fmt.Printf("{\"bytes\":\"%02x\"}\n", b)
			}
			fmt.Println(`{"done":true,"tokens":20,"eog":true}`)
			continue
		}
		if !strings.Contains(string(prompt), "Be helpful") || !strings.Contains(string(prompt), "Earlier answer") {
			os.Exit(2)
		}
		fmt.Println(`{"model":"qwen-test"}`)
		// Split an actual Unicode character across events.
		for _, p := range []string{"Hello ", "\xe2", "\x82\xac"} {
			fmt.Printf("{\"bytes\":\"%s\"}\n", hex.EncodeToString([]byte(p)))
		}
		fmt.Println(`{"done":true,"tokens":3,"eog":true}`)
	}
}

func TestToolTextStreaming(t *testing.T) {
	input := `Before <tool_call>{"name":"lookup","arguments":{}}</tool_call> between <tool_call>{"name":"lookup","arguments":{}}</tool_call> after <too`
	for size := 1; size <= len(input); size++ {
		var filter toolTextStream
		var text strings.Builder
		for offset := 0; offset < len(input); offset += size {
			end := offset + size
			if end > len(input) {
				end = len(input)
			}
			text.WriteString(filter.push(input[offset:end], false))
		}
		text.WriteString(filter.push("", true))
		if text.String() != "Before  between  after <too" {
			t.Fatalf("chunk size %d: %q", size, text.String())
		}
	}
	var filter toolTextStream
	if got := filter.push("Immediate text", false); got != "Immediate text" {
		t.Fatalf("ordinary text buffered: %q", got)
	}
}

func TestChatProcessReuseAndIsolation(t *testing.T) {
	t.Setenv("DAN_TEST_CLIENT", "1")
	exe, _ := os.Executable()
	a := &localAPI{client: exe, discover: "127.0.0.1:1", timeout: 3 * time.Second}
	t.Cleanup(func() {
		if a.process != nil {
			a.process.idle.Stop()
			a.process.close()
			<-a.process.done
		}
	})
	handler := a.handler()
	send := func(user, key string) *chatProcess {
		t.Helper()
		body := fmt.Sprintf(`{"model":"dan-auto","prompt_cache_key":%q,"messages":[{"role":"system","content":"Be helpful"},{"role":"assistant","content":"Earlier answer"},{"role":"user","content":"Hello"}]}`, key)
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("X-OpenWebUI-User-Id", user)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Fatalf("request: %d %s", w.Code, w.Body.String())
		}
		return a.process
	}
	first := send("alice", "chat-1")
	if send("alice", "chat-1") != first {
		t.Fatal("same chat restarted client")
	}
	second := send("bob", "chat-1")
	if second == first {
		t.Fatal("different user reused client")
	}
	select {
	case <-first.done:
	default:
		t.Fatal("old client not reaped")
	}
	third := send("bob", "chat-2")
	if third == second {
		t.Fatal("different chat reused client")
	}
	// Expiry releases the retained process and the next request starts cleanly.
	third.idle.Stop()
	third.idle = time.AfterFunc(time.Millisecond, third.close)
	select {
	case <-third.done:
	case <-time.After(3 * time.Second):
		t.Fatal("idle client not released")
	}
	if send("bob", "chat-2") == third {
		t.Fatal("expired client reused")
	}
}

func TestSamplingRequestValidation(t *testing.T) {
	for _, body := range []string{
		`{"temperature":-1}`, `{"top_p":1.1}`, `{"min_p":-0.1}`,
		`{"seed":4294967296}`, `{"repeat_penalty":0}`, `{"top_k":-1}`,
		`{"response_format":{"type":"json_schema","json_schema":{}}}`,
		`{"response_format":{"type":"unknown"}}`,
	} {
		var input localRequest
		if err := json.Unmarshal([]byte(body), &input); err != nil {
			t.Fatal(err)
		}
		if validateSampling(input) == nil {
			t.Fatalf("accepted invalid request: %s", body)
		}
	}
	var input localRequest
	if err := json.Unmarshal([]byte(`{"temperature":0.7,"seed":123,"top_p":0.9,"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}`), &input); err != nil {
		t.Fatal(err)
	}
	if err := validateSampling(input); err != nil {
		t.Fatal(err)
	}
}

func TestLocalToolRoundTrip(t *testing.T) {
	t.Setenv("DAN_TEST_CLIENT", "1")
	exe, _ := os.Executable()
	a := &localAPI{client: exe, discover: "127.0.0.1:1", timeout: 3 * time.Second}
	t.Cleanup(func() {
		if a.process != nil {
			a.process.close()
			<-a.process.done
		}
	})
	tool := `{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}`
	for _, stream := range []bool{false, true} {
		body := fmt.Sprintf(`{"model":"dan-auto","stream":%t,"messages":[{"role":"user","content":[{"type":"text","text":"TOOL_TEST"}]}],"tools":[%s],"tool_choice":"auto"}`, stream, tool)
		w := httptest.NewRecorder()
		a.handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if w.Code != 200 || !strings.Contains(w.Body.String(), `"finish_reason":"tool_calls"`) || strings.Contains(w.Body.String(), "<tool_call>") {
			t.Fatalf("tool response: %d %s", w.Code, w.Body.String())
		}
		if stream {
			if !strings.Contains(w.Body.String(), `"index":0`) || !strings.Contains(w.Body.String(), "data: [DONE]") {
				t.Fatal(w.Body.String())
			}
			continue
		}
		var response struct {
			Choices []struct {
				Message struct {
					ToolCalls []toolCall `json:"tool_calls"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		calls := response.Choices[0].Message.ToolCalls
		if len(calls) != 1 || calls[0].Function.Arguments != `{"query":"hello €"}` {
			t.Fatal(w.Body.String())
		}
		followup, _ := json.Marshal(map[string]any{"model": "dan-auto", "messages": []any{
			map[string]any{"role": "user", "content": "TOOL_TEST"},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": calls},
			map[string]any{"role": "tool", "tool_call_id": calls[0].ID, "content": "Result from lookup"},
		}})
		w = httptest.NewRecorder()
		a.handler().ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(followup))))
		if w.Code != 200 || !strings.Contains(w.Body.String(), "Found the answer.") {
			t.Fatal(w.Body.String())
		}
	}
	for _, output := range []string{`<tool_call>{"name":"unknown","arguments":{}}</tool_call>`, `<tool_call>{"name":"lookup","arguments":{}}`, `<tool_call>{"name":"lookup","arguments":"bad"}</tool_call>`} {
		if _, _, err := parseToolOutput(output, map[string]bool{"lookup": true}, "id"); err == nil {
			t.Fatalf("invalid call accepted: %s", output)
		}
	}
}

func TestLocalAPIProcessAndStreaming(t *testing.T) {
	t.Setenv("DAN_TEST_CLIENT", "1")
	exe, _ := os.Executable()
	a := &localAPI{client: exe, discover: "127.0.0.1:1", manifests: []string{"catalog.json"}, timeout: 3 * time.Second, key: "secret"}
	t.Cleanup(func() {
		if a.process != nil {
			a.process.close()
			<-a.process.done
		}
	})
	for _, stream := range []bool{false, true} {
		body := fmt.Sprintf(`{"model":"dan-auto","stream":%t,"messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"Earlier question"},{"role":"assistant","content":"Earlier answer"},{"role":"user","content":"Now answer"}],"temperature":0.7}`, stream)
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer secret")
		w := httptest.NewRecorder()
		a.handler().ServeHTTP(w, req)
		if w.Code != 200 || !strings.Contains(w.Body.String(), "qwen-test") {
			t.Fatalf("response: %d %s", w.Code, w.Body.String())
		}
		if stream {
			if !strings.Contains(w.Body.String(), `"content":"€"`) || !strings.Contains(w.Body.String(), "data: [DONE]") {
				t.Fatal(w.Body.String())
			}
		} else if !strings.Contains(w.Body.String(), "Hello €") {
			t.Fatal(w.Body.String())
		}
		if len(a.turn) != 0 || len(a.queued) != 0 {
			t.Fatal("request slot leaked")
		}
	}
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	a.handler().ServeHTTP(w, req)
	if w.Code != 401 {
		t.Fatal("missing authentication accepted")
	}
	a.key = ""
	for i := 0; i < cap(a.queued); i++ {
		a.queued <- struct{}{}
	}
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"dan-auto","messages":[{"role":"user","content":"hello"}]}`))
	w = httptest.NewRecorder()
	a.handler().ServeHTTP(w, req)
	if w.Code != 429 {
		t.Fatal("unbounded request queue accepted")
	}
	for len(a.queued) > 0 {
		<-a.queued
	}
	a.timeout = 100 * time.Millisecond
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"dan-auto","messages":[{"role":"user","content":"WAIT private contents"}]}`))
	w = httptest.NewRecorder()
	started := time.Now()
	a.handler().ServeHTTP(w, req)
	if w.Code != 503 || time.Since(started) > 3*time.Second || strings.Contains(w.Body.String(), "private contents") {
		t.Fatalf("timeout: %d %s", w.Code, w.Body.String())
	}
	a.timeout = 3 * time.Second
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"dan-auto","messages":[{"role":"user","content":"CONTEXT_FULL private contents"}]}`))
	w = httptest.NewRecorder()
	a.handler().ServeHTTP(w, req)
	if w.Code != 400 || !strings.Contains(w.Body.String(), `"code":"context_length_exceeded"`) || strings.Contains(w.Body.String(), "private contents") {
		t.Fatalf("context error: %d %s", w.Code, w.Body.String())
	}
}

func TestLocalAPIQueuesOverlappingRequests(t *testing.T) {
	t.Setenv("DAN_TEST_CLIENT", "1")
	exe, _ := os.Executable()
	a := &localAPI{client: exe, discover: "127.0.0.1:1", timeout: 3 * time.Second}
	t.Cleanup(func() {
		if a.process != nil {
			a.process.close()
			<-a.process.done
		}
	})
	handler := a.handler()
	body := `{"model":"dan-auto","messages":[{"role":"system","content":"Be helpful"},{"role":"assistant","content":"Earlier answer"},{"role":"user","content":"SLOW answer"}]}`
	results := make(chan *httptest.ResponseRecorder, 2)
	send := func() {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		results <- w
	}
	go send()
	deadline := time.Now().Add(time.Second)
	for len(a.turn) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(a.turn) == 0 {
		t.Fatal("first request never acquired inference slot")
	}
	go send()
	for i := 0; i < 2; i++ {
		w := <-results
		if w.Code != 200 || !strings.Contains(w.Body.String(), "Hello €") {
			t.Fatalf("queued request: %d %s", w.Code, w.Body.String())
		}
	}
	if len(a.turn) != 0 || len(a.queued) != 0 {
		t.Fatal("queued request leaked a slot")
	}
	// A disconnected waiter must leave without starting a subprocess or releasing
	// the active request's slot.
	a.turn <- struct{}{}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	ctx, cancel := context.WithCancel(req.Context())
	cancel()
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req.WithContext(ctx))
	if w.Code != 504 || len(a.queued) != 0 || len(a.turn) != 1 {
		t.Fatal("cancelled waiter leaked or stole an inference slot")
	}
	<-a.turn
}
