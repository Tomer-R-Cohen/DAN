package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"
)

// Local mode retains at most one idle-expiring conversation in process/worker RAM.
// No prompt files or request/content logs.
// ponytail: one request owns the sidecar return port; per-client sidecars if concurrency matters.
type localAPI struct {
	client, discover, key string
	manifests             []string
	timeout               time.Duration
	contextSize           int
	init                  sync.Once
	queued                chan struct{}
	turn                  chan struct{}
	process               *chatProcess // owned while holding turn
	// The model each conversation used, so it stays the same after the client process
	// expires (selection plan M5). Scope hash -> model SHA-256 only; no content.
	pinMu sync.Mutex
	pins  map[[32]byte]modelPin
}

type modelPin struct {
	sha256 string
	at     time.Time
}

const (
	maxPins = 256
	pinTTL  = 24 * time.Hour
)

// Model names the local API offers. dan-auto uses only routes predicted to meet the speed
// target and fails otherwise; dan-any is the explicit choice to accept a slower or not yet
// measured route.
var localModels = map[string]string{"dan-auto": "target", "dan-any": "any"}

func (a *localAPI) pinned(scope [32]byte) string {
	a.pinMu.Lock()
	defer a.pinMu.Unlock()
	pin, ok := a.pins[scope]
	if !ok || time.Since(pin.at) > pinTTL {
		delete(a.pins, scope)
		return ""
	}
	return pin.sha256
}

func (a *localAPI) pin(scope [32]byte, sha string) {
	a.pinMu.Lock()
	defer a.pinMu.Unlock()
	if a.pins == nil {
		a.pins = map[[32]byte]modelPin{}
	}
	if _, ok := a.pins[scope]; !ok && len(a.pins) >= maxPins {
		var oldest [32]byte
		first := true
		for key, value := range a.pins {
			if first || value.at.Before(a.pins[oldest].at) {
				oldest, first = key, false
			}
		}
		delete(a.pins, oldest)
	}
	a.pins[scope] = modelPin{sha, time.Now()}
}

type localRequest struct {
	Model               string           `json:"model"`
	Messages            []toolMessage    `json:"messages"`
	MaxTokens           int              `json:"max_tokens"`
	MaxCompletionTokens int              `json:"max_completion_tokens"`
	Stream              bool             `json:"stream"`
	Tools               []toolDefinition `json:"tools"`
	ToolChoice          json.RawMessage  `json:"tool_choice"`
	ResponseFormat      json.RawMessage  `json:"response_format"`
	Temperature         *float64         `json:"temperature,omitempty"`
	TopP                *float64         `json:"top_p,omitempty"`
	MinP                *float64         `json:"min_p,omitempty"`
	TopK                *int             `json:"top_k,omitempty"`
	Seed                *int64           `json:"seed,omitempty"`
	FrequencyPenalty    *float64         `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64         `json:"presence_penalty,omitempty"`
	RepeatPenalty       *float64         `json:"repeat_penalty,omitempty"`
	ParallelToolCalls   bool             `json:"parallel_tool_calls,omitempty"`
	User                string           `json:"user,omitempty"`
	PromptCacheKey      string           `json:"prompt_cache_key,omitempty"`
}

func (a *localAPI) handler() http.Handler {
	a.init.Do(func() {
		a.queued = make(chan struct{}, 8) // bounds waiting requests and their in-memory prompts
		a.turn = make(chan struct{}, 1)
	})
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		jsonReply(w, 200, map[string]string{"status": "ok", "mode": "decentralized"})
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		jsonReply(w, 200, map[string]any{"object": "list", "data": []any{
			map[string]string{"id": "dan-auto", "object": "model", "owned_by": "dan"},
			map[string]string{"id": "dan-any", "object": "model", "owned_by": "dan"}}})
	})
	mux.HandleFunc("POST /v1/chat/completions", a.chat)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !(&config{apiKey: a.key}).authorized(r) {
			apiError(w, 401, "invalid API key")
			return
		}
		mux.ServeHTTP(w, r)
	})
}

func (a *localAPI) chat(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	var input localRequest
	d := json.NewDecoder(r.Body)
	if d.Decode(&input) != nil || d.Decode(&struct{}{}) != io.EOF {
		apiError(w, 400, "invalid JSON; messages must contain text content")
		return
	}
	policy, known := localModels[input.Model]
	if !known {
		apiError(w, 404, "unknown model; use dan-auto or dan-any")
		return
	}
	if err := validateSampling(input); err != nil {
		apiError(w, 400, err.Error())
		return
	}
	if input.MaxTokens == 0 {
		input.MaxTokens = input.MaxCompletionTokens
	}
	if input.MaxTokens == 0 {
		input.MaxTokens = 256
	}
	if input.MaxTokens < 1 || input.MaxTokens > 4096 {
		apiError(w, 400, "max_tokens must be between 1 and 4096")
		return
	}
	toolNames, err := validateChat(input)
	if err != nil {
		apiError(w, 400, err.Error())
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), a.timeout)
	defer cancel()
	select {
	case a.queued <- struct{}{}:
		defer func() { <-a.queued }()
	default:
		w.Header().Set("Retry-After", "5")
		apiError(w, 429, "DAN request queue is full; retry shortly")
		return
	}
	select {
	case a.turn <- struct{}{}:
		defer func() { <-a.turn }()
	case <-ctx.Done():
		apiError(w, 504, "request cancelled or timed out while waiting for inference")
		return
	}
	if ctx.Err() != nil {
		return
	}
	scope := chatScope(input, r)
	process, err := a.chatClient(scope, policy)
	if err != nil {
		apiError(w, 503, "could not start DAN client")
		return
	}
	stopCancel := context.AfterFunc(ctx, process.close)
	clean := false
	defer func() {
		stopCancel()
		if !clean {
			process.close()
		} else {
			process.idle = time.AfterFunc(5*time.Minute, process.close)
		}
	}()
	if err = process.send(input); err != nil {
		apiError(w, 503, "could not send chat request")
		return
	}
	model := process.model
	id := fmt.Sprintf("chatcmpl-dan-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	started, done, eog := false, false, false
	tokens := 0
	promptTokens, cachedTokens := 0, 0
	contextFull := false
	invalidChat := false
	belowTarget, unavailable := false, false
	reason := ""
	var pending []byte
	var output strings.Builder
	var toolText toolTextStream
	decoder := process.output
	for {
		var event struct {
			Model        string `json:"model"`
			Bytes        string `json:"bytes"`
			Done         bool   `json:"done"`
			Tokens       int    `json:"tokens"`
			EOG          bool   `json:"eog"`
			Error        string `json:"error"`
			Reason       string `json:"reason"`
			SHA256       string `json:"sha256"`
			PromptTokens int    `json:"prompt_tokens"`
			CachedTokens int    `json:"cached_tokens"`
		}
		if err = decoder.Decode(&event); err != nil {
			break
		}
		if done {
			err = errors.New("events after completion")
			break
		}
		if event.Error != "" {
			contextFull = event.Error == "context_length_exceeded"
			invalidChat = event.Error == "invalid_chat_request"
			belowTarget = event.Error == "below_target"
			unavailable = event.Error == "model_unavailable"
			reason = event.Reason
			err = errors.New("client inference failure")
			break
		}
		if event.Model != "" {
			model = event.Model
			process.model = model
			if len(event.SHA256) == 64 {
				a.pin(scope, event.SHA256)
			}
		}
		if event.Bytes != "" {
			var piece []byte
			piece, err = hex.DecodeString(event.Bytes)
			if err != nil {
				break
			}
			pending = append(pending, piece...)
			if !utf8.Valid(pending) {
				continue
			}
			output.Write(pending)
			text := string(pending)
			if len(toolNames) > 0 {
				text = toolText.push(text, false)
			}
			if input.Stream && text != "" {
				if !started {
					w.Header().Set("Content-Type", "text/event-stream")
					w.Header().Set("Cache-Control", "no-cache")
					started = true
				}
				if err = streamEvent(w, id, model, created, map[string]string{"role": "assistant", "content": text}, nil); err != nil {
					break
				}
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			pending = nil
		}
		if event.Done {
			done = true
			tokens = event.Tokens
			eog = event.EOG
			promptTokens, cachedTokens = event.PromptTokens, event.CachedTokens
			err = io.EOF // logical request boundary; the process stays alive
			break
		}
	}
	if err != io.EOF {
		cancel()
	}
	if err != io.EOF || !done || len(pending) != 0 {
		if invalidChat && !started {
			apiError(w, 400, "invalid chat request or unsupported model template/JSON schema")
			return
		}
		if belowTarget && !started {
			apiError(w, 503, "No DAN route is predicted to meet the speed target ("+strings.Join(strings.Fields(reason), " ")+
				"). Choose the dan-any model to accept a slower or not yet measured route.")
			return
		}
		if unavailable && !started {
			apiError(w, 503, "The model this conversation used is not available now. Start a new chat to use another model.")
			return
		}
		if contextFull {
			failure := map[string]any{"error": map[string]string{"message": "Conversation and tool definitions exceed the model context window. Start a new chat, shorten the history, or disable unused tools.", "type": "invalid_request_error", "code": "context_length_exceeded"}}
			if started {
				payload, _ := json.Marshal(failure)
				fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", payload)
			} else {
				jsonReply(w, 400, failure)
			}
			return
		}
		if started {
			payload, _ := json.Marshal(map[string]any{"error": map[string]string{"message": "DAN inference failed or timed out", "type": "server_error"}})
			fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", payload)
		} else {
			apiError(w, 503, "DAN inference failed or timed out; check that a compatible worker is online")
		}
		return
	}
	clean = true
	finish := "stop"
	if !eog && tokens >= input.MaxTokens {
		finish = "length"
	}
	content := output.String()
	var calls []toolCall
	if len(toolNames) > 0 {
		content, calls, err = parseToolOutput(content, toolNames, id)
		if err != nil {
			message := "model returned an invalid or incomplete tool call; retry or increase max_tokens"
			if started {
				payload, _ := json.Marshal(map[string]any{"error": map[string]string{"message": message, "type": "server_error"}})
				fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", payload)
			} else {
				apiError(w, 502, message)
			}
			return
		}
		if len(calls) > 0 {
			finish = "tool_calls"
		}
	}
	if input.Stream {
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
		}
		if len(toolNames) > 0 {
			delta := map[string]any{"role": "assistant"}
			if remaining := toolText.push("", true); remaining != "" {
				delta["content"] = remaining
			}
			if len(calls) > 0 {
				chunks := make([]map[string]any, 0, len(calls))
				for index, call := range calls {
					chunks = append(chunks, map[string]any{"index": index, "id": call.ID, "type": call.Type, "function": call.Function})
				}
				delta["tool_calls"] = chunks
			}
			_ = streamEvent(w, id, model, created, delta, nil)
		}
		_ = streamEvent(w, id, model, created, map[string]string{}, finish)
		fmt.Fprint(w, "data: [DONE]\n\n")
		return
	}
	responseMessage := map[string]any{"role": "assistant", "content": content}
	if len(calls) > 0 {
		responseMessage["tool_calls"] = calls
		if content == "" {
			responseMessage["content"] = nil
		}
	}
	jsonReply(w, 200, map[string]any{"id": id, "object": "chat.completion", "created": created, "model": model,
		"usage": map[string]any{"prompt_tokens": promptTokens, "completion_tokens": tokens,
			"total_tokens": promptTokens + tokens, "prompt_tokens_details": map[string]int{"cached_tokens": cachedTokens}},
		"choices": []any{map[string]any{"index": 0, "message": responseMessage, "finish_reason": finish}}})
}

func serveLocal(listen, client, discover string, manifests []string, key string, timeout time.Duration, contextSize int) error {
	host, _, err := net.SplitHostPort(listen)
	if err != nil {
		return err
	}
	ip := net.ParseIP(host)
	if host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return errors.New("local API must listen on loopback")
	}
	if timeout <= 0 {
		return errors.New("timeout must be positive")
	}
	if contextSize < 1 {
		return errors.New("context must be positive")
	}
	for _, path := range append([]string{client}, manifests...) {
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("missing file: %s", path)
		}
	}
	a := &localAPI{client: client, discover: discover, manifests: manifests, key: key, timeout: timeout, contextSize: contextSize}
	server := &http.Server{Addr: listen, Handler: a.handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: timeout + 5*time.Second, IdleTimeout: 60 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server.BaseContext = func(net.Listener) context.Context { return ctx }
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	fmt.Printf("DAN local API: http://%s/v1 (model: dan-auto)\n", listen)
	err = server.ListenAndServe()
	stop() // cancel active handlers before taking ownership of the retained client
	a.turn <- struct{}{}
	if a.process != nil {
		if a.process.idle != nil {
			a.process.idle.Stop()
		}
		a.process.close()
		<-a.process.done
	}
	<-a.turn
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
