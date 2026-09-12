package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
	"unicode/utf8"
)

const (
	magic       = 0x44414e31
	version     = 2
	headerSize  = 48
	typeError   = 0
	typePrompt  = 4
	typeMetrics = 10
	typeResult  = 15
	typeStream  = 23
	typeChunk   = 24
	maxBodySize = 1 << 20
)

type config struct {
	coordinator  string
	coordinators []string
	model        string
	apiKey       string
	timeout      time.Duration
	requestID    atomic.Uint64
	replicaID    atomic.Uint64
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model     string    `json:"model"`
	Messages  []message `json:"messages"`
	MaxTokens int       `json:"max_tokens"`
	Stream    bool      `json:"stream"`
}

type frame struct {
	typeID  uint16
	request uint64
	rows    uint32
	payload []byte
}

type coordinatorError string

func (e coordinatorError) Error() string { return string(e) }

type coordinatorStatus struct {
	QueueDepth          float64 `json:"queue_depth"`
	QueueCapacity       float64 `json:"queue_capacity"`
	QueueDepthMax       float64 `json:"queue_depth_max"`
	LatencyP50MS        float64 `json:"request_latency_p50_ms"`
	LatencyP95MS        float64 `json:"request_latency_p95_ms"`
	LatencyP99MS        float64 `json:"request_latency_p99_ms"`
	RequestsCompleted   float64 `json:"requests_completed"`
	RequestsFailed      float64 `json:"requests_failed"`
	RequestsCancelled   float64 `json:"requests_cancelled"`
	RequestsTimedOut    float64 `json:"requests_timed_out"`
	ReplicaReformations float64 `json:"replica_reformations"`
	ResidentSessions    float64 `json:"resident_sessions"`
	KVMemoryBytes       float64 `json:"kv_memory_bytes"`
	GeneratedTokens     float64 `json:"generated_tokens"`
	TokensPerSecond     float64 `json:"aggregate_generated_tokens_per_second"`
	SpeculativeEnabled  bool    `json:"speculative_enabled"`
	PipelineDepth       float64 `json:"pipeline_depth"`
	SpeculativeRounds   float64 `json:"speculative_rounds"`
	DraftTokensProposed float64 `json:"draft_tokens_proposed"`
	DraftTokensAccepted float64 `json:"draft_tokens_accepted"`
	ActiveRequest       bool    `json:"active_request"`
	ReplicaAvailable    bool    `json:"replica_available"`
}

func writeFrame(w io.Writer, f frame) error {
	header := make([]byte, headerSize)
	binary.BigEndian.PutUint32(header, magic)
	binary.BigEndian.PutUint16(header[4:], version)
	binary.BigEndian.PutUint16(header[6:], f.typeID)
	binary.BigEndian.PutUint64(header[16:], f.request)
	binary.BigEndian.PutUint32(header[28:], f.rows)
	binary.BigEndian.PutUint64(header[40:], uint64(len(f.payload)))
	if written, err := w.Write(header); err != nil {
		return err
	} else if written != len(header) {
		return io.ErrShortWrite
	}
	written, err := w.Write(f.payload)
	if err == nil && written != len(f.payload) {
		err = io.ErrShortWrite
	}
	return err
}

func readFrame(r io.Reader) (frame, error) {
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(r, header); err != nil {
		return frame{}, err
	}
	if binary.BigEndian.Uint32(header) != magic || binary.BigEndian.Uint16(header[4:]) != version {
		return frame{}, errors.New("invalid DAN response")
	}
	size := binary.BigEndian.Uint64(header[40:])
	if size > maxBodySize {
		return frame{}, errors.New("DAN response is too large")
	}
	f := frame{typeID: binary.BigEndian.Uint16(header[6:]), request: binary.BigEndian.Uint64(header[16:]), rows: binary.BigEndian.Uint32(header[28:]), payload: make([]byte, size)}
	_, err := io.ReadFull(r, f.payload)
	return f, err
}

func (c *config) endpoints() []string {
	if len(c.coordinators) != 0 {
		return c.coordinators
	}
	return []string{c.coordinator}
}

func (c *config) generateOn(ctx context.Context, endpoint string, id uint64,
	prompt string, tokens int, onChunk func(frame) error) (frame, bool, error) {
	dialer := net.Dialer{Timeout: 5 * time.Second}
	conn, err := dialer.DialContext(ctx, "tcp", endpoint)
	if err != nil {
		return frame{}, false, err
	}
	defer conn.Close()
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = conn.Close()
		case <-done:
		}
	}()
	deadline := time.Now().Add(c.timeout)
	if end, ok := ctx.Deadline(); ok && end.Before(deadline) {
		deadline = end
	}
	if err := conn.SetDeadline(deadline); err != nil {
		return frame{}, false, err
	}
	typeID := uint16(typePrompt)
	if onChunk != nil {
		typeID = typeStream
	}
	if err := writeFrame(conn, frame{typeID: typeID, request: id, rows: uint32(tokens), payload: []byte(prompt)}); err != nil {
		return frame{}, false, err
	}
	emitted := false
	for {
		result, err := readFrame(conn)
		if err != nil {
			return frame{}, emitted, err
		}
		if result.request != id {
			return frame{}, emitted, errors.New("mismatched DAN response")
		}
		if result.typeID == typeChunk && onChunk != nil {
			if result.rows != 1 {
				return frame{}, emitted, errors.New("invalid DAN stream chunk")
			}
			if err := onChunk(result); err != nil {
				return frame{}, true, err
			}
			emitted = true
			continue
		}
		if result.typeID == typeError {
			return frame{}, emitted, coordinatorError(result.payload)
		}
		if result.typeID != typeResult {
			return frame{}, emitted, errors.New("unexpected DAN response")
		}
		return result, emitted, nil
	}
}

func retryableCoordinatorError(err error) bool {
	message := err.Error()
	return message == "replica_unavailable" || message == "queue_full" ||
		message == "connection_queue_full" || message == "queue_timeout" ||
		message == "provider_disconnected" || strings.HasPrefix(message, "provider_failure:")
}

func (c *config) generate(ctx context.Context, prompt string, tokens int,
	onChunk func(frame) error) (frame, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	endpoints := c.endpoints()
	start := int(c.replicaID.Add(1)-1) % len(endpoints)
	id := c.requestID.Add(1)
	var last error
	for offset := range endpoints {
		result, emitted, err := c.generateOn(ctx, endpoints[(start+offset)%len(endpoints)], id, prompt, tokens, onChunk)
		if err == nil {
			return result, nil
		}
		last = err
		var backendError coordinatorError
		if emitted || ctx.Err() != nil || (errors.As(err, &backendError) && !retryableCoordinatorError(err)) {
			return frame{}, err
		}
	}
	return frame{}, last
}

func coordinatorMetrics(ctx context.Context, endpoint string) (coordinatorStatus, error) {
	var status coordinatorStatus
	dialer := net.Dialer{Timeout: 2 * time.Second}
	conn, err := dialer.DialContext(ctx, "tcp", endpoint)
	if err != nil {
		return status, err
	}
	defer conn.Close()
	deadline := time.Now().Add(2 * time.Second)
	if end, ok := ctx.Deadline(); ok && end.Before(deadline) {
		deadline = end
	}
	if err := conn.SetDeadline(deadline); err != nil {
		return status, err
	}
	if err := writeFrame(conn, frame{typeID: typeMetrics}); err != nil {
		return status, err
	}
	result, err := readFrame(conn)
	if err != nil || result.typeID != typeMetrics {
		return status, errors.New("DAN coordinator unavailable")
	}
	if err := json.Unmarshal(result.payload, &status); err != nil {
		return status, errors.New("invalid DAN coordinator metrics")
	}
	return status, nil
}

func readyEndpoint(ctx context.Context, endpoint string) error {
	status, err := coordinatorMetrics(ctx, endpoint)
	if err != nil {
		return err
	}
	if !status.ReplicaAvailable {
		return errors.New("DAN replica unavailable")
	}
	return nil
}

func (c *config) ready(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	results := make(chan error, len(c.endpoints()))
	for _, endpoint := range c.endpoints() {
		go func() { results <- readyEndpoint(ctx, endpoint) }()
	}
	for range c.endpoints() {
		if <-results == nil {
			return nil
		}
	}
	return errors.New("no DAN replica is ready")
}

func qwenPrompt(messages []message) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("messages must not be empty")
	}
	var prompt strings.Builder
	for _, m := range messages {
		if (m.Role != "system" && m.Role != "user" && m.Role != "assistant") || m.Content == "" {
			return "", errors.New("each message needs a supported role and non-empty content")
		}
		fmt.Fprintf(&prompt, "<|im_start|>%s\n%s<|im_end|>\n", m.Role, m.Content)
	}
	prompt.WriteString("<|im_start|>assistant\n")
	return prompt.String(), nil
}

func jsonReply(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func apiError(w http.ResponseWriter, status int, message string) {
	jsonReply(w, status, map[string]any{"error": map[string]string{"message": message, "type": "invalid_request_error"}})
}

func (c *config) authorized(r *http.Request) bool {
	return c.apiKey == "" || subtle.ConstantTimeCompare(
		[]byte(r.Header.Get("Authorization")), []byte("Bearer "+c.apiKey)) == 1
}

func streamEvent(w http.ResponseWriter, id, model string, created int64,
	delta map[string]string, finish any) error {
	payload, err := json.Marshal(map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	})
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", payload)
	return err
}

func (c *config) streamChat(w http.ResponseWriter, r *http.Request,
	prompt string, tokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		apiError(w, http.StatusInternalServerError, "streaming is unavailable")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	created := time.Now().Unix()
	started := false
	var pending []byte
	emit := func(chunk frame) error {
		if err := r.Context().Err(); err != nil {
			return err
		}
		pending = append(pending, chunk.payload...)
		if !utf8.Valid(pending) {
			return nil
		}
		id := fmt.Sprintf("chatcmpl-dan-%d", chunk.request)
		delta := map[string]string{"content": string(pending)}
		if !started {
			delta["role"] = "assistant"
			started = true
		}
		if err := streamEvent(w, id, c.model, created, delta, nil); err != nil {
			return err
		}
		pending = pending[:0]
		flusher.Flush()
		return nil
	}
	result, err := c.generate(r.Context(), prompt, tokens, emit)
	if err != nil {
		payload, _ := json.Marshal(map[string]any{"error": map[string]string{"message": err.Error(), "type": "server_error"}})
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", payload)
		flusher.Flush()
		return
	}
	id := fmt.Sprintf("chatcmpl-dan-%d", result.request)
	if len(pending) != 0 {
		_ = streamEvent(w, id, c.model, created,
			map[string]string{"content": string(bytes.ToValidUTF8(pending, []byte("�")))}, nil)
	}
	finish := "stop"
	if result.rows == uint32(tokens) {
		finish = "length"
	}
	_ = streamEvent(w, id, c.model, created, map[string]string{}, finish)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (c *config) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		if err := c.ready(r.Context()); err != nil {
			jsonReply(w, http.StatusServiceUnavailable, map[string]string{"status": "unavailable"})
			return
		}
		jsonReply(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		if !c.authorized(r) {
			apiError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		type sample struct {
			index  int
			status coordinatorStatus
			err    error
		}
		results := make(chan sample, len(c.endpoints()))
		for index, endpoint := range c.endpoints() {
			go func() {
				status, err := coordinatorMetrics(r.Context(), endpoint)
				results <- sample{index: index, status: status, err: err}
			}()
		}
		samples := make([]sample, len(c.endpoints()))
		for range samples {
			result := <-results
			samples[result.index] = result
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		for _, result := range samples {
			label := fmt.Sprintf("{replica=\"%d\"}", result.index)
			up := 1
			if result.err != nil {
				up = 0
			}
			fmt.Fprintf(w, "dan_replica_up%s %d\n", label, up)
			if result.err != nil {
				continue
			}
			available, active, speculative := 0, 0, 0
			if result.status.ReplicaAvailable {
				available = 1
			}
			if result.status.ActiveRequest {
				active = 1
			}
			if result.status.SpeculativeEnabled {
				speculative = 1
			}
			fmt.Fprintf(w, "dan_replica_available%s %d\n", label, available)
			fmt.Fprintf(w, "dan_active_request%s %d\n", label, active)
			fmt.Fprintf(w, "dan_queue_depth%s %g\n", label, result.status.QueueDepth)
			fmt.Fprintf(w, "dan_queue_capacity%s %g\n", label, result.status.QueueCapacity)
			fmt.Fprintf(w, "dan_queue_depth_max%s %g\n", label, result.status.QueueDepthMax)
			fmt.Fprintf(w, "dan_request_latency_p50_ms%s %g\n", label, result.status.LatencyP50MS)
			fmt.Fprintf(w, "dan_request_latency_p95_ms%s %g\n", label, result.status.LatencyP95MS)
			fmt.Fprintf(w, "dan_request_latency_p99_ms%s %g\n", label, result.status.LatencyP99MS)
			fmt.Fprintf(w, "dan_requests_completed_total%s %g\n", label, result.status.RequestsCompleted)
			fmt.Fprintf(w, "dan_requests_failed_total%s %g\n", label, result.status.RequestsFailed)
			fmt.Fprintf(w, "dan_requests_cancelled_total%s %g\n", label, result.status.RequestsCancelled)
			fmt.Fprintf(w, "dan_requests_timed_out_total%s %g\n", label, result.status.RequestsTimedOut)
			fmt.Fprintf(w, "dan_replica_reformations_total%s %g\n", label, result.status.ReplicaReformations)
			fmt.Fprintf(w, "dan_resident_sessions%s %g\n", label, result.status.ResidentSessions)
			fmt.Fprintf(w, "dan_kv_memory_bytes%s %g\n", label, result.status.KVMemoryBytes)
			fmt.Fprintf(w, "dan_generated_tokens_total%s %g\n", label, result.status.GeneratedTokens)
			fmt.Fprintf(w, "dan_generated_tokens_per_second%s %g\n", label, result.status.TokensPerSecond)
			fmt.Fprintf(w, "dan_speculative_enabled%s %d\n", label, speculative)
			fmt.Fprintf(w, "dan_pipeline_depth%s %g\n", label, result.status.PipelineDepth)
			fmt.Fprintf(w, "dan_speculative_rounds_total%s %g\n", label, result.status.SpeculativeRounds)
			fmt.Fprintf(w, "dan_draft_tokens_proposed_total%s %g\n", label, result.status.DraftTokensProposed)
			fmt.Fprintf(w, "dan_draft_tokens_accepted_total%s %g\n", label, result.status.DraftTokensAccepted)
		}
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		if !c.authorized(r) {
			apiError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		jsonReply(w, http.StatusOK, map[string]any{"object": "list", "data": []any{map[string]any{"id": c.model, "object": "model", "owned_by": "dan"}}})
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if !c.authorized(r) {
			apiError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
		var input chatRequest
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		if decoder.Decode(&input) != nil || decoder.Decode(&struct{}{}) != io.EOF {
			apiError(w, http.StatusBadRequest, "invalid JSON request")
			return
		}
		if input.Model != c.model {
			apiError(w, http.StatusNotFound, "unknown model")
			return
		}
		if input.MaxTokens == 0 {
			input.MaxTokens = 256
		}
		if input.MaxTokens < 1 || input.MaxTokens > 4096 {
			apiError(w, http.StatusBadRequest, "max_tokens must be between 1 and 4096")
			return
		}
		prompt, err := qwenPrompt(input.Messages)
		if err != nil {
			apiError(w, http.StatusBadRequest, err.Error())
			return
		}
		if input.Stream {
			c.streamChat(w, r, prompt, input.MaxTokens)
			return
		}
		result, err := c.generate(r.Context(), prompt, input.MaxTokens, nil)
		if err != nil {
			apiError(w, http.StatusServiceUnavailable, err.Error())
			return
		}
		finishReason := "stop"
		if result.rows == uint32(input.MaxTokens) {
			finishReason = "length"
		}
		jsonReply(w, http.StatusOK, map[string]any{
			"id": fmt.Sprintf("chatcmpl-dan-%d", result.request), "object": "chat.completion",
			"created": time.Now().Unix(), "model": c.model,
			"choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": string(result.payload)}, "finish_reason": finishReason}},
		})
	})
	return mux
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	var coordinators []string
	flag.Func("coordinator", "DAN binary coordinator address; repeat for more replicas", func(value string) error {
		if strings.TrimSpace(value) == "" {
			return errors.New("coordinator address must not be empty")
		}
		coordinators = append(coordinators, value)
		return nil
	})
	model := flag.String("model", "", "served model ID")
	apiKey := flag.String("api-key", os.Getenv("DAN_API_KEY"), "Bearer token (or DAN_API_KEY)")
	timeout := flag.Duration("timeout", 5*time.Minute, "inference timeout")
	flag.Parse()
	if len(coordinators) == 0 {
		coordinators = []string{"127.0.0.1:50100"}
	}
	if *model == "" {
		log.Fatal("-model is required")
	}
	host, _, err := net.SplitHostPort(*listen)
	if err != nil {
		log.Fatal(err)
	}
	if ip := net.ParseIP(host); *apiKey == "" && host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		log.Fatal("-api-key or DAN_API_KEY is required for non-loopback listening")
	}
	c := &config{coordinators: coordinators, model: *model, apiKey: *apiKey, timeout: *timeout}
	server := &http.Server{Addr: *listen, Handler: c.handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: *timeout + 5*time.Second, IdleTimeout: 60 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	log.Printf("DAN API ready on %s", *listen)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
