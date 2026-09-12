package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestChatCompletionUsesDANWire(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	seen := make(chan frame, 1)
	go func() {
		conn, acceptErr := listener.Accept()
		if acceptErr != nil {
			return
		}
		defer conn.Close()
		input, readErr := readFrame(conn)
		if readErr != nil {
			return
		}
		seen <- input
		_ = writeFrame(conn, frame{typeID: typeResult, request: input.request, rows: 1, payload: []byte(" Paris")})
	}()

	c := &config{coordinator: listener.Addr().String(), model: "qwen", apiKey: "secret", timeout: time.Second}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	body, _ := json.Marshal(chatRequest{Model: "qwen", Messages: []message{{Role: "user", Content: "Capital of France?"}}, MaxTokens: 8})
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer secret")
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", response.StatusCode)
	}
	var output struct {
		Choices []struct {
			Message message `json:"message"`
		} `json:"choices"`
	}
	if json.NewDecoder(response.Body).Decode(&output) != nil || len(output.Choices) != 1 || output.Choices[0].Message.Content != " Paris" {
		t.Fatalf("unexpected response: %#v", output)
	}
	input := <-seen
	if input.typeID != typePrompt || input.rows != 8 || !bytes.Contains(input.payload, []byte("Capital of France?")) {
		t.Fatalf("unexpected DAN frame: %#v", input)
	}
}

func TestGenerateRoundRobinsCoordinators(t *testing.T) {
	var listeners []net.Listener
	for _, answer := range []string{" replica-a", " replica-b"} {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		listeners = append(listeners, listener)
		defer listener.Close()
		go func() {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			defer conn.Close()
			input, err := readFrame(conn)
			if err == nil {
				_ = writeFrame(conn, frame{typeID: typeResult, request: input.request, rows: 1, payload: []byte(answer)})
			}
		}()
	}
	c := &config{coordinators: []string{listeners[0].Addr().String(), listeners[1].Addr().String()}, timeout: time.Second}
	for _, want := range []string{" replica-a", " replica-b"} {
		result, err := c.generate(context.Background(), "prompt", 1, nil)
		if err != nil || string(result.payload) != want {
			t.Fatalf("result = %q, err = %v, want %q", result.payload, err, want)
		}
	}
}

func TestGenerateSkipsUnreachableCoordinator(t *testing.T) {
	dead, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	deadAddress := dead.Addr().String()
	dead.Close()
	live, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer live.Close()
	go func() {
		conn, err := live.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		input, err := readFrame(conn)
		if err == nil {
			_ = writeFrame(conn, frame{typeID: typeResult, request: input.request, rows: 1, payload: []byte(" fallback")})
		}
	}()
	c := &config{coordinators: []string{deadAddress, live.Addr().String()}, timeout: time.Second}
	result, err := c.generate(context.Background(), "prompt", 1, nil)
	if err != nil || string(result.payload) != " fallback" {
		t.Fatalf("result = %q, err = %v", result.payload, err)
	}
}

func TestGenerateSkipsUnavailableCoordinator(t *testing.T) {
	var listeners []net.Listener
	for _, response := range []frame{
		{typeID: typeError, payload: []byte("replica_unavailable")},
		{typeID: typeResult, rows: 1, payload: []byte(" fallback")},
	} {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		listeners = append(listeners, listener)
		defer listener.Close()
		go func() {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			defer conn.Close()
			input, err := readFrame(conn)
			if err == nil {
				response.request = input.request
				_ = writeFrame(conn, response)
			}
		}()
	}
	c := &config{coordinators: []string{listeners[0].Addr().String(), listeners[1].Addr().String()}, timeout: time.Second}
	result, err := c.generate(context.Background(), "prompt", 1, nil)
	if err != nil || string(result.payload) != " fallback" {
		t.Fatalf("result = %q, err = %v", result.payload, err)
	}
}

func TestGenerateDoesNotFailOverAfterStreaming(t *testing.T) {
	first, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	go func() {
		conn, err := first.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		input, err := readFrame(conn)
		if err == nil {
			_ = writeFrame(conn, frame{typeID: typeChunk, request: input.request, rows: 1, payload: []byte("visible")})
		}
	}()
	c := &config{coordinators: []string{first.Addr().String(), second.Addr().String()}, timeout: time.Second}
	_, err = c.generate(context.Background(), "prompt", 1, func(frame) error { return nil })
	if err == nil {
		t.Fatal("stream ending without a final result succeeded")
	}
	tcp := second.(*net.TCPListener)
	_ = tcp.SetDeadline(time.Now().Add(50 * time.Millisecond))
	if conn, err := second.Accept(); err == nil {
		conn.Close()
		t.Fatal("request failed over after streaming content")
	}
}

func TestHealthReflectsReplicaReadiness(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	backendErrors := make(chan error, 2)
	go func() {
		for _, available := range []bool{true, false} {
			conn, err := listener.Accept()
			if err != nil {
				backendErrors <- err
				return
			}
			input, err := readFrame(conn)
			if err == nil && input.typeID != typeMetrics {
				err = errors.New("health did not request DAN metrics")
			}
			if err == nil {
				payload, _ := json.Marshal(map[string]bool{"replica_available": available})
				err = writeFrame(conn, frame{typeID: typeMetrics, payload: payload})
			}
			conn.Close()
			backendErrors <- err
		}
	}()
	c := &config{coordinator: listener.Addr().String(), model: "qwen"}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	for _, want := range []int{http.StatusOK, http.StatusServiceUnavailable} {
		response, err := http.Get(server.URL + "/health")
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != want {
			t.Fatalf("health status = %d, want %d", response.StatusCode, want)
		}
		if err := <-backendErrors; err != nil {
			t.Fatal(err)
		}
	}
}

func TestMetricsAreAuthenticatedAndScrapeable(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		input, err := readFrame(conn)
		if err == nil && input.typeID == typeMetrics {
			_ = writeFrame(conn, frame{typeID: typeMetrics, payload: []byte(
				`{"replica_available":true,"queue_depth":3,"queue_capacity":8,"requests_completed":7,"resident_sessions":2,"kv_memory_bytes":4096,"generated_tokens":42,"speculative_enabled":true,"pipeline_depth":4,"speculative_rounds":5,"draft_tokens_proposed":20,"draft_tokens_accepted":16}`)})
		}
	}()
	c := &config{coordinator: listener.Addr().String(), model: "qwen", apiKey: "secret"}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	response, err := http.Get(server.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthorized metrics status = %d", response.StatusCode)
	}
	req, _ := http.NewRequest(http.MethodGet, server.URL+"/metrics", nil)
	req.Header.Set("Authorization", "Bearer secret")
	response, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil ||
		!bytes.Contains(body, []byte("dan_replica_available{replica=\"0\"} 1")) ||
		!bytes.Contains(body, []byte("dan_queue_capacity{replica=\"0\"} 8")) ||
		!bytes.Contains(body, []byte("dan_requests_completed_total{replica=\"0\"} 7")) ||
		!bytes.Contains(body, []byte("dan_resident_sessions{replica=\"0\"} 2")) ||
		!bytes.Contains(body, []byte("dan_kv_memory_bytes{replica=\"0\"} 4096")) ||
		!bytes.Contains(body, []byte("dan_generated_tokens_total{replica=\"0\"} 42")) ||
		!bytes.Contains(body, []byte("dan_speculative_enabled{replica=\"0\"} 1")) ||
		!bytes.Contains(body, []byte("dan_draft_tokens_accepted_total{replica=\"0\"} 16")) {
		t.Fatalf("metrics = %q, err = %v", body, err)
	}
}

func TestChatCompletionStreamsBeforeFinalResult(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	allowFinal := make(chan struct{})
	go func() {
		conn, acceptErr := listener.Accept()
		if acceptErr != nil {
			return
		}
		defer conn.Close()
		input, readErr := readFrame(conn)
		if readErr != nil || input.typeID != typeStream {
			return
		}
		_ = writeFrame(conn, frame{typeID: typeChunk, request: input.request, rows: 1, payload: []byte(" Par")})
		<-allowFinal
		_ = writeFrame(conn, frame{typeID: typeChunk, request: input.request, rows: 1, payload: []byte("is")})
		_ = writeFrame(conn, frame{typeID: typeResult, request: input.request, rows: 2})
	}()
	c := &config{coordinator: listener.Addr().String(), model: "qwen", timeout: time.Second}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	body, _ := json.Marshal(chatRequest{Model: "qwen", Messages: []message{{Role: "user", Content: "Capital?"}}, MaxTokens: 8, Stream: true})
	response, err := http.Post(server.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	reader := bufio.NewReader(response.Body)
	first, err := reader.ReadString('\n')
	if err != nil || !bytes.Contains([]byte(first), []byte(" Par")) {
		t.Fatalf("first event = %q, err = %v", first, err)
	}
	close(allowFinal)
	rest, err := io.ReadAll(reader)
	if err != nil || !bytes.Contains(rest, []byte("is")) || !bytes.Contains(rest, []byte("[DONE]")) {
		t.Fatalf("remaining events = %q, err = %v", rest, err)
	}
}

func TestStreamingDisconnectClosesDANRequest(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	backendClosed := make(chan error, 1)
	go func() {
		conn, acceptErr := listener.Accept()
		if acceptErr != nil {
			backendClosed <- acceptErr
			return
		}
		defer conn.Close()
		input, readErr := readFrame(conn)
		if readErr != nil {
			backendClosed <- readErr
			return
		}
		if err := writeFrame(conn, frame{typeID: typeChunk, request: input.request, rows: 1, payload: []byte("first")}); err != nil {
			backendClosed <- err
			return
		}
		_ = conn.SetReadDeadline(time.Now().Add(time.Second))
		var one [1]byte
		_, readErr = conn.Read(one[:])
		if readErr == nil {
			readErr = errors.New("DAN connection remained open")
		} else if netErr, ok := readErr.(net.Error); ok && netErr.Timeout() {
			readErr = errors.New("DAN connection did not close after HTTP cancellation")
		} else {
			readErr = nil
		}
		backendClosed <- readErr
	}()

	c := &config{coordinator: listener.Addr().String(), model: "qwen", timeout: time.Second}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	body, _ := json.Marshal(chatRequest{Model: "qwen", Messages: []message{{Role: "user", Content: "Long answer"}}, MaxTokens: 4096, Stream: true})
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader(body))
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	reader := bufio.NewReader(response.Body)
	if first, err := reader.ReadString('\n'); err != nil || !strings.Contains(first, "first") {
		t.Fatalf("first event = %q, err = %v", first, err)
	}
	cancel()
	response.Body.Close()
	if err := <-backendClosed; err != nil {
		t.Fatal(err)
	}
}

func TestAuthFailsClosed(t *testing.T) {
	c := &config{model: "qwen", apiKey: "secret", timeout: time.Second}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	response, _ := http.Post(server.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(`{"model":"qwen"}`))
	if response.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d", response.StatusCode)
	}
	response.Body.Close()
}

func TestChatCompletionRejectsTrailingJSON(t *testing.T) {
	c := &config{model: "qwen", timeout: time.Second}
	server := httptest.NewServer(c.handler())
	defer server.Close()
	response, err := http.Post(server.URL+"/v1/chat/completions", "application/json",
		bytes.NewBufferString(`{"model":"qwen"}{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusBadRequest {
		t.Fatalf("trailing JSON status = %d", response.StatusCode)
	}
}
