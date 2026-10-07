package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"sync"
	"time"
)

// One bounded, idle-expiring RAM cache for this local API principal. The worker
// verifies token-prefix equality and trims all stages before every request.
type chatProcess struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	output *json.Decoder
	scope  [32]byte
	model  string
	done   chan struct{}
	stop   sync.Once
	idle   *time.Timer
}

func (p *chatProcess) close() {
	p.stop.Do(func() {
		_ = p.stdin.Close()
		// Closing output makes the client's existing token sink cancel generation.
		// EOF on input then lets it destroy the remote session and receive its ack.
		_ = p.stdout.Close()
		select {
		case <-p.done:
		case <-time.After(2 * time.Second):
			_ = p.cmd.Process.Kill()
		}
	})
}

func chatScope(input localRequest, r *http.Request) [32]byte {
	firstUser := ""
	for _, message := range input.Messages {
		if message.Role == "user" {
			firstUser, _ = textContent(message.Content)
			break
		}
	}
	data, _ := json.Marshal([]string{r.Header.Get("X-OpenWebUI-User-Id"), input.User, input.PromptCacheKey, firstUser})
	return sha256.Sum256(data)
}

// Caller holds a.turn. There is never more than one active client process.
func (a *localAPI) chatClient(scope [32]byte) (*chatProcess, error) {
	if p := a.process; p != nil {
		expired := p.idle != nil && !p.idle.Stop()
		select {
		case <-p.done:
			expired = true
		default:
		}
		if p.scope == scope && !expired {
			return p, nil
		}
		p.close()
		<-p.done // return listener and sessions must be released before opening another
		a.process = nil
	}
	args := []string{"--api-chat", "--replica", "--discover", a.discover}
	if a.contextSize > 0 {
		args = append(args, "--context", fmt.Sprint(a.contextSize))
	}
	for _, path := range a.manifests {
		args = append(args, "--manifest", path)
	}
	cmd := exec.Command(a.client, args...)
	cmd.Stderr = io.Discard
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, err
	}
	p := &chatProcess{cmd: cmd, stdin: stdin, stdout: stdout, output: json.NewDecoder(stdout), scope: scope, model: "dan-auto", done: make(chan struct{})}
	go func() { _ = cmd.Wait(); close(p.done) }()
	a.process = p
	return p, nil
}

func (p *chatProcess) send(input localRequest) error {
	body, err := json.Marshal(input)
	if err != nil {
		return err
	}
	if len(body) > 1024*1024 {
		return fmt.Errorf("chat request too large")
	}
	if _, err = fmt.Fprintf(p.stdin, "%d %d\n", input.MaxTokens, len(body)); err != nil {
		return err
	}
	_, err = p.stdin.Write(body)
	return err
}
