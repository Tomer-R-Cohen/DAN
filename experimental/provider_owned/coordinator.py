#!/usr/bin/env python3
"""Metadata-only coordinator for DAN provider-owned execution v0."""

import argparse
import json
import socket
import struct
import sys
import time

MAGIC = 0x44414E30
VERSION = 1
HEADER = struct.Struct("!IHHQIIIHHQ")
MAX_PAYLOAD = 64 * 1024 * 1024
PROMPT, TOKEN, ACTIVATION, RESULT = 1, 2, 3, 4
F32LE = 1


def encode_frame(kind, request, position=0, rows=0, cols=0, dtype=0, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload exceeds 64 MiB")
    return HEADER.pack(MAGIC, VERSION, kind, request, position, rows, cols, dtype, 0, len(payload)) + payload


def decode_header(data):
    if len(data) != HEADER.size:
        raise ValueError("wrong header size")
    magic, version, kind, request, position, rows, cols, dtype, reserved, size = HEADER.unpack(data)
    if magic != MAGIC or version != VERSION or reserved != 0:
        raise ValueError("bad frame header")
    if size > MAX_PAYLOAD:
        raise ValueError("payload exceeds 64 MiB")
    return kind, request, position, rows, cols, dtype, size


def recv_exact(sock, size):
    parts = []
    while size:
        part = sock.recv(size)
        if not part:
            raise ConnectionError("provider disconnected during frame")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def recv_frame(sock):
    fields = decode_header(recv_exact(sock, HEADER.size))
    return fields[:-1], recv_exact(sock, fields[-1])


def connect(endpoint):
    host, port = endpoint.rsplit(":", 1)
    return socket.create_connection((host, int(port)), timeout=30)


def load_manifest(path):
    with open(path, encoding="utf-8") as source:
        manifest = json.load(source)
    required = {"model_id", "architecture", "layers", "hidden_size", "split_layer", "quantization"}
    if (required - manifest.keys() or manifest["architecture"] != "qwen2"
            or not 0 < manifest["split_layer"] < manifest["layers"]
            or manifest["hidden_size"] < 1):
        raise ValueError("invalid provider-owned v0 manifest")
    return manifest


def exchange(sock, frame):
    start = time.perf_counter_ns()
    sock.sendall(frame)
    header, payload = recv_frame(sock)
    return header, payload, time.perf_counter_ns() - start


def require_activation(header, payload, request, position, hidden):
    kind, got_request, got_position, rows, cols, dtype = header
    if kind == 0:
        raise RuntimeError(payload.decode("utf-8", "replace"))
    if (kind, got_request, got_position, cols, dtype) != (ACTIVATION, request, position, hidden, F32LE):
        raise RuntimeError("invalid stage A activation metadata")
    if rows == 0 or len(payload) != 8 + rows * cols * 4:
        raise RuntimeError("invalid stage A activation size")
    return rows, struct.unpack("!Q", payload[:8])[0]


def require_result(header, payload, request):
    kind, got_request, position, rows, cols, dtype = header
    if kind == 0:
        raise RuntimeError(payload.decode("utf-8", "replace"))
    if kind != RESULT or got_request != request or rows or cols or dtype or len(payload) < 13:
        raise RuntimeError("invalid stage B result")
    token = struct.unpack("!I", payload[:4])[0]
    compute_ns = struct.unpack("!Q", payload[4:12])[0]
    return token, compute_ns, bool(payload[12]), payload[13:].decode("utf-8", "replace"), position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--provider-a", required=True)
    parser.add_argument("--provider-b", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    manifest = load_manifest(args.manifest)

    request = int(time.time_ns() & 0xFFFFFFFFFFFFFFFF) or 1
    metrics = {"model": manifest["model_id"], "split": manifest["split_layer"], "generated_tokens": 0,
        "prefill_activation_bytes": 0, "decode_activation_bytes": 0, "a_compute_ms": [],
        "b_compute_ms": [], "activation_route_ms": [], "total_token_ms": [], "token_ids": []}
    output = []
    with connect(args.provider_a) as a, connect(args.provider_b) as b:
        position = 0
        frame = encode_frame(PROMPT, request, payload=args.prompt.encode())
        ah, ap, a_ns = exchange(a, frame)
        rows, a_compute = require_activation(ah, ap, request, 0, manifest["hidden_size"])
        metrics["prompt_tokens"] = rows
        metrics["prefill_activation_bytes"] = rows * manifest["hidden_size"] * 4
        bh, bp, b_ns = exchange(b, encode_frame(*ah, payload=ap))
        token, b_compute, eog, text, position = require_result(bh, bp, request)
        metrics["prefill_ms"] = (a_ns + b_ns) / 1e6
        metrics["a_prefill_compute_ms"] = a_compute / 1e6
        metrics["b_prefill_compute_ms"] = b_compute / 1e6
        output.append(text)
        metrics["token_ids"].append(token)
        metrics["generated_tokens"] = 1

        while metrics["generated_tokens"] < args.tokens and not eog:
            turn = time.perf_counter_ns()
            ah, ap, a_ns = exchange(a, encode_frame(TOKEN, request, position, payload=struct.pack("!I", token)))
            rows, a_compute = require_activation(ah, ap, request, position, manifest["hidden_size"])
            relay = time.perf_counter_ns()
            bh, bp, b_ns = exchange(b, encode_frame(*ah, payload=ap))
            route_ns = time.perf_counter_ns() - relay
            token, b_compute, eog, text, position = require_result(bh, bp, request)
            output.append(text)
            metrics["token_ids"].append(token)
            metrics["generated_tokens"] += 1
            metrics["decode_activation_bytes"] += rows * manifest["hidden_size"] * 4
            metrics["a_compute_ms"].append(a_compute / 1e6)
            metrics["b_compute_ms"].append(b_compute / 1e6)
            metrics["activation_route_ms"].append(max(0, (route_ns - b_compute) / 1e6))
            metrics["total_token_ms"].append((time.perf_counter_ns() - turn) / 1e6)

    generated = "".join(output)
    count = metrics["generated_tokens"]
    metrics["decode_bytes_per_token"] = manifest["hidden_size"] * 4
    metrics["decode_tok_s"] = 1000 / (sum(metrics["total_token_ms"]) / len(metrics["total_token_ms"])) if len(metrics["total_token_ms"]) else 0
    for name in ("a_compute_ms", "b_compute_ms", "activation_route_ms", "total_token_ms"):
        metrics[name + "_mean"] = sum(metrics[name]) / len(metrics[name]) if metrics[name] else 0
    print(generated)
    print(json.dumps(metrics, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as target:
            json.dump({"output": generated, "metrics": metrics}, target, indent=2)
    if count < args.tokens and not eog:
        raise SystemExit("generation stopped before requested token count")


if __name__ == "__main__":
    try:
        main()
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        print(f"experimental runtime unavailable: {exc}", file=sys.stderr)
        raise SystemExit(1)
