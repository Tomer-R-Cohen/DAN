"""Checks a running serve-mode dan-stage-worker's reservation rules (no model load).

usage: python lease_integration.py HOST:PORT MODEL_SHA256
The worker must be idle, reachable directly (no --peer-header), and allow at least one
session with the model's context.
"""

import os
import socket
import struct
import sys
import threading
import time

MAGIC, VERSION = 0x44414E31, 2
PROVIDER_AVAILABLE, ASSIGN_STAGE, ERROR, ACK, RESERVE, RELEASE_ROUTE = 16, 17, 0, 9, 25, 26
HEADER = struct.Struct(">IHHQQIIIHHQ")


def send(sock, frame_type, payload=b""):
    sock.sendall(HEADER.pack(MAGIC, VERSION, frame_type, 0, 0, 0, 0, 0, 0, 0, len(payload)) + payload)


def receive(sock):
    header = b""
    while len(header) < HEADER.size:
        chunk = sock.recv(HEADER.size - len(header))
        if not chunk:
            return None, b""
        header += chunk
    fields = HEADER.unpack(header)
    payload = b""
    while len(payload) < fields[10]:
        chunk = sock.recv(fields[10] - len(payload))
        if not chunk:
            return None, b""
        payload += chunk
    return fields[2], payload


def connect(endpoint):
    host, port = endpoint.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=10)
    frame_type, payload = receive(sock)
    assert frame_type == PROVIDER_AVAILABLE, frame_type
    hello = {}
    for line in payload.decode().split("\n"):
        key, value = line.split("=", 1)
        hello.setdefault(key, []).append(value)
    return sock, hello


def request(sha, lease_ms=5000, sessions=1, extra=""):
    route = os.urandom(16).hex()
    text = (f"route_id={route}\nmodel_sha256={sha}\nbegin=0\nend=2\ncontext=512\n"
            f"sessions={sessions}\nlease_ms={lease_ms}{extra}")
    return route, text.encode()


def state(endpoint):
    sock, hello = connect(endpoint)
    sock.close()
    return hello["state"][0]


def main():
    endpoint, sha = sys.argv[1], sys.argv[2].lower()
    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise SystemExit("FAIL " + message)
        checks += 1
        print("PASS " + message)

    sock, hello = connect(endpoint)
    check(hello["state"] == ["available"] and sha in hello.get("model", [])
          and hello["abi"][0].startswith("dan-stage-v1/"), "greeting reports state, model and ABI")

    _, payload = request(sha, extra="\nurl=https://attacker.example/model.gguf")
    send(sock, RESERVE, payload)
    frame_type, reply = receive(sock)
    check(frame_type == ERROR and reply == b"invalid_reservation", "a client cannot supply a download URL")
    _, payload = request("0" * 64)
    send(sock, RESERVE, payload)
    check(receive(sock) == (ERROR, b"unknown_model"), "a model outside the catalog is refused")
    _, payload = request(sha, sessions=10000)
    send(sock, RESERVE, payload)
    check(receive(sock) == (ERROR, b"limits_exceeded"), "requests above the worker's limits are refused")
    sock.close()

    sock, _ = connect(endpoint)
    _, payload = request(sha)
    send(sock, ASSIGN_STAGE, payload)
    frame_type, _ = receive(sock)
    check(frame_type == ERROR and receive(sock)[0] is None, "assign without a reservation is refused")
    sock.close()

    results = []
    sockets = [connect(endpoint)[0] for _ in range(4)]
    barrier = threading.Barrier(len(sockets))

    def reserve(client):
        _, body = request(sha)
        barrier.wait()
        send(client, RESERVE, body)
        results.append(receive(client))

    threads = [threading.Thread(target=reserve, args=(client,)) for client in sockets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    check(sorted(frame for frame, _ in results) == [ERROR, ERROR, ERROR, ACK]
          and all(reply == b"busy" for frame, reply in results if frame == ERROR),
          "concurrent reservations: exactly one wins, the rest are busy")
    check(state(endpoint) == "reserved", "a reserved worker greets as reserved")
    for client in sockets:
        client.close()
    time.sleep(0.5)
    check(state(endpoint) == "available", "closing the connection releases the lease")

    sock, _ = connect(endpoint)
    route, payload = request(sha)
    send(sock, RESERVE, payload)
    check(receive(sock)[0] == ACK, "reservation accepted")
    send(sock, RELEASE_ROUTE, os.urandom(16).hex().encode())
    check(receive(sock) == (ERROR, b"route_not_held"), "only the held route can be released")
    send(sock, RELEASE_ROUTE, route.encode())
    check(receive(sock)[0] == ACK and state(endpoint) == "available", "explicit release frees the worker")
    sock.close()

    sock, _ = connect(endpoint)
    _, payload = request(sha, lease_ms=1500)
    send(sock, RESERVE, payload)
    check(receive(sock)[0] == ACK, "short reservation accepted")
    time.sleep(2.5)
    sock.settimeout(5)
    check(receive(sock)[0] is None and state(endpoint) == "available",
          "an unassigned reservation expires and the worker drops the connection")
    sock.close()
    print(f"lease checks passed: {checks}")


if __name__ == "__main__":
    main()
