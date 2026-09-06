#!/usr/bin/env python3
"""Small llama-server-compatible HTTP runtime used by CPU-only tests."""

import argparse
import http.server
import json
import os
import pathlib
import signal
import threading
import time


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True, type=int)
parser.add_argument('--state-file', type=pathlib.Path)
parser.add_argument('--startup-delay', type=float, default=0)
parser.add_argument('--exit-before-ready', action='store_true')
parser.add_argument('--exit-after', type=int, default=0)
parser.add_argument('--malformed-after', type=int, default=0)
args, _ = parser.parse_known_args()


def record(message):
    if args.state_file:
        with args.state_file.open('a') as output:
            output.write(message + '\n')


record(f'START {os.getpid()}')
if args.exit_before_ready:
    raise SystemExit(1)
time.sleep(args.startup_delay)
requests = 0


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, body):
        encoded = body.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path != '/health':
            self.send_error(404)
            return
        self.reply('{"status":"ok"}')

    def do_POST(self):
        global requests
        if self.path != '/completion':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            prompt = json.loads(self.rfile.read(length))['prompt']
        except (ValueError, KeyError, json.JSONDecodeError):
            self.send_error(400)
            return
        requests += 1
        record(f'REQUEST {requests} {prompt}')
        if args.malformed_after == requests:
            self.reply('{malformed')
        else:
            self.reply(json.dumps({'content': f'response:{prompt}'}))
        if args.exit_after == requests:
            threading.Thread(target=lambda: (time.sleep(0.05), os._exit(1)), daemon=True).start()


def stopped(*_):
    record(f'STOP {os.getpid()}')
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stopped)
with Server((args.host, args.port), Handler) as server:
    server.serve_forever()
