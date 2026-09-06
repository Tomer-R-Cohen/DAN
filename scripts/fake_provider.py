#!/usr/bin/env python3
"""Small control-plane provider simulator; it never loads or transfers weights."""

import argparse
import pathlib
import socket
import struct
import threading
import time


def send(sock, lock, message):
    data = message.encode()
    with lock:
        sock.sendall(struct.pack('!I', len(data)) + data)


def receive(sock):
    def exact(size):
        data = b''
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise EOFError
            data += chunk
        return data
    return exact(struct.unpack('!I', exact(4))[0]).decode()


def fields(message):
    result = {}
    for line in message.splitlines()[1:]:
        key, separator, value = line.partition('=')
        if not separator or not key or key in result:
            raise ValueError('malformed control message')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--gpu', default='fake-gpu')
    parser.add_argument('--vram-mib', required=True, type=int)
    parser.add_argument('--cache-file', required=True, type=pathlib.Path)
    parser.add_argument('--heartbeat-interval', type=float, default=0.25)
    args = parser.parse_args()

    cached = args.cache_file.read_text().strip() if args.cache_file.exists() else ''
    sock = socket.create_connection(('127.0.0.1', args.port), timeout=5)
    sock.settimeout(None)
    lock = threading.Lock()
    stopped = threading.Event()
    send(sock, lock, 'HELLO')
    capabilities = (f'CAPABILITIES\nprovider_id={args.id}\ndevice_type=GPU\n'
                    f'gpu_name={args.gpu}\nvram={args.vram_mib} MiB\n'
                    f'vram_mib={args.vram_mib}\nmodel_name=fake\nbackend=FAKE\ncontrol_plane=1')
    if cached:
        capabilities += f'\ncached_shard={cached}'
    send(sock, lock, capabilities)

    def heartbeat():
        while not stopped.wait(args.heartbeat_interval):
            try:
                send(sock, lock, 'HEARTBEAT')
            except OSError:
                return

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    assignment = None
    try:
        while True:
            message = receive(sock)
            if message == 'BYE':
                break
            if message.startswith('ASSIGN_SHARD\n'):
                assignment = fields(message)
                identity = '|'.join(assignment[key]
                                    for key in ('model_id', 'version', 'shard_id', 'hash'))
                if cached == identity:
                    states = ('CACHED',)
                else:
                    states = ('ASSIGNED', 'DOWNLOADING', 'CACHED')
                    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    args.cache_file.write_text(identity + '\n')
                    cached = identity
                for state in states:
                    send(sock, lock, 'SHARD_STATE\n' + '\n'.join((
                        f'model_id={assignment["model_id"]}',
                        f'version={assignment["version"]}',
                        f'shard_id={assignment["shard_id"]}',
                        f'hash={assignment["hash"]}', f'state={state}')))
                continue
            if message.startswith(('LOAD_SHARD\n', 'UNLOAD_SHARD\n')):
                command = fields(message)
                if assignment is None or any(command.get(key) != assignment.get(key)
                        for key in ('model_id', 'version', 'shard_id', 'hash')):
                    raise ValueError('managed command does not match assignment')
                states = ('LOADING', 'READY') if message.startswith('LOAD_SHARD\n') else ('CACHED',)
                for state in states:
                    send(sock, lock, 'SHARD_STATE\n' + '\n'.join((
                        f'model_id={assignment["model_id"]}',
                        f'version={assignment["version"]}',
                        f'shard_id={assignment["shard_id"]}',
                        f'hash={assignment["hash"]}', f'state={state}')))
                continue
            if message.startswith('PROMPT\n'):
                _, request_id, _ = message.split('\n', 2)
                send(sock, lock, f'RESPONSE\n{request_id}\nfake response')
                continue
            raise ValueError('unexpected coordinator message')
    except (EOFError, OSError):
        pass
    finally:
        stopped.set()
        sock.close()


if __name__ == '__main__':
    main()
