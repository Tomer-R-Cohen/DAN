"""Compare opt-in batched ring decode with serial decode using a real Qwen2.5 0.5B.

No timing benchmarks. Exercises first, middle and last stage sequence isolation.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading

from output_rows_integration import HEADER, read_exact, exchange


def frame(kind, session=0, request=0, position=0, rows=0, cols=0, dtype=0, payload=b''):
    return HEADER.pack(0x44414E31, 2, kind, session, request, position,
                       rows, cols, dtype, 0, len(payload)) + payload


def receive(sock):
    header = HEADER.unpack(read_exact(sock, HEADER.size))
    payload = read_exact(sock, header[-1])
    assert header[2] != 0, payload
    return header, payload


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, PATH=str(args.runtime.resolve() / 'bin') + os.pathsep + os.environ['PATH'])
    for begin, end in [(0, 8), (8, 16), (16, 24)]:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            listener.settimeout(60)
            control_port, ring_port = port(), port()
            worker = subprocess.Popen([str(args.runtime.resolve() / 'dan-stage-worker.exe'),
                '--model', str(args.model.resolve()), '--stage-start', str(begin), '--stage-end', str(end),
                '--ctx', '128', '--max-sessions', '2', '--gpu-layers', '0', '--port', str(control_port),
                '--ring-listen', f'127.0.0.1:{ring_port}', '--next', f'127.0.0.1:{listener.getsockname()[1]}',
                '--continuous-batching', 'true'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            logs = []
            def drain():
                for line in worker.stderr: logs.append(line)
            threading.Thread(target=drain, daemon=True).start()
            try:
                with listener.accept()[0] as sink:
                    sink.settimeout(60)
                    assert receive(sink)[0][2] == 9
                    sink.sendall(frame(9))
                    assert receive(sink)[0][2] == 9
                    with socket.create_connection(('127.0.0.1', control_port), timeout=60) as control:
                        for session in (1, 2): assert exchange(control, 1, session)[0][2] == 9
                        with socket.create_connection(('127.0.0.1', ring_port), timeout=60) as ring:
                            ring.sendall(frame(9))
                            assert receive(ring)[0][2] == 9
                            ring.sendall(frame(9))
                            outputs = []
                            for batched in (False, True):
                                positions = []
                                for session in (1, 2):
                                    if batched: assert exchange(control, 2, session)[0][2] == 9
                                    if begin == 0:
                                        seed = frame(4, session, 1, payload=b'Hello ' * session)
                                    else:
                                        seed = frame(6, session, 1, rows=1, cols=896, dtype=1,
                                                     payload=b'\0' * (8 + 896 * 4))
                                    control.sendall(seed)
                                    header, _ = receive(sink)
                                    positions.append(header[5] if end == 24 else header[5] + header[6])
                                frames = []
                                for session, position in enumerate(positions, 1):
                                    if begin == 0:
                                        frames.append(frame(5, session, 1, position, payload=struct.pack('>I', 100 + session)))
                                    else:
                                        frames.append(frame(6, session, 1, position, 1, 896, 1,
                                            b'\0' * 8 + struct.pack('<f', session * 0.01) * 896))
                                result = []
                                if batched:
                                    ring.sendall(b''.join(frames))
                                    result = [receive(sink), receive(sink)]
                                else:
                                    for data in frames:
                                        control.sendall(data)
                                        result.append(receive(sink))
                                outputs.append(result)
                                for session in (1, 2): assert exchange(control, 8, session, 1)[0][2] == 9
                            for (serial_h, serial), (batch_h, batch) in zip(*outputs):
                                assert serial_h == batch_h, (serial_h, batch_h)
                                if end == 24:
                                    assert serial[:4] == batch[:4] and serial[12:] == batch[12:]
                                else:
                                    left, right = struct.unpack('<896f', serial[8:]), struct.unpack('<896f', batch[8:])
                                    assert max(abs(a-b) for a, b in zip(left, right)) < 0.001
                            _, data = exchange(control, 10)
                            assert json.loads(data)['decode_batches'] > 0, data
                print(f'PASS batched stage {begin}:{end}: same outputs, isolated sequences')
            except Exception:
                print(b''.join(logs[-15:]).decode(errors='replace'))
                raise
            finally:
                worker.kill()
                worker.wait(timeout=30)


if __name__ == '__main__':
    main()
