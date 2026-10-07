"""Check a static worker's bounded outputs with a real model, without timing it.

Usage: python output_rows_integration.py HOST:PORT [--hidden N]
Use --hidden for a tail-only worker; omit it for a whole-model worker. Start the
worker with --ctx 128 --max-sessions 2. No prompt/response content is saved.
"""

import argparse
import socket
import struct


HEADER = struct.Struct(">IHHQQIIIHHQ")


def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("worker closed the connection")
        data.extend(chunk)
    return bytes(data)


def exchange(sock, kind, session=0, request=0, position=0, rows=0, cols=0,
             dtype=0, payload=b""):
    sock.sendall(HEADER.pack(0x44414E31, 2, kind, session, request, position,
                             rows, cols, dtype, 0, len(payload)) + payload)
    header = HEADER.unpack(read_exact(sock, HEADER.size))
    assert header[:2] == (0x44414E31, 2), header[:2]
    return header, read_exact(sock, header[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint")
    parser.add_argument("--hidden", type=int, default=0)
    args = parser.parse_args()
    host, port = args.endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=120) as sock:
        def control(kind, session=0, request=0):
            header, _ = exchange(sock, kind, session, request)
            assert header[2] == 9, header

        def prompt(session):
            if args.hidden:
                return exchange(sock, 6, session, 1, rows=1, cols=args.hidden,
                                dtype=1, payload=b"\0" * (8 + args.hidden * 4))
            # More than 32 prompt tokens must still yield only one sampled output.
            return exchange(sock, 4, session, 1, payload=b"Hello " * 48)

        control(1, 1)
        control(1, 2)
        header, payload = prompt(1)
        assert header[2] == 7 and header[6] == 0, header
        position = header[5]
        token = struct.unpack_from("<I", payload)[0]
        header, _ = prompt(2)
        assert header[2] == 7 and header[6] == 0, header
        control(8, 2, 1)
        control(2, 2)
        control(3, 2)

        def verify(rows):
            if args.hidden:
                return exchange(sock, 20, 1, 1, position, rows, args.hidden, 1,
                                b"\0" * (8 + rows * args.hidden * 4))
            return exchange(sock, 5, 1, 1, position, rows,
                            payload=struct.pack("<I", token) * rows)

        header, payload = verify(32)
        assert header[2] == 7 and header[6] == 32, (header, payload)
        assert len(payload) == 8 + 32 * 4, len(payload)
        position = header[5]
        header, payload = verify(33)
        assert header[2] == 0 and b"supported width" in payload, (header, payload)
        # An invalid frame closes an unbound control connection and clears its
        # sessions. Recovery follows that existing contract through a new connection.
        try:
            assert sock.recv(1) == b"", "invalid connection was not closed"
        except ConnectionResetError:
            pass
    with socket.create_connection((host, int(port)), timeout=120) as sock:
        control(1, 1)
        header, _ = prompt(1)
        assert header[2] == 7 and header[6] == 0, header
        control(8, 1, 1)
        control(2, 1)
        header, _ = prompt(1)
        assert header[2] == 7 and header[6] == 0, header
        control(8, 1, 1)
        control(3, 1)
        control(11)
    print("PASS prompt outputs, session isolation/reset, 32 outputs, oversized rejection/recovery")


if __name__ == "__main__":
    main()
