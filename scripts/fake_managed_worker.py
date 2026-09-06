#!/usr/bin/env python3
"""Tiny TCP listener with llama.cpp rpc-server-compatible test arguments."""

import argparse
import socket


parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True, type=int)
parser.add_argument('--device', required=True)
args, _ = parser.parse_known_args()

with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen()
    while True:
        connection, _ = listener.accept()
        connection.close()
