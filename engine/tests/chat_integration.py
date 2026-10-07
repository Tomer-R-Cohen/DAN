"""Real-model chat correctness checks; no timings or prompt files.

python engine/tests/chat_integration.py --runtime build-runtime --model FILE
Runs one worker and then two split stages, retaining a client for repeated turns.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import queue


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--runtime', type=Path, required=True)
    args.add_argument('--model', type=Path, required=True)
    args.add_argument('--gpu-layers', type=int, default=0)
    args = args.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PATH=str(args.runtime.resolve() / 'bin') + os.pathsep + os.environ['PATH'])
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    for split in (False, True):
        workers, client = [], None
        try:
            endpoints = []
            for begin, end in ([(0, 12), (12, 24)] if split else [(0, 24)]):
                with socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    port = listener.getsockname()[1]
                worker = subprocess.Popen([str(args.runtime.resolve() / 'dan-stage-worker.exe'),
                    '--model', str(args.model.resolve()), '--stage-start', str(begin), '--stage-end', str(end),
                    '--ctx', '512', '--max-sessions', '2', '--gpu-layers', str(args.gpu_layers), '--port', str(port)],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
                workers.append(worker)
                ready = queue.Queue()
                def drain(process=worker, signal=ready):
                    for line in process.stderr:
                        if b'listening on' in line: signal.put(True)
                    signal.put(False)
                threading.Thread(target=drain, daemon=True).start()
                assert ready.get(timeout=60), 'worker exited during startup'
                endpoints += ['--provider', f'127.0.0.1:{port}']
            client = subprocess.Popen([str(args.runtime.resolve() / 'dan-client.exe'), '--api-chat',
                '--manifest', str(root / 'config/provider-owned-qwen2.5-0.5b-q4km.json'), *endpoints],
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=flags)
            events = queue.Queue()
            def read_events():
                for line in client.stdout: events.put(json.loads(line))
                events.put(None)
            threading.Thread(target=read_events, daemon=True).start()
            def request(body):
                data = json.dumps(body).encode()
                client.stdin.write(f'32 {len(data)}\n'.encode() + data)
                client.stdin.flush()
                content = bytearray()
                while True:
                    event = events.get(timeout=120)
                    assert event is not None and 'error' not in event, event
                    if 'bytes' in event: content.extend(bytes.fromhex(event['bytes']))
                    if event.get('done'): return content.decode(), event
            body = {'messages': [{'role': 'user', 'content': 'Return the required JSON object.'}],
                'temperature': 0.7, 'seed': 123, 'top_p': 0.9,
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'check', 'strict': True,
                    'schema': {'type': 'object', 'properties': {'answer': {'const': 4}},
                        'required': ['answer'], 'additionalProperties': False}}}}
            first, cold = request(body)
            second, warm = request(body)
            assert first.lstrip().startswith('{') and second.lstrip().startswith('{'), (first, second, cold, warm)
            assert json.loads(first) == json.loads(second) == {'answer': 4}, (first, second)
            assert cold['cached_tokens'] == 0 and warm['cached_tokens'] > 0, (cold, warm)
            assert warm['prompt_tokens'] == cold['prompt_tokens']
            body['messages'][0]['content'] = 'A different prompt; follow the schema.'
            third, changed = request(body)
            assert json.loads(third) == {'answer': 4}, third
            assert changed['cached_tokens'] < changed['prompt_tokens']
            # A real template/tool grammar path, with required rather than guessed tool use.
            tool = {'type': 'function', 'function': {'name': 'answer', 'description': 'Report the answer',
                'parameters': {'type': 'object', 'properties': {'value': {'const': 4}},
                    'required': ['value'], 'additionalProperties': False}}}
            result, _ = request({'messages': [{'role': 'user', 'content': 'Call answer with value 4.'}],
                'tools': [tool], 'tool_choice': 'required', 'temperature': 0})
            assert '<tool_call>' in result and 'answer' in result, result
            call = json.loads(result.split('<tool_call>', 1)[1].split('</tool_call>', 1)[0])
            assert call['name'] == 'answer' and call['arguments'] == {'value': 4}, result
            plain = {'messages': [{'role': 'user', 'content': 'Reply with one word: blue.'}], 'temperature': 0}
            text1, _ = request(plain)
            text2, reused = request(plain)
            assert text1 == text2 and text1.strip() and reused['cached_tokens'] > 0, (text1, text2)
            client.stdin.close()
            assert client.wait(timeout=30) == 0
            print('PASS split chat' if split else 'PASS whole-model chat')
        finally:
            for process in ([client] if client else []) + workers:
                if process.poll() is None: process.kill()
                process.wait(timeout=30)


if __name__ == '__main__':
    main()
