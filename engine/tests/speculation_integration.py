"""Compare upstream speculation with plain generation using cached model artifacts.

Uses DAN placement/ring, two successive requests, and no performance assertions.
Requires the existing adaptive-check cache (1.5B target and 0.5B draft).
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

from batching_integration import port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PATH=str(args.runtime.resolve() / 'bin') + os.pathsep + os.environ['PATH'])
    target = str(root / 'config/provider-owned-qwen2.5-1.5b-q4km.json')
    draft = str(root / 'config/provider-owned-qwen2.5-0.5b-q4km.json')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    reference = None
    for mode in ('plain', 'draft', 'ngram'):
        control, ring, returning = port(), port(), port()
        worker = subprocess.Popen([str(args.runtime.resolve() / 'dan-stage-worker.exe'),
            '--control-listen', f'127.0.0.1:{control}', '--ring-listen', f'127.0.0.1:{ring}',
            '--catalog', target, '--catalog', draft, '--cache-dir', str(args.cache.resolve()),
            '--provider-id', 'correctness-worker', '--gpu', 'CPU', '--vram-mib', '16000',
            '--max-sessions', '1', '--gpu-layers', '0', '--draft-width', '4', '--adaptive-draft', 'false',
            '--ngram-draft', str(mode == 'ngram').lower()],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
        ready, logs = queue.Queue(), []
        def drain():
            for line in worker.stderr:
                logs.append(line)
                if b'serving placement requests' in line: ready.put(True)
            ready.put(False)
        threading.Thread(target=drain, daemon=True).start()
        try:
            assert ready.get(timeout=60), b''.join(logs[-10:])
            with tempfile.TemporaryDirectory(prefix='dan-correctness-') as temporary:
                report = Path(temporary) / 'result.json'
                command = [str(args.runtime.resolve() / 'dan-client.exe'), '--manifest', target,
                    '--candidate', f'127.0.0.1:{control}', '--min-stages', '1',
                    '--ring-return', f'127.0.0.1:{returning}', '--require-direct',
                    '--requests', '2', '--tokens', '32', '--persistent', '--report', str(report),
                    '--prompt', 'Repeat this sequence exactly: red blue green red blue green red blue green red blue green',
                    '--prompt', 'The capital of France is']
                if mode == 'draft': command += ['--manifest', draft, '--speculate']
                client = subprocess.run(command, env=env, capture_output=True, timeout=180, creationflags=flags)
                assert client.returncode == 0, client.stderr.decode(errors='replace')
                output = json.loads(report.read_text())['outputs']
                if reference is None: reference = output
                else: assert output == reference, (mode, output, reference)
                print(f'PASS {mode}: persistent requests match plain generation')
        except Exception:
            print(b''.join(logs[-18:]).decode(errors='replace'))
            raise
        finally:
            worker.kill()
            worker.wait(timeout=30)


if __name__ == '__main__':
    main()
