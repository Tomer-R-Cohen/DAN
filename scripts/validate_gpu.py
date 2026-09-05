#!/usr/bin/env python3
"""Run ten strictly sequential requests through real DAN processes; retain evidence."""
import argparse
import os
import signal
import json
import pathlib
import queue
import re
import subprocess
import threading
import time

PROMPTS = [
    'Explain TCP reliability in two sentences.',
    'What is 17 times 23? Show a brief calculation.',
    'Write a Python function that returns the largest of three numbers.',
    'Explain the difference between RAM and GPU VRAM.',
    'Summarize: A worker accepts a job, computes a result, and returns it to a coordinator.',
    'Give three edge cases for testing a request queue.',
    'Translate to French: The server is ready to receive requests.',
    'Explain why a mutex can prevent a data race.',
    'A train travels 120 km in 90 minutes. What is its average speed in km/h?',
    'List two advantages and two limitations of running a language model locally.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default='build')
    parser.add_argument('--registry', required=True)
    parser.add_argument('--model-id', default='dan-main')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', default='19099')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--cpu-smoke', action='store_true', help='Local integration check only; not GPU acceptance')
    args = parser.parse_args()
    rows = [line.split('|') for line in pathlib.Path(args.registry).read_text().splitlines()
            if line and not line.startswith('#')]
    row = next(r for r in rows if r[0] == args.model_id)
    if len(row) != 11 or row[9] != 'yes':
        raise ValueError('Select a valid single-provider registry entry')
    model = pathlib.Path(row[3]).resolve(strict=True)
    shard = re.fullmatch(r'(.+)-00001-of-(\d{5})\.gguf', model.name)
    weights = [model]
    if shard:
        weights = [model.with_name(f'{shard[1]}-{i:05d}-of-{shard[2]}.gguf')
                   for i in range(1, int(shard[2]) + 1)]
        for weight in weights:
            weight.resolve(strict=True)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', mode='cpu-smoke' if args.cpu_smoke else 'cuda-validation',
                  model_id=row[0], family=row[2], quantization=row[5], model_bytes=sum(p.stat().st_size for p in weights),
                  weight_files=[str(p) for p in weights],
                  context_length=int(row[7]), registry=row, requests=[])
    gpu = 'CPU smoke test'
    vram = 'not available'
    events = queue.Queue()
    children = []
    logs = []
    readers = []
    monitor = None

    def launch(name, command):
        log = (output / (name + '.log')).open('w')
        logs.append(log)
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        children.append(proc)
        def consume():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                events.put((name, line))
            events.put((name, None))
        reader = threading.Thread(target=consume, daemon=True)
        readers.append(reader)
        reader.start()
        report[name + '_command'] = command
        return proc

    def wait_for(name, pattern):
        deadline = time.monotonic() + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(pattern)
            try:
                source, line = events.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError('Timed out waiting for ' + pattern) from None
            if line is None:
                raise RuntimeError(source + ' exited unexpectedly')
            if source == name and re.search(pattern, line):
                return line

    try:
        if not args.cpu_smoke:
            report['gpu_before'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv'], text=True)
            gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
            vram = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.total', '--format=csv,noheader'], text=True).strip()
            memory_log = (output / 'vram.csv').open('w')
            logs.append(memory_log)
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,memory.total,memory.free,memory.used,utilization.gpu',
                                        '--format=csv', '-l', '1'], stdout=memory_log)
        coordinator = launch('coordinator', [str(pathlib.Path(args.build).resolve() / 'coordinator'),
                                            args.port, '--models', args.registry])
        wait_for('coordinator', 'Listening for providers')
        options = ['--gpu-layers', '0'] if args.cpu_smoke else ['--gpu-layers', '999', '--runtime-device', 'CUDA0']
        provider = launch('provider', [str(pathlib.Path(args.build).resolve() / 'provider'), args.runtime,
                                      str(model), '127.0.0.1', args.port, '--id', 'gpu-validation',
                                      '--model-name', row[4], '--device', 'CPU' if args.cpu_smoke else 'GPU',
                                      '--backend', 'CPU' if args.cpu_smoke else 'CUDA',
                                      '--gpu', gpu.strip().replace('\n', '; '), '--vram', vram,
                                      '--ctx-size', row[7], '--n-predict', '-1'] + options)
        wait_for('coordinator', r'Provider 1 registered')
        report['model_load'] = 'runtime ready and registered; verify offload in provider.log'
        for number, prompt in enumerate(PROMPTS, 1):
            started = time.monotonic()
            coordinator.stdin.write('/model ' + args.model_id + ' ' + prompt + '\n')
            coordinator.stdin.flush()
            wait_for('coordinator', r'Response for request ' + str(number) + r' from provider')
            timing = wait_for('coordinator', r'Performance: provider')
            report['requests'].append(dict(id=number, prompt=prompt, status='response-received',
                                           wall_seconds=time.monotonic() - started, timing=timing.strip()))
            print('Completed request', number, flush=True)
        coordinator.stdin.write('exit\n')
        coordinator.stdin.flush()
        codes = [p.wait(timeout=args.timeout) for p in children]
        report['exit_codes'] = codes
        if any(codes):
            raise RuntimeError('Nonzero process exit')
        report['status'] = 'ten-responses-received; inspect GPU evidence and answer quality'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        if monitor:
            monitor.terminate()
            monitor.wait()
        for reader in readers:
            reader.join(timeout=5)
        for log in logs:
            log.close()
        provider_log = output / 'provider.log'
        if provider_log.exists():
            match = re.search(r'runtime ready in ([\d.]+) ms', provider_log.read_text())
            report['runtime_ready_ms'] = float(match[1]) if match else None
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
