"""CPU-only tests for persistent managed distributed serving."""

import hashlib
import os
import pathlib
import signal
import socket
import subprocess
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        return reservation.getsockname()[1]


def wait_for(path, text, timeout=10, count=1):
    deadline = time.monotonic() + timeout
    content = ''
    while time.monotonic() < deadline:
        content = path.read_text() if path.exists() else ''
        if content.count(text) >= count:
            return content
        time.sleep(0.05)
    raise AssertionError(f'timed out waiting for {text!r}\n{content}')


def child_pids(pid):
    return [int(value) for value in
            pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]


class PersistentServingTests(unittest.TestCase):
    def provider_command(self, provider_id, coordinator_port, worker_port, cache, vram):
        return [str(ROOT / 'build/managed_provider'), '--id', provider_id,
                '--gpu', f'GPU-{provider_id}', '--vram-mib', str(vram),
                '--cache-dir', str(cache), '--worker',
                str(ROOT / 'scripts/fake_managed_worker.py'), '--worker-port',
                str(worker_port), '--port', str(coordinator_port), '--worker-timeout', '2']

    def coordinator_command(self, port, manifest, model, runtime_port, *runtime_args):
        command = [str(ROOT / 'build/coordinator'), str(port), '--managed-model',
                   str(manifest), '--managed-runtime',
                   str(ROOT / 'scripts/fake_persistent_runtime.py'), str(model),
                   str(runtime_port), '--managed-runtime-timeout', '3',
                   '--managed-request-timeout', '3', '--heartbeat-timeout', '3']
        for argument in runtime_args:
            command += ['--managed-runtime-arg', str(argument)]
        return command

    def make_manifest(self, work, count=1):
        lines = ['model|dan-main|v1']
        minimums = (24000, 20000, 16000, 12000)
        for index in range(count):
            artifact = work / f'shard-{index}'
            artifact.write_bytes(f'persistent shard {index}\n'.encode())
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            minimum = minimums[index] if count == 4 else 1
            lines.append(f'shard|{index}|{artifact.stat().st_size}|{digest}|file://{artifact}|{minimum}')
        manifest = work / 'manifest'
        manifest.write_text('\n'.join(lines) + '\n')
        model = work / 'model.gguf'
        model.write_bytes(b'fake coordinator-side model')
        return manifest, model

    def test_four_providers_reuse_one_runtime_for_ten_requests_and_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            manifest, model = self.make_manifest(work, 4)
            port, runtime_port = free_port(), free_port()
            state = work / 'runtime.state'
            log = work / 'coordinator.log'
            providers = [None] * 4
            handles = []
            with log.open('w') as output:
                coordinator = subprocess.Popen(self.coordinator_command(
                    port, manifest, model, runtime_port, '--state-file', state),
                    stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, text=True)
                try:
                    wait_for(log, 'Listening for providers')
                    vrams = (24576, 20480, 16384, 12288)
                    for index in reversed(range(4)):
                        handle = (work / f'provider-{index}.log').open('w')
                        handles.append(handle)
                        providers[index] = subprocess.Popen(self.provider_command(
                            f'node-{chr(97 + index)}', port, free_port(),
                            work / f'cache-{index}', vrams[index]), stdout=handle,
                            stderr=subprocess.STDOUT)
                    wait_for(log, 'Managed runtime READY')
                    initial_runtime = int(state.read_text().splitlines()[0].split()[1])
                    initial_workers = [child_pids(provider.pid)[0] for provider in providers]
                    cache_times = {path: path.stat().st_mtime_ns for path in work.glob(
                        'cache-*/*/*/*/*.artifact')}

                    commands = ['/providers', '/runtime start'] + [
                        f'/model dan-main prompt-{number}' for number in range(1, 11)]
                    coordinator.stdin.write('\n'.join(commands) + '\n')
                    coordinator.stdin.flush()
                    result = wait_for(log, 'Response for request 10 from managed dan-main')
                    coordinator.stdin.write('/providers\n')
                    coordinator.stdin.flush()
                    result = wait_for(log, 'requests served: 10')
                    self.assertIn('replica: READY\nruntime: READY', result)
                    self.assertIn(f'runtime pid: {initial_runtime}', result)
                    for number in range(1, 11):
                        self.assertIn(f'Response for request {number} from managed dan-main:', result)
                        self.assertIn(f'response:prompt-{number}', result)
                    self.assertEqual(state.read_text().count('START '), 1)
                    self.assertEqual(state.read_text().count('REQUEST '), 10)
                    self.assertEqual([child_pids(provider.pid)[0] for provider in providers],
                                     initial_workers)
                    self.assertEqual({path: path.stat().st_mtime_ns for path in cache_times},
                                     cache_times)
                    for index in range(4):
                        self.assertEqual(result.count(f'node-{chr(97 + index)} shard {index} is DOWNLOADING'), 1)
                        self.assertEqual(result.count(f'node-{chr(97 + index)} shard {index} is LOADING'), 1)

                    os.kill(initial_runtime, signal.SIGKILL)
                    wait_for(log, 'Managed runtime ERROR: persistent runtime exited unexpectedly')
                    coordinator.stdin.write('/model dan-main unavailable\n/runtime start\n')
                    coordinator.stdin.flush()
                    wait_for(state, 'START ', count=2)
                    wait_for(log, 'Managed runtime READY', count=2)
                    coordinator.stdin.write('/model dan-main recovered\n')
                    coordinator.stdin.flush()
                    result = wait_for(log, 'Response for request 12 from managed dan-main')
                    self.assertIn('response:recovered', result)
                    self.assertNotEqual(int(state.read_text().splitlines()[-2].split()[1]),
                                        initial_runtime)
                    self.assertEqual([child_pids(provider.pid)[0] for provider in providers],
                                     initial_workers)
                    self.assertEqual(result.count(' is DOWNLOADING'), 4)

                    os.kill(providers[0].pid, signal.SIGSTOP)
                    wait_for(log, 'heartbeat timed out; marked OFFLINE')
                    wait_for(log, 'Managed runtime STOPPED: replica is not ready')
                    coordinator.stdin.write('/model dan-main provider-lost\nexit\n')
                    coordinator.stdin.flush()
                    self.assertEqual(coordinator.wait(timeout=10), 0)
                    final = log.read_text()
                    self.assertIn('rejected: dan-main replica is NOT_READY', final)
                finally:
                    if coordinator.poll() is None:
                        coordinator.kill(); coordinator.wait()
                    if coordinator.stdin and not coordinator.stdin.closed:
                        coordinator.stdin.close()
                    for provider in providers:
                        if provider and provider.poll() is None:
                            provider.kill(); provider.wait()
                    for handle in handles:
                        handle.close()

    def run_single(self, runtime_args=(), executable=None):
        context = tempfile.TemporaryDirectory()
        work = pathlib.Path(context.name)
        manifest, model = self.make_manifest(work)
        port, runtime_port = free_port(), free_port()
        state = work / 'runtime.state'
        log = work / 'coordinator.log'
        command = self.coordinator_command(port, manifest, model, runtime_port,
                                           '--state-file', state, *runtime_args)
        if executable:
            command[command.index(str(ROOT / 'scripts/fake_persistent_runtime.py'))] = executable
        output = log.open('w')
        coordinator = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                       stderr=subprocess.STDOUT, text=True)
        wait_for(log, 'Listening for providers')
        provider_output = (work / 'provider.log').open('w')
        provider = subprocess.Popen(self.provider_command('node', port, free_port(),
            work / 'cache', 1), stdout=provider_output, stderr=subprocess.STDOUT)
        return context, work, log, state, coordinator, provider, output, provider_output

    def cleanup_single(self, values):
        context, _, _, _, coordinator, provider, output, provider_output = values
        if coordinator.poll() is None:
            coordinator.kill(); coordinator.wait()
        if coordinator.stdin and not coordinator.stdin.closed:
            coordinator.stdin.close()
        if provider.poll() is None:
            provider.kill(); provider.wait()
        output.close(); provider_output.close(); context.cleanup()

    def test_requests_rejected_until_replica_and_runtime_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            manifest, model = self.make_manifest(work)
            log = work / 'log'
            with log.open('w') as output:
                coordinator = subprocess.Popen(self.coordinator_command(
                    free_port(), manifest, model, free_port()), stdin=subprocess.PIPE,
                    stdout=output, stderr=subprocess.STDOUT, text=True)
                try:
                    wait_for(log, 'Listening for providers')
                    coordinator.stdin.write('/model dan-main no-provider\nexit\n')
                    coordinator.stdin.flush()
                    coordinator.wait(timeout=5)
                    self.assertIn('replica is NOT_READY', log.read_text())
                finally:
                    if coordinator.poll() is None:
                        coordinator.kill(); coordinator.wait()
                    if coordinator.stdin and not coordinator.stdin.closed:
                        coordinator.stdin.close()

        values = self.run_single(('--startup-delay', '1'))
        try:
            _, _, log, state, coordinator, _, _, _ = values
            wait_for(log, 'Managed runtime STARTING')
            coordinator.stdin.write('/model dan-main still-starting\n')
            coordinator.stdin.flush()
            self.assertIn('runtime is STARTING', wait_for(log, 'runtime is STARTING'))
            wait_for(log, 'Managed runtime READY')
            coordinator.stdin.write('exit\n'); coordinator.stdin.flush()
            self.assertEqual(coordinator.wait(timeout=5), 0)
            self.assertIn('STOP ', state.read_text())
        finally:
            self.cleanup_single(values)

    def test_runtime_startup_and_response_failures(self):
        for runtime_args, executable, expected in (
                ((), '/does/not/exist', 'persistent runtime exited unexpectedly'),
                (('--exit-before-ready',), None, 'persistent runtime exited unexpectedly')):
            values = self.run_single(runtime_args, executable)
            try:
                _, _, log, _, coordinator, _, _, _ = values
                wait_for(log, f'Managed runtime ERROR: {expected}')
                coordinator.stdin.write('exit\n'); coordinator.stdin.flush()
                coordinator.wait(timeout=5)
            finally:
                self.cleanup_single(values)

        values = self.run_single(('--malformed-after', '1'))
        try:
            _, _, log, _, coordinator, _, _, _ = values
            wait_for(log, 'Managed runtime READY')
            coordinator.stdin.write('/model dan-main malformed\n')
            coordinator.stdin.flush()
            result = wait_for(log, 'Managed dan-main request failed')
            self.assertIn('runtime ERROR', result)
            coordinator.stdin.write('exit\n'); coordinator.stdin.flush()
            coordinator.wait(timeout=5)
        finally:
            self.cleanup_single(values)


if __name__ == '__main__':
    unittest.main()
