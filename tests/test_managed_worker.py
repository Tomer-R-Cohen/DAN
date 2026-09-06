"""CPU-only regression coverage for the real managed-worker lifecycle."""

import hashlib
import os
import pathlib
import signal
import socket
import struct
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
    while time.monotonic() < deadline:
        content = path.read_text() if path.exists() else ''
        if content.count(text) >= count:
            return content
        time.sleep(0.05)
    raise AssertionError(f'timed out waiting for {text!r}\n{content}')


def children(pid):
    value = pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    return [int(child) for child in value]


class ManagedWorkerTests(unittest.TestCase):
    def provider_command(self, provider_id, coordinator_port, worker_port, cache, worker=None,
                         vram=1):
        return [str(ROOT / 'build/managed_provider'), '--id', provider_id,
                '--gpu', f'GPU-{provider_id}', '--vram-mib', str(vram),
                '--cache-dir', str(cache), '--worker',
                str(worker or ROOT / 'scripts/fake_managed_worker.py'),
                '--worker-port', str(worker_port), '--port', str(coordinator_port),
                '--worker-timeout', '2']

    def test_four_provider_download_worker_failure_and_cache_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            coordinator_port = free_port()
            worker_ports = [free_port() for _ in range(4)]
            manifest = work / 'manifest'
            lines = ['model|dan-main|v1']
            for index, minimum in enumerate((24000, 20000, 16000, 12000)):
                artifact = work / f'shard-{index}'
                artifact.write_bytes(f'managed shard {index}\n'.encode())
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                lines.append(f'shard|{index}|{artifact.stat().st_size}|{digest}|file://{artifact}|{minimum}')
            manifest.write_text('\n'.join(lines) + '\n')
            coordinator_log = work / 'coordinator.log'
            providers = [None] * 4
            logs = []
            with coordinator_log.open('w') as output:
                coordinator = subprocess.Popen([
                    str(ROOT / 'build/coordinator'), str(coordinator_port),
                    '--managed-model', str(manifest), '--heartbeat-timeout', '1'],
                    stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, text=True)
                try:
                    wait_for(coordinator_log, 'Listening for providers')
                    for index in reversed(range(4)):
                        log = work / f'node-{index}.log'
                        logs.append(log)
                        handle = log.open('w')
                        providers[index] = (subprocess.Popen(self.provider_command(
                            f'node-{chr(97 + index)}', coordinator_port, worker_ports[index],
                            work / f'cache-{index}', vram=(24576, 20480, 16384, 12288)[index]),
                            stdout=handle, stderr=subprocess.STDOUT), handle)
                    wait_for(coordinator_log, ' is READY', count=4)
                    coordinator.stdin.write('/providers\n')
                    coordinator.stdin.flush()
                    wait_for(coordinator_log, 'state: READY')

                    node_a = providers[0][0]
                    first_worker = children(node_a.pid)[0]
                    os.kill(first_worker, signal.SIGKILL)
                    wait_for(coordinator_log, 'node-a shard 0 is ERROR')
                    coordinator.stdin.write('/providers\n/load node-a\n')
                    coordinator.stdin.flush()
                    wait_for(coordinator_log, 'node-a shard 0 is READY', count=2)
                    recovered_worker = children(node_a.pid)[0]
                    coordinator.stdin.write('/load node-a\n')
                    coordinator.stdin.flush()
                    time.sleep(0.4)
                    self.assertEqual(children(node_a.pid), [recovered_worker])

                    coordinator.stdin.write('/unload node-a\n')
                    coordinator.stdin.flush()
                    wait_for(coordinator_log, 'node-a shard 0 is CACHED', count=3)
                    deadline = time.monotonic() + 3
                    while children(node_a.pid) and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertEqual(children(node_a.pid), [])
                    coordinator.stdin.write('/load node-a\n')
                    coordinator.stdin.flush()
                    wait_for(coordinator_log, 'node-a shard 0 is READY', count=4)

                    node_a.terminate()
                    node_a.wait(timeout=5)
                    providers[0][1].close()
                    wait_for(coordinator_log, ' disconnected')
                    time.sleep(0.2)
                    restart_log = work / 'node-a-restart.log'
                    restart_handle = restart_log.open('w')
                    providers[0] = (subprocess.Popen(self.provider_command(
                        'node-a', coordinator_port, worker_ports[0], work / 'cache-0', vram=24576),
                        stdout=restart_handle, stderr=subprocess.STDOUT), restart_handle)
                    # Initial, recovery, duplicate, unload/reload, and provider restart.
                    wait_for(coordinator_log, 'node-a shard 0 is READY', count=5)
                    coordinator.stdin.write('/providers\n')
                    coordinator.stdin.flush()
                    wait_for(coordinator_log, 'state: READY', count=2)
                    coordinator.stdin.write('exit\n')
                    coordinator.stdin.flush()
                    self.assertEqual(coordinator.wait(timeout=10), 0)
                    final = coordinator_log.read_text()
                    self.assertEqual(final.count('node-a shard 0 is DOWNLOADING'), 1)
                    self.assertIn('state: NOT_READY', final)
                    self.assertGreaterEqual(final.count('state: READY'), 2)
                finally:
                    if coordinator.poll() is None:
                        coordinator.kill()
                        coordinator.wait()
                    if coordinator.stdin and not coordinator.stdin.closed:
                        coordinator.stdin.close()
                    for entry in providers:
                        if not entry:
                            continue
                        process, handle = entry
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        if not handle.closed:
                            handle.close()

    def run_failure(self, source, expected_hash, worker, pre_cache=None):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            cache = work / 'cache'
            if pre_cache is not None:
                target = cache / 'dan-main/v1/only' / f'{expected_hash}.artifact'
                target.parent.mkdir(parents=True)
                target.write_bytes(pre_cache)
            manifest = work / 'manifest'
            manifest.write_text(f'model|dan-main|v1\nshard|only|0|{expected_hash}|{source}|1\n')
            log = work / 'log'
            port = free_port()
            with log.open('w') as output:
                coordinator = subprocess.Popen([str(ROOT / 'build/coordinator'), str(port),
                    '--managed-model', str(manifest)], stdin=subprocess.PIPE, stdout=output,
                    stderr=subprocess.STDOUT, text=True)
                provider = None
                try:
                    wait_for(log, 'Listening for providers')
                    provider = subprocess.Popen(self.provider_command(
                        'failure-node', port, free_port(), cache, worker=worker),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    wait_for(log, 'failure-node shard only is ERROR', timeout=6)
                    coordinator.stdin.write('exit\n')
                    coordinator.stdin.flush()
                    coordinator.wait(timeout=5)
                    coordinator.stdin.close()
                    target = cache / 'dan-main/v1/only' / f'{expected_hash}.artifact'
                    return log.read_text(), target.read_bytes() if target.exists() else None
                finally:
                    if coordinator.poll() is None:
                        coordinator.kill(); coordinator.wait()
                    if coordinator.stdin and not coordinator.stdin.closed:
                        coordinator.stdin.close()
                    if provider and provider.poll() is None:
                        provider.kill(); provider.wait()

    def test_download_hash_and_worker_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / 'artifact'
            artifact.write_bytes(b'valid artifact')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            log, _ = self.run_failure(f'file://{artifact}', '0' * 64,
                                      ROOT / 'scripts/fake_managed_worker.py')
            self.assertIn('DOWNLOADING', log)
            log, _ = self.run_failure('file:///does/not/exist', digest,
                                      ROOT / 'scripts/fake_managed_worker.py')
            self.assertIn('DOWNLOADING', log)
            log, _ = self.run_failure(f'file://{artifact}', digest, '/does/not/exist')
            self.assertIn('is LOADING', log)
            log, _ = self.run_failure(f'file://{artifact}', digest, '/bin/false')
            self.assertIn('is LOADING', log)

    def test_corrupt_cache_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / 'artifact'
            artifact.write_bytes(b'valid artifact')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            # A missing worker leaves the verified replacement available for inspection.
            log, cached = self.run_failure(f'file://{artifact}', digest,
                                           '/does/not/exist', pre_cache=b'corrupt')
            self.assertIn('is DOWNLOADING', log)
            self.assertEqual(cached, artifact.read_bytes())

    def test_malformed_load_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / 'artifact'
            artifact.write_bytes(b'artifact')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0)); listener.listen()
                port = listener.getsockname()[1]
                provider = subprocess.Popen(self.provider_command(
                    'malformed', port, free_port(), pathlib.Path(directory) / 'cache'),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                connection, _ = listener.accept()
                with connection:
                    def receive():
                        size = struct.unpack('!I', connection.recv(4))[0]
                        data = b''
                        while len(data) < size:
                            data += connection.recv(size - len(data))
                        return data.decode()
                    def send(message):
                        data = message.encode()
                        connection.sendall(struct.pack('!I', len(data)) + data)
                    self.assertEqual(receive(), 'HELLO')
                    self.assertTrue(receive().startswith('CAPABILITIES\n'))
                    send(f'ASSIGN_SHARD\nmodel_id=dan-main\nversion=v1\nshard_id=only\n'
                         f'size_bytes=0\nhash={digest}\nsource=file://{artifact}')
                    while not receive().endswith('state=CACHED'):
                        pass
                    send('LOAD_SHARD\nmodel_id=dan-main')
                self.assertNotEqual(provider.wait(timeout=5), 0)


if __name__ == '__main__':
    unittest.main()
