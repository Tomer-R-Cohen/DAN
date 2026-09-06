"""CPU-only regression checks for the GPU deployment path (no CUDA claims)."""
import json
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


class PreparationTests(unittest.TestCase):
    def test_managed_provider_reports_cached_runtime_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            runtime = work / 'runtime'
            runtime.write_text("""#!/usr/bin/env python3
import sys
print('\\n> ', end='', flush=True)
for line in sys.stdin:
    print('fake response\\n> ', end='', flush=True)
""")
            runtime.chmod(0o755)
            model = work / 'test.gguf'
            model.touch()
            manifest = work / 'manifest'
            manifest.write_text('model|dan-main|v1\nshard|only|0|testhash|placeholder://only|1\n')
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            coordinator = subprocess.Popen([
                str(ROOT / 'build/coordinator'), str(port), '--managed-model',
                str(manifest)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            provider = None
            try:
                while 'Listening for providers' not in coordinator.stdout.readline():
                    if coordinator.poll() is not None:
                        self.fail('Coordinator exited before listening')
                provider = subprocess.Popen([
                    str(ROOT / 'build/provider'), str(runtime), str(model),
                    '127.0.0.1', str(port), '--id', 'managed-real', '--control-plane',
                    '--vram-mib', '1', '--cached-shard',
                    'dan-main|v1|only|testhash'], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                time.sleep(1.2)
                output, _ = coordinator.communicate('/providers\nexit\n', timeout=10)
                self.assertEqual(coordinator.returncode, 0, output)
                self.assertIn('managed-real', output)
                self.assertIn('ready shards: 1\nstate: READY', output)
                self.assertEqual(provider.wait(timeout=5), 0)
            finally:
                if coordinator.poll() is None:
                    coordinator.kill()
                    coordinator.wait()
                if provider and provider.poll() is None:
                    provider.kill()
                    provider.wait()
                coordinator.stdout.close()
                if coordinator.stdin and not coordinator.stdin.closed:
                    coordinator.stdin.close()

    def test_four_provider_control_plane_reconnects_cached_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            coordinator = subprocess.Popen([
                str(ROOT / 'build/coordinator'), str(port), '--managed-model',
                str(ROOT / 'config/dan-main.example.manifest'),
                '--heartbeat-timeout', '1'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            providers = [None] * 4
            try:
                while 'Listening for providers' not in coordinator.stdout.readline():
                    if coordinator.poll() is not None:
                        self.fail('Coordinator exited before listening')
                vrams = (24576, 20480, 16384, 12288)
                # Lowest VRAM connects first; manifest requirements still produce
                # capability-based shard placement rather than A/B or arrival slots.
                for index in reversed(range(4)):
                    vram = vrams[index]
                    provider_id = f'node-{chr(ord("a") + index)}'
                    providers[index] = subprocess.Popen([
                        'python3', str(ROOT / 'scripts/fake_provider.py'),
                        '--id', provider_id, '--port', str(port), '--gpu',
                        f'GPU-{index}', '--vram-mib', str(vram), '--cache-file',
                        str(work / f'{provider_id}.cache')], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    time.sleep(0.1)
                time.sleep(1)
                coordinator.stdin.write('/providers\n')
                coordinator.stdin.flush()

                with socket.create_connection(('127.0.0.1', port), timeout=5) as malformed:
                    for message in ('HELLO', 'CAPABILITIES\nprovider_id=bad-control\n'
                            'device_type=GPU\ngpu_name=fake\nvram=1 MiB\nvram_mib=1\n'
                            'model_name=fake\nbackend=FAKE\ncontrol_plane=1',
                            'SHARD_STATE\nstate=READY'):
                        data = message.encode()
                        malformed.sendall(struct.pack('!I', len(data)) + data)
                time.sleep(0.2)

                os.kill(providers[0].pid, signal.SIGSTOP)
                time.sleep(1.5)
                coordinator.stdin.write('/providers\n')
                coordinator.stdin.flush()
                providers[0].kill()
                providers[0].wait(timeout=5)

                providers[0] = subprocess.Popen([
                    'python3', str(ROOT / 'scripts/fake_provider.py'), '--id',
                    'node-a', '--port', str(port), '--gpu', 'GPU-0', '--vram-mib',
                    '24576', '--cache-file', str(work / 'node-a.cache')],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)
                coordinator.stdin.write('/providers\nexit\n')
                coordinator.stdin.flush()
                output, _ = coordinator.communicate(timeout=10)
                self.assertEqual(coordinator.returncode, 0, output)
                states = [line for line in output.splitlines() if line.startswith('state: ')]
                self.assertEqual(states, ['state: READY', 'state: NOT_READY', 'state: READY'])
                self.assertIn('heartbeat timed out; marked OFFLINE', output)
                self.assertIn('sent invalid shard state', output)
                self.assertEqual(output.count('node-a shard 0 is DOWNLOADING'), 1)
                for index, name in enumerate(('node-a', 'node-b', 'node-c', 'node-d')):
                    self.assertRegex(output, rf'{name}\s+GPU-[0-3].*\s{index}\s+READY')
            finally:
                if coordinator.poll() is None:
                    coordinator.kill()
                    coordinator.wait()
                for provider in providers:
                    if provider and provider.poll() is None:
                        provider.kill()
                        provider.wait()
                coordinator.stdout.close()
                if coordinator.stdin and not coordinator.stdin.closed:
                    coordinator.stdin.close()

    def test_piped_prompt_burst_and_shutdown(self):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        coordinator = subprocess.Popen([str(ROOT / 'build/coordinator'), str(port)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
        try:
            while 'Listening for providers' not in coordinator.stdout.readline():
                if coordinator.poll() is not None:
                    self.fail('Coordinator exited before listening')
            with socket.create_connection(('127.0.0.1', port), timeout=5) as peer:
                def send(message):
                    data = message.encode()
                    peer.sendall(struct.pack('!I', len(data)) + data)
                def receive():
                    def exact(size):
                        data = b''
                        while len(data) < size:
                            chunk = peer.recv(size - len(data))
                            if not chunk:
                                self.fail('Unexpected disconnect')
                            data += chunk
                        return data
                    return exact(struct.unpack('!I', exact(4))[0]).decode()
                send('HELLO')
                send('CAPABILITIES\nprovider_id=test\ndevice_type=CPU\ngpu_name=none\n'
                     'vram=0\nmodel_name=test\nbackend=CPU')
                # One write, no terminal line buffering; exit has no final newline.
                coordinator.stdin.write('first\nsecond\nexit')
                coordinator.stdin.close()
                for number, prompt in enumerate(['first', 'second'], 1):
                    self.assertEqual(receive(), f'PROMPT\n{number}\n{prompt}')
                    send(f'RESPONSE\n{number}\nanswer')
                self.assertEqual(receive(), 'BYE')
            self.assertEqual(coordinator.wait(timeout=5), 0)
        finally:
            if coordinator.poll() is None:
                coordinator.kill()
                coordinator.wait()
            coordinator.stdout.close()
            if not coordinator.stdin.closed:
                coordinator.stdin.close()

    def test_driver_and_runtime_options(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            runtime = work / 'runtime'
            runtime.write_text('''#!/usr/bin/env python3
import sys
assert sys.argv[sys.argv.index('--ctx-size') + 1] == '4096'
assert sys.argv[sys.argv.index('--gpu-layers') + 1] == '0'
assert sys.argv[sys.argv.index('--n-predict') + 1] == '-1'
print('\\n> ', end='', flush=True)
for line in sys.stdin:
    assert line.endswith(' \\n')
    print('Test response\\n> ', end='', flush=True)
''')
            runtime.chmod(0o755)
            model = work / 'test.gguf'
            model.touch()
            registry = work / 'models.conf'
            registry.write_text(f'dan-main|test|fake|{model}|test.gguf|fake|1|4096|test|yes|no\n')
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            result = subprocess.run(['python3', str(ROOT / 'scripts/validate_gpu.py'),
                                     '--registry', str(registry), '--runtime', str(runtime),
                                     '--output', str(work / 'evidence'), '--port', str(port),
                                     '--cpu-smoke', '--timeout', '10'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((work / 'evidence/results.json').read_text())
            self.assertEqual(len(report['requests']), 10)
            self.assertEqual(report['exit_codes'], [0, 0])
            self.assertEqual(report['mode'], 'cpu-smoke')

    def test_runtime_failure_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            model = work / 'test.gguf'
            model.touch()
            registry = work / 'models.conf'
            registry.write_text(f'dan-main|test|fake|{model}|test.gguf|fake|1|4096|test|yes|no\n')
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            result = subprocess.run(['python3', str(ROOT / 'scripts/validate_gpu.py'),
                                     '--registry', str(registry), '--runtime', '/bin/false',
                                     '--output', str(work / 'evidence'), '--port', str(port),
                                     '--cpu-smoke', '--timeout', '5'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((work / 'evidence/results.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertIn('error', report)

    def test_bad_registry_and_invalid_runtime_option(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = pathlib.Path(directory) / 'bad.conf'
            registry.write_text('bad|main|family|/model|name|Q4|16GB|4096|CUDA|yes|no\n')
            result = subprocess.run([str(ROOT / 'build/coordinator'), '--models', str(registry)],
                                    capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid numeric', result.stderr)
            manifest = pathlib.Path(directory) / 'bad.manifest'
            manifest.write_text('model|another-model|v1\nshard|0|bad|hash|source|1\n')
            result = subprocess.run([str(ROOT / 'build/coordinator'), '--managed-model',
                                     str(manifest)], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid managed-model manifest', result.stderr)
        result = subprocess.run([str(ROOT / 'build/provider'), '/unused', '/unused',
                                 '--ctx-size', '-1'], capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
