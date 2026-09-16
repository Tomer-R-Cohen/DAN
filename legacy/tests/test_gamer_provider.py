"""CPU-only onboarding and reconnect coverage for dan-provider."""

import hashlib
import os
import pathlib
import re
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


class GamerProviderTests(unittest.TestCase):
    def mock_smi(self, work, output, status=0):
        path = work / f'nvidia-smi-{len(list(work.glob("nvidia-smi-*")))}'
        path.write_text(f'#!/usr/bin/env python3\nimport sys\n'
                        f'sys.stdout.write({output!r})\nsys.exit({status})\n')
        path.chmod(0o755)
        return path

    def check_command(self, work, smi, *extra):
        return [str(ROOT / 'build/dan-provider'), '--coordinator', '127.0.0.1:9000',
                '--advertise-host', '127.0.0.1', '--rpc-worker', '/bin/true',
                '--state-dir', str(work / 'state'), '--cache-dir', str(work / 'cache'),
                '--nvidia-smi', str(smi), '--check', *extra]

    def test_gpu_detection_identity_reserve_and_device_override(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            smi = self.mock_smi(work,
                '2, NVIDIA GeForce RTX 4090, 24576, GPU-b\n'
                '0, NVIDIA GeForce RTX 2070, 8192, GPU-a\n')
            first = subprocess.run(self.check_command(work, smi), text=True,
                                   capture_output=True, check=True).stdout
            second = subprocess.run(self.check_command(work, smi), text=True,
                                    capture_output=True, check=True).stdout
            self.assertIn('GPU: NVIDIA GeForce RTX 2070', first)
            self.assertIn('Device: CUDA0', first)
            self.assertIn('VRAM total: 8192 MiB', first)
            self.assertIn('Reserved: 1536 MiB', first)
            self.assertIn('Available to DAN: 6656 MiB', first)
            first_id = next(line for line in first.splitlines()
                            if line.startswith('Provider ID:'))
            self.assertIn(first_id, second)

            override = subprocess.run(self.check_command(work, smi, '--device', '2',
                '--reserve-vram-mib', '2048'), text=True, capture_output=True,
                check=True).stdout
            self.assertIn('GPU: NVIDIA GeForce RTX 4090', override)
            self.assertIn('Device: CUDA2', override)
            self.assertIn('Available to DAN: 22528 MiB', override)

            config = work / 'provider.conf'
            config.write_text(f'coordinator=127.0.0.1:9000\nadvertise_host=127.0.0.1\n'
                              f'rpc_worker=/bin/true\nnvidia_smi={smi}\n'
                              f'state_dir={work / "configured-state"}\n'
                              f'cache_dir={work / "configured-cache"}\n'
                              'provider_name=friend-pc\ndevice=2\nreserve_vram_mib=2048\n')
            configured = subprocess.run([str(ROOT / 'build/dan-provider'), '--config',
                str(config), '--check'], text=True, capture_output=True, check=True).stdout
            self.assertIn('Name: friend-pc', configured)
            self.assertIn('Device: CUDA2', configured)

    def test_gpu_and_configuration_failures_are_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            empty = self.mock_smi(work, '')
            malformed = self.mock_smi(work, 'not,csv\n')
            good = self.mock_smi(work, '0, RTX 2070, 8192, GPU-a\n')
            for command, expected in (
                (self.check_command(work, empty), 'NVIDIA GPU detection failed'),
                (self.check_command(work, malformed), 'malformed nvidia-smi GPU output'),
                (self.check_command(work, good, '--device', '4'),
                 'Configured CUDA device 4'),
                (self.check_command(work, good, '--reserve-vram-mib', '8192'),
                 'VRAM reserve must be smaller')):
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

            config = work / 'bad.conf'
            config.write_text('coordinator=127.0.0.1:9000\nunknown=yes\n')
            result = subprocess.run([str(ROOT / 'build/dan-provider'), '--config',
                str(config), '--check'], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unknown provider configuration key', result.stderr)
            subprocess.run(['bash', '-n', str(ROOT / 'scripts/setup_provider.sh')], check=True)
            setup = subprocess.run([str(ROOT / 'scripts/setup_provider.sh')],
                                   text=True, capture_output=True)
            self.assertNotEqual(setup.returncode, 0)
            self.assertIn('Usage:', setup.stderr)

    def test_spare_reconnects_with_cache_and_keeps_one_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            artifact = work / 'artifact'
            artifact.write_bytes(b'gamer provider artifact')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = work / 'manifest'
            manifest.write_text(f'model|dan-main|v1\nshard|0|{artifact.stat().st_size}|'
                                f'{digest}|file://{artifact}|1\n')
            smi = self.mock_smi(work, '0, NVIDIA GeForce RTX 2070, 8192, GPU-test\n')
            port = free_port()
            processes, handles = [], []

            def coordinator(name):
                log = work / f'{name}.log'
                handle = log.open('w')
                handles.append(handle)
                process = subprocess.Popen([str(ROOT / 'build/coordinator'), str(port),
                    '--managed-model', str(manifest), '--heartbeat-timeout', '2'],
                    stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT, text=True)
                processes.append(process)
                wait_for(log, 'Listening for providers')
                return process, log

            first, first_log = coordinator('coordinator-1')
            incumbent_handle = (work / 'incumbent.log').open('w')
            handles.append(incumbent_handle)
            incumbent = subprocess.Popen([str(ROOT / 'build/managed_provider'), '--id',
                'incumbent', '--gpu', 'GPU', '--vram-mib', '9000', '--cache-dir',
                str(work / 'incumbent-cache'), '--worker',
                str(ROOT / 'scripts/fake_managed_worker.py'), '--worker-port',
                str(free_port()), '--port', str(port), '--worker-timeout', '2'],
                stdout=incumbent_handle, stderr=subprocess.STDOUT)
            processes.append(incumbent)
            wait_for(first_log, 'incumbent shard 0 is READY')

            gamer_log = work / 'gamer.log'
            gamer_handle = gamer_log.open('w')
            handles.append(gamer_handle)
            gamer = subprocess.Popen([str(ROOT / 'build/dan-provider'), '--coordinator',
                f'127.0.0.1:{port}', '--advertise-host', '127.0.0.1', '--rpc-worker',
                str(ROOT / 'scripts/fake_managed_worker.py'), '--state-dir',
                str(work / 'gamer-state'), '--cache-dir', str(work / 'gamer-cache'),
                '--provider-name', 'tomer-pc', '--nvidia-smi', str(smi),
                '--reconnect-seconds', '1'], stdout=gamer_handle,
                stderr=subprocess.STDOUT)
            processes.append(gamer)
            unrelated = subprocess.Popen(['sleep', '60'])
            processes.append(unrelated)
            try:
                wait_for(first_log, 'Name: tomer-pc')
                first.stdin.write('/providers\n'); first.stdin.flush()
                persistent_id = (work / 'gamer-state/provider-id').read_text().strip()
                self.assertRegex(wait_for(first_log, 'SPARE'),
                                 rf'{re.escape(persistent_id)}.*SPARE.*ONLINE')

                first.kill(); first.wait(timeout=5)
                first.stdin.close()
                second, second_log = coordinator('coordinator-2')
                wait_for(second_log, 'shard 0 is READY')
                worker_pid = child_pids(gamer.pid)[0]
                self.assertEqual(gamer_log.read_text().count('State: DOWNLOADING'), 1)

                second.kill(); second.wait(timeout=5)
                second.stdin.close()
                third, third_log = coordinator('coordinator-3')
                wait_for(third_log, 'Assigned shard 0', timeout=10)
                self.assertIn('from exact cache', wait_for(third_log, 'shard 0 is READY'))
                self.assertEqual(child_pids(gamer.pid), [worker_pid])
                self.assertEqual(gamer_log.read_text().count('State: DOWNLOADING'), 1)
                self.assertEqual((work / 'gamer-state/provider-id').read_text().strip(),
                                 next(line.split(': ', 1)[1] for line in
                                      gamer_log.read_text().splitlines()
                                      if line.startswith('Provider ID:')))

                gamer.send_signal(signal.SIGTERM)
                self.assertEqual(gamer.wait(timeout=5), 0)
                self.assertFalse(pathlib.Path(f'/proc/{worker_pid}').exists())
                self.assertIsNone(unrelated.poll())
                third.stdin.write('exit\n'); third.stdin.flush()
                self.assertEqual(third.wait(timeout=5), 0)
                third.stdin.close()
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill(); process.wait()
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                for handle in handles:
                    handle.close()


if __name__ == '__main__':
    unittest.main()
