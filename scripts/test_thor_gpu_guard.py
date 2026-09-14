#!/usr/bin/env python3
"""CPU-only regressions for physical-GPU reservation; no CUDA calls are made."""

from contextlib import contextmanager, redirect_stderr
import fcntl
import io
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import thor_gpu_guard as guard


class FakeLock:
    def __init__(self, grants=()):
        self.grants = list(grants)
        self.held = False
        self.releases = 0

    def try_acquire(self):
        self.held = self.grants.pop(0) if self.grants else True
        return self.held

    def release(self):
        self.held = False
        self.releases += 1


class Scenario:
    def __init__(self, activity, grants=()):
        self.lock = FakeLock(grants)
        self.activity = iter(activity)
        self.now = 0.0
        self.sleeps = []
        self.probes = []

    def probe(self, timeout):
        self.probes.append((self.lock.held, timeout))
        item = next(self.activity)
        if isinstance(item, BaseException):
            raise item
        return item

    def sleep(self, seconds):
        assert not self.lock.held, "the guard monopolized the GPU lock while polling"
        self.sleeps.append(seconds)
        self.now += seconds

    def run(self, wait=5, poll=2):
        guard.reserve_idle(self.lock, wait, poll, probe=self.probe,
                           clock=lambda: self.now, sleep=self.sleep)


class ReservationTests(unittest.TestCase):
    def test_idle_is_rechecked_under_lock(self):
        scenario = Scenario([False, False])
        scenario.run()
        self.assertTrue(scenario.lock.held)
        self.assertEqual([held for held, _ in scenario.probes], [False, True])
        self.assertEqual(scenario.sleeps, [])

    def test_busy_gpu_does_not_reserve_the_lock(self):
        scenario = Scenario([True, False, False])
        scenario.run()
        self.assertEqual(scenario.sleeps, [2])
        self.assertTrue(scenario.lock.held)

    def test_lock_contention_is_bounded_and_retried(self):
        scenario = Scenario([False, False, False], grants=[False, True])
        scenario.run()
        self.assertEqual(scenario.sleeps, [2])
        self.assertTrue(scenario.lock.held)

    def test_workload_appearing_during_acquisition_releases_lock(self):
        scenario = Scenario([False, True, False, False])
        scenario.run()
        self.assertEqual(scenario.lock.releases, 1)
        self.assertEqual(scenario.sleeps, [2])
        self.assertTrue(scenario.lock.held)

    def test_permanently_busy_gpu_expires_without_oversleep(self):
        scenario = Scenario([True] * 3)
        with self.assertRaises(guard.ResourceUnavailable):
            scenario.run()
        self.assertEqual(scenario.sleeps, [2, 2, 1])
        self.assertFalse(scenario.lock.held)

    def test_permanent_lock_contention_expires(self):
        scenario = Scenario([False] * 3, grants=[False] * 3)
        with self.assertRaises(guard.ResourceUnavailable):
            scenario.run()
        self.assertEqual(scenario.now, 5)
        self.assertFalse(scenario.lock.held)

    def test_query_failure_before_acquisition_never_grants(self):
        scenario = Scenario([guard.ProbeFailed("unqueryable")])
        with self.assertRaises(guard.ProbeFailed):
            scenario.run()
        self.assertFalse(scenario.lock.held)

    def test_query_failure_under_lock_releases(self):
        scenario = Scenario([False, guard.ProbeFailed("unqueryable")])
        with self.assertRaises(guard.ProbeFailed):
            scenario.run()
        self.assertEqual(scenario.lock.releases, 1)
        self.assertFalse(scenario.lock.held)

    def test_cancellation_during_recheck_releases(self):
        scenario = Scenario([False, KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            scenario.run()
        self.assertFalse(scenario.lock.held)

    def test_late_idle_probe_is_not_accepted(self):
        scenario = Scenario([])
        calls = []

        def slow_probe(timeout):
            calls.append(timeout)
            if scenario.lock.held:
                scenario.now = 5
            return False

        with self.assertRaises(guard.ResourceUnavailable):
            guard.reserve_idle(scenario.lock, 5, 2, probe=slow_probe,
                               clock=lambda: scenario.now, sleep=scenario.sleep)
        self.assertFalse(scenario.lock.held)
        self.assertEqual(calls, [5, 5])

    def test_query_timeout_is_bounded_by_remaining_budget(self):
        scenario = Scenario([True, False, False])
        scenario.run()
        self.assertEqual([timeout for _, timeout in scenario.probes], [5, 3, 3])

    def test_invalid_time_budgets_are_rejected(self):
        for value in ("nan", "inf", "-inf", "0", "-1", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                guard.positive_seconds(value)


class ProcessTests(unittest.TestCase):
    def test_probe_errors_are_not_idle(self):
        errors = [FileNotFoundError(), subprocess.CalledProcessError(1, "nvidia-smi"),
                  subprocess.TimeoutExpired("nvidia-smi", 1)]
        for error in errors:
            with self.subTest(error=error), mock.patch.object(guard.subprocess, "run", side_effect=error):
                with self.assertRaises(guard.ProbeFailed):
                    guard.compute_active(1)

    def test_probe_retains_argv_and_timeout_without_shell(self):
        with mock.patch.object(guard.subprocess, "run") as run:
            run.return_value.stdout = "  \n"
            self.assertFalse(guard.compute_active(3))
            run.return_value.stdout = "123, external-program\n"
            self.assertTrue(guard.compute_active(3))
            self.assertEqual(run.call_args.kwargs["timeout"], 3)
            self.assertNotIn("shell", run.call_args.kwargs)
            self.assertTrue(run.call_args.kwargs["check"])

    def test_missing_lock_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError), guard.device_lock(path):
                self.fail("missing device accepted")
            self.assertFalse(path.exists())

    def test_regular_lock_file_is_rejected_without_truncating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-device"
            path.write_text("preserve me")
            with self.assertRaises(ValueError), guard.device_lock(path):
                self.fail("regular file accepted")
            self.assertEqual(path.read_text(), "preserve me")

    def test_symlink_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "link"
            path.symlink_to("/dev/null")
            with self.assertRaises(OSError), guard.device_lock(path):
                self.fail("symlink accepted")

    def test_real_flock_contends_across_file_descriptions(self):
        with tempfile.NamedTemporaryFile() as first, open(first.name, "r+b") as second:
            one, two = guard.Flock(first.fileno()), guard.Flock(second.fileno())
            self.assertTrue(one.try_acquire())
            self.assertFalse(two.try_acquire())
            one.release()
            self.assertTrue(two.try_acquire())
            two.release()

    def test_exec_inherits_lock_and_preserves_command_exit_status(self):
        with tempfile.NamedTemporaryFile() as lockfile:
            child = r'''
import fcntl, os, sys
fd = int(sys.argv[1])
assert os.get_inheritable(fd)
other = os.open(sys.argv[2], os.O_RDWR)
try:
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(23)
sys.exit(99)
'''
            launcher = "import sys; import thor_gpu_guard as g; lock=g.Flock(int(sys.argv[1])); assert lock.try_acquire(); g.exec_reserved(sys.argv[2:], lock)"
            result = subprocess.run(
                [sys.executable, "-c", launcher, str(lockfile.fileno()), sys.executable,
                 "-c", child, str(lockfile.fileno()), lockfile.name],
                cwd=Path(__file__).parent, pass_fds=(lockfile.fileno(),), timeout=10,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            # The inherited open description is shared with this test process.
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)

    def test_timeout_never_executes_and_is_not_success(self):
        @contextmanager
        def fake_device(_):
            yield FakeLock()
        with mock.patch.object(guard, "device_lock", fake_device), \
                mock.patch.object(guard, "reserve_idle", side_effect=guard.ResourceUnavailable("busy")), \
                mock.patch.object(guard, "exec_reserved") as execute, redirect_stderr(io.StringIO()) as output:
            self.assertEqual(guard.main(["--", "do-not-run"]), 75)
            execute.assert_not_called()
            self.assertIn("reason=resource-unavailable", output.getvalue())
            self.assertIn("execution_started=false", output.getvalue())

    def test_probe_failure_is_not_a_resource_exception(self):
        @contextmanager
        def fake_device(_):
            yield FakeLock()
        with mock.patch.object(guard, "device_lock", fake_device), \
                mock.patch.object(guard, "reserve_idle", side_effect=guard.ProbeFailed("failed")), \
                mock.patch.object(guard, "exec_reserved") as execute, redirect_stderr(io.StringIO()) as output:
            self.assertEqual(guard.main(["--", "do-not-run"]), 1)
            execute.assert_not_called()
            self.assertIn("reason=guard-error", output.getvalue())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (Path(__file__).resolve().parents[1] /
                         ".github/workflows/sciagent-thor-gate.yml").read_text()

    def test_every_hardware_command_uses_the_guard(self):
        lines = self.workflow.splitlines()
        commands = [i for i, line in enumerate(lines)
                    if "cargo test " in line or "cargo run " in line]
        self.assertEqual(len(commands), 7)
        for i in commands:
            self.assertIn("python3 scripts/thor_gpu_guard.py --", lines[i - 1])
        self.assertNotIn("flock -x 9", self.workflow)
        self.assertEqual(self.workflow.count('      - "scripts/thor_gpu_guard.py"'), 2)
        self.assertLess(self.workflow.index("name: SCIAGENT CUDA clippy"),
                        self.workflow.index("name: Wait for CUDA runtime availability"))

    def runtime_probe(self, producer):
        line = next(line for line in self.workflow.splitlines()
                    if "ldconfig -p 2>/dev/null" in line)
        predicate = line.split("|| ! ", 1)[1].removesuffix("; then")
        command = (shlex.quote(sys.executable) + " -c " + shlex.quote(producer))
        predicate = predicate.replace("ldconfig -p 2>/dev/null", command, 1)
        return subprocess.run(["bash", "-o", "pipefail", "-c", predicate],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=10)

    def test_runtime_probe_consumes_large_output_without_sigpipe(self):
        producer = ("import os,signal; signal.signal(signal.SIGPIPE,signal.SIG_DFL); "
                    "os.write(1,b'libcuda.so\\n'); "
                    "[os.write(1,b'x'*8192+b'\\n') for _ in range(256)]")
        result = self.runtime_probe(producer)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_probe_rejects_missing_cuda(self):
        result = self.runtime_probe("print('libother.so')")
        self.assertEqual(result.returncode, 1)

    def test_runtime_probe_does_not_mask_producer_failure(self):
        result = self.runtime_probe("import sys; print('libcuda.so'); sys.exit(7)")
        self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
