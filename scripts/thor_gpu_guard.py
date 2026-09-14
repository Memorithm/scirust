#!/usr/bin/env python3
"""Run a foreground command under a cooperatively acquired physical GPU lock.

Linux only; no third-party modules. This does not stop or signal other workloads.
An advisory lock cannot exclude programs which do not use the same lock inode.
"""

import argparse
from contextlib import contextmanager
import fcntl
import math
import os
import stat
import subprocess
import sys
import time


class ResourceUnavailable(RuntimeError):
    """The bounded reservation window elapsed before an idle device was acquired."""


class ProbeFailed(RuntimeError):
    """Device occupancy could not be verified; never treat this as idle."""


class Flock:
    """An advisory lock on an already opened file description."""

    def __init__(self, fd):
        self.fd = fd

    def try_acquire(self):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def release(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)


@contextmanager
def device_lock(path):
    # Do not create, truncate or follow a symlink to a replacement lock file.
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        if not stat.S_ISCHR(os.fstat(fd).st_mode):
            raise ValueError("GPU lock target must be a character device")
        yield Flock(fd)
    finally:
        os.close(fd)


def compute_active(timeout):
    """Query all visible compute applications conservatively, with a timeout."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ProbeFailed("nvidia-smi occupancy query failed") from exc
    return bool(result.stdout.strip())


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("time budgets must be finite and greater than zero")
    return seconds


def reserve_idle(lock, wait_seconds, poll_seconds, *, probe=compute_active,
                 clock=time.monotonic, sleep=time.sleep):
    """Return with lock held only after an idle observation under that lock.

    The deadline includes both observations and lock contention. Polling never
    holds the lock. A probe error releases a newly acquired lock and propagates.
    """
    wait_seconds = positive_seconds(wait_seconds)
    poll_seconds = positive_seconds(poll_seconds)
    deadline = clock() + wait_seconds

    def observe():
        remaining = deadline - clock()
        if remaining <= 0:
            raise ResourceUnavailable("idle GPU reservation deadline elapsed")
        active = probe(min(30.0, remaining))
        if clock() >= deadline:
            raise ResourceUnavailable("idle GPU reservation deadline elapsed")
        return active

    while True:
        if not observe() and lock.try_acquire():
            try:
                if not observe():
                    return
            except BaseException:
                lock.release()
                raise
            # A non-cooperating workload started between observation and lock.
            lock.release()
        remaining = deadline - clock()
        if remaining <= 0:
            raise ResourceUnavailable("idle GPU reservation deadline elapsed")
        sleep(min(poll_seconds, remaining))


def exec_reserved(command, lock):
    """Replace the guard, keeping the same lock through the command's lifetime.

    Passing argv directly avoids shell evaluation. exec also preserves the
    command's exit status and normal runner cancellation/signal behavior.
    """
    os.set_inheritable(lock.fd, True)
    os.execvp(command[0], command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=os.environ.get("SCIRUST_THOR_GPU_LOCK", "/dev/nvidia0"))
    parser.add_argument("--wait-seconds", type=positive_seconds,
                        default=os.environ.get("SCIRUST_THOR_IDLE_WAIT_SECONDS", "3600"))
    parser.add_argument("--poll-seconds", type=positive_seconds, default="15")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a foreground command is required after --")
    try:
        with device_lock(args.lock) as lock:
            reserve_idle(lock, args.wait_seconds, args.poll_seconds)
            print("gpu_reservation=acquired occupancy=idle_under_lock", flush=True)
            exec_reserved(command, lock)
    except ResourceUnavailable as exc:
        print(f"::error::reason=resource-unavailable execution_started=false: {exc}", file=sys.stderr)
        return 75
    except (ProbeFailed, OSError, ValueError) as exc:
        print(f"::error::reason=guard-error execution_started=false: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
