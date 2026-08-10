"""Advisory file locks that serialize the long-running jobs sharing one checkout."""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Iterator
from pathlib import Path

# scripts/run_job.sh wraps every cron in this lock, so holding it serializes a
# one-off archive write against a live confer / prophecize / close-lines.
ARCHIVE_LOCK_FILE = "/tmp/sportstradamus-archive.lock"

# Held for a whole confirm/supersede walk. The walk snapshots a cell's serve-read
# artifacts, retrains, and restores them when the candidate loses — correct only
# while nothing else writes that set. A hand-typed meditate takes this lock to
# announce that it does, so the two can't interleave and tear a cell's pickle
# away from its test-set CSV.
TRAINING_ARTIFACTS_LOCK_FILE = "/tmp/sportstradamus-training-artifacts.lock"

# Set on every meditate subprocess a sweep or confirm walk launches. Those runs
# are the lock holder's own work, so they must not contend with it.
ORCHESTRATED_ENV_VAR = "SPORTSTRADAMUS_ORCHESTRATED"

# Relocates every lock file. Two checkouts on one box own separate data dirs and should not
# block each other, and parallel test workers must not contend on the real paths — the same
# escape hatch SPORTSTRADAMUS_ARCHIVE_DB gives the archive.
_LOCK_DIR_ENV = "SPORTSTRADAMUS_LOCK_DIR"

_POLL_S = 2

# Resolved path -> the descriptor holding it, for locks kept until this process exits.
_HELD_FOR_PROCESS: dict[str, int] = {}


def _resolve(path: str) -> str:
    override = os.environ.get(_LOCK_DIR_ENV)
    return str(Path(override) / Path(path).name) if override else path


def _holder(path: str) -> str:
    """The stamped holder as ``" by <holder>"``, or ``""`` when unstamped or unreadable."""
    with contextlib.suppress(OSError):
        if stamp := Path(path).read_text().strip():
            return f" by {stamp}"
    return ""


def _acquire(path: str, timeout_s: float, label: str, contention_hint: str) -> int:
    """Flock ``path`` exclusively and stamp the holder into it, or raise once ``timeout_s`` is spent."""
    path = _resolve(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                held, waited = _holder(path), f" after {timeout_s:.0f}s" if timeout_s else ""
                os.close(fd)
                raise RuntimeError(f"{path} still held{held}{waited}. {contention_hint}") from None
            time.sleep(_POLL_S)
    os.ftruncate(fd, 0)
    os.write(fd, f"pid {os.getpid()}: {label}".encode())
    return fd


@contextlib.contextmanager
def file_lock(path: str, *, timeout_s: float, label: str, contention_hint: str) -> Iterator[None]:
    """Hold ``path``'s flock for the block, waiting up to ``timeout_s`` before raising."""
    fd = _acquire(path, timeout_s, label, contention_hint)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def hold_for_process(path: str, *, timeout_s: float, label: str, contention_hint: str) -> None:
    """Take ``path``'s flock for the rest of this process, or no-op if already held.

    The descriptor is deliberately never closed — the kernel drops the lock at exit,
    which is the wanted lifetime, and staying out of a ``with`` block spares the
    caller from indenting an entire command body. Recording it also makes a second
    call a no-op rather than a self-denial: flock treats each descriptor
    independently, so re-opening the same file would refuse this process its own lock.
    """
    resolved = _resolve(path)
    if resolved not in _HELD_FOR_PROCESS:
        _HELD_FOR_PROCESS[resolved] = _acquire(path, timeout_s, label, contention_hint)
