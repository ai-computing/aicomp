"""Real pipe/process failures; the tiny child replaces only the expensive engine."""
import os
import subprocess
import sys
import time
from concurrent.futures import TimeoutError
from multiprocessing import shared_memory

import pytest
from opt_prime.accel.pd.contracts import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker

CHILD = '''
import json, os, sys, time
from pathlib import Path
mode = os.environ.get("PREFILL_TEST_MODE", "ok")
if mode == "stubborn":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
gate = Path(os.environ["PREFILL_TEST_GATE"])
if mode == "startup_stall":
    time.sleep(120)
if mode == "startup_exit":
    sys.exit(7)
if mode == "logs":
    print('{"compile_start": true}', flush=True)
    print('[]', flush=True)
print('{"ready": true}', flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if msg["id"] == 0 and mode in ("cancel", "exit", "malformed", "stubborn"):
        gate.with_suffix(".seen").touch()
        while not gate.exists():
            time.sleep(.005)
    if mode == "exit":
        sys.exit(8)
    if mode == "malformed":
        print(json.dumps({"id": msg["id"], "n": "bad"}), flush=True)
        continue
    print(json.dumps({"id": msg["id"], "n": 1, "token": 17}), flush=True)
'''


@pytest.fixture
def children(tmp_path, monkeypatch):
    """Observe real allocated resources, and clean up even against the buggy baseline."""
    procs, names = [], []
    popen, shm = subprocess.Popen, shared_memory.SharedMemory

    def spawn(*args, **kwargs):
        p = popen(*args, **kwargs)
        procs.append(p)
        return p

    def allocate(*args, **kwargs):
        s = shm(*args, **kwargs)
        if kwargs.get("create"):
            names.append(s.name)
        return s

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(shared_memory, "SharedMemory", allocate)
    script = tmp_path / "child"
    script.write_text(f"#!{sys.executable}\n" + CHILD)
    script.chmod(0o755)
    workers = []

    def make(mode="ok", **kwargs):
        env = dict(os.environ, PREFILL_TEST_MODE=mode, PREFILL_TEST_GATE=str(tmp_path / "gate"))
        w = SubprocessPrefillWorker(str(script), [], slots=2, layers=1, kv_heads=1,
                                   head_dim=2, max_context=8, env=env, **kwargs)
        workers.append(w)
        return w

    yield make, procs, names, tmp_path / "gate"
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=5)
    for w in workers:
        w.close()
    for name in names:
        try:
            s = shm(name=name)
            s.close()
            s.unlink()
        except FileNotFoundError:
            pass


def request():
    return PDRequest("test", (1, 2), 1)


def release(gate):
    deadline = time.monotonic() + 3
    while not gate.with_suffix(".seen").exists():
        assert time.monotonic() < deadline, "child did not receive request"
        time.sleep(.005)
    gate.touch()


def test_json_sdk_logs_do_not_break_startup(children):
    make, _, _, _ = children
    w = make("logs", startup_timeout=1)
    assert w.submit(request(), slot=0).result(timeout=2).token == 17


def test_cancelled_response_does_not_strand_next_request(children):
    make, _, _, gate = children
    w = make("cancel")
    first = w.submit(request(), slot=0)
    assert first.cancel()
    second = w.submit(request(), slot=1)
    release(gate)
    assert second.result(timeout=2).token == 17
    assert first.cancelled()


@pytest.mark.parametrize("mode", ["exit", "malformed"])
def test_terminal_failure_settles_all_requests_and_rejects_new_ones(children, mode):
    make, _, _, gate = children
    w = make(mode)
    futures = [w.submit(request(), slot=i) for i in range(2)]
    release(gate)
    for f in futures:
        with pytest.raises(RuntimeError):
            f.result(timeout=2)
    with pytest.raises(RuntimeError):
        w.submit(request(), slot=0).result(timeout=2)
    replacement = make()
    assert replacement.submit(request(), slot=0).result(timeout=2).token == 17


@pytest.mark.parametrize("mode", ["startup_stall", "startup_exit"])
def test_failed_startup_releases_child_and_shared_memory(children, mode):
    make, procs, names, _ = children
    with pytest.raises((TimeoutError, RuntimeError)):
        make(mode, startup_timeout=.2)
    assert procs[-1].poll() is not None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[-1])


def test_close_is_idempotent_and_rejects_new_work(children):
    make, _, names, _ = children
    w = make()
    w.close()
    w.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[-1])
    with pytest.raises(RuntimeError):
        w.submit(request(), slot=0).result(timeout=2)


def test_close_kills_unresponsive_worker_and_settles_pending(children):
    make, procs, names, _ = children
    w = make("stubborn")
    pending = w.submit(request(), slot=0)
    w.close(timeout=.1)
    with pytest.raises(RuntimeError):
        pending.result(timeout=2)
    assert procs[-1].poll() is not None
    assert not w._recv.is_alive()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[-1])


def test_spawn_failure_does_not_leak_shared_memory(children, monkeypatch):
    make, _, names, _ = children

    def fail(*args, **kwargs):
        raise OSError("cannot start child")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError, match="cannot start child"):
        make()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[-1])
