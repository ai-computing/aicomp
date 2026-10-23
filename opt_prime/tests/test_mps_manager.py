#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.mps_manager: daemon detection and stale cleanup stay scoped to this job's pipe directory.
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime import mps_manager  # noqa: E402


def _fake_run(calls):
    # Every command "succeeds" as if another job's MPS daemon were running on this host.
    def run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")
    return run


def test_daemon_running_only_from_own_control_socket(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    assert not mps_manager._mps_daemon_running(str(tmp_path))      # another job's daemon is not ours
    (tmp_path / "control").touch()
    assert mps_manager._mps_daemon_running(str(tmp_path))


def test_stale_cleanup_does_not_kill_other_daemons(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    monkeypatch.setattr(mps_manager.time, "sleep", lambda s: None)
    (tmp_path / "control").touch()
    mps_manager.MPSManager(pipe_dir=str(tmp_path), log_dir=str(tmp_path / "log"))._force_cleanup_stale_daemon()
    assert not any(c[0] in ("pkill", "killall") for c in calls), calls
    assert not (tmp_path / "control").exists()
