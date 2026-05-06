#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#

"""
MPS (NVIDIA Multi-Process Service) lifecycle manager for opt_prime inference.

This module provides:
  - MPSManager: per-node lifecycle for the MPS daemon (start, stop, env mgmt)
  - resolve_visible_devices(): reconcile --num-gpus / --gpu-ids with CVD env
  - restore_visible_devices(): undo CVD changes made by resolve_visible_devices()
  - File barrier helpers for synchronizing local ranks before NCCL init

Design reference: opt_prime/docs/mps_inference_design.md

Key invariants:
  - MPS is enabled per-process via --use-mps (parsed in example entry points)
  - Daemon lifecycle is managed by local rank 0 only (one daemon per node)
  - CVD inheritance: daemon inherits CUDA_VISIBLE_DEVICES at spawn time
  - Env snapshot/restore: only opt_prime-set variables are restored; user-set
    values are never touched
  - All cleanup is idempotent and safe to call from atexit / signal handlers
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state for resolve_visible_devices / restore_visible_devices
# ---------------------------------------------------------------------------

# Tracks whether resolve_visible_devices() set CVD itself.
# - None: CVD was already set by user (or not set, and we did not change it)
# - str:  the value before we set it (currently always None for "was unset")
_resolve_snapshot: dict = {}


# ---------------------------------------------------------------------------
# CVD resolution helpers
# ---------------------------------------------------------------------------

def _parse_gpu_ids(gpu_ids_str: str) -> list:
    """Parse '0,2,4,6' into [0, 2, 4, 6]. Raise ValueError on bad input."""
    out = []
    for tok in gpu_ids_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            raise ValueError(f"Invalid GPU id in --gpu-ids: '{tok}'")
    return out


def _normalize_cvd(cvd_str: str) -> list:
    """Normalize an env CVD string into a sorted list of ints (for comparison)."""
    return sorted(_parse_gpu_ids(cvd_str))


def _physical_gpu_count() -> int:
    """Return total physical GPU count via nvidia-smi.

    Used only for sanity checking --num-gpus / --gpu-ids when CVD is not set.
    Returns 0 if nvidia-smi is unavailable (in which case checks are skipped).
    """
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return 0
        return len([line for line in out.stdout.strip().split("\n") if line.strip()])
    except (subprocess.TimeoutExpired, OSError):
        return 0


def _parse_visible_gpu_indices():
    """Return a set of physical GPU indices visible per CUDA_VISIBLE_DEVICES.
    Returns None if CVD is unset (means all physical GPUs are visible).

    Used only for sanity checking; does not initialize CUDA.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return None
    indices = set()
    for tok in cvd.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lstrip("-").isdigit():
            indices.add(int(tok))
    return indices


def _abort(msg: str, rank_zero_only: bool = False) -> None:
    """Print error message and exit. By default rank 0 only prints (for noise
    reduction), but all ranks call sys.exit(1) so the whole job aborts.

    Always flush stderr before exit; under torchrun's elastic launcher,
    buffered output can be lost when sys.exit triggers SIGTERM cascade
    across siblings.
    """
    rank = int(os.environ.get("RANK", "0"))
    # When abort is critical (driver/env issue), make it visible across all
    # ranks regardless of rank_zero_only — prevents losing the message if
    # rank 0's output gets clipped by torchrun's signal handling.
    should_print = (not rank_zero_only) or rank == 0
    if should_print:
        # Surround the message with a banner so it stands out among torchrun
        # control-plane logs (W0506.../E0506...).
        banner = "=" * 70
        full = (
            f"\n{banner}\n"
            f"[opt_prime][MPS] ERROR (rank={rank}, pid={os.getpid()}):\n"
            f"{msg}\n"
            f"{banner}\n"
        )
        try:
            print(full, file=sys.stderr, flush=True)
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass
    # Small delay so the banner makes it past torchrun's stdout/stderr
    # capture before this rank exits.
    try:
        time.sleep(0.1)
    except Exception:
        pass
    sys.exit(1)


def _info(msg: str) -> None:
    """Rank-0-only informational message."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[opt_prime][MPS] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    """Rank-0-only warning."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[opt_prime][MPS] WARN: {msg}", file=sys.stderr)


def resolve_visible_devices(num_gpus: Optional[int],
                            gpu_ids: Optional[str],
                            use_mps: bool = False) -> None:
    """
    Reconcile --num-gpus / --gpu-ids arguments with the env CUDA_VISIBLE_DEVICES.

    Must be called BEFORE any CUDA API call (i.e., before importing torch.cuda
    or invoking torch.cuda.* functions). Sets os.environ['CUDA_VISIBLE_DEVICES']
    if needed.

    Resolution rules (CLI args take precedence over env, with WARN on override):

      env CVD     | --gpu-ids        | --num-gpus       | action
      ------------+------------------+------------------+------------------------
      unset       | unset            | unset            | no change (Case 6)
      unset       | "0,2,4"          | -                | CVD = "0,2,4" (Case 4)
      unset       | -                | N                | CVD = "0,1,...,N-1" (5)
      set         | unset            | unset            | keep CVD (Case 1)
      set         | matches CVD      | -                | keep CVD, INFO (Case 2a)
      set         | differs from CVD | -                | override CVD, WARN (Case 2b)
      set         | -                | matches count    | keep CVD, INFO (Case 3a)
      set         | -                | differs in count | override CVD, WARN (Case 3b)

    --gpu-ids takes precedence over --num-gpus when both given (with WARN).

    Whenever this function changes os.environ['CUDA_VISIBLE_DEVICES'], it
    records the original value (or None if previously unset) in _resolve_snapshot
    so restore_visible_devices() can undo the change later.

    Aborts (sys.exit(1)) on syntactic errors in --gpu-ids or out-of-range IDs.

    MPS strict mode: when use_mps is True, this function NEVER modifies the
    env CUDA_VISIBLE_DEVICES. Empirically, mutating CVD via os.environ from
    inside the worker process causes the MPS-routed libcuda to see 0 GPUs
    (driver/container-toolkit specific behavior). Required workflow with MPS:
    set CVD in the shell BEFORE invoking torchrun. opt_prime then validates
    that --num-gpus / --gpu-ids agree with the shell-set CVD but does not
    change it.
    """
    global _resolve_snapshot

    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    # Diagnostic banner — prints once per process. Helps verify that the
    # up-to-date code is loaded. Limited to rank 0 to avoid noise.
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[opt_prime][MPS] DIAG: pid={os.getpid()}, "
              f"env CUDA_VISIBLE_DEVICES={env_cvd!r}, "
              f"--num-gpus={num_gpus}, --gpu-ids={gpu_ids!r}, "
              f"use_mps={use_mps}",
              file=sys.stderr)

    # --num-gpus / --gpu-ids both given: --gpu-ids takes precedence
    if num_gpus is not None and gpu_ids is not None:
        _warn("both --num-gpus and --gpu-ids specified; using --gpu-ids and "
              "ignoring --num-gpus.")
        num_gpus = None  # disable further num_gpus path

    # Validate --gpu-ids syntax up front
    parsed_gpu_ids: Optional[list] = None
    if gpu_ids is not None:
        try:
            parsed_gpu_ids = _parse_gpu_ids(gpu_ids)
        except ValueError as e:
            _abort(str(e), rank_zero_only=True)
        if not parsed_gpu_ids:
            _abort("--gpu-ids is empty.", rank_zero_only=True)

    # Validate --num-gpus
    if num_gpus is not None and num_gpus <= 0:
        _abort(f"--num-gpus must be positive, got {num_gpus}.",
               rank_zero_only=True)

    # Physical GPU validation — applies whether we're overriding env CVD or
    # setting it fresh. nvidia-smi reports all physical GPUs (independent of
    # CVD), so we can verify the requested IDs exist before changing CVD.
    # Skipped silently when nvidia-smi is unavailable (phys==0).
    if parsed_gpu_ids is not None:
        phys = _physical_gpu_count()
        if phys > 0:
            for g in parsed_gpu_ids:
                if g < 0 or g >= phys:
                    _abort(f"--gpu-ids contains id {g} which is out of range "
                           f"[0, {phys}). Physical GPU count on this node is "
                           f"{phys}.", rank_zero_only=True)
    if num_gpus is not None:
        phys = _physical_gpu_count()
        if phys > 0 and num_gpus > phys:
            _abort(f"--num-gpus={num_gpus} exceeds physical GPU count "
                   f"({phys}) on this node.", rank_zero_only=True)

    def _set_cvd_with_snapshot(new_cvd: str) -> None:
        """Snapshot the current CVD (or its absence) and apply the new value.

        Idempotent w.r.t. _resolve_snapshot: only the first change is recorded
        so restore_visible_devices() returns to the pre-opt_prime state.
        """
        if "CUDA_VISIBLE_DEVICES" not in _resolve_snapshot:
            _resolve_snapshot["CUDA_VISIBLE_DEVICES"] = env_cvd
        os.environ["CUDA_VISIBLE_DEVICES"] = new_cvd

    def _reexec_with_new_cvd(new_cvd: str, reason: str) -> None:
        """Replace the current process with a fresh one that has new_cvd in env.

        This is the only reliable way to change CUDA_VISIBLE_DEVICES: the CUDA
        driver caches CVD at process start (libcuda reads getenv() once and may
        not honor later changes via os.environ). Re-exec gives libcuda a fresh
        process where CVD is correct from the very beginning.

        Compatibility:
        - PID is preserved across execv → torchrun's worker monitoring is
          unaffected
        - File descriptors (stdout/stderr) are inherited → torchrun still
          captures output
        - Env vars (LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, etc.) are
          inherited → NCCL rendezvous coordination is preserved

        Re-exec is idempotent: in the new process, env CVD == --gpu-ids/
        --num-gpus, so resolve_visible_devices takes the no-op branch and
        does not re-exec again.
        """
        rank = int(os.environ.get("RANK", "0"))
        # Print on ALL ranks so user sees re-exec is happening; multiple lines
        # are acceptable here because the message is critical for diagnostics
        # and only fires once per worker.
        print(f"[opt_prime][MPS] >>> RE-EXEC <<< pid={os.getpid()}, "
              f"rank={rank}: setting CUDA_VISIBLE_DEVICES='{new_cvd}' and "
              f"calling os.execv. Reason: {reason}",
              file=sys.stderr, flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = new_cvd
        # Sentinel marker so the new process can confirm via DIAG that it is
        # the post-execv invocation.
        os.environ["_OPT_PRIME_MPS_REEXEC"] = str(int(os.environ.get(
            "_OPT_PRIME_MPS_REEXEC", "0")) + 1)
        # Best-effort flush of any buffered output before exec
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Replace this process with a fresh Python invocation. Same argv,
        # same env (with the CVD we just set). Never returns.
        os.execv(sys.executable, [sys.executable] + sys.argv)

    if env_cvd is not None:
        # env CVD is set
        env_list_sorted = _normalize_cvd(env_cvd)

        if parsed_gpu_ids is not None:
            arg_list_sorted = sorted(parsed_gpu_ids)
            if env_list_sorted == arg_list_sorted:
                # Case 2a: same selection — keep CVD
                _info(f"CVD '{env_cvd}' matches --gpu-ids "
                      f"(exposes {len(env_list_sorted)} GPU(s)).")
            else:
                # Case 2b: --gpu-ids differs from env CVD.
                if use_mps:
                    # MPS strict mode: cannot change CVD reliably.
                    _abort(
                        f"--use-mps requires CUDA_VISIBLE_DEVICES to be set "
                        f"in the shell, and it must match --gpu-ids. "
                        f"Got env CUDA_VISIBLE_DEVICES='{env_cvd}' but "
                        f"--gpu-ids='{gpu_ids}'.\n"
                        f"  Fix: export CUDA_VISIBLE_DEVICES='{gpu_ids}' "
                        f"before torchrun, OR drop --gpu-ids to keep the "
                        f"shell-set CVD.\n"
                        f"  Reason: mutating CVD via os.environ inside the "
                        f"worker process is not honored by libcuda when MPS "
                        f"is active (driver/container-toolkit specific).",
                        rank_zero_only=True)
                # Non-MPS path: re-exec to apply new CVD cleanly.
                new_cvd = ",".join(str(g) for g in parsed_gpu_ids)
                _reexec_with_new_cvd(
                    new_cvd,
                    f"--gpu-ids='{gpu_ids}' overrides "
                    f"CUDA_VISIBLE_DEVICES='{env_cvd}'",
                )
                # never reached
            return

        if num_gpus is not None:
            if len(env_list_sorted) == num_gpus:
                # Case 3a: count matches — keep CVD (preserves user's selection)
                _info(f"CVD '{env_cvd}' matches --num-gpus={num_gpus}; "
                      f"keeping existing GPU selection.")
            else:
                # Case 3b: count differs.
                if use_mps:
                    _abort(
                        f"--use-mps requires CUDA_VISIBLE_DEVICES to be set "
                        f"in the shell, and its count must match --num-gpus. "
                        f"Got env CUDA_VISIBLE_DEVICES='{env_cvd}' "
                        f"({len(env_list_sorted)} GPU(s)) but "
                        f"--num-gpus={num_gpus}.\n"
                        f"  Fix: align them in the shell, e.g. "
                        f"`export CUDA_VISIBLE_DEVICES=0,1,...,{num_gpus-1}` "
                        f"before torchrun.\n"
                        f"  Reason: mutating CVD via os.environ inside the "
                        f"worker process is not honored by libcuda when MPS "
                        f"is active (driver/container-toolkit specific).",
                        rank_zero_only=True)
                # Non-MPS path: re-exec.
                new_cvd = ",".join(str(i) for i in range(num_gpus))
                _reexec_with_new_cvd(
                    new_cvd,
                    f"--num-gpus={num_gpus} overrides "
                    f"CUDA_VISIBLE_DEVICES='{env_cvd}' "
                    f"({len(env_list_sorted)} GPU(s))",
                )
                # never reached
            return

        # Case 1: CVD set, no args. Nothing to do.
        return

    # env CVD is unset (physical-range validation already done above)
    if parsed_gpu_ids is not None:
        # Case 4: --gpu-ids, env CVD unset.
        if use_mps:
            _abort(
                f"--use-mps requires CUDA_VISIBLE_DEVICES to be set in the "
                f"shell. opt_prime cannot reliably set CVD via os.environ "
                f"when MPS is active (driver/container-toolkit specific).\n"
                f"  Fix: export CUDA_VISIBLE_DEVICES='{gpu_ids}' before "
                f"torchrun, then keep --gpu-ids='{gpu_ids}' for opt_prime "
                f"validation (or omit --gpu-ids entirely).",
                rank_zero_only=True)
        new_cvd = ",".join(str(g) for g in parsed_gpu_ids)
        _set_cvd_with_snapshot(new_cvd)
        _info(f"CUDA_VISIBLE_DEVICES set to '{new_cvd}' (from --gpu-ids).")
        return

    if num_gpus is not None:
        # Case 5: --num-gpus, env CVD unset.
        if use_mps:
            default_cvd = ",".join(str(i) for i in range(num_gpus))
            _abort(
                f"--use-mps requires CUDA_VISIBLE_DEVICES to be set in the "
                f"shell. opt_prime cannot reliably set CVD via os.environ "
                f"when MPS is active (driver/container-toolkit specific).\n"
                f"  Fix: export CUDA_VISIBLE_DEVICES='{default_cvd}' before "
                f"torchrun, then keep --num-gpus={num_gpus} for opt_prime "
                f"validation (or omit it entirely).",
                rank_zero_only=True)
        new_cvd = ",".join(str(i) for i in range(num_gpus))
        _set_cvd_with_snapshot(new_cvd)
        _info(f"CUDA_VISIBLE_DEVICES set to '{new_cvd}' (from --num-gpus={num_gpus}).")
        return

    # Case 6: nothing to do


def restore_visible_devices() -> None:
    """Undo the CVD change made by resolve_visible_devices(), if any.

    Idempotent. Safe to call multiple times.
    """
    global _resolve_snapshot
    if "CUDA_VISIBLE_DEVICES" in _resolve_snapshot:
        original = _resolve_snapshot.pop("CUDA_VISIBLE_DEVICES")
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


# ---------------------------------------------------------------------------
# File barrier — used to synchronize local ranks before NCCL is initialized
# ---------------------------------------------------------------------------

_READY_SUFFIX = ".opt_prime_mps_ready"
_ABORT_SUFFIX = ".opt_prime_mps_abort"


def _ready_path(pipe_dir: str) -> str:
    return os.path.join(pipe_dir, _READY_SUFFIX)


def _abort_path(pipe_dir: str) -> str:
    return os.path.join(pipe_dir, _ABORT_SUFFIX)


def signal_mps_ready(pipe_dir: str) -> None:
    """Called by local rank 0 after MPS daemon is up and running."""
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        with open(_ready_path(pipe_dir), "w") as f:
            f.write(f"{os.getpid()}\n")
    except OSError as e:
        logger.warning(f"signal_mps_ready: failed to create ready file: {e}")


def signal_mps_abort(pipe_dir: str) -> None:
    """Called by local rank 0 if MPS setup fails. Other ranks read this signal."""
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        with open(_abort_path(pipe_dir), "w") as f:
            f.write(f"{os.getpid()}\n")
    except OSError as e:
        logger.warning(f"signal_mps_abort: failed to create abort file: {e}")


def wait_for_mps_ready(pipe_dir: str, timeout_s: float = 60.0,
                       poll_interval_s: float = 0.1) -> None:
    """Block until MPS daemon is signaled ready by local rank 0.

    Called by non-zero local ranks. Aborts if abort signal is seen or timeout.
    """
    deadline = time.monotonic() + timeout_s
    ready = _ready_path(pipe_dir)
    abort = _abort_path(pipe_dir)
    while time.monotonic() < deadline:
        if os.path.exists(abort):
            print(f"[opt_prime][MPS] Abort signal received from local rank 0. "
                  f"Exiting.", file=sys.stderr)
            sys.exit(1)
        if os.path.exists(ready):
            return
        time.sleep(poll_interval_s)
    print(f"[opt_prime][MPS] Timed out ({timeout_s}s) waiting for MPS daemon "
          f"to become ready.", file=sys.stderr)
    sys.exit(1)


def cleanup_barrier_files(pipe_dir: str) -> None:
    """Remove ready/abort sentinel files. Idempotent."""
    for p in (_ready_path(pipe_dir), _abort_path(pipe_dir)):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"cleanup_barrier_files: failed to unlink {p}: {e}")


# ---------------------------------------------------------------------------
# MPS availability detection
# ---------------------------------------------------------------------------

def _mps_daemon_running(pipe_dir: str) -> bool:
    """Best-effort check whether an MPS daemon is already running.

    Looks for the pipe directory's control socket and probes pgrep.
    Returns True if either signal is present.
    """
    # 1) control socket presence (most reliable when MPS daemon started cleanly)
    if os.path.exists(os.path.join(pipe_dir, "control")):
        return True
    # 2) Process check
    try:
        out = subprocess.run(
            ["pgrep", "-x", "nvidia-cuda-mps-server"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    try:
        out = subprocess.run(
            ["pgrep", "-x", "nvidia-cuda-mps-control"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False


# ---------------------------------------------------------------------------
# MPSManager
# ---------------------------------------------------------------------------

class MPSManager:
    """Per-node lifecycle manager for the MPS daemon.

    Only local rank 0 should instantiate and use this class. Other ranks
    rely on file barrier (wait_for_mps_ready) to synchronize.

    Lifecycle:
        mgr = MPSManager(pipe_dir, log_dir, thread_pct)
        ok, msg = MPSManager.check_availability()
        if not ok: ...abort...
        mgr.start()                # spawns daemon if not present
        mgr.register_cleanup()     # atexit + signal handlers
        # ...inference runs...
        mgr.stop()                 # idempotent; restores env

    Env snapshot/restore:
        Only env vars that this manager set are restored on stop().
        User-exported values are never overwritten and never restored.
    """

    def __init__(self, pipe_dir: str = "/tmp/nvidia-mps",
                 log_dir: str = "/tmp/nvidia-mps-log",
                 thread_pct: Optional[int] = None):
        self.pipe_dir = pipe_dir
        self.log_dir = log_dir
        self.thread_pct = thread_pct

        self._owned: bool = False     # True if we started the daemon
        self._stopped: bool = False
        self._cleanup_registered: bool = False
        self._lock = threading.Lock()
        self._env_snapshot: dict = {}  # {key: original_value_or_None}
        self._prev_signal_handlers: dict = {}  # {signum: prev_handler}

    # ------------------- availability -------------------

    @staticmethod
    def check_availability() -> tuple:
        """Return (available: bool, error_message: str).

        IMPORTANT: This method MUST NOT call torch.cuda.* (or anything that
        initializes the CUDA driver in this process). Doing so would create
        a direct CUDA context in rank 0 BEFORE the MPS daemon is started,
        which can corrupt the daemon's GPU ownership for subsequent clients.
        We use nvidia-smi (a pure query, no CUDA driver init) for all probing.

        Required:
          - nvidia-cuda-mps-control binary in PATH
          - nvidia-smi callable
          - All visible GPUs have compute capability >= 7.0 (Volta+)
        """
        # 1) nvidia-cuda-mps-control binary
        if shutil.which("nvidia-cuda-mps-control") is None:
            return False, ("nvidia-cuda-mps-control not found in PATH. "
                           "Install NVIDIA CUDA toolkit (or load CUDA module).")

        # 2) nvidia-smi callable
        if shutil.which("nvidia-smi") is None:
            return False, "nvidia-smi not found in PATH."

        # 3) Query physical GPUs (index, name, compute_cap) via nvidia-smi.
        # This does NOT initialize the CUDA driver.
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0:
                return False, f"nvidia-smi failed: {out.stderr.strip()}"
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"nvidia-smi could not be invoked: {e}"

        lines = [line.strip() for line in out.stdout.strip().split("\n")
                 if line.strip()]
        if not lines:
            return False, "No NVIDIA GPUs detected via nvidia-smi."

        # 4) Filter by CVD if set, then verify compute capability of visible GPUs
        visible_indices = _parse_visible_gpu_indices()
        found_visible = False
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            cap_str = parts[2]

            if visible_indices is not None and idx not in visible_indices:
                continue
            found_visible = True

            try:
                major = int(cap_str.split(".")[0])
            except (ValueError, IndexError):
                continue
            if major < 7:
                return False, (f"MPS requires Volta or newer (CC >= 7.0). "
                               f"GPU {idx} ({name}) has CC {cap_str}.")

        if not found_visible:
            return False, ("No GPUs visible after applying CUDA_VISIBLE_DEVICES. "
                           "Check that CVD references existing GPU indices.")

        # 5) Soft warning: many environments (NVIDIA driver/CUDA Container
        # Toolkit/MPS combinations) only support MPS reliably when
        # CUDA_VISIBLE_DEVICES is the contiguous default 0,1,...,N-1.
        # Non-default mappings (e.g., '4,5,6,7' or '0,2,4,6') have been
        # observed to cause libcuda to report 0 GPUs to NCCL/lazy_init even
        # though NVML and torch.cuda.device_count() report the correct count.
        # We do NOT abort because some environments may support it; we WARN
        # with a clear hint to the user.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            try:
                ids = _parse_gpu_ids(cvd)
                expected = list(range(len(ids)))
                if ids != expected:
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(
                            f"\n[opt_prime][MPS] WARNING: "
                            f"CUDA_VISIBLE_DEVICES='{cvd}' is not the default "
                            f"contiguous form '0,1,...,{len(ids)-1}'.\n"
                            f"[opt_prime][MPS]          Some NVIDIA driver / "
                            f"CUDA Container Toolkit / MPS combinations only "
                            f"support MPS with the default CVD pattern.\n"
                            f"[opt_prime][MPS]          If you encounter "
                            f"'ProcessGroupNCCL ... no GPUs found' or "
                            f"'device < num_gpus INTERNAL ASSERT FAILED' "
                            f"errors, try:\n"
                            f"[opt_prime][MPS]            export "
                            f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in expected)}\n"
                            f"[opt_prime][MPS]          For non-contiguous "
                            f"GPU subsets, use container-level GPU mapping "
                            f"(e.g., docker run --gpus '\"device={cvd}\"' ...).\n",
                            file=sys.stderr, flush=True,
                        )
            except ValueError:
                pass

        return True, ""

    # ------------------- env helpers -------------------

    def _set_env(self, key: str, value: str) -> None:
        """Set env var, snapshotting the original value (only on first change)."""
        if key not in self._env_snapshot:
            self._env_snapshot[key] = os.environ.get(key, None)
        os.environ[key] = value

    def _restore_env(self) -> None:
        """Restore env vars to pre-start state. Idempotent."""
        for key, original in list(self._env_snapshot.items()):
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        self._env_snapshot.clear()

    # ------------------- start / stop -------------------

    def start(self) -> None:
        """Start the MPS daemon if not already running.

        Sets CUDA_MPS_PIPE_DIRECTORY, CUDA_MPS_LOG_DIRECTORY,
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (if configured), and
        NCCL_LAUNCH_MODE=PARALLEL (if not already set by user).

        Daemon inherits CUDA_VISIBLE_DEVICES at spawn time.
        """
        with self._lock:
            # Set MPS-related env (snapshotted, will be restored on stop)
            self._set_env("CUDA_MPS_PIPE_DIRECTORY", self.pipe_dir)
            self._set_env("CUDA_MPS_LOG_DIRECTORY", self.log_dir)
            if self.thread_pct is not None:
                self._set_env("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                              str(self.thread_pct))
            # NCCL_LAUNCH_MODE: only set if user did not set
            if "NCCL_LAUNCH_MODE" not in os.environ:
                self._set_env("NCCL_LAUNCH_MODE", "PARALLEL")

            try:
                os.makedirs(self.pipe_dir, exist_ok=True)
                os.makedirs(self.log_dir, exist_ok=True)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to create MPS pipe/log directory: {e}") from e

            # Detect existing daemon
            if _mps_daemon_running(self.pipe_dir):
                # Try a health probe; if stale, clean up and retry
                if self._daemon_responsive():
                    print(f"[opt_prime][MPS] Found running MPS daemon at "
                          f"{self.pipe_dir}; reusing (external ownership).",
                          file=sys.stderr)
                    self._owned = False
                    return
                else:
                    print(f"[opt_prime][MPS] Detected stale MPS daemon at "
                          f"{self.pipe_dir}. Attempting cleanup before "
                          f"restart.", file=sys.stderr)
                    self._force_cleanup_stale_daemon()

            # Spawn new daemon. It forks to background and returns immediately.
            try:
                # Pass a copy of os.environ so daemon inherits CVD and MPS vars.
                proc = subprocess.run(
                    ["nvidia-cuda-mps-control", "-d"],
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                raise RuntimeError(
                    f"Failed to start MPS daemon: {e}") from e

            if proc.returncode != 0:
                raise RuntimeError(
                    f"nvidia-cuda-mps-control -d failed: "
                    f"stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}")

            # Brief grace period for daemon initialization
            time.sleep(0.5)
            self._owned = True
            print(f"[opt_prime][MPS] MPS daemon started "
                  f"(pipe_dir={self.pipe_dir}).", file=sys.stderr)

    def _daemon_responsive(self, timeout_s: float = 2.0) -> bool:
        """Return True if 'get_server_list' query succeeds within timeout."""
        try:
            proc = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="get_server_list\n",
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
            # Successful response is non-empty stdout with no error string.
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _force_cleanup_stale_daemon(self) -> None:
        """Best-effort cleanup of a stale daemon: try quit, then pkill (own user)."""
        try:
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                capture_output=True,
                text=True,
                timeout=5,
                env=os.environ.copy(),
            )
        except (subprocess.TimeoutExpired, OSError):
            pass
        # Force kill stragglers (only this user's processes)
        for proc_name in ("nvidia-cuda-mps-server", "nvidia-cuda-mps-control"):
            try:
                subprocess.run(
                    ["pkill", "-u", str(os.geteuid()), "-x", proc_name],
                    capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
        # Remove control socket if present
        ctrl = os.path.join(self.pipe_dir, "control")
        try:
            os.unlink(ctrl)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        time.sleep(0.5)

    def stop(self) -> None:
        """Stop the MPS daemon (if we started it) and restore env. Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

            # 1) Shut down daemon if we own it
            if self._owned:
                try:
                    subprocess.run(
                        ["nvidia-cuda-mps-control"],
                        input="quit\n",
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env=os.environ.copy(),
                    )
                    print(f"[opt_prime][MPS] MPS daemon shut down.",
                          file=sys.stderr)
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning(f"MPSManager.stop: failed to quit daemon: {e}")
                self._owned = False

            # 2) Remove sentinel barrier files (only if we own pipe dir)
            cleanup_barrier_files(self.pipe_dir)

            # 3) Restore env vars
            self._restore_env()

            # 4) Restore signal handlers
            for signum, prev in self._prev_signal_handlers.items():
                try:
                    signal.signal(signum, prev)
                except (ValueError, OSError):
                    pass
            self._prev_signal_handlers.clear()

    # ------------------- cleanup registration -------------------

    def register_cleanup(self) -> None:
        """Install atexit + signal handlers that call stop() on shutdown."""
        if self._cleanup_registered:
            return
        self._cleanup_registered = True

        atexit.register(self.stop)

        def _handler(signum, frame):
            self.stop()
            # Re-raise default behavior
            prev = self._prev_signal_handlers.get(signum)
            if callable(prev):
                try:
                    prev(signum, frame)
                    return
                except SystemExit:
                    raise
                except Exception:
                    pass
            # Fallback: exit non-zero (preserves convention for SIGINT/SIGTERM)
            sys.exit(128 + signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.signal(signum, _handler)
                self._prev_signal_handlers[signum] = prev
            except (ValueError, OSError):
                # Cannot install (e.g., not in main thread). Best effort only.
                pass


# ---------------------------------------------------------------------------
# High-level setup helper for example entry points
# ---------------------------------------------------------------------------

def setup_mps_for_inference(use_mps: bool,
                            pipe_dir: str = "/tmp/nvidia-mps",
                            log_dir: str = "/tmp/nvidia-mps-log",
                            thread_pct: Optional[int] = None,
                            barrier_timeout_s: float = 60.0
                            ) -> Optional["MPSManager"]:
    """Drive the per-node MPS setup flow described in design 4.1 / 5.3.

    To be called from inference example entry points BEFORE any CUDA work
    and BEFORE dist.init_process_group.

    Behavior:
      - If use_mps is False: returns None (no-op).
      - On local rank 0: validates availability, starts daemon, registers
        cleanup, signals ready via file barrier.
      - On other local ranks: waits for ready (or abort) signal.
      - Aborts (sys.exit(1)) on any unrecoverable error.

    Returns:
      MPSManager instance on local rank 0 (caller may keep a reference but
      does not need to call stop() — atexit handles it).
      None on other ranks or when use_mps is False.
    """
    if not use_mps:
        return None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        # Clear any stale barrier files from previous runs
        cleanup_barrier_files(pipe_dir)

        ok, msg = MPSManager.check_availability()
        if not ok:
            print(f"[opt_prime][MPS] MPS unavailable: {msg}", file=sys.stderr)
            print(f"[opt_prime][MPS] Aborting because --use-mps was requested "
                  f"but MPS cannot be enabled in this environment.",
                  file=sys.stderr)
            # Make sure the pipe_dir exists so abort signal can be written
            try:
                os.makedirs(pipe_dir, exist_ok=True)
            except OSError:
                pass
            signal_mps_abort(pipe_dir)
            sys.exit(1)

        mgr = MPSManager(pipe_dir=pipe_dir, log_dir=log_dir, thread_pct=thread_pct)
        try:
            mgr.start()
        except Exception as e:
            print(f"[opt_prime][MPS] Failed to start MPS daemon: {e}",
                  file=sys.stderr)
            try:
                os.makedirs(pipe_dir, exist_ok=True)
            except OSError:
                pass
            signal_mps_abort(pipe_dir)
            sys.exit(1)

        mgr.register_cleanup()
        signal_mps_ready(pipe_dir)
        return mgr

    # non-zero local rank: wait for daemon ready
    wait_for_mps_ready(pipe_dir, timeout_s=barrier_timeout_s)
    return None
