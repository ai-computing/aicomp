#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Native helpers, built with gcc on first use into ~/.cache/opt_prime and loaded with ctypes (no build tooling
beyond gcc):

* csrc/host_experts.c - host experts kernel; `load()` returns None when the CPU lacks AVX-512 BF16 or the build
  fails, and callers use the PyTorch path.
* csrc/dispatch.c - host dispatcher of the blocking tiers (decode fast path); `load_dispatch()`, None if unavailable.
"""
import ctypes
import hashlib
import os
import subprocess
import threading

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SRC = os.path.join(_CSRC, "host_experts.c")
_LIB = None
_TRIED = False
_DISP, _DISP_TRIED = None, False
_LOCK = threading.Lock()


def _build(src_path=_SRC, name="host_experts", flags=("-fopenmp", "-lm")) -> str:
    src = open(src_path, "rb").read()
    tag = hashlib.sha1(src + repr(flags).encode()).hexdigest()[:12]
    out_dir = os.path.join(os.path.expanduser(os.environ.get("OPT_PRIME_CACHE", "~/.cache/opt_prime")), "moe_offload")
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, f"{name}_{tag}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", src_path, "-o", tmp, *flags],
                       check=True, capture_output=True)
        os.replace(tmp, so)
    return so


def load_dispatch():
    """csrc/dispatch.c with argtypes set, or None."""
    global _DISP, _DISP_TRIED
    with _LOCK:
        if _DISP_TRIED:
            return _DISP
        _DISP_TRIED = True
        try:
            lib = ctypes.CDLL(_build(os.path.join(_CSRC, "dispatch.c"), "moe_dispatch", ("-lpthread",)))
            P, I = ctypes.c_void_p, ctypes.c_int
            lib.moe_disp_new.restype = P
            lib.moe_disp_start.argtypes = [P, I, I, I, P, P, P, P, P, I, P, P, P, P]
            lib.moe_disp_wait.argtypes = [P, I, I, P]
            lib.moe_cpu_ctx_new.restype = P
            lib.moe_cpu_ctx_new.argtypes = [P, P, P, I, I, I]
            lib.moe_pipe_new.restype = P
            lib.moe_pipe_new.argtypes = [I, I, I]
            lib.moe_pipe_ctx_new.restype = P
            lib.moe_pipe_ctx_new.argtypes = [P, I]
            lib.moe_pipe_lock.argtypes = [P]
            lib.moe_pipe_unlock.argtypes = [P]
            _DISP = lib
        except Exception as e:
            print(f"[moe_offload] native dispatcher unavailable: {type(e).__name__}: {e}")
            _DISP = None
        return _DISP


def fn_addr(lib, name) -> int:
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value


def load():
    global _LIB, _TRIED
    with _LOCK:
        if _TRIED:
            return _LIB
        _TRIED = True
        try:
            lib = ctypes.CDLL(_build())
            if not lib.moe_host_supported():
                return None
            P, I, I64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
            lib.moe_host_forward.argtypes = [I, I, I, I, I, P, P, P, P, P, P, I]
            lib.moe_host_forward_bw.argtypes = [I, I, I, I, I, P, P, P, P, P, P, I]
            lib.moe_host_place.argtypes = [P, P, I64, I64, I]
            _LIB = lib
        except Exception:
            _LIB = None
        return _LIB
