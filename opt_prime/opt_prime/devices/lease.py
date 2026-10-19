#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Device leases: which role owns which accelerator.

`acquire("rngd:1", role="experts", owner="moe_offload tier 1")` returns a DeviceLease and holds an advisory lock
file (~/.cache/opt_prime/devices/rngd1.lock, flock) until release, so two Optimus components - in this process or
another on the host - never drive the same chip (e.g. a furiosa-torch PD engine and an rngd-opt expert server).
The lock is advisory: vendor runtimes started outside Optimus do not see it; pass `probe=` to also check the
vendor's own notion of busy before taking the lease.

Each acquisition gets a generation number that increases per device across processes (stored in the lock file),
so a stale holder can be told from the current one. Kinds in SHARED_KINDS are not locked: the host CPU, and GPUs,
which several processes share by memory budget. NPUs whose runtime takes the whole chip (rngd, atom) are exclusive.
"""
import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .capabilities import ROLES

SHARED_KINDS = {"cpu", "cuda"}
_HELD: Dict[str, "DeviceLease"] = {}
_LOCK = threading.Lock()


class DeviceBusy(RuntimeError):
    pass


def _lock_dir():
    root = os.path.expanduser(os.environ.get("OPT_PRIME_CACHE", "~/.cache/opt_prime"))
    d = os.path.join(root, "devices")
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class DeviceLease:
    device: str                      # "rngd:1"
    role: str
    owner: str
    generation: int
    pid: int = field(default_factory=os.getpid)
    _fd: Optional[int] = field(default=None, repr=False)
    released: bool = False

    def release(self):
        with _LOCK:
            if self.released:
                return
            self.released = True
            if _HELD.get(self.device) is self:
                del _HELD[self.device]
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _parse(device):
    kind, _, idx = str(device).partition(":")
    if not kind or not idx.isdigit():
        raise ValueError(f"device must be '<kind>:<index>', got {device!r}")
    return kind, int(idx)


_PROBES: Dict[str, Callable[[str], Optional[str]]] = {}


def register_probe(kind: str):
    """default busy probe of a device kind: fn("kind:index") -> reason it is busy, or None."""
    def deco(fn):
        _PROBES[kind] = fn
        return fn
    return deco


@register_probe("rngd")
def _rngd_busy(name):
    """PEs held by any process (Optimus or not): /sys/class/rngd_mgmt/rngd!npu<N>mgmt/pe_occupancy (1 = held)."""
    from .discovery import _p
    idx = name.split(":")[1]
    try:
        occ = open(_p(f"/sys/class/rngd_mgmt/rngd!npu{idx}mgmt/pe_occupancy")).read().split()
    except OSError:
        return None                      # no sysfs (older driver / container): cannot tell, do not refuse
    held = [i for i, v in enumerate(occ) if v.strip() == "1"]
    return f"PEs {held} are held by another process (pe_occupancy)" if held else None


@register_probe("atom")
def _atom_busy(name):
    """rbln-smi: the physical card is busy when it is not 'normal', or when memory is in use and some context on its
    NPU id belongs to another process (memory held by this process - e.g. another tier of the same model - is not a
    reason to refuse). None when rbln-smi is not installed."""
    import shutil
    import subprocess
    if shutil.which("rbln-smi") is None:
        return None
    physical = "rbln" + name.split(":")[1]
    state = json.loads(subprocess.check_output(["rbln-smi", "--json"], text=True, timeout=30))
    devs = [d for d in state.get("devices", []) if d.get("device") == physical]
    if not devs:
        return f"{physical} not listed by rbln-smi"
    d = devs[0]
    if d.get("status") != "normal":
        return f"{physical} status {d.get('status')}"
    if not int(d.get("memory", {}).get("used", 0) or 0):
        return None
    npu = d.get("npu")
    others = [c for c in state.get("contexts", []) if str(c.get("npu")) == str(npu) and str(c.get("pid")) != str(os.getpid())]
    if others:
        return f"{physical}: contexts of pid {sorted({c.get('pid') for c in others})}"
    mine = [c for c in state.get("contexts", []) if str(c.get("pid")) == str(os.getpid())]
    return None if mine else f"{physical}: memory in use ({d['memory']['used']} B) by an unknown owner"


def acquire(device: str, role: str, owner: str, probe="auto") -> DeviceLease:
    """Lease `device` for `role`. Raises DeviceBusy if another Optimus component holds it, or if the probe returns a
    reason it is busy. probe: "auto" (the kind's registered probe, e.g. RNGD PE occupancy), None (no probe), or
    fn("kind:index") -> reason | None."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; roles: {sorted(ROLES)}")
    kind, idx = _parse(device)
    name = f"{kind}:{idx}"
    if kind in SHARED_KINDS:
        return DeviceLease(name, role, owner, 0)
    with _LOCK:
        cur = _HELD.get(name)
        if cur is not None:
            raise DeviceBusy(f"{name} is leased to {cur.owner} ({cur.role}) in this process")
        path = os.path.join(_lock_dir(), f"{kind}{idx}.lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                info = json.loads(os.pread(fd, 4096, 0) or b"{}")
            except ValueError:
                info = {}
            os.close(fd)
            raise DeviceBusy(f"{name} is leased by pid {info.get('pid')} ({info.get('owner')}, {info.get('role')})")
        try:
            if probe == "auto":
                probe = _PROBES.get(kind)
            if probe is not None:
                reason = probe(name)
                if reason:
                    raise DeviceBusy(f"{name}: {reason}")
            try:
                prev = json.loads(os.pread(fd, 4096, 0) or b"{}")
            except ValueError:
                prev = {}
            gen = int(prev.get("generation", 0)) + 1
            data = json.dumps(dict(pid=os.getpid(), owner=owner, role=role, generation=gen, since=time.time()))
            os.ftruncate(fd, 0)
            os.pwrite(fd, data.encode(), 0)
        except BaseException:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise
        lease = DeviceLease(name, role, owner, gen, _fd=fd)
        _HELD[name] = lease
        return lease


def held() -> Dict[str, DeviceLease]:
    """leases held by this process."""
    with _LOCK:
        return dict(_HELD)
