#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Accelerator discovery: one discoverer per device kind, registered by name, run lazily.

Discoverers read device nodes / sysfs only; they do not open devices, start SDK runtimes or call vendor CLIs, so
discovery is cheap and never disturbs another process's device. Whether a device is idle is a separate,
vendor-specific question (see `lease.acquire(..., probe=...)`).

Built-in kinds:
  cuda - torch.cuda (only when torch is already imported or CUDA_VISIBLE_DEVICES is set: no CUDA init otherwise)
  rngd - FuriosaAI RNGD, /dev/rngd/npu<N>mgmt
  atom - Rebellions ATOM, /sys/class/rebellions/rsd/rsd_group (physical rbln<N>, host group 0)
  cpu  - the host, one device per NUMA node
A new kind registers `@register_discoverer("kind")` returning a list of DeviceInfo.
"""
import csv
import io
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .capabilities import check_capabilities

_ROOT = os.environ.get("OPT_PRIME_DEVICE_ROOT", "")      # test hook: prefix for /dev and /sys


def _p(path: str) -> str:
    return _ROOT + path


@dataclass(frozen=True)
class DeviceInfo:
    kind: str                       # "cuda", "rngd", "atom", "cpu", ...
    index: int                      # the kind's own numbering (rngd:1 = /dev/rngd/npu1*, atom:3 = rbln3)
    vendor: str = ""
    memory_bytes: Optional[int] = None
    numa_node: Optional[int] = None
    capabilities: frozenset = frozenset()
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "capabilities", check_capabilities(self.capabilities))

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.index}"


_DISCOVERERS: Dict[str, Callable[[], List[DeviceInfo]]] = {}


def register_discoverer(kind: str):
    def deco(fn):
        _DISCOVERERS[kind] = fn
        return fn
    return deco


def discover(kinds=None) -> List[DeviceInfo]:
    """devices of the given kinds (default: every registered kind), kinds whose discoverer fails are skipped."""
    out = []
    for kind in (kinds or sorted(_DISCOVERERS)):
        if kind not in _DISCOVERERS:
            raise KeyError(f"no discoverer for {kind!r}; registered: {sorted(_DISCOVERERS)}")
        try:
            out.extend(_DISCOVERERS[kind]())
        except Exception as e:                       # a broken driver must not hide the other kinds
            print(f"[opt_prime.devices] {kind} discovery failed: {type(e).__name__}: {e}", file=sys.stderr)
    return out


def _numa_of_pci(bdf: Optional[str]) -> Optional[int]:
    if not bdf:
        return None
    try:
        v = int(open(_p(f"/sys/bus/pci/devices/{bdf}/numa_node")).read())
        return v if v >= 0 else None
    except (OSError, ValueError):
        return None


@register_discoverer("cuda")
def _cuda():
    if "torch" not in sys.modules and "CUDA_VISIBLE_DEVICES" not in os.environ:
        return []                                    # do not import torch / init CUDA just to look
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bdf = None
        if hasattr(p, "pci_bus_id"):
            bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        out.append(DeviceInfo("cuda", i, "nvidia" if torch.version.cuda else "amd", p.total_memory,
                              _numa_of_pci(bdf), frozenset({"prefill", "decode", "experts", "resident_weights"}),
                              {"model": p.name, "pci": bdf}))
    return out


@register_discoverer("rngd")
def _rngd():
    d = _p("/dev/rngd")
    if not os.path.isdir(d):
        return []
    idx = sorted({int(n[3:-4]) for n in os.listdir(d) if n.startswith("npu") and n.endswith("mgmt")
                  and n[3:-4].isdigit()})
    out = []
    for i in idx:
        bdf = None
        try:      # /sys/class/rngd_mgmt/rngd!npu<N>mgmt/busname = 0000:50:00.0
            bdf = open(_p(f"/sys/class/rngd_mgmt/rngd!npu{i}mgmt/busname")).read().strip() or None
        except OSError:
            pass
        out.append(DeviceInfo("rngd", i, "furiosa", 48 << 30, _numa_of_pci(bdf),
                              frozenset({"prefill", "decode", "experts", "graph_compile", "custom_kernels",
                                         "resident_weights", "p2p_dma"}),
                              {"model": "RNGD", "pci": bdf, "dev": f"/dev/rngd/npu{i}mgmt"}))
    return out


@register_discoverer("atom")
def _atom():
    path = _p("/sys/class/rebellions/rsd/rsd_group")
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(io.StringIO(open(path).read())))
    out = []
    for r in rows:
        dev = r.get("DEVICE", "")
        if r.get("GROUP") != "0" or not dev.startswith("rbln") or not dev[4:].isdigit():
            continue                                 # only host group 0 (as the PD worker requires)
        out.append(DeviceInfo("atom", int(dev[4:]), "rebellions", None, None,
                              frozenset({"prefill", "decode", "graph_compile", "resident_weights", "p2p_dma"}),
                              {"model": "ATOM", "visible_npu_id": r.get("NPU ID"), "dev": f"/dev/{dev}"}))
    return out


@register_discoverer("cpu")
def _cpu():
    nodes = []
    base = _p("/sys/devices/system/node")
    if os.path.isdir(base):
        nodes = sorted(int(n[4:]) for n in os.listdir(base) if n.startswith("node") and n[4:].isdigit())
    mem = None
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass
    return [DeviceInfo("cpu", 0, "host", mem, None, frozenset({"experts", "decode"}),
                       {"numa_nodes": nodes or [0], "cores": os.cpu_count()})]
