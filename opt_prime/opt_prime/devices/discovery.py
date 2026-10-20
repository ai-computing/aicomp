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

ai-rm: when an rm-core accelerator provider is installed (rm-accel registers one in the entry-point group
`airm.accelerator_providers`), it decides which accelerators exist, so Optimus and the rest of ai-rm see the same
devices (restricted to those this process can address). Optimus adds what it needs on top - its kind:index names, role capabilities, NUMA node, device paths - by
matching the provider's records to the sysfs ones (PCI address, else index). Kinds the provider does not report
(and the host CPU) come from the built-in discoverers. OPT_PRIME_DEVICE_SOURCE=local skips the provider;
OPT_PRIME_ACCELERATOR_PROVIDER picks one by name when several are installed.
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
    kinds = list(kinds or sorted(_DISCOVERERS))
    out = []
    for kind in kinds:
        if kind not in _DISCOVERERS:
            raise KeyError(f"no discoverer for {kind!r}; registered: {sorted(_DISCOVERERS)}")
        try:
            out.extend(_DISCOVERERS[kind]())
        except Exception as e:                       # a broken driver must not hide the other kinds
            print(f"[opt_prime.devices] {kind} discovery failed: {type(e).__name__}: {e}", file=sys.stderr)
    accs = _provider_accelerators()
    return out if accs is None else _merge_provider(out, accs, kinds)


#: rm-core Accelerator.vendor -> Optimus device kind
VENDOR_KINDS = {"nvidia": "cuda", "amd": "cuda", "furiosa": "rngd", "rebellions": "atom"}
#: role capabilities of a kind when only the provider reports a device
_KIND_CAPS = {
    "cuda": frozenset({"prefill", "decode", "experts", "resident_weights"}),
    "rngd": frozenset({"prefill", "decode", "experts", "graph_compile", "custom_kernels", "resident_weights", "p2p_dma"}),
    "atom": frozenset({"prefill", "decode", "graph_compile", "resident_weights", "p2p_dma"}),
}


def _provider_accelerators():
    """rm_core.Accelerator list from an installed ai-rm accelerator provider, or None (none installed / disabled)."""
    if os.environ.get("OPT_PRIME_DEVICE_SOURCE", "auto") == "local":
        return None
    try:
        from rm_core import registry
    except ImportError:
        return None
    group = "airm.accelerator_providers"
    names = sorted(registry.available(group))
    if not names:
        return None
    name = os.environ.get("OPT_PRIME_ACCELERATOR_PROVIDER") or names[0]
    try:
        return list(registry.load(group, name).discover())
    except Exception as e:                           # fall back to sysfs rather than see no devices
        print(f"[opt_prime.devices] accelerator provider {name} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _norm_bdf(bdf) -> Optional[str]:
    return bdf.lower() if bdf else None


def _merge_provider(local: List[DeviceInfo], accs, kinds) -> List[DeviceInfo]:
    """provider records (which devices exist, model, memory, peak) + local records (capabilities, NUMA, paths)."""
    by_pci = {_norm_bdf(d.extra.get("pci")): d for d in local if d.extra.get("pci")}
    by_name = {d.name: d for d in local}
    merged, covered = [], set()
    for a in accs:
        kind = VENDOR_KINDS.get(a.vendor)
        if kind is None or kind not in kinds:
            continue
        try:
            index = int(a.id.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        bdf = _norm_bdf(a.extra.get("pci_bdf"))
        if bdf and any(d.kind == kind and d.extra.get("pci") for d in local):
            base = by_pci.get(bdf)                   # both sides know the PCI address: only that identifies it
        else:
            base = by_name.get(f"{kind}:{index}")
        if base is not None and base.kind == kind:
            index = base.index                       # Optimus numbering follows the device nodes
        elif any(d.kind == kind for d in local):
            continue                                 # not addressable here (e.g. outside CUDA_VISIBLE_DEVICES)
        else:
            base = None
        extra = dict(base.extra) if base else {"pci": a.extra.get("pci_bdf")}
        extra.update(model=a.model, airm_id=a.id, memory_kind=a.memory_kind, interconnect=list(a.interconnect),
                     peak_tflops=dict(a.peak_tflops))
        mem = (base.memory_bytes if base else None) or a.memory_bytes      # usable (local) before nominal
        merged.append(DeviceInfo(kind, index, a.vendor, mem,
                                 base.numa_node if base else _numa_of_pci(extra.get("pci")),
                                 base.capabilities if base else _KIND_CAPS.get(kind, frozenset()), extra))
        covered.add(kind)
    return merged + [d for d in local if d.kind not in covered]


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
