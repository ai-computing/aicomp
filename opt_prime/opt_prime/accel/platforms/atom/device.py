"""Rebellions ATOM: discovery (rsd_group, host group 0), the busy probe (rbln-smi) and the explicit selection of a
physical card for PD decode (select_idle_atom) - one place that reads RSD groups."""

import csv
import io
import json
import os
import re
from pathlib import Path
import subprocess


def select_idle_atom(physical="rbln3"):
    if not isinstance(physical, str) or not re.fullmatch(
        r"rbln(?:0|[1-9][0-9]*)", physical
    ):
        raise ValueError("unknown physical ATOM device")
    group_text = Path("/sys/class/rebellions/rsd/rsd_group").read_text()
    rows = list(csv.DictReader(io.StringIO(group_text)))
    matches = [r for r in rows if r["DEVICE"] == physical and r["GROUP"] == "0"]
    if len(matches) != 1:
        raise RuntimeError("selected physical card must be in host group0")
    visible = int(matches[0]["NPU ID"])
    state = json.loads(subprocess.check_output(["rbln-smi", "--json"], text=True))
    devices = [d for d in state["devices"] if d["device"] == physical]
    if (
        len(devices) != 1
        or devices[0]["status"] != "normal"
        or int(devices[0]["memory"]["used"])
    ):
        raise RuntimeError("selected card is not idle/normal")
    if any(int(c["npu"]) == visible for c in state["contexts"]):
        raise RuntimeError("selected visible NPU ID has an active context")
    os.environ["RBLN_DEVICES"] = str(visible)
    os.environ["RBLN_NPUS_PER_DEVICE"] = "1"
    os.environ.pop("RBLN_DEVICE_MAP", None)
    os.environ.pop("RBLN_DUMMY_DEVICE", None)
    return dict(physical=physical, visible=visible, groups=group_text, before=state)


def verify_atom_binding(selection):
    current = Path("/sys/class/rebellions/rsd/rsd_group").read_text()

    def canonical(text):
        return sorted(
            (r["GROUP"], r["NPU ID"], r["DEVICE"])
            for r in csv.DictReader(io.StringIO(text))
        )

    if canonical(current) != canonical(selection["groups"]):
        raise RuntimeError("RSD mapping changed during startup")
    paths = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            path = os.readlink(fd)
        except OSError:
            continue
        if path.startswith("/dev/rbln"):
            paths.append(path)
    if set(paths) != {f"/dev/{selection['physical']}"}:
        raise RuntimeError(f"unexpected physical ATOM descriptors: {paths}")
    return dict(
        selection,
        opened=paths,
        loaded=json.loads(subprocess.check_output(["rbln-smi", "--json"], text=True)),
    )


# -- discovery and busy probe (opt_prime.accel.platforms) ---------------------------------------------------

from opt_prime.accel.platforms.lease import register_probe  # noqa: E402
from opt_prime.accel.platforms.registry import DeviceInfo, _p, register_discoverer  # noqa: E402


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
    mine = [c for c in state.get("contexts", [])
            if str(c.get("npu")) == str(npu) and str(c.get("pid")) == str(os.getpid())]
    return None if mine else f"{physical}: memory in use ({d['memory']['used']} B) by an unknown owner"
