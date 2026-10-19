"""Resolve an explicitly selected physical ATOM without changing RSD groups."""

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
