#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Devices shared by every Optimus execution path (GPU pipeline, PD engines, MoE expert tiers).

* discovery    - what accelerators this host exposes (`discover()`), one discoverer per device kind
* capabilities - the shared vocabulary engines and backends declare (`CAPABILITIES`)
* leases       - which role owns which chip (`acquire()`): exclusive across Optimus processes on the host

Importing this package never initializes CUDA or an NPU SDK.
"""
from .capabilities import CAPABILITIES, ROLES, check_capabilities  # noqa: F401
from .discovery import DeviceInfo, discover, register_discoverer  # noqa: F401
from .lease import DeviceBusy, DeviceLease, acquire, held, register_probe  # noqa: F401
