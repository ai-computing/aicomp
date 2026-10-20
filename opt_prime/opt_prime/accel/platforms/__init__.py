#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Accelerators: what this host has, who holds which chip, and each vendor's implementations.

* registry      - `discover()`: devices from the discoverers below, or from an ai-rm accelerator provider (rm-accel)
* lease         - `acquire()`: one exclusive role per chip across Optimus processes, with vendor busy probes
* capabilities  - the vocabulary of roles and device capabilities
* <vendor>/     - cuda, cpu, rngd, atom: `device.py` (discovery, busy probe), `experts.py` (MoE expert tier),
                  `engine.py` (whole-model engine), `pd*.py` (PD prefill / decode adapters)
* npu_graph     - shared base of the graph-compiled NPU expert tiers (RNGD, ATOM)

Feature packages (moe, engines, pd, train) never import a vendor SDK; vendor modules here import theirs lazily, so
importing this package never initializes CUDA or an NPU SDK.
"""
from opt_prime.accel.platforms.capabilities import CAPABILITIES, ROLES, check_capabilities  # noqa: F401
from opt_prime.accel.platforms.registry import DeviceInfo, discover, register_discoverer  # noqa: F401
from opt_prime.accel.platforms.lease import DeviceBusy, DeviceLease, acquire, held, register_probe  # noqa: F401
from opt_prime.accel.platforms.atom import device as _atom_device  # noqa: F401,E402  (register discoverers/probes)
from opt_prime.accel.platforms.cpu import device as _cpu_device  # noqa: F401,E402
from opt_prime.accel.platforms.cuda import device as _cuda_device  # noqa: F401,E402
from opt_prime.accel.platforms.rngd import device as _rngd_device  # noqa: F401,E402


def load_expert_backends():
    """import every vendor's MoE expert tier so it registers with opt_prime.accel.moe.backends (lazy: the tiers
    import moe.backends, which calls this only when a backend is looked up)."""
    from opt_prime.accel.platforms.atom import experts as _a  # noqa: F401
    from opt_prime.accel.platforms.cpu import experts as _c  # noqa: F401
    from opt_prime.accel.platforms.cuda import experts as _g  # noqa: F401
    from opt_prime.accel.platforms.rngd import experts as _r, experts_opt as _ro  # noqa: F401
