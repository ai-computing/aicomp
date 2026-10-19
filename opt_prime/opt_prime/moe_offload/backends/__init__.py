#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Device side: one ExpertBackend per device type, looked up by name ("gpu"/"cuda"/"xpu", "cpu", "rngd", ...)."""
from .base import ExpertBackend, Handle, get_backend, register_backend, registered  # noqa: F401
from . import atom, cpu, cuda, rngd, rngd_opt  # noqa: F401  (register the built-in backends)
from .npu_graph import GraphNPUBackend  # noqa: F401  (base for graph-compiled NPUs: RNGD, later ATOM, ...)
