#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Kernel providers: which implementation runs a block (opt_prime.model block spec) on a device kind, and at what
cost. Planners ask `providers_for(block, device_kind)` and pick by cost; a provider is only offered for blocks its
`supports()` accepts, so a new device or kernel is one registration and never changes other code.

Built-in providers:
  torch        any block, cuda / cpu (reference and fallback: the model's own module)
  experts:<b>  routed experts through a moe_offload ExpertBackend <b> (gpu, cpu, rngd, rngd-opt, ...)
"""
from .registry import (BLOCK_KINDS, KernelProvider, block_kind, providers_for, register_provider,  # noqa: F401
                       registered_providers)
from . import builtin  # noqa: F401  (registers the built-in providers)
