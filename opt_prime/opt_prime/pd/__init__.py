"""Optional, vendor-neutral synchronous disaggregated inference contracts.

Importing this module does not import torch or a vendor SDK.
"""
from .runtime import KVSpec, KVLease, PrefillResult, PDRequest, PDRuntime

__all__ = ['KVSpec', 'KVLease', 'PrefillResult', 'PDRequest', 'PDRuntime']
