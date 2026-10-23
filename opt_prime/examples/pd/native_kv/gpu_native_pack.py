"""Owned CUDA-pinned packing directly from validated NumPy KV views."""
import numpy as np
import torch


def pack_native_views(views):
    if len(views.slots) != 72 or any(slot is None for slot in views.slots):
        raise ValueError('incomplete native KV')
    if not isinstance(views.slots[0], np.ndarray) or views.slots[0].ndim != 3:
        raise ValueError('invalid native KV slot')
    n = views.slots[0].shape[1]
    if not 1 <= n < 2048: raise ValueError('prefix outside capacity')
    for slot in views.slots:
        if not isinstance(slot, np.ndarray) or slot.shape != (8,n,128) or slot.dtype != np.uint16:
            raise ValueError('invalid native KV slot')
    host = torch.empty((72,n,8,128),dtype=torch.bfloat16,pin_memory=True)
    target = host.view(torch.uint16).numpy()
    for i,slot in enumerate(views.slots):
        np.copyto(target[i],slot.transpose(1,0,2),casting='no')
    return host
