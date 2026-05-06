#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

# Lazy package init.
#
# We deliberately AVOID eager `from opt_prime.opti_pri import Optimus_p` /
# `from opt_prime.inference import Optimus_Inference` here. Both of those
# submodules transitively `import torch` at module load time, which would
# initialize CUDA-related state inside PyTorch.
#
# That breaks the inference example flow:
#   - example file does `from opt_prime.mps_manager import resolve_visible_devices`
#     to set CUDA_VISIBLE_DEVICES BEFORE `import torch`
#   - but the first `from opt_prime.<anything>` triggers this __init__.py
#   - if __init__.py eagerly imported opti_pri/inference, torch would be
#     imported here — already too late to change CVD safely
#
# PEP 562 module-level __getattr__ defers the actual import to first access:
#   from opt_prime import Optimus_p             # triggers torch import here
#   from opt_prime.opti_pri import Optimus_p    # also imports torch — but
#                                                 caller is expected to have
#                                                 finalized CVD by then
#   from opt_prime.mps_manager import ...       # NO torch import (mps_manager
#                                                 has no torch at module level)

__all__ = ["Optimus_p", "Optimus_Inference"]


def __getattr__(name):
    if name == "Optimus_p":
        from opt_prime.opti_pri import Optimus_p
        return Optimus_p
    if name == "Optimus_Inference":
        from opt_prime.inference import Optimus_Inference
        return Optimus_Inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
