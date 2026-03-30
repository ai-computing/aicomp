#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
#
#  LoRA (Low-Rank Adaptation) support for OptimusPrime pipeline-parallel training.
#
#  This module provides lightweight LoRA adapters that can be applied to
#  FX GraphModule submods after pipeline partitioning.
#
#  TP-aware: When Tensor Parallelism is active, LoRA adapters are automatically
#  parallelized to match the base linear's sharding plan.
#    - ColwiseParallel targets: lora_B gets ColwiseParallel
#    - RowwiseParallel targets: lora_A gets RowwiseParallel
#

import torch
import torch.nn as nn
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


# TP plan classification for LLaMA target modules
_COLWISE_TARGETS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
_ROWWISE_TARGETS = {"o_proj", "down_proj"}


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapters.

    The base linear weight is frozen. Only lora_A and lora_B are trainable.

    When tp_style is set, the corresponding LoRA sub-linear is expected to be
    parallelized externally (by parallelize_module) after construction.
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.scaling = alpha / r
        self.merged = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Create lora_A/lora_B with same dtype as base weight
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=dtype)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base weights
        for p in self.base_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base_linear(x)
        if self.merged:
            return base_out
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the base linear layer (for inference)."""
        if self.merged:
            return
        with torch.no_grad():
            base_dtype = self.base_linear.weight.dtype

            # Extract local tensors (handles DTensor from TP parallelization)
            lora_A_w = self.lora_A.weight
            lora_B_w = self.lora_B.weight
            if hasattr(lora_A_w, 'to_local'):
                lora_A_w = lora_A_w.to_local()
            if hasattr(lora_B_w, 'to_local'):
                lora_B_w = lora_B_w.to_local()
            lora_A_w = lora_A_w.to(base_dtype)
            lora_B_w = lora_B_w.to(base_dtype)

            delta = (lora_B_w @ lora_A_w) * self.scaling

            # Apply to base weight's local data
            base_w = self.base_linear.weight
            if hasattr(base_w, 'to_local'):
                base_w.to_local().add_(delta)
            else:
                base_w.data += delta
        self.merged = True

    def state_dict_lora_only(self, prefix=''):
        """Return only LoRA adapter weights."""
        return {
            f'{prefix}lora_A.weight': self.lora_A.weight.data,
            f'{prefix}lora_B.weight': self.lora_B.weight.data,
        }


def _match_target(name: str, target_modules: List[str]) -> Optional[str]:
    """Check if a flattened module name matches any target pattern.

    Returns the matched target string, or None.
    """
    parts = name.replace('.', '_').split('_')
    for target in target_modules:
        target_parts = target.split('_')
        if parts[-len(target_parts):] == target_parts:
            return target
    return None


def _get_tp_style(target: str) -> Optional[str]:
    """Determine TP parallelization style for a matched target.

    Returns 'colwise', 'rowwise', or None.
    """
    if target in _COLWISE_TARGETS:
        return "colwise"
    elif target in _ROWWISE_TARGETS:
        return "rowwise"
    return None


def apply_lora_to_submod(submod: nn.Module, config: LoRAConfig,
                         tp_mesh=None) -> Set[str]:
    """Apply LoRA adapters to matching nn.Linear modules in a GraphModule submod.

    After split_module, module names inside submod are flattened with underscores,
    e.g., 'model_layers_0_self_attn_q_proj'.  We match the suffix against
    config.target_modules.

    When tp_mesh is provided, LoRA adapters are parallelized to match:
      - ColwiseParallel targets: lora_B → ColwiseParallel
      - RowwiseParallel targets: lora_A → RowwiseParallel

    Args:
        submod: The pipeline stage's GraphModule (run_info.submod).
        config: LoRA configuration.
        tp_mesh: DeviceMesh for tensor parallelism (None if TP not used).

    Returns:
        Set of module names that were replaced with LoRA adapters.
    """
    replaced = set()

    # Collect (name, module, matched_target) first to avoid mutation during iteration
    targets = []
    for name, module in submod.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        matched = _match_target(name, config.target_modules)
        if matched is not None:
            targets.append((name, module, matched))

    for name, module, matched in targets:
        lora_module = LoRALinear(module, config.r, config.alpha, config.dropout)

        # Move LoRA adapters to the same device as the base linear
        device = next(module.parameters()).device
        lora_module.lora_A.to(device)
        lora_module.lora_B.to(device)
        if hasattr(lora_module, 'lora_dropout') and isinstance(lora_module.lora_dropout, nn.Dropout):
            lora_module.lora_dropout.to(device)

        # Navigate to parent and replace
        atoms = name.split('.')
        if len(atoms) == 1:
            setattr(submod, name, lora_module)
        else:
            parent = submod
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], lora_module)

        replaced.add(name)

    # Apply TP parallelization to LoRA sub-modules if needed
    if tp_mesh is not None:
        _apply_tp_to_lora(submod, targets, tp_mesh)

    # Freeze ALL non-LoRA parameters in the submod
    for pname, param in submod.named_parameters():
        if 'lora_A' not in pname and 'lora_B' not in pname:
            param.requires_grad = False

    return replaced


def _apply_tp_to_lora(submod: nn.Module, targets: list, tp_mesh):
    """Parallelize LoRA adapters to match base linear's TP sharding.

    ColwiseParallel targets (q/k/v/gate/up_proj):
      - lora_B: ColwiseParallel (output column-sharded)
      - lora_A: plain tensor + gradient all-reduce hook (keeps TP ranks in sync)

    RowwiseParallel targets (o/down_proj):
      - lora_A: RowwiseParallel (input row-sharded + allreduce)
      - lora_B: plain tensor + gradient all-reduce hook (keeps TP ranks in sync)

    Non-parallelized adapters stay as plain tensors but register a gradient hook
    that all-reduces their gradients across TP ranks after backward. Without this,
    TP ranks diverge after the first optimizer step, producing degenerate output.
    """
    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel,
    )
    import torch.distributed as dist

    tp_plan = {}
    sync_params = []  # parameters that need gradient all-reduce across TP

    for name, module, matched in targets:
        tp_style = _get_tp_style(matched)
        if tp_style == "colwise":
            tp_plan[f"{name}.lora_B"] = ColwiseParallel()
            sync_params.append(f"{name}.lora_A")
        elif tp_style == "rowwise":
            tp_plan[f"{name}.lora_A"] = RowwiseParallel()
            sync_params.append(f"{name}.lora_B")

    if tp_plan:
        parallelize_module(module=submod, device_mesh=tp_mesh, parallelize_plan=tp_plan)

    # Register gradient all-reduce hooks on non-parallelized adapters
    # so TP ranks stay synchronized during training
    tp_group = tp_mesh.get_group()
    tp_size = tp_mesh.size()
    hook_count = 0
    for mod_path in sync_params:
        mod = submod
        for attr in mod_path.split('.'):
            mod = getattr(mod, attr)
        for param in mod.parameters():
            if param.requires_grad:
                def _make_hook(p_tp_group, p_tp_size):
                    def hook(grad):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=p_tp_group)
                        grad.div_(p_tp_size)
                        return grad
                    return hook
                param.register_hook(_make_hook(tp_group, tp_size))
                hook_count += 1

    print(f"[LoRA-TP] Parallelized {len(tp_plan)} + gradient-synced {hook_count} "
          f"LoRA sub-modules")


def get_lora_parameters(submod: nn.Module):
    """Return only LoRA trainable parameters from the submod."""
    for name, param in submod.named_parameters():
        if param.requires_grad:
            yield param


def get_lora_named_parameters(submod: nn.Module):
    """Return only LoRA trainable named parameters from the submod."""
    for name, param in submod.named_parameters():
        if param.requires_grad:
            yield name, param


def get_lora_state_dict(submod: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA adapter weights from the submod.

    Uses PyTorch's native state_dict() which handles DTensor correctly,
    then filters to LoRA keys only.
    """
    full_sd = submod.state_dict()
    state = {k: v for k, v in full_sd.items() if 'lora_A' in k or 'lora_B' in k}
    return state


def load_lora_state_dict(submod: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Load LoRA adapter weights into the submod.

    Uses PyTorch's native load_state_dict(strict=False) which handles
    DTensor correctly (including shard/replicate placement).
    """
    # load_state_dict with strict=False ignores missing keys (base weights)
    # and only loads the provided LoRA keys
    missing, unexpected = submod.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(unexpected)
    print(f"[LoRA] Loaded {loaded} tensors, unexpected {len(unexpected)}")


def merge_lora_weights(submod: nn.Module):
    """Merge all LoRA adapters into base weights (for inference deployment)."""
    for name, module in submod.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()


def count_parameters(submod: nn.Module):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in submod.parameters())
    trainable = sum(p.numel() for p in submod.parameters() if p.requires_grad)
    return total, trainable
