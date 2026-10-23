#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Find routed-expert modules (transformers >= 5.0 experts interface) without model-specific code.

An experts module holds all experts of one MoE layer as stacked tensors whose first dimension is the expert index
(`gate_up_proj`, `down_proj`, optional biases) and is called as `experts(hidden[T,H], top_k_index[T,k],
top_k_weights[T,k]) -> [T,H]`; indices equal to `num_experts` are skipped by the eager implementation.
"""
from dataclasses import dataclass
from typing import List

import torch.nn as nn


def is_experts_module(m: nn.Module) -> bool:
    E = getattr(m, "num_experts", None)
    gu = getattr(m, "gate_up_proj", None)
    dn = getattr(m, "down_proj", None)
    return (isinstance(E, int) and isinstance(gu, nn.Parameter) and isinstance(dn, nn.Parameter)
            and gu.dim() == 3 and dn.dim() == 3 and gu.shape[0] == E and dn.shape[0] == E)


def per_expert_tensors(m: nn.Module):
    """names of the parameters / buffers stacked over experts (first dim == num_experts)."""
    E = m.num_experts
    ps = [n for n, p in m.named_parameters(recurse=False) if p.dim() >= 1 and p.shape[0] == E]
    bs = [n for n, b in m.named_buffers(recurse=False) if b.dim() >= 1 and b.shape[0] == E]
    return ps, bs


@dataclass
class ExpertsInfo:
    name: str            # qualified module name in the model
    module: nn.Module
    layer_id: int        # index among the model's MoE layers
    num_experts: int
    expert_bytes: int    # bytes of one expert (all per-expert tensors)


def find_moe_experts(model: nn.Module) -> List[ExpertsInfo]:
    out = []
    for name, m in model.named_modules():
        if not is_experts_module(m):
            continue
        ps, bs = per_expert_tensors(m)
        nbytes = sum(getattr(m, n).numel() * getattr(m, n).element_size() for n in ps + bs)
        out.append(ExpertsInfo(name, m, len(out), m.num_experts, nbytes // m.num_experts))
    return out
