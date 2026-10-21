#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""The contract between models (adapters) and devices (backends): what one MoE layer's experts look like."""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExpertSpec:
    num_experts: int
    hidden: int                  # H
    ffn: int                     # F (per expert)
    activation: str = "silu"     # applied to the gate half
    gate: str = "swiglu"         # "swiglu": act(gate) * up with gate_up = [gate; up]; "custom": model-specific
    has_bias: bool = False
    transposed: bool = False     # weights stored [in, out] instead of [out, in]
    dtype: torch.dtype = torch.bfloat16

    @property
    def expert_bytes(self) -> int:
        n = 3 * self.hidden * self.ffn + (2 * self.ffn + self.hidden if self.has_bias else 0)
        return n * torch.tensor([], dtype=self.dtype).element_size()

    @property
    def standard(self) -> bool:
        """plain SwiGLU experts: the form native kernels implement."""
        return self.gate == "swiglu" and self.activation == "silu" and not self.has_bias and not self.transposed

    def signature(self) -> str:
        return (f"E{self.num_experts}_H{self.hidden}_F{self.ffn}_{self.gate}-{self.activation}"
                f"{'_bias' if self.has_bias else ''}{'_T' if self.transposed else ''}_{str(self.dtype).split('.')[-1]}")


@dataclass
class ExpertWeights:
    """A subset of one layer's experts, local ids 0..n-1 in `global_ids` order (host tensors)."""
    spec: ExpertSpec
    global_ids: list
    gate_up: torch.Tensor                    # [n, 2F, H] (or [n, H, 2F] if transposed)
    down: torch.Tensor                       # [n, H, F]  (or [n, F, H])
    gate_up_bias: Optional[torch.Tensor] = None
    down_bias: Optional[torch.Tensor] = None
    module: Optional[nn.Module] = field(default=None, repr=False)   # the model's experts module (torch backends)
