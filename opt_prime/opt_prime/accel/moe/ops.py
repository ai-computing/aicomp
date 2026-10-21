#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""The custom op that stands in for an experts module in exported graphs, and the module that calls it.

`opt_prime::offloaded_experts(hidden, top_k_index, top_k_weights, layer_id)` is opaque to torch.export / the
OptimusPrime IR (shape-only fake implementation), so data-dependent routing is never traced and the expert weights
are not graph parameters.
"""
import inspect
import itertools
import threading
from typing import Dict, List

import torch
import torch.nn as nn

from opt_prime.accel.moe.layer import OffloadedLayer

_LAYERS: Dict[int, OffloadedLayer] = {}
_IDS = itertools.count()
_LOCK = threading.Lock()


def register_layer(layer: OffloadedLayer) -> int:
    with _LOCK:
        lid = next(_IDS)
    _LAYERS[lid] = layer
    return lid


def get_layer(layer_id: int) -> OffloadedLayer:
    return _LAYERS[layer_id]


# cudagraph_unsafe: the op synchronizes with the host (routing), so torch.compile(mode="reduce-overhead") captures
# CUDA graphs for the regions between offloaded layers and runs this op eagerly in between (graph partition).
_TAGS = (torch.Tag.cudagraph_unsafe,) if hasattr(torch.Tag, "cudagraph_unsafe") else ()
_OP_KW = {"tags": _TAGS} if "tags" in inspect.signature(torch.library.custom_op).parameters else {}   # torch < 2.6


@torch.library.custom_op("opt_prime::offloaded_experts", mutates_args=(), **_OP_KW)
def offloaded_experts(hidden: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor,
                      layer_id: int) -> torch.Tensor:
    return _LAYERS[layer_id].forward(hidden, top_k_index, top_k_weights)


@offloaded_experts.register_fake
def _(hidden, top_k_index, top_k_weights, layer_id):
    return torch.empty_like(hidden)


# Training through offloaded experts (experts frozen): gradients w.r.t. hidden and routing weights, computed by the
# layer from the host tier's copy of the experts the call used (OffloadedLayer.backward).
def _setup_context(ctx, inputs, output):
    hidden, top_k_index, top_k_weights, layer_id = inputs
    ctx.save_for_backward(hidden, top_k_index, top_k_weights)
    ctx.layer_id = layer_id


def _backward(ctx, grad_out):
    hidden, top_k_index, top_k_weights = ctx.saved_tensors
    gx, gw = _LAYERS[ctx.layer_id].backward(hidden, top_k_index, top_k_weights, grad_out)
    return gx, None, gw, None


offloaded_experts.register_autograd(_backward, setup_context=_setup_context)


class OffloadedExperts(nn.Module):
    """Drop-in replacement for an experts module: same call signature, no parameters."""

    def __init__(self, layer_id: int, num_experts: int, hot: List[int]):
        super().__init__()
        self.layer_id, self.num_experts, self.hot = layer_id, num_experts, hot

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return torch.ops.opt_prime.offloaded_experts(hidden_states, top_k_index, top_k_weights, self.layer_id)

    def extra_repr(self):
        return f"layer_id={self.layer_id}, experts={self.num_experts}, hot={len(self.hot)}"
