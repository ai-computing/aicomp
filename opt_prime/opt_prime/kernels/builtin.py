#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Built-in providers: the torch reference for every block, and the MoE offload expert backends for routed experts."""
from ..moe_offload.spec import ExpertSpec
from .registry import BLOCK_KINDS, KernelProvider, block_kind, register_provider

# rough device memory bandwidths for the torch provider's cost estimate (bytes/s); calibrate per host
_TORCH_BW = {"cuda": 1.5e12, "cpu": 6.0e10}
_TORCH_CALL_S = {"cuda": 30e-6, "cpu": 20e-6}        # per block call: Python dispatch + launches


def _block_bytes(block) -> int:
    """weight bytes a block streams per call (decode is memory bound); 0 when unknown."""
    kind = block_kind(block)
    if kind == "experts":
        return block.expert_bytes          # per expert used
    if kind == "attention" and block.kind == "gqa":
        q, kv = block.heads * block.head_dim, block.kv_heads * block.head_dim
        return 2 * block.hidden * (2 * q + 2 * kv)
    if kind == "mlp" and block.ffn:
        return 2 * block.hidden * block.ffn * (3 if block.kind == "gated" else 2)
    if kind == "norm":
        return 2 * block.hidden
    return 0


class TorchProvider(KernelProvider):
    """The model's own torch module (eager / torch.compile). Runs everything torch runs on the device."""
    name = "torch"
    block_kinds = frozenset(BLOCK_KINDS)

    def __init__(self, device_kind):
        self.device_kind = device_kind

    def cost(self, block, tokens=1):
        return _TORCH_CALL_S[self.device_kind] + _block_bytes(block) / _TORCH_BW[self.device_kind]


class ExpertBackendProvider(KernelProvider):
    """Routed experts through a moe_offload ExpertBackend (the backend instance is created per tier at planning time;
    this provider answers support and cost from a template instance)."""
    block_kinds = frozenset({"experts"})

    def __init__(self, backend_name, device_kind, **template_kw):
        self.backend_name, self.device_kind, self.template_kw = backend_name, device_kind, template_kw
        self.name = f"experts:{backend_name}"
        self._tpl = None

    def _template(self):
        if self._tpl is None:
            from ..moe_offload.backends import get_backend
            self._tpl = get_backend(self.backend_name, **self.template_kw)
        return self._tpl

    def supports(self, block):
        return isinstance(block, ExpertSpec) and self._template().supports(block)

    def cost(self, block, tokens=1):
        k = max(1, tokens)                    # decode: about one expert per routed pair
        return self._template().cost(block, k, k)


for _dev in ("cuda", "cpu"):
    register_provider(TorchProvider(_dev))
register_provider(ExpertBackendProvider("gpu", "cuda"))
register_provider(ExpertBackendProvider("cpu", "cpu"))
register_provider(ExpertBackendProvider("rngd", "rngd", emulate=True))      # emulate: no device touched to answer
register_provider(ExpertBackendProvider("rngd-opt", "rngd", emulate=True))
