#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Block-level model description: what a model computes, independent of devices and kernels.

A ModelSpec is a list of LayerSpecs (one per decoder layer) between an embedding and an LM head. Each layer holds
block specs - attention, norms, and either a dense MLP or a router + routed experts (+ shared experts). Kernel
providers and planners work on these specs; blocks whose kind is "custom" (an architecture no provider knows yet)
are still described by their module path, so they can run as an opaque torch module wherever torch runs.

The KV-cache semantics a PD transfer depends on (which layers keep K/V, their geometry, RoPE, windows) are derived
from the same spec: `ModelSpec.kv_semantics()` / `kv_semantic_id()`.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from opt_prime.accel.spec.experts import ExpertSpec  # noqa: F401  (routed experts keep the spec already used by the offload)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class NormSpec:
    kind: str                     # "rms" | "layer" | "custom"
    hidden: int
    eps: Optional[float] = None
    module_class: str = ""


@dataclass(frozen=True)
class RopeSpec:
    theta: Optional[float] = None
    scaling: Optional[str] = None          # json of rope parameters beyond theta (type, factor, ...), sorted keys
    partial_rotary_factor: float = 1.0
    rotary_dim: Optional[int] = None       # explicit rotary width when it differs from head_dim (e.g. MLA rope dim)


@dataclass(frozen=True)
class AttentionSpec:
    kind: str                     # "gqa" (q/k/v/o projections) | "mla" (latent KV) | "linear" | "custom"
    hidden: int
    heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0             # q/k head dim (MLA: nope + rope)
    v_head_dim: int = 0
    qkv_bias: bool = False
    o_bias: bool = False
    qk_norm: bool = False         # per-head RMSNorm on q and k (Qwen3, OLMoE, ...)
    sliding_window: Optional[int] = None
    rope: Optional[RopeSpec] = None
    kv_lora_rank: int = 0         # MLA
    q_lora_rank: int = 0          # MLA (0 = no q compression)
    module_class: str = ""

    @property
    def keeps_kv(self) -> bool:
        return self.kind in ("gqa", "mla")


@dataclass(frozen=True)
class DenseMLPSpec:
    kind: str                     # "gated" (act(gate) * up -> down) | "plain" (up -> act -> down) | "custom"
    hidden: int
    ffn: int
    activation: str = "silu"
    bias: bool = False
    module_class: str = ""


@dataclass(frozen=True)
class RouterSpec:
    num_experts: int
    top_k: int
    scoring: str = "softmax"      # "softmax" | "sigmoid" | "custom"
    norm_topk_prob: bool = True
    correction_bias: bool = False  # DeepSeek-V3 style e_score_correction_bias used for selection only
    n_group: int = 1              # group-limited routing (DeepSeek-V3 / Solar-Open / GLM-4.5)
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    module_class: str = ""


@dataclass(frozen=True)
class MoESpec:
    router: RouterSpec
    experts: ExpertSpec
    shared_ffn: int = 0           # total intermediate size of the shared expert(s); 0 = none
    shared_gate: bool = False     # Qwen2-MoE style sigmoid gate on the shared expert output
    module_class: str = ""


@dataclass(frozen=True)
class LayerSpec:
    index: int
    attention: AttentionSpec
    input_norm: Optional[NormSpec]
    post_attention_norm: Optional[NormSpec]
    mlp: Optional[DenseMLPSpec] = None
    moe: Optional[MoESpec] = None
    module_path: str = ""         # e.g. "model.layers.3"

    @property
    def is_moe(self) -> bool:
        return self.moe is not None


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    hidden: int
    vocab: int
    layers: Tuple[LayerSpec, ...]
    final_norm: Optional[NormSpec]
    tied_embeddings: bool = False
    dtype: str = "bfloat16"
    max_position: Optional[int] = None
    revision: str = ""            # checkpoint identity when known (path name / commit hash)
    notes: Tuple[str, ...] = field(default=())

    # ---- summaries ----
    @property
    def moe_layers(self) -> List[LayerSpec]:
        return [l for l in self.layers if l.is_moe]

    @property
    def custom_blocks(self) -> List[str]:
        """blocks no provider is expected to know (run as opaque torch modules)."""
        out = []
        for l in self.layers:
            if l.attention.kind == "custom":
                out.append(f"layer {l.index} attention ({l.attention.module_class})")
            if l.mlp is not None and l.mlp.kind == "custom":
                out.append(f"layer {l.index} mlp ({l.mlp.module_class})")
            if l.moe is not None and (l.moe.router.scoring == "custom" or not l.moe.experts.standard):
                out.append(f"layer {l.index} moe ({l.moe.module_class})")
        return out

    def signature(self) -> str:
        return _digest(asdict(self))

    # ---- KV semantics (PD contract) ----
    def kv_semantics(self) -> dict:
        """What a KV cache of this model means, independent of its physical layout on a device: per layer the
        attention kind and geometry that writes K/V, RoPE and window; layers with recurrent state instead of K/V are
        listed separately (a PD transfer must carry that state too)."""
        kv, state = [], []
        for l in self.layers:
            a = l.attention
            if a.keeps_kv:
                kv.append(dict(layer=l.index, kind=a.kind, kv_heads=a.kv_heads, head_dim=a.head_dim,
                               v_head_dim=a.v_head_dim, kv_lora_rank=a.kv_lora_rank, window=a.sliding_window,
                               rope=asdict(a.rope) if a.rope else None, qk_norm=a.qk_norm))
            else:
                state.append(dict(layer=l.index, kind=a.kind, module_class=a.module_class))
        return dict(version=1, model_type=self.model_type, revision=self.revision, dtype=self.dtype,
                    kv_layers=kv, state_layers=state)

    def kv_semantic_id(self) -> str:
        """versioned digest of kv_semantics(); two engines may exchange KV only if these match."""
        return "opt-kv-v1:" + _digest(self.kv_semantics())
