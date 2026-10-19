#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""MoE expert offloading for OptimusPrime.

Keeps the frequently routed ("hot") experts of every MoE layer in accelerator memory and computes the rest
("cold") on the host CPU, overlapping both. Works on any HuggingFace MoE model whose experts module follows the
transformers 5.x experts interface: `experts(hidden[T,H], top_k_index[T,k], top_k_weights[T,k]) -> [T,H]` with
fused `gate_up_proj[E, 2F, H]` and `down_proj[E, H, F]` parameters (Qwen2/3-MoE, Mixtral, DeepSeek-V3, GLM-4-MoE,
Solar-Open, ...). Routers and shared experts are left untouched.

    from opt_prime.moe_offload import MoEOffloadConfig, apply_moe_offload
    report = apply_moe_offload(model, MoEOffloadConfig(gpu_budget_gb=20, profile=counts))

or `Optimus_Inference(model, ..., moe_offload=MoEOffloadConfig(...))`.

Requirements: transformers >= 5.0 (the fused experts interface above; transformers 4.x keeps experts in an
nn.ModuleList and is not supported), torch >= 2.4 (torch.library.custom_op), CUDA for the device side.
Validated with torch 2.10.0+cu128, transformers 5.1.0, NVIDIA A100 80GB (CUDA 12.8), see
examples/moe_offload_check.py and examples/moe_offload_generate.py. The official OptimusPrime examples target
transformers 4.46.2; this feature and the dynamo_capture fixes it relies on target transformers 5.x.

Layout (see DESIGN.md): spec (ExpertSpec: the model/device contract) -> adapters (model side) -> backends (device
side: cuda, cpu, rngd, ...) -> planner (heterogeneous placement over tiers) -> layer (N-tier execution) -> ops
(custom op + OffloadedExperts) -> api (apply_moe_offload).

Numerics: the accelerator half runs the model's configured experts implementation on the hot subset and the host
half runs the eager implementation on the cold pairs; the two partial sums are added in the hidden dtype, so results
match the unmodified model up to bf16 rounding of that final addition (exact in fp32 to ~1e-7).
"""

__all__ = ["MoEOffloadConfig", "TierSpec", "OffloadStats", "apply_moe_offload", "OffloadReport", "ExpertProfiler",
           "find_moe_experts", "plan_hot_experts", "ExpertSpec", "ModelAdapter", "register_adapter",
           "ExpertBackend", "register_backend"]

_WHERE = {"MoEOffloadConfig": "config", "TierSpec": "config", "OffloadStats": "config", "apply_moe_offload": "api",
          "OffloadReport": "api", "ExpertProfiler": "profiler", "find_moe_experts": "discover",
          "plan_hot_experts": "placement", "ExpertSpec": "spec", "ModelAdapter": "adapters",
          "register_adapter": "adapters", "ExpertBackend": "backends", "register_backend": "backends"}


def __getattr__(name):
    if name in _WHERE:
        import importlib
        return getattr(importlib.import_module(f"{__name__}.{_WHERE[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
