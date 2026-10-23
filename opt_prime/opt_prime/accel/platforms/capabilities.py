#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Device-level capability vocabulary: what a device kind can do (discoverers and expert backends declare these
names only, so a typo fails at declaration instead of silently disabling a route). Endpoint-level contracts - input
and context limits, transports, external KV visibility, cancellation - belong to the engine that runs on the device
(opt_prime.accel.pd.capabilities.EngineCapabilities), not here. A capability is a claim verified for the declared
SDK/driver range; unverified features are not declared."""

#: roles a device can be leased for (one exclusive role per accelerator at a time)
ROLES = {
    "prefill": "computes prompt KV for a PD request",
    "decode": "generates tokens from an admitted KV cache",
    "generate": "whole-request engine (prefill and decode on the same device, e.g. the GPU pipeline)",
    "experts": "MoE expert tier: holds expert weights and runs routed (token, expert) pairs",
    "dense": "dense layers of a model whose experts are offloaded elsewhere",
    "train": "training (forward + backward + optimizer step) of a model or adapter",
}

CAPABILITIES = {
    # compute
    "prefill": "can run prefill for the model",
    "decode": "can run decode steps for the model",
    "experts": "can run routed MoE experts (an ExpertBackend exists for it)",
    "graph_compile": "runs torch modules through an ahead-of-time graph compiler (furiosa-torch, RBLN)",
    "custom_kernels": "runs generated or hand-written device kernels (furiosa-opt)",
    "continuous_batching": "the vendor engine schedules tokens of several requests itself",
    "p2p_dma": "can take part in device-to-device PCIe DMA (its buffers can be exported or imported)",
    # memory
    "resident_weights": "keeps weights resident between calls (no per-call weight upload)",
}


def check_capabilities(names) -> frozenset:
    """frozenset of known capability names; raises ValueError on unknown ones."""
    names = frozenset(names)
    unknown = sorted(n for n in names if n not in CAPABILITIES)
    if unknown:
        raise ValueError(f"unknown capabilities {unknown}; known: {sorted(CAPABILITIES)}")
    return names
