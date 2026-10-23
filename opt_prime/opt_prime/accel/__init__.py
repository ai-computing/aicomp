#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Heterogeneous accelerators for OptimusPrime: everything that is not the GPU pipeline engine itself.

    spec/       one description of a model: blocks, attention / KV geometry, MoE routing (ModelSpec, ExpertSpec)
    platforms/  devices and leases (rm-accel aware) and each vendor's implementations: cuda/, cpu/, rngd/, atom/
    moe/        MoE expert offload (planner, layer execution, custom op, model adapters) and MoE training experts
    engines/    whole-model engine selection (Optimus pipeline; vendor engines registered by platforms)
    pd/         prefill/decode disaggregation: contracts, runtime, torch engine, RNGD->ATOM NPU pipeline
    train/      training jobs and backends (Optimus pipeline on GPUs, npu-train on NPUs)
    serve/      serving front ends (llm-d sidecar)

Dependencies point one way: spec <- platforms <- moe / engines / pd / train <- serve. Feature packages do not import
vendor SDKs; vendor modules under platforms/ import theirs lazily (pd/npu is the cross-vendor RNGD->ATOM pipeline).
The engine (opti_pri, inference) imports this package only through its optional `moe_offload=` argument.
"""
