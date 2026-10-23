#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Block-level model descriptions shared by every execution path (GPU pipeline, MoE offload, PD engines, NPU
layer engines). See blocks.py for the specs and hf.py for building them from transformers models."""
from opt_prime.accel.spec.blocks import (AttentionSpec, DenseMLPSpec, ExpertSpec, LayerSpec, MoESpec, ModelSpec,  # noqa: F401
                     NormSpec, RopeSpec, RouterSpec)
from opt_prime.accel.spec.hf import ROUTER_SCORING, model_spec, model_spec_from_pretrained  # noqa: F401
