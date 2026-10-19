#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Execution engines: whole-request runners chosen per model (opt_prime.model.ModelSpec) and devices.

  optimus     Optimus_Inference (GPU pipeline + optional MoE expert tiers): runs any model torch runs
  rngd-layer  a native RNGD layer engine (all blocks on RNGD in one process, cold experts on the host CPU), for the
              model geometries it was built for (engine_manifest.json next to its data)

`select_engine(spec)` returns the fastest engine that supports the model, with the reasons the others were not
chosen. PD composes engines of this kind as prefill / decode endpoints.
"""
from .engines import (ExecutionEngine, OptimusEngine, RngdLayerEngine, engine_geometry,  # noqa: F401
                      register_engine, registered_engines, select_engine)
