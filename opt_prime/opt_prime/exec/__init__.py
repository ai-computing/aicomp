#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Execution engines: whole-request runners chosen per model (opt_prime.model.ModelSpec) and devices.

  optimus     Optimus_Inference (GPU pipeline + optional MoE expert tiers): runs any model torch runs
  rngd-layer  a native RNGD layer engine (all blocks on RNGD in one process, cold experts on the host CPU), for the
              model geometries it was built for (engine_manifest.json next to its data)
  atom-rbln   Rebellions ATOM through optimum-rbln: the decoder compiled into RBLN prefill/decode graphs (dense
              and the MoE families optimum-rbln compiles), cached per model / tensor-parallel size / max_seq_len

`select_engine(spec)` returns the fastest engine that supports the model, with the reasons the others were not
chosen. PD composes engines of this kind as prefill / decode endpoints.
"""
from .engines import (AtomRblnEngine, ExecutionEngine, OptimusEngine, RngdLayerEngine,  # noqa: F401
                      engine_geometry, register_engine, registered_engines, select_engine)
