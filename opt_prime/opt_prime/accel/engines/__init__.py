#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Execution engines: whole-request runners chosen per model (opt_prime.accel.spec.ModelSpec) and devices.

  optimus     (engines.optimus) Optimus_Inference (GPU pipeline + optional MoE expert tiers): runs any model torch runs
  rngd-layer  (platforms.rngd.engine) a native RNGD layer engine (all blocks on RNGD in one process, cold experts on the host CPU), for the
              model geometries it was built for (engine_manifest.json next to its data)
  atom-rbln   (platforms.atom.engine) Rebellions ATOM through optimum-rbln: the decoder compiled into RBLN prefill/decode graphs (dense
              and the MoE families optimum-rbln compiles), cached per model / tensor-parallel size / max_seq_len

`select_engine(spec)` returns the fastest engine that supports the model, with the reasons the others were not
chosen. PD composes engines of this kind as prefill / decode endpoints.
"""
from opt_prime.accel.engines.base import (ExecutionEngine, engine_geometry, register_engine,  # noqa: F401
                                         registered_engines, select_engine)
from opt_prime.accel.engines.optimus import OptimusEngine  # noqa: F401

# The platform engines import engines.base, so they are re-exported lazily: importing one of them first must not
# re-enter its own half-initialised module through this package (#102). SDKs load only when an engine starts.
_WHERE = {"AtomRblnEngine": "opt_prime.accel.platforms.atom.engine",
          "RngdLayerEngine": "opt_prime.accel.platforms.rngd.engine"}


def __getattr__(name):
    if name in _WHERE:
        import importlib
        return getattr(importlib.import_module(_WHERE[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
