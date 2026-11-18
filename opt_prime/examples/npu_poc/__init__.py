#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""opt_prime × Korean NPUs — PP + TP PoC (phases 2 and 3).

The layout is the integration plan: each module maps to one integration unit.

    backends/base.py          NpuBackend · CompiledStage protocol        → integration item #1
    backends/rbln.py          ATOM+ adapter                              (stays outside the engine)
    compat.py                 version fixes C1~C9                        → basis for engine change (expose capture options)
    patches.py                per-backend model patch sets               → integration item #5
    capture.py                capture option override + split            → engine change (expose capture options)
                              (engine unmodified)
    stage_io.py               boundary contract — scalar · bool · dtype  → integration items #1·#6
    placement.py              (PP,TP) → card placement                   → integration items #2·#4
    runner.py                 submod replacement + FX forward            → integration item #3 (largest)
    equivalence.py            3-tier verdict · metrics · thresholds      (stays in the PoC)
    generate.py               functional KV-cache generation (PP)        → integration item #9
    resident.py               resident cache — SDPA → vendor paged op    (optional · vendor-specific op)
    resident_generate.py      resident cache × TP generation (NPU)       (optional)
    resident_pp_generate.py   resident cache × PP × TP generation (NPU)  (optional)

Engine integration items (referred to as "integration item #N" in comments):

    #1 Backend abstraction      lift NpuBackend/CompiledStage into the engine
    #2 Topology·Comm            remove the "cuda"/NCCL assumption (NPU-to-NPU transfer goes via host)
    #3 Execution model          per-rank processes ↔ single process, multiple devices
    #4 Heterogeneous placement  express configs where only some stages run on CPU
    #5 Per-backend patch sets   keep model patches per backend and transformers version
    #6 dtype policy             boundary dtype and backend-forced dtype
    #7 Partitioner objective    compute balance + compilability, not transfer volume
    #8 Compile cache
    #9 KV cache · generation

Principle: the engine (`opt_prime/`) is not modified at all.
"""
