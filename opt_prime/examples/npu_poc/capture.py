"""Capture + split — the boundary where the engine is used unmodified.

This file does three things.

  1. Puts a fake object in the `Optimus_p` slot the engine expects
     (_StubOptimus).  This lets the engine run without distributed
     initialization, and is why the engine can stay unmodified.
  2. Overrides capture options in-process only (C2+C3).
  3. Calls the engine's retrieve_IR → split_IR.

Capture-option override
───────────────────────
The engine's `IR._build_dynamic_shapes` and `IR._create_example_inputs` are
hardcoded in `IR.py`, so callers have no supported way to change them.  We
therefore leave `IR.py` untouched and reassign those two methods on the class
object loaded into memory by import.  They are restored when the context
manager exits.

  (1) _build_dynamic_shapes → None
       The engine default marks both batch and seq as dynamic.  That is the
       right default for GPU training and GPT-2 passes with it, but with
       torch 2.10 + Llama the export is rejected because the core guard
       Eq(Min(64*s, 256*s), 64*s) cannot be proven.  NPU binaries are tied
       to a fixed shape anyway, so static capture is not a workaround but the
       correct setting for this target.

  (2) _create_example_inputs → deployment shape
       The counterpart of (1).  HF dummy_inputs uses seq=5, which is harmless
       under dynamic capture, but once shapes are static it is baked in as a
       constant of view(batch, 5, -1, 64) and breaks at runtime with seq=16.
       We call the engine function first and only fix the shapes of its
       result — the input-name set and lookup logic remain the engine's.

  Warning: the effect applies to every IR instance in this process, because
    the class itself is patched.  _create_example_inputs is also called by
    the ShapeProp step of milp_split and hierarchical_split (IR.py:2293,
    2579), so with those partitioners the override carries into the split
    stage.  simple and llama-tp-split do not call it.

  → Suggested engine improvement: expose both options as constructor
    arguments.  Half of this file would then disappear, and so would the
    boundary-scalar workaround (see stage_io.py).
"""

from __future__ import annotations

import contextlib
import time

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Fake Optimus_p handed to the engine
# ─────────────────────────────────────────────────────────────────────
#
# Note the direction: this is not "a stub the engine requires" but "a fake
# object the PoC gives to the engine".  It goes into the Optimus_p slot that
# IR(model, optimus) expects.  The engine actually reads only the following
# (verified by instrumentation):
#
#     model2type        return value of retrieve_IR (used in computation)
#     tpl.rank          only inside print()
#     is_first_stage()  logging branch in llama_tp_split
#     get_rank()        logging
#     tpl.num_stage / tpl.get_stage / run_info.*
#                       setup_submod family — not reached on this PoC path,
#                       but safe defaults are provided in case the
#                       partitioner is changed.

class _StubTpl:
    rank = 0
    local_rank = 0
    world_size = 1
    pp_size = 1
    dp_size = 1
    tp_size = 1
    num_stage = 1

    def get_stage(self, *a, **kw):
        return 0


class _StubRunInfo:
    name = None
    submod = None
    node = None
    getitem_dic = None


class _StubOptimus:
    """Mimics only what the engine reads from the Optimus_p slot."""

    model2type = {"hf": 50, "sy": 51, "vt": 52}

    def __init__(self, num_stage: int = 1):
        self.tpl = _StubTpl()
        self.tpl.num_stage = num_stage
        self.tpl.pp_size = num_stage
        self.run_info = _StubRunInfo()

    def is_first_stage(self) -> bool:
        return True

    def is_last_stage(self) -> bool:
        return self.tpl.num_stage == 1

    def get_rank(self) -> int:
        return 0


# ─────────────────────────────────────────────────────────────────────
# Capture-option override (C2 + C3)
# ─────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def capture_options(static: bool, batch: int, seq: int, verbose: bool = True):
    """Replace the two capture options of the `IR` class for this block only.

    With static=False nothing is changed — capture uses the engine default
    (batch+seq dynamic), and boundary scalar freezing (C8) becomes necessary.
    """
    from opt_prime.IR import IR

    if not static:
        if verbose:
            print("[capture] engine-default dynamic capture — boundary scalar freezing required",
                  flush=True)
        yield
        return

    orig_dyn = IR._build_dynamic_shapes
    orig_ex = IR._create_example_inputs

    def _no_dynamic(self, input_names, example_inputs, model=None):
        return None

    def _deploy_shape(self, model, input_names):
        # Run the engine logic first — dummy_inputs lookup, the
        # max_position_embeddings bound check and position_ids generation are
        # the engine's as-is; only the shapes are fixed.
        inputs = orig_ex(self, model, input_names)
        vocab = getattr(getattr(model, "config", None), "vocab_size", 1000)
        fixed = {}
        for name, t in inputs.items():
            if not isinstance(t, torch.Tensor) or t.dim() < 2:
                fixed[name] = t
            elif name == "position_ids":
                fixed[name] = (torch.arange(seq, dtype=t.dtype)
                               .unsqueeze(0).expand(batch, -1).contiguous())
            elif name in ("attention_mask", "token_type_ids"):
                fixed[name] = torch.ones(batch, seq, dtype=t.dtype)
            elif name == "input_ids":
                fixed[name] = torch.randint(0, min(vocab, 1000), (batch, seq),
                                            dtype=t.dtype)
            else:
                fixed[name] = t
        return fixed

    IR._build_dynamic_shapes = _no_dynamic
    IR._create_example_inputs = _deploy_shape
    if verbose:
        print(f"[capture] capture-option override: static shape (batch={batch}, seq={seq}) "
              f"— IR.py unmodified, in-process only", flush=True)
    try:
        yield
    finally:
        IR._build_dynamic_shapes = orig_dyn
        IR._create_example_inputs = orig_ex


# ─────────────────────────────────────────────────────────────────────
# Dead call_module removal (C1 — specific to torch 2.8)
# ─────────────────────────────────────────────────────────────────────

def prune_dead_call_modules(gm: nn.Module) -> int:
    """Erase call_module nodes that have no users.

    torch 2.8 attributes `aten._assert_tensor_metadata` to the
    nn_module_stack, and the engine's reconstruction treats it as a second
    region of the same module, producing embed(embed(ids)).  Nobody uses the
    result, but the FX forward still executes it, so embedding receives float
    indices and crashes.  FX's eliminate_dead_code() skips call_module nodes
    as side-effecting, so we erase them directly.  On torch 2.10 and 2.13
    there are none.
    """
    removed = 0
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_module" and len(node.users) == 0:
            gm.graph.erase_node(node)
            removed += 1
    if removed:
        gm.recompile()
    return removed


# ─────────────────────────────────────────────────────────────────────
# Capture + split
# ─────────────────────────────────────────────────────────────────────

class CaptureResult:
    def __init__(self, ir, split_gm, model_type, capture_s, split_s,
                 n_nodes, n_pruned, stage_names):
        self.ir = ir
        self.split_gm = split_gm
        self.model_type = model_type
        self.capture_s = capture_s
        self.split_s = split_s
        self.n_nodes = n_nodes
        self.n_pruned = n_pruned
        self.stage_names = stage_names


def capture_and_split(model: nn.Module, num_stages: int, plan, *,
                      batch: int, seq: int, partitioner: str = "simple",
                      for_training: bool = False,
                      cost_alpha: float = 0.0,
                      use_kv_cache: bool = False) -> CaptureResult:
    """Use the engine's IR capture + PP split as-is.  Zero lines of engine changes."""
    from opt_prime.IR import IR

    stub = _StubOptimus(num_stages)
    ir = IR(model, stub)

    t0 = time.time()
    with capture_options(plan.static_capture, batch, seq):
        model_type = ir.retrieve_IR(model, use_kv_cache=use_kv_cache,
                                    dynamo_capture=plan.use_dynamo,
                                    for_training=for_training)
    capture_s = time.time() - t0

    n_nodes = len(list(ir.gm.graph.nodes))
    print(f"[capture] {('export' if plan.use_dynamo else 'HFTracer')} "
          f"{capture_s:.1f}s · nodes={n_nodes} · model_type={model_type}",
          flush=True)

    t0 = time.time()
    if partitioner.startswith("balanced"):
        # PoC-side compute-balance partition.  The engine is called with
        # 'simple', and only the split_module call is intercepted to change
        # the partition (see partition.py).
        from .partition import balanced
        with balanced(num_stages, mode=partitioner, alpha=cost_alpha):
            ir.split_IR(model, "simple", num_stages)
    else:
        ir.split_IR(model, partitioner, num_stages)
    split_s = time.time() - t0
    split_gm = ir.model_ir[0]

    names = [n.target for n in split_gm.graph.nodes
             if n.op == "call_module" and str(n.target).startswith("submod_")]
    print(f"[capture] split('{partitioner}', {num_stages}) {split_s:.1f}s "
          f"→ {names}", flush=True)

    # C1 is applied per stage after the split — the dead nodes appear inside
    # stages, and this ordering is the path validated in the earlier PoC.
    n_pruned = 0
    if plan.prune_dead_call_modules:
        for nm in names:
            k = prune_dead_call_modules(split_gm.get_submodule(nm))
            if k:
                print(f"[capture]   {nm}: pruned {k} dead call_module(s)", flush=True)
            n_pruned += k

    if len(names) != num_stages:
        raise SystemExit(
            f"[capture] partitioner '{partitioner}' produced only {len(names)} stage(s) "
            f"(requested {num_stages}).  On the dynamo path llama-tp-split looks up "
            f"flattened names (layers_7_mlp_up_proj) by splitting on dots, so it "
            f"may not work — try simple instead.")

    return CaptureResult(ir, split_gm, model_type, capture_s, split_s,
                         n_nodes, n_pruned, names)
