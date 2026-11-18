"""Compute-balanced partitioning — integration item #7 "partitioner objective".

Why
───
None of the engine's four partitioners uses **compute balance as its objective**:

    simple          equal number of call_modules
    llama-tp-split  equal split at layer boundaries
    milp            minimize cross-stage activation bytes
    hierarchical    block super-graph MILP (also transfer volume)

Pipeline throughput, however, is set by **`max(t_stage)`**.  On Llama-3.2-1B
PP=4, `lm_head` (128256×2048) sits in the last stage, making that stage
**2×** the others (5.4ms ↔ 2.5~2.9ms); that one imbalance cuts the
theoretical ceiling from 2.91× to **2.12×**.

The ATOM+ partitioner study concluded that "the objective should be compute
balance, not transfer volume", but **the compute-balanced side was never
actually built.**  This file builds it.

How — engine unmodified
───────────────────────
We do not copy `simple_split` wholesale.  After partitioning, that function
does lengthy parameter moves and reference cleanup, and at the end it
**rebuilds `metadata_range` from the result and self-checks with
`assert len(...) == num_stage`**.

So there is exactly one interception point — the `split_module` symbol in
the `opt_prime.IR` namespace.  Swapping the `part_fn` the engine passes for
ours leaves **all engine code below it running unchanged.**

  Note: the cost model is the parameter count — a proxy for matmul work.
    `nn.Embedding` is set to 0: despite many parameters it is a gather, not
    compute.  This is a static estimate, not measured time, so it is a
    feasibility measurement of how much of the ceiling comes back, not an
    optimal partition.
"""

from __future__ import annotations

import contextlib

import torch.nn as nn


def out_elems(gm, node) -> float:
    """Output element count of this call_module — a proxy for activation cost."""
    v = (node.meta or {}).get("val", None)
    if v is None:
        v = (node.meta or {}).get("tensor_meta", None)
    if v is None or not hasattr(v, "shape"):
        return 0.0
    n = 1
    for d in v.shape:
        try:
            n *= int(d)
        except Exception:
            return 0.0
    return float(n)


def make_cost(alpha: float):
    """cost = parameter count + alpha × output element count.

    alpha=0 gives the plain parameter model.  The parameter model was
    measured not to fit at long seq, so an activation term is added to try.
    """
    def f(gm, node):
        return param_cost(gm, node) + alpha * out_elems(gm, node)
    return f


def param_cost(gm, node) -> float:
    """Compute proxy for one call_module = parameter count.

    Embedding is 0 — even with large weights (128256×2048) execution is a gather.
    """
    try:
        m = gm.get_submodule(node.target)
    except AttributeError:
        return 0.0
    if isinstance(m, nn.Embedding):
        return 0.0
    return float(sum(p.numel() for p in m.parameters(recurse=True)))


def assign_balanced(gm, num_stage: int, cost_fn=param_cost) -> dict:
    """Assign a stage index to each node so that cumulative cost is balanced.

    Nodes are assigned the current stage in order; once cumulative cost
    crosses the target line, the stage advances from the next node on.
    Placeholders come first, so they naturally go to stage 0, and output
    goes to the last stage.
    """
    nodes = list(gm.graph.nodes)
    cm = [n for n in nodes if n.op == "call_module"]
    costs = {n.name: cost_fn(gm, n) for n in cm}
    total = sum(costs.values())
    if total <= 0 or len(cm) < num_stage:
        raise SystemExit(f"[partition] balanced partition impossible — {len(cm)} call_module(s) · "
                         f"total cost {total}")

    target = total / num_stage
    assign, stage, acc, placed = {}, 0, 0.0, 0
    for n in nodes:
        assign[n.name] = stage
        if n.op != "call_module":
            continue
        acc += costs[n.name]
        placed += 1
        # At least one module must be left for each remaining stage —
        # otherwise the engine's `assert len(metadata_range) == num_stage` fails.
        remaining_mods = len(cm) - placed
        remaining_stages = num_stage - 1 - stage
        if stage < num_stage - 1 and (acc >= target * (stage + 1)
                                      or remaining_mods <= remaining_stages):
            stage += 1
    return assign


def describe(gm, assign: dict, num_stage: int, cost_fn=param_cost) -> str:
    per = [0.0] * num_stage
    cnt = [0] * num_stage
    for n in gm.graph.nodes:
        if n.op == "call_module":
            k = assign[n.name]
            per[k] += cost_fn(gm, n)
            cnt[k] += 1
    tot = sum(per) or 1.0
    return " · ".join(f"s{k}: {cnt[k]} mods {per[k]/1e6:.0f}M ({per[k]/tot*100:.1f}%)"
                      for k in range(num_stage))


def _layer_end_nodes(gm) -> dict:
    """The last call_module of each decoder layer — cut candidates.

    Cutting in the middle of a layer makes the residual and several
    intermediate activations cross the boundary, increasing transfer volume.
    Cutting at layer boundaries keeps crossings few (measured at PP=4:
    simple 2203 KiB ↔ llama-tp-split 795 KiB).
    """
    import re
    last = {}
    order = {}
    for i, n in enumerate(gm.graph.nodes):
        if n.op != "call_module":
            continue
        m = re.match(r"(?:model[._])?layers[._](\d+)[._]", str(n.target))
        if m:
            last[int(m.group(1))] = n.name
            order[n.name] = i
    return last, order


def assign_balanced_layer(gm, num_stage: int, cost_fn=param_cost) -> dict:
    """Compute balance ∧ layer boundaries — both objectives together.

    Picks as cut points the layer ends closest to the cumulative-cost target
    lines.  Gives up a little balance to cut transfer volume substantially.
    """
    nodes = list(gm.graph.nodes)
    cm = [n for n in nodes if n.op == "call_module"]
    costs = {n.name: cost_fn(gm, n) for n in cm}
    total = sum(costs.values())
    last, _ = _layer_end_nodes(gm)
    if not last or len(last) < num_stage:
        raise SystemExit(f"[partition] found only {len(last)} layer boundaries — "
                         f"not enough for PP={num_stage}")

    # Cumulative cost at the end of each layer
    cum, acc = {}, 0.0
    for n in nodes:
        if n.op == "call_module":
            acc += costs[n.name]
        if n.name in last.values():
            cum[n.name] = acc
    target = total / num_stage

    cuts = []
    for k in range(1, num_stage):
        want = target * k
        cand = [(abs(c - want), nm) for nm, c in cum.items() if nm not in cuts]
        cuts.append(min(cand)[1])
    cuts = set(cuts)

    assign, stage = {}, 0
    for n in nodes:
        assign[n.name] = stage
        if n.name in cuts and stage < num_stage - 1:
            stage += 1
    return assign


@contextlib.contextmanager
def balanced(num_stage: int, cost_fn=None, verbose: bool = True,
             mode: str = "balanced", alpha: float = 0.0):
    """Make `IR.split_IR(..., 'simple', N)` cut for compute balance.

    Engine files are not touched — only the `split_module` symbol in the
    `opt_prime.IR` namespace is swapped for the duration of this block.
    """
    import opt_prime.IR as IRmod

    if cost_fn is None:
        cost_fn = make_cost(alpha) if alpha else param_cost
    orig = IRmod.split_module

    def patched(gm, root_m, part_fn, *a, **kw):
        assign = _ASSIGN[mode](gm, num_stage, cost_fn)
        if verbose:
            print(f"[partition] {mode} partition (alpha={alpha:g}): "
                  f"{describe(gm, assign, num_stage, cost_fn)}", flush=True)
        return orig(gm, root_m, lambda n: assign[n.name], *a, **kw)

    IRmod.split_module = patched
    try:
        yield
    finally:
        IRmod.split_module = orig


_ASSIGN = {
    "balanced": assign_balanced,
    "balanced-layer": assign_balanced_layer,
}
