"""Generation — one prefill + N decode steps.  Integration item #9.

Every measurement so far was a single forward pass.  This is the first place
that **produces a second token.**

Design — "zero model modifications" is the invariant
────────────────────────────────────────────────────
The cache is made **functional, as graph inputs/outputs**.  The engine's
`CachedScaledDotProductAttention` is inserted into the stage graph **after**
capture (FX surgery) and keeps its state in Python values — the write position
is a Python int advanced by `self.cache_pos += …`, and `KVCacheManager` holds
the cache tensors in a Python dict and slices them with Python ints.  When a
stage is compiled for an NPU, those values are fixed at capture time, so the
binary cannot advance the position or carry the cache from one call to the
next.  HF, on the other hand, already has a pure-function path that takes
`past_key_values` as an argument — we use that as is.

  prefill : (input_ids[1,S])                      → (logits, 2L K·V tensors)
  decode  : (input_ids[1,1], position_ids, 2L past K·V tensors)
                                                  → (logits, 2L updated K·V tensors)

Note: vendor-specific implementations (`paged_attn_*`, `mark_static_address`)
   are **not used.**  That route requires rewriting attention in the vendor's
   form, which conflicts with the "no model modifications" goal.  The cost is
   that the cache round-trips through the host every step, but measured
   bandwidth is not the bottleneck — 99x the transfer for 1.63x the time.

Note: **shapes are fixed, so each past length needs its own binary.**  Hence
   buckets — handled by `buckets.py`.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class PrefillModule(nn.Module):
    """Run the prompt through once and return the cache as outputs.

    Calls the HF model as is — zero patches.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids):
        out = self.model(input_ids=input_ids, position_ids=position_ids,
                         use_cache=True)
        return (out.logits, *_flat(out.past_key_values))


class DecodeModule(nn.Module):
    """Process one token.  The past K·V are taken as flat arguments.

    Warning: the varargs name `past` matters — for NPUs that bind `input_info`
      names to the forward signature, the input names must start with `past…`.

    `attention_mask` is taken because of bucketing — when the cache is padded
    to the bucket size, the empty slots must be masked.  With fixed shapes each
    past length needs its own binary; buckets reduce that to a few.
    """

    def __init__(self, model, n_layers: int):
        super().__init__()
        self.model = model
        self.n = n_layers

    def forward(self, input_ids, position_ids, attention_mask, *past):
        pkv = _as_cache(tuple((past[2 * i], past[2 * i + 1]) for i in range(self.n)))
        out = self.model(input_ids=input_ids, position_ids=position_ids,
                         attention_mask=attention_mask,
                         past_key_values=pkv, use_cache=True)
        return (out.logits, *_flat(out.past_key_values))


def _flat(pkv):
    """Flatten a DynamicCache / legacy tuple into (k0, v0, k1, v1, …)."""
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()
    elif hasattr(pkv, "layers"):                 # transformers 5.x DynamicCache
        pkv = [(layer.keys, layer.values) for layer in pkv.layers]
    flat = []
    for k, v in pkv:
        flat += [k, v]
    return tuple(flat)


def _as_cache(pkv):
    """Past K·V pairs in the form the model accepts.

    transformers 4.x takes the legacy tuple as is; 5.x accepts only Cache
    objects, so the pairs are wrapped in a DynamicCache there.
    """
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache, "from_legacy_cache"):
        return pkv
    return DynamicCache(ddp_cache_data=pkv)


def _cache_export_support() -> bool:
    """Let torch.export carry a DynamicCache output (transformers 5.x).

    Returns True if the cache is flattened as all keys then all values — the
    caller then interleaves them into (k0, v0, k1, v1, …).  transformers 4.x
    returns the legacy tuple and needs nothing.
    """
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache, "from_legacy_cache"):
        return False
    from transformers.integrations.executorch import register_dynamic_cache_export_support
    register_dynamic_cache_export_support()
    return True


def _interleave(past: list) -> list:
    """(k0, …, kN, v0, …, vN) → (k0, v0, k1, v1, …)."""
    n = len(past) // 2
    return [t for i in range(n) for t in (past[i], past[n + i])]


def past_names(n_layers: int) -> list[str]:
    """Names for NPUs that bind inputs by name — they must start with `past`."""
    return [f"past_{'kv'[i % 2]}_{i // 2}" for i in range(2 * n_layers)]


def cache_shape(cfg, batch: int, seq: int) -> tuple[int, int, int, int]:
    """(B, kv_heads, seq, head_dim) — K or V of one layer."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return (batch, cfg.num_key_value_heads, seq, head_dim)


def cache_bytes_per_token(cfg, dtype: torch.dtype) -> int:
    """KV bytes per token — the basic unit for capacity calculations."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    el = torch.empty((), dtype=dtype).element_size()
    return 2 * cfg.num_key_value_heads * head_dim * cfg.num_hidden_layers * el


# ─────────────────────────────────────────────────────────────────────
# Generation loop
# ─────────────────────────────────────────────────────────────────────

def run(model, cfg, backend, tok_ids, n_new: int, *, devices,
        tp_size: int = 1, dtype: torch.dtype, bucket_list=None,
        verbose: bool = True, force: list | None = None):
    """One prefill + n_new decode steps.  Zero model modifications.

    `devices` is the list of cards used by this single unit and `tp_size` is
    the degree the backend shards across them — with PP=1 the two have the
    same length.  TP lives inside the rebel binary, so here the argument is
    only passed through (`placement.py`).

    If `force` is given, decoding is teacher-forced — the next step's input is
    the reference token rather than the NPU's own choice.  Greedy decoding is
    chaotic: once it diverges everything after differs, so the token match
    rate of free-running generation measures amplification, not compute
    accuracy.  Teacher forcing compares every step in the same context.

    Returns: (list of generated tokens, metrics dict)
    """
    import time
    from . import buckets as BK

    L = cfg.num_hidden_layers
    B, P0 = tok_ids.shape
    names_past = past_names(L)
    bl = tuple(bucket_list) if bucket_list else BK.DEFAULT
    stats = {"compile_s": 0.0, "prefill_ms": 0.0, "decode_ms": [], "binaries": {}}
    if force is not None:
        stats["logits"] = []           # per-step last-position logits (teacher-forced verdict)

    DT = {torch.float32: "float32", torch.bfloat16: "bfloat16",
          torch.int64: "int64", torch.long: "int64"}

    # ── prefill ──────────────────────────────────────────────────────
    pos = torch.arange(P0, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    pre = PrefillModule(model).eval()
    info = [("input_ids", [B, P0], "int64"), ("position_ids", [B, P0], "int64")]
    t0 = time.time()
    cp = backend.compile(pre, [tok_ids, pos], name=f"prefill_s{P0}",
                         devices=list(devices), tp_size=tp_size,
                         input_names=["input_ids", "position_ids"])
    stats["compile_s"] += cp.stats.total_s
    stats["binaries"][f"prefill_s{P0}"] = cp.stats.total_s
    t1 = time.time(); out = cp.run(tok_ids, pos); stats["prefill_ms"] = (time.time()-t1)*1000
    out = list(out) if isinstance(out, (list, tuple)) else [out]
    logits, past = out[0], out[1:]
    cur = int(logits[0, -1].argmax().item())
    gen = [cur]
    valid = P0
    if force is not None:
        stats["logits"].append(logits[0, -1].float().clone())
    if verbose:
        print(f"[gen] prefill S={P0} · compile {cp.stats.total_s:.1f}s · "
              f"run {stats['prefill_ms']:.1f}ms · first token {cur}", flush=True)

    if force is not None and force:
        # Teacher forcing starts at the first decode input: feed the reference
        # token, not the prefill's own choice, so the KV cache holds the reference
        # context only (the NPU's choice above is recorded in `gen`).
        cur = force[0]

    # ── decode ───────────────────────────────────────────────────────
    dec = DecodeModule(model, L).eval()
    compiled: dict[int, object] = {}
    for step in range(n_new - 1):
        bk = BK.pick(valid, bl)
        padded = BK.pad_cache(past, valid, bk)
        am = BK.mask(valid, bk, B)
        ids = torch.tensor([[cur]], dtype=torch.long)
        ppos = torch.tensor([[valid]], dtype=torch.long)
        if bk not in compiled:
            info = ([("input_ids", [B, 1], "int64"),
                     ("position_ids", [B, 1], "int64"),
                     ("attention_mask", [B, bk + 1], "int64")]
                    + [(n, list(t.shape), DT[t.dtype])
                       for n, t in zip(names_past, padded)])
            cd = backend.compile(dec, [ids, ppos, am, *padded],
                                 name=f"decode_b{bk}", devices=list(devices),
                                 tp_size=tp_size,
                                 input_names=["input_ids", "position_ids",
                                              "attention_mask"] + names_past)
            compiled[bk] = cd
            stats["compile_s"] += cd.stats.total_s
            stats["binaries"][f"decode_b{bk}"] = cd.stats.total_s
            if verbose:
                print(f"[gen] bucket {bk}: compiled new binary in "
                      f"{cd.stats.total_s:.1f}s", flush=True)
        t1 = time.time()
        o = compiled[bk].run(ids, ppos, am, *padded)
        stats["decode_ms"].append((time.time() - t1) * 1000)
        o = list(o) if isinstance(o, (list, tuple)) else [o]
        logits, oc = o[0], o[1:]
        past = BK.take_valid(oc, valid, bk)      # keep only the valid part + new token
        valid += 1
        cur = int(logits[0, -1].argmax().item())
        gen.append(cur)
        if force is not None:
            stats["logits"].append(logits[0, -1].float().clone())
        if force is not None and step + 1 < len(force):
            cur = force[step + 1]          # ← next input follows the reference
    return gen, stats


# ─────────────────────────────────────────────────────────────────────
# PP × generation — the substance of integration item #3
# ─────────────────────────────────────────────────────────────────────
#
# Until now the generation path did not use the PP split.  Joining the two is
# the substance of engine integration — the engine runs PP as one process per
# rank + send/recv, whereas on an NPU the device is an argument, so a single
# process iterating over all stages is the natural fit (`runner.py`).
#
# The gate was whether the engine can capture the decode graph.  Since
# `_get_export_input_names` reads the forward signature of a generic module,
# a wrapper with the past K·V as explicit arguments suffices — with `*past`
# varargs the names would collapse into a single 'past'.

def make_decode_cls(n_layers: int):
    """Dynamically build a decode wrapper that takes the past K·V as explicit arguments."""
    names = past_names(n_layers)
    args = ", ".join(names)
    src = (
        "class _Decode(nn.Module):\n"
        "    def __init__(self, model, n):\n"
        "        super().__init__(); self.model = model; self.n = n\n"
        f"    def forward(self, input_ids, position_ids, attention_mask, {args}):\n"
        f"        flat = [{args}]\n"
        "        pkv = _as_cache(tuple((flat[2*i], flat[2*i+1]) for i in range(self.n)))\n"
        "        o = self.model(input_ids=input_ids, position_ids=position_ids,\n"
        "                       attention_mask=attention_mask,\n"
        "                       past_key_values=pkv, use_cache=True)\n"
        "        return (o.logits, *_flat(o.past_key_values))\n"
    )
    ns = {"nn": nn, "torch": torch, "_flat": _flat, "_as_cache": _as_cache}
    exec(src, ns)
    return ns["_Decode"], names


@contextlib.contextmanager
def decode_example_inputs(cfg, batch: int, bucket: int, dtype):
    """Third capture-option override — supplies example inputs for the past K·V.

    The engine's `_create_example_inputs` gives `randn(2, in_features)` to
    unknown input names of a generic module, which does not fit the 4-D past K·V.
    → Exposing the capture options as engine arguments (see capture.py)
      would resolve this as well.
    """
    from opt_prime.IR import IR
    orig = IR._create_example_inputs
    sh = cache_shape(cfg, batch, bucket)

    def patched(self, model_, input_names):
        out = {}
        for nm in input_names:
            if nm == "input_ids":
                out[nm] = torch.randint(0, 1000, (batch, 1))
            elif nm == "position_ids":
                out[nm] = torch.zeros(batch, 1, dtype=torch.long)
            elif nm == "attention_mask":
                out[nm] = torch.ones(batch, bucket + 1, dtype=torch.long)
            else:
                out[nm] = torch.zeros(*sh, dtype=dtype)
        return out

    IR._create_example_inputs = patched
    try:
        yield
    finally:
        IR._create_example_inputs = orig


def split_decode(model, cfg, bucket: int, pp: int, batch: int, dtype):
    """Capture the decode graph and split it for PP.  No engine modifications."""
    import io, contextlib as _c
    from opt_prime.IR import IR
    from .capture import _StubOptimus, capture_options

    L = cfg.num_hidden_layers
    Cls, names = make_decode_cls(L)
    dec = Cls(model, L).eval()
    ir = IR(dec, _StubOptimus(pp))
    buf = io.StringIO()
    with _c.redirect_stdout(buf), capture_options(True, batch, 1, verbose=False), \
            decode_example_inputs(cfg, batch, bucket, dtype):
        ir.retrieve_IR(dec, use_kv_cache=True, dynamo_capture=True, for_training=False)
        ir.split_IR(dec, "simple", pp)
    sg = ir.model_ir[0]
    stages = [n.target for n in sg.graph.nodes
              if n.op == "call_module" and str(n.target).startswith("submod_")]
    # ── Fix C1 — remove dead call_modules (torch 2.8 only) ──────────────
    # Warning: the decode path was broken because this step was missing.
    #   `capture_and_split` applies it to every stage after the split, but
    #   here the engine API was called directly and the fix was skipped.
    #   torch 2.8 produces embed(embed(ids)); nobody uses it, but the FX
    #   forward still executes it, so Embedding receives bf16 indices and
    #   crashes (Expected … Long, Int; but got CPUBFloat16Type).
    #   torch 2.10/2.13 produce none, so the condition mirrors the capture path.
    from .capture import prune_dead_call_modules
    from .compat import detect
    if detect().torch_mm == (2, 8):
        n_pruned = sum(prune_dead_call_modules(sg.get_submodule(nm))
                       for nm in stages)
        if n_pruned:
            print(f"[genpp] decode: pruned {n_pruned} dead call_module(s) (C1)",
                  flush=True)
    return dec, sg, stages, names


def _compile_stages(backend, split_gm, stages, caller, driver, stage_devs,
                    tp_size: int = 1, cpu_stages=()):
    """Compile the stages and swap them for NPU wrappers.  Execution model of integration item #3.

    `stage_devs[i]` is the group of cards used by the i-th NPU stage — its
    length equals `tp_size`.  **The PoC core splits PP; the backend splits TP**
    (`placement.py`).  Stages whose index is in `cpu_stages` are not compiled;
    the original GraphModule runs them on the host.
    """
    npu = [nm for i, nm in enumerate(stages) if i not in set(cpu_stages)]
    if len(stage_devs) != len(npu):
        raise SystemExit(f"[generate] {len(npu)} NPU stages but "
                         f"{len(stage_devs)} placements")
    from . import runner, stage_io
    specs, _, _ = stage_io.collect(split_gm, stages, caller, driver)
    total = 0.0
    compiled = {}
    for i, nm in enumerate(npu):
        spec = specs[nm]
        target = stage_io.specialize(split_gm.get_submodule(nm), spec)
        tens = [torch.zeros(s, dtype=getattr(torch, d)) for (_, s, d) in spec.tensors]
        cs = backend.compile(target, tens, name=nm,
                             devices=stage_devs[i], tp_size=tp_size,
                             input_names=[n for (n, _, _) in spec.tensors])
        compiled[nm] = cs
        total += cs.stats.total_s
    runner.attach(split_gm, compiled, specs)
    return specs, total


def _ordered(driver: dict, split_gm) -> list:
    """Order the arguments by the top-level placeholders."""
    want = [n.target for n in split_gm.graph.nodes if n.op == "placeholder"]
    miss = [w for w in want if w not in driver]
    if miss:
        raise SystemExit(f"[generate] inputs required by the graph are missing: {miss}")
    return [driver[w] for w in want]


def run_pp(model, cfg, backend, tok_ids, n_new: int, *, pp: int,
           stage_devs: list, tp_size: int = 1, dtype: torch.dtype, bucket: int,
           verbose: bool = True, force: list | None = None, cpu_stages=()):
    """PP × TP × generation — puts both prefill and decode onto the grid.

    Note: until 2026-10-08 the generation path used neither the PP split nor
      TP.  PP was attached first; here the TP axis is passed through.

    `stage_devs[i]` = card group of the i-th NPU stage (length = `tp_size`).
    `cpu_stages` = stage indices kept on the host, in both prefill and decode.
    """
    import time
    from . import buckets as BK, capture as CAP, compat

    L = cfg.num_hidden_layers
    B, P0 = tok_ids.shape
    st = {"compile_s": 0.0, "prefill_ms": 0.0, "decode_ms": [], "stages": {}}

    # ── prefill: engine capture + PP split ──────────────────────────
    # Note: the engine captures the original model, so whether the cache is
    #   output follows the config.  The PP=1 path avoided this by having
    #   PrefillModule pass use_cache=True explicitly.
    model.config.use_cache = True
    split_kv = _cache_export_support()
    plan = compat.build_plan(backend.caps(), dtype=dtype)
    plan.compute_dtype = dtype
    cap = CAP.capture_and_split(model, pp, plan, batch=B, seq=P0,
                                use_kv_cache=True)
    pos = torch.arange(P0, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    dv = {"input_ids": tok_ids, "position_ids": pos}
    dv = {k: v for k, v in dv.items()
          if k in [n.target for n in cap.split_gm.graph.nodes if n.op == "placeholder"]}
    _, t = _compile_stages(backend, cap.split_gm, cap.stage_names, model, dv,
                           stage_devs, tp_size, cpu_stages)
    st["compile_s"] += t; st["stages"]["prefill"] = len(cap.stage_names)
    t0 = time.time()
    with torch.no_grad():
        out = cap.split_gm(*_ordered(dv, cap.split_gm))
    st["prefill_ms"] = (time.time() - t0) * 1000
    out = list(out) if isinstance(out, (list, tuple)) else [out]
    logits, past = out[0], out[1:]
    if split_kv:
        past = _interleave(past)
    cur = int(logits[0, -1].argmax().item()); gen = [cur]; valid = P0
    if force is not None:
        st["logits"] = [logits[0, -1].float().clone()]
    if verbose:
        print(f"[genpp] prefill PP={pp} × TP={tp_size} · "
              f"stages {cap.stage_names} · "
              f"compile {t:.1f}s · run {st['prefill_ms']:.1f}ms · "
              f"{len(out)} outputs (logits + {len(past)} past)", flush=True)
    if len(past) != 2 * L:
        raise SystemExit(
            f"[generate] prefill returned only {len(past)} past tensors "
            f"(need {2*L}).  Check whether the top-level output of the PP split "
            f"was narrowed to logits only — verify that use_kv_cache capture took effect.")

    if force is not None and force:
        # Teacher forcing starts at the first decode input: feed the reference
        # token, not the prefill's own choice, so the KV cache holds the reference
        # context only (the NPU's choice above is recorded in `gen`).
        cur = force[0]

    # ── decode: separate capture + PP split ─────────────────────────
    dec, dsg, dstages, pnames = split_decode(model, cfg, bucket, pp, B, dtype)
    padded = BK.pad_cache(past, valid, bucket)
    am = BK.mask(valid, bucket, B)
    dv2 = {"input_ids": torch.tensor([[cur]], dtype=torch.long),
           "position_ids": torch.tensor([[valid]], dtype=torch.long),
           "attention_mask": am}
    dv2.update({n: t for n, t in zip(pnames, padded)})
    _, t = _compile_stages(backend, dsg, dstages, dec, dv2, stage_devs, tp_size,
                           cpu_stages)
    st["compile_s"] += t; st["stages"]["decode"] = len(dstages)
    if verbose:
        print(f"[genpp] decode PP={pp} × TP={tp_size} · "
              f"stages {dstages} · bucket {bucket} · "
              f"compile {t:.1f}s", flush=True)

    for step in range(n_new - 1):
        dv2["input_ids"] = torch.tensor([[cur]], dtype=torch.long)
        dv2["position_ids"] = torch.tensor([[valid]], dtype=torch.long)
        dv2["attention_mask"] = BK.mask(valid, bucket, B)
        for n_, t_ in zip(pnames, BK.pad_cache(past, valid, bucket)):
            dv2[n_] = t_
        t0 = time.time()
        with torch.no_grad():
            o = dsg(*_ordered(dv2, dsg))
        st["decode_ms"].append((time.time() - t0) * 1000)
        o = list(o) if isinstance(o, (list, tuple)) else [o]
        past = BK.take_valid(o[1:], valid, bucket)
        valid += 1
        cur = int(o[0][0, -1].argmax().item())
        gen.append(cur)
        if force is not None:
            st["logits"].append(o[0][0, -1].float().clone())
        if force is not None and step + 1 < len(force):
            cur = force[step + 1]
    return gen, st
