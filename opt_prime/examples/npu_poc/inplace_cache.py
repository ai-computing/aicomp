"""In-place KV cache — generation with the cache kept on the device (PP × TP).

`generate.py` keeps the cache **functional**: every decode step sends the past
K·V to the device and copies the updated K·V back.  This path keeps each
stage's caches **on that stage's device** for the whole generation and updates
them **in place**; per step only the token id, the position, a 2-D mask, the
RoPE cos/sin, stage-boundary hidden states and the logits move.

Only standard PyTorch ops are used (`index_copy_`) — no vendor attention ops.
The model source is not modified: the graphs are the ones opt_prime captures
(`IR.retrieve_IR`, export path), rewritten **before** the PP split:

  1  fold the lazy-init cache cats   cat([empty, past_i]) → past_i
  2  RoPE out of the decode graph    model.rotary_emb → inputs rope_cos/rope_sin
                                     (computed on the host for the step position)
  3  decode update                   cat([past_i, new]) → index_copy_(past_i, 2, pos, new)
  4  prefill write                   index_copy_(cache_i, 2, arange(P), new)
                                     (the prompt starts at slot 0)
  5  outputs                         logits only — the caches never leave a stage

then split with **llama-tp-split** (whole decoder layers per stage), so every
layer's cache is read and written by one stage, and prefill and decode put the
same layers on the same stage.

Mask: the decode graph is captured with a past length of C−1, so after the
update the attention spans exactly the C cache slots.  The query's static
position is C−1 (after every slot), so the causal part admits all slots and the
2-D mask `[1]*(p+1) + [0]*(C−p−1)` selects the valid ones.

Shape notes (measured on RNGD, SDK 2026.4):
  · the write position is built right before each write, so it stays in the
    write's stage (a shared one would leave a view of a stage input as a stage
    output, which the runtime does not hand back);
  · the prefill write uses a constant `arange(P)`; this form compiled on every
    layout tested (fusion 4 PP=4 · TP=2 PP=3 · TP=4 PP=2), the others did not
    on all of them;
  · RoPE is computed on the host for decode (and prefill) — with RoPE inside a
    decode stage of six or more layers the TP ≥ 2 compiles did not finish.

Scope: Llama-family models (llama-tp-split), batch 1, backends whose compiled
stages update their inputs in place (validated: rngd).
"""

from __future__ import annotations

import contextlib
import io
import re
import time

import torch

_aten = torch.ops.aten
_KV = re.compile(r"layers\.(\d+)\.self_attn\.(k|v)_proj$")


# ── graph helpers ───────────────────────────────────────────────────────
def _phs(gm):
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _is_cache_cat(n) -> bool:
    return (n.op == "call_function" and n.target == _aten.cat.default
            and isinstance(n.args[0], (list, tuple)) and len(n.args[0]) == 2
            and (n.args[1] if len(n.args) > 1 else 0) in (-2, 2))


def _empty(n) -> bool:
    tm = n.meta.get("tensor_meta")
    return tm is not None and hasattr(tm, "shape") and tm.shape.numel() == 0


def _layer_kv(n, depth: int = 40):
    """(layer, 'k'|'v') of the new K/V that feeds a cache cat."""
    seen, front = set(), [n]
    for _ in range(depth):
        nxt = []
        for x in front:
            if x.op == "call_module":
                m = _KV.search(str(x.target))
                if m:
                    return int(m.group(1)), m.group(2)
            for y in x.all_input_nodes:
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        front = nxt
    raise SystemExit(f"[inplace] no k/v projection above {n.name}")


def _add_placeholders(gm, names):
    g = gm.graph
    last = _phs(gm)[-1]
    out = {}
    for nm in names:
        with g.inserting_after(last):
            last = out[nm] = g.placeholder(nm)
    return out


def _fold_init(gm) -> int:
    n = 0
    for c in list(gm.graph.nodes):
        if _is_cache_cat(c):
            a, b = c.args[0]
            if _empty(a) and b.op == "placeholder" and b.name.startswith("past_"):
                c.replace_all_uses_with(b)
                gm.graph.erase_node(c)
                n += 1
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return n


def _lift_rope(gm):
    g = gm.graph
    rot = [n for n in g.nodes if n.op == "call_module" and str(n.target).endswith("rotary_emb")]
    if len(rot) != 1:
        raise SystemExit(f"[inplace] expected one rotary_emb call, found {len(rot)}")
    rot = rot[0]
    new = _add_placeholders(gm, ["rope_cos", "rope_sin"])
    for u in list(rot.users):
        if not (u.op == "call_function" and u.args[1] in (0, 1)):
            raise SystemExit(f"[inplace] unexpected rotary_emb user {u.format_node()}")
        u.replace_all_uses_with(new["rope_cos" if u.args[1] == 0 else "rope_sin"])
        g.erase_node(u)
    g.erase_node(rot)
    g.eliminate_dead_code()
    gm.recompile()


def _keep_logits(gm):
    o = [n for n in gm.graph.nodes if n.op == "output"][0]
    o.args = ((o.args[0][0],),)
    gm.graph.eliminate_dead_code()
    gm.recompile()


def _decode_updates(gm) -> int:
    g = gm.graph
    pos = [n for n in _phs(gm) if n.target == "position_ids"][0]
    n = 0
    for c in list(g.nodes):
        if _is_cache_cat(c):
            a, b = c.args[0]
            if a.op == "placeholder" and a.name.startswith("past_"):
                with g.inserting_before(c):
                    w = g.call_function(_aten.reshape.default, (pos, [-1]))
                    ic = g.call_function(_aten.index_copy_.default, (a, 2, w, b))
                c.replace_all_uses_with(ic)
                g.erase_node(c)
                n += 1
    _keep_logits(gm)
    return n


def _prefill_writes(gm, n_layers: int, prompt_len: int) -> int:
    g = gm.graph
    cats = [(_layer_kv(c.args[0][1]), c) for c in g.nodes
            if _is_cache_cat(c) and _empty(c.args[0][0])]
    cache = _add_placeholders(gm, [f"past_{kv}_{i}" for i in range(n_layers) for kv in "kv"])
    for (li, kv), c in cats:
        new, nxt = c.args[0][1], c.next
        with g.inserting_before(nxt):
            w = g.call_function(_aten.arange.default, (prompt_len,), {"dtype": torch.long})
            g.call_function(_aten.index_copy_.default, (cache[f"past_{kv}_{li}"], 2, w, new))
        c.replace_all_uses_with(new)
        g.erase_node(c)
    _keep_logits(gm)
    return len(cats)


# ── capture → rewrite → split ───────────────────────────────────────────
def _decode_cls(n_layers: int):
    """Decode wrapper.  llama-tp-split needs a Llama* class with `.model.layers`."""
    from .generate import past_names, _flat, _as_cache
    names = past_names(n_layers)
    args = ", ".join(names)
    src = ("class LlamaInplaceDecode(nn.Module):\n"
           "    def __init__(self, m, n):\n"
           "        super().__init__(); self.model = m.model; self.lm_head = m.lm_head; self.n = n\n"
           f"    def forward(self, input_ids, position_ids, attention_mask, {args}):\n"
           f"        flat = [{args}]\n"
           "        pkv = _as_cache(tuple((flat[2*i], flat[2*i+1]) for i in range(self.n)))\n"
           "        o = self.model(input_ids=input_ids, position_ids=position_ids,\n"
           "                       attention_mask=attention_mask, past_key_values=pkv, use_cache=True)\n"
           "        return (self.lm_head(o.last_hidden_state), *_flat(o.past_key_values))\n")
    ns = {"nn": torch.nn, "torch": torch, "_flat": _flat, "_as_cache": _as_cache}
    exec(src, ns)
    return ns["LlamaInplaceDecode"], names


def _capture(mod, pp, cfg, seq, decode_past=None, dtype=None):
    from opt_prime.IR import IR
    from .capture import _StubOptimus, capture_options
    from .generate import decode_example_inputs
    ir = IR(mod, _StubOptimus(pp))
    ctx = (decode_example_inputs(cfg, 1, decode_past, dtype) if decode_past
           else contextlib.nullcontext())
    with contextlib.redirect_stdout(io.StringIO()), capture_options(True, 1, seq, verbose=False), ctx:
        ir.retrieve_IR(mod, use_kv_cache=True, dynamo_capture=True, for_training=False)
    return ir


def _split(ir, mod, pp):
    with contextlib.redirect_stdout(io.StringIO()):
        ir.split_IR(mod, "llama-tp-split", pp)
    sg = ir.model_ir[0]
    st = [n.target for n in sg.graph.nodes
          if n.op == "call_module" and str(n.target).startswith("submod_")]
    if len(st) != pp:
        raise SystemExit(f"[inplace] llama-tp-split produced {len(st)} stages (requested {pp})")
    return sg, st


def _owned(sg, nm):
    call = [n for n in sg.graph.nodes if n.op == "call_module" and n.target == nm][0]
    return [a.target for a in call.args
            if getattr(a, "op", "") == "placeholder" and a.target.startswith("past_")]


def host_rope(model, pos2d, dtype):
    """cos/sin for the given positions, computed on the host."""
    with torch.no_grad():
        h = torch.zeros(1, pos2d.shape[1], model.config.hidden_size, dtype=dtype)
        return model.model.rotary_emb(h, pos2d)


def build(model, cfg, pp: int, capacity: int, prompt_len: int):
    """Captured, rewritten and split prefill/decode graphs.

    Returns (prefill split, prefill stages, decode split, decode stages,
    cache names owned by each stage).
    """
    from torch.fx.passes.shape_prop import ShapeProp
    from .generate import _cache_export_support, cache_shape
    if not model.__class__.__name__.startswith("Llama"):
        raise SystemExit("[inplace] the in-place cache path supports Llama-family models only "
                         "(it splits with llama-tp-split)")
    L = cfg.num_hidden_layers
    dt = model.dtype
    model.config.use_cache = True
    _cache_export_support()

    # decode — captured with a past length of C−1 (attention spans C slots after the update)
    Cls, names = _decode_cls(L)
    dmod = Cls(model, L).eval()
    dir_ = _capture(dmod, pp, cfg, 1, decode_past=capacity - 1, dtype=dt)
    dgm = dir_.gm
    feed = {"input_ids": torch.zeros(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, capacity, dtype=torch.long)}
    feed.update({n: torch.zeros(*cache_shape(cfg, 1, capacity - 1), dtype=dt) for n in names})
    ShapeProp(dgm).propagate(*[feed[n.target] for n in _phs(dgm)])
    nf = _fold_init(dgm)
    _lift_rope(dgm)
    nu = _decode_updates(dgm)

    # prefill
    pir = _capture(model, pp, cfg, prompt_len)
    pgm = pir.gm
    pfeed = {"input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
             "position_ids": torch.arange(prompt_len).unsqueeze(0)}
    ShapeProp(pgm).propagate(*[pfeed[n.target] for n in _phs(pgm)])
    _lift_rope(pgm)
    nw = _prefill_writes(pgm, L, prompt_len)
    if not nf == nu == nw == 2 * L:
        raise SystemExit(f"[inplace] rewrite counts {nf}/{nu}/{nw} (expect {2*L} each)")

    psg, pst = _split(pir, model, pp)
    dsg, dst = _split(dir_, dmod, pp)
    own = [_owned(dsg, nm) for nm in dst]
    if own != [_owned(psg, nm) for nm in pst] or sorted(sum(own, [])) != sorted(names):
        raise SystemExit("[inplace] prefill and decode do not own the same caches per stage")
    return psg, pst, dsg, dst, own


# ── execution ───────────────────────────────────────────────────────────
class _Stage(torch.nn.Module):
    """Placed in `split_gm.submod_k`.  Caches already on the stage's device pass
    through unchanged and are updated in place by the compiled stage."""

    def __init__(self, compiled, multi: bool):
        super().__init__()
        self.compiled, self.multi = compiled, multi

    def forward(self, *args):
        out = self.compiled.run(*args)
        if self.multi:
            return out if isinstance(out, (list, tuple)) else [out]
        return out[0] if isinstance(out, (list, tuple)) else out


def _record(sg, st, feed):
    """Inputs and output kind of each stage in a CPU run — the compile examples."""
    rec = {}
    hs = [sg.get_submodule(nm).register_forward_hook(
        lambda mod, a, out, nm=nm: rec.__setitem__(nm, ([x.clone() for x in a],
                                                       isinstance(out, (list, tuple)))))
          for nm in st]
    _call(sg, feed)
    for h in hs:
        h.remove()
    return rec


def _call(sg, feed):
    with torch.no_grad():
        out = sg(*[feed[n.target] for n in _phs(sg)])
    return out[0] if isinstance(out, (list, tuple)) else out


def _mask(p: int, capacity: int):
    m = torch.zeros(1, capacity, dtype=torch.long)
    m[:, :p + 1] = 1
    return m


def run_pp(model, cfg, backend, tok_ids, n_new: int, *, pp: int, stage_devs: list,
           tp_size: int = 1, capacity: int, cpu_stages=(), verbose: bool = True,
           force: list | None = None):
    """One prefill + n_new−1 decode steps with the in-place cache.

    Same return shape as `generate.run_pp`: (tokens, metrics).
    `stage_devs[i]` = device group of the i-th NPU stage; `cpu_stages` stay on the host.
    """
    from .generate import cache_shape
    B, P = tok_ids.shape
    if B != 1:
        raise SystemExit("[inplace] batch 1 only")
    if P + n_new > capacity:
        raise SystemExit(f"[inplace] cache capacity {capacity} < prompt {P} + {n_new} tokens")
    dt = model.dtype
    st = {"compile_s": 0.0, "prefill_ms": 0.0, "decode_ms": [], "stages": {}}
    if force is not None:
        st["logits"] = []

    psg, pst, dsg, dst, own = build(model, cfg, pp, capacity, P)
    npu = [i for i in range(pp) if i not in set(cpu_stages)]
    if len(stage_devs) != len(npu):
        raise SystemExit(f"[inplace] {len(npu)} NPU stages but {len(stage_devs)} placements")

    # compile examples come from a CPU run of the split graphs (zero caches)
    pos0 = torch.arange(P).unsqueeze(0)
    z = {n: torch.zeros(*cache_shape(cfg, 1, capacity), dtype=dt) for s in own for n in s}
    pc, ps = host_rope(model, pos0, dt)
    prec = _record(psg, pst, {"input_ids": tok_ids, "position_ids": pos0,
                              "rope_cos": pc, "rope_sin": ps, **z})
    dpos = torch.tensor([[P]])
    dc, ds = host_rope(model, dpos, dt)
    drec = _record(dsg, dst, {"input_ids": tok_ids[:, -1:], "position_ids": dpos,
                              "attention_mask": _mask(P, capacity), "rope_cos": dc,
                              "rope_sin": ds, **{n: torch.zeros_like(t) for n, t in z.items()}})
    dev = {i: torch.device("cpu") for i in range(pp)}
    for j, i in enumerate(npu):
        for tag, sg, names, rec in (("prefill", psg, pst, prec), ("decode", dsg, dst, drec)):
            nm = names[i]
            args, multi = rec[nm]
            cs = backend.compile(sg.get_submodule(nm), args, name=f"inplace_{tag}_{nm}",
                                 devices=stage_devs[j], tp_size=tp_size)
            setattr(sg, nm, _Stage(cs, multi))
            st["compile_s"] += cs.stats.total_s
            dev[i] = cs.device
    st["stages"] = {"prefill": len(pst), "decode": len(dst)}

    # caches — allocated once on the owning stage's device, written in place
    cache = {n: torch.zeros(*cache_shape(cfg, 1, capacity), dtype=dt).to(dev[i])
             for i in range(pp) for n in own[i]}
    ptr = {n: t.data_ptr() for n, t in cache.items()}
    t0 = time.time()
    logits = _call(psg, {"input_ids": tok_ids, "position_ids": pos0,
                         "rope_cos": pc, "rope_sin": ps, **cache})
    st["prefill_ms"] = (time.time() - t0) * 1000
    lg = logits[0, -1].float().cpu()
    cur = int(lg.argmax())
    gen = [cur]
    if force is not None:
        st["logits"].append(lg)
        cur = force[0]
    if verbose:
        print(f"[inplace] PP={pp} × TP={tp_size} · capacity {capacity} · "
              f"stages {dst} · compile {st['compile_s']:.1f}s · "
              f"prefill {st['prefill_ms']:.1f}ms", flush=True)
    for s in range(1, n_new):
        p = P + s - 1
        pos = torch.tensor([[p]])
        c, sn = host_rope(model, pos, dt)
        t1 = time.time()
        logits = _call(dsg, {"input_ids": torch.tensor([[cur]]), "position_ids": pos,
                             "attention_mask": _mask(p, capacity),
                             "rope_cos": c, "rope_sin": sn, **cache})
        st["decode_ms"].append((time.time() - t1) * 1000)
        lg = logits[0, -1].float().cpu()
        cur = int(lg.argmax())
        gen.append(cur)
        if force is not None:
            st["logits"].append(lg)
            if s < len(force):
                cur = force[s]
    if any(cache[n].data_ptr() != q for n, q in ptr.items()):
        raise SystemExit("[inplace] a cache tensor was reallocated — the update was not in place")
    return gen, st
