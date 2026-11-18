"""Resident KV cache path -- **optional.  It does not touch the default path.**

> Note: **this file is the "second path".**  Not a single line of it runs unless
> you use `poc.py --resident-cache` (CPU rewrite check) or
> `resident_generate.py` / `resident_pp_generate.py` (NPU execution).  The default
> path (the functional KV cache in `generate.py`) **works unchanged**, and the
> main code still runs if this file is deleted entirely.
>
> **Why a second path** -- it uses **vendor-specific ops** (`paged_causal_attn_*`,
> `mark_static_address`).  The project goal is *"avoid vendor-specific LLM
> implementations on the NPU as far as possible"*, so this **cannot be the
> default.**  Instead it is a **control that quantifies what you gain by giving
> up that goal** -- namely **(1) 1.6x longer context and (2) TP>1 on rebel-compiler 0.10.x**
> (with 0.12.0 the functional cache also runs with TP>1).
>
> **Entry points for this path** --
>   `poc.py --resident-cache`     checks only the rewrite logic against the original on CPU (impl="ref")
>   `resident_generate.py`     resident cache × TP generation on the NPU (PP=1)
>   `resident_pp_generate.py`        PP × TP × generation on the NPU

What it does -- swaps only the attention, on top of the captured graph
──────────────────────────────────────────────────────
**The model source is not modified.**  It finds `aten.scaled_dot_product_attention`
in the FX graph captured by opt_prime and replaces it with the vendor paged op.

Graph shape confirmed by measurement (Llama-3.2-1B · torch 2.8 · tf 4.46.2):

    q    ← aten.add(mul, mul)            (B, NH, S, D)   ← RoPE output
    k    ← reshape ← expand ← (B, KVH, 1, S, D)          ← repeat_kv
    v    ← reshape ← expand ← (B, KVH, 1, S, D)
    mask ← slice chain
    → aten.scaled_dot_product_attention(q, k, v, mask)
    → consumer: aten.transpose

Note: **the input of `repeat_kv`'s `expand` is already `(B,KVH,1,S,D)`** -- **exactly**
  the shape the vendor op expects.  Unwinding to it removes the duplication.

Rewrite result:

    q5 = view(q, (B, KVH, G, S, D))
    o5 = paged_causal_attn_{prefill,decode}(q5, k5, v5, kc5, vc5,
                                            seq, scale, block_table, BLK[, False])
    o  = reshape(o5, (B, NH, S, D))        → to the original consumer

The `expand` and `reshape` nodes **become dead and are removed by DCE.**

Warning: **the mask is dropped** -- the vendor `paged_causal_*` op owns causality.
  It is **equivalent only for unpadded prefill**; padded batches need revisiting
  (the same class of limitation as the bucket mask in `buckets.py`).

Warning: **the vendor ops are device kernels** -- their CPU registration provides shapes
  only (measured 2026-10-12: CPU outputs are not meaningful and the cache is not updated).
  So **verification of the rewrite logic is separated from verification of the
  vendor kernel**:

    impl="ref"     pure-torch reference implementation → **equivalence closed on CPU**
    impl="vendor"  torch.ops.rbln_custom_ops.* → **compiled and run on the NPU**

  The reference implementation also documents the semantics of the op.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Block size = **the prompt length limit**.  The default is 128; raising it
# raises the limit accordingly (measured up to 512).
# Prompts longer than one block are not supported on this path, so
#   `check_prefill_len` rejects them.
BLOCK = int(os.environ.get("NPU_POC_BLOCK", "128"))


# ─────────────────────────────────────────────────────────────────────
# 1. Reference implementation -- vendor op semantics in pure torch
# ─────────────────────────────────────────────────────────────────────

def paged_ref(q5, k5, v5, kc, vc, seq, scale, block_table, block_size,
              *, prefill: bool, is_bidirectional: bool = False):
    """Pure-torch reference for `paged_causal_attn_{prefill,decode}`.

    The layout is exactly the vendor's:
        q5  (B, KVH, G, S, D)
        k5  (B, KVH, 1, S, D)      v5 likewise
        kc  (NBLK, KVH, 1, BLK, D) vc likewise   ← cache with unsqueeze(2)
        seq (B, 1) int             ← logical position where writing of new tokens starts

    Three things it does:
      (1) write k5/v5 to logical cache positions [seq, seq+S) (via block_table)
      (2) q5 attends over cache positions [0, seq+S)
      (3) within the new-token span: causal (prefill) / attend to all (decode)

    Warning: **writes in place** -- modifies `kc`/`vc`.  Same behavior as the
      vendor op, which is what makes the resident cache work.
    """
    B, KVH, G, S, D = q5.shape
    NBLK = kc.shape[0]
    start = int(seq.flatten()[0])
    total = start + S

    # (1) Write -- logical position p → (block b, offset o) → physical block block_table[0, b]
    for i in range(S):
        p = start + i
        b, o = p // block_size, p % block_size
        phys = int(block_table[0, b])
        kc[phys, :, 0, o, :] = k5[0, :, 0, i, :]
        vc[phys, :, 0, o, :] = v5[0, :, 0, i, :]

    # (2) Read -- gather [0, total) in logical order
    kk = torch.empty(B, KVH, total, D, dtype=q5.dtype)
    vv = torch.empty(B, KVH, total, D, dtype=q5.dtype)
    for p in range(total):
        b, o = p // block_size, p % block_size
        phys = int(block_table[0, b])
        kk[0, :, p, :] = kc[phys, :, 0, o, :]
        vv[0, :, p, :] = vc[phys, :, 0, o, :]

    # (3) Attention -- flatten the GQA group axis into the head axis and call standard SDPA
    q4 = q5.reshape(B, KVH * G, S, D)
    k4 = kk.repeat_interleave(G, dim=1)
    v4 = vv.repeat_interleave(G, dim=1)
    if prefill and not is_bidirectional:
        # new token i attends up to logical position start+i
        pos = torch.arange(total).view(1, total)
        row = (start + torch.arange(S)).view(S, 1)
        mask = torch.where(pos <= row, 0.0, float("-inf")).to(q5.dtype)
        mask = mask.view(1, 1, S, total)
    else:
        mask = None
    o4 = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask,
                                        is_causal=False, scale=float(scale))
    return o4.reshape(B, KVH, G, S, D)


def paged_vendor(q5, k5, v5, kc, vc, seq, scale, block_table, block_size,
                 *, prefill: bool, is_bidirectional: bool = False):
    """Call the vendor op.  Warning: no CPU kernel -- compile path only."""
    op = (torch.ops.rbln_custom_ops.paged_causal_attn_prefill if prefill
          else torch.ops.rbln_custom_ops.paged_causal_attn_decode)
    a = (q5, k5, v5, kc, vc, seq, scale, block_table, block_size)
    return op(*a, is_bidirectional) if prefill else op(*a)


# ─────────────────────────────────────────────────────────────────────
# 2. FX rewriter
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RewriteStats:
    sdpa_found: int = 0
    replaced: int = 0
    repeat_kv_removed: int = 0
    mask_dropped: int = 0
    skipped: list = field(default_factory=list)

    def __str__(self):
        s = (f"rewrote {self.replaced} of {self.sdpa_found} SDPA · "
             f"removed {self.repeat_kv_removed} repeat_kv · "
             f"dropped {self.mask_dropped} masks")
        return s + (f" · skipped {self.skipped}" if self.skipped else "")


def _unwrap_repeat_kv(node):
    """Unwind `reshape ← expand ← X` and return X.

    HF `repeat_kv` is `h[:, :, None].expand(b,kvh,n,s,d).reshape(b,kvh*n,s,d)`,
    so **the input of `expand` is already `(B,KVH,1,S,D)`** -- the vendor's shape.
    If it cannot be unwound, returns `None` and the caller skips that SDPA.
    """
    if node.op != "call_function" or "reshape" not in str(node.target):
        return None
    src = node.args[0]
    if not hasattr(src, "op") or "expand" not in str(src.target):
        return None
    return src.args[0]


def rewrite(gm, *, kvh: int, n_heads: int, head_dim: int, batch: int,
            prefill: bool, impl: str = "ref", block_size: int = BLOCK,
            verbose: bool = True, with_mask: str = "",
            _break: str = "") -> RewriteStats:
    """Rewrite the SDPAs of a captured graph into the paged op and add cache inputs.

    Added placeholders: `seq`, `block_table`, and per layer `past_key_values_{k,v}_i`.
    Warning: **K/V are not emitted as graph outputs** -- this is what made the
      path usable with TP>1 on rebel-compiler 0.10.x.

    `with_mask` selects how the mask is passed to the op:

        ""        no mask · `is_bidirectional=False`  ← single chunk only
        "pos"     `(1, max_seq)` · `is_bidirectional=True`   (use_position_ids)
        "causal"  `(1,1,chunk,max_seq)` → `unsqueeze(2)` inside the op ·
                  `is_bidirectional=False`
    **Required for chunked prefill** -- the mask must open up the previous
    chunks.  Without a mask **the second chunk cannot see the past** (measured:
    cos 0.249).

    Modifies `gm` **in place**.  To restore, the caller re-captures the original
    or keeps a `copy.deepcopy`.
    """
    g = gm.graph
    group = n_heads // kvh
    st = RewriteStats()
    if impl not in ("ref", "vendor"):
        raise ValueError(f"impl={impl!r} — expected 'ref' or 'vendor'")

    sdpas = [n for n in g.nodes if n.op == "call_function"
             and "scaled_dot_product_attention" in str(n.target)]
    st.sdpa_found = len(sdpas)
    if not sdpas:
        return st

    # ── Insert the new placeholders after the existing ones ────────────
    phs = [n for n in g.nodes if n.op == "placeholder"]
    anchor = phs[-1]
    new_ph = {}
    with g.inserting_after(anchor):
        # insert in reverse so the declaration order comes out right
        for i in reversed(range(len(sdpas))):
            for kv in ("v", "k"):
                nm = f"past_key_values_{kv}_{i}"
                new_ph[nm] = g.placeholder(nm)
        new_ph["block_table"] = g.placeholder("block_table")
        if with_mask:
            new_ph["attn_mask"] = g.placeholder("attn_mask")
            mask_node = new_ph["attn_mask"]
        new_ph["seq"] = g.placeholder("seq")

    scale = 1.0 / math.sqrt(head_dim)
    # Warning: the vendor op accepts `scale` **only as a Tensor** (measured: a float
    #   gives `Expected a value of type 'Tensor'`).  So it is registered as a buffer
    #   and referenced via `get_attr`, which carries through `torch.export` and rebel tracing.
    # `_break` is for **negative controls** only -- it deliberately breaks the
    # rewrite to confirm that the check is not vacuous.
    if _break == "scale":
        scale *= 2.0
    elif _break == "group":
        group = max(1, group // 2)

    if impl == "vendor":
        gm.register_buffer("_paged_scale", torch.tensor(scale), persistent=False)
        with g.inserting_after(anchor):
            scale_node = g.get_attr("_paged_scale")
    else:
        scale_node = scale          # a Python float suffices for the reference implementation
    for i, nd in enumerate(sdpas):
        q, k, v = nd.args[0], nd.args[1], nd.args[2]
        # Note: count the mask before erasing -- reading args after erase gives 0
        #   (the first version did that and reported "dropped 0 masks").
        had_mask = len(nd.args) > 3 and hasattr(nd.args[3], "op")
        k5, v5 = _unwrap_repeat_kv(k), _unwrap_repeat_kv(v)
        if k5 is None or v5 is None:
            st.skipped.append(nd.name)        # Note: never skip silently
            continue
        with g.inserting_before(nd):
            q5 = g.call_function(torch.ops.aten.view.default,
                                 (q, [batch, kvh, group, -1, head_dim]))
            base = (q5, k5, v5, new_ph[f"past_key_values_k_{i}"],
                    new_ph[f"past_key_values_v_{i}"], new_ph["seq"],
                    scale_node, new_ph["block_table"], block_size)
            if impl == "vendor":
                # Note: emit the **op node directly**, not a Python wrapper -- the
                #   compiler traces it, so the `call_function` target must be a torch op.
                op = (torch.ops.rbln_custom_ops.paged_causal_attn_prefill
                      if prefill else
                      torch.ops.rbln_custom_ops.paged_causal_attn_decode)
                if with_mask == "causal":
                    # the op takes the mask as 5-D: unsqueeze(2)
                    mk = g.call_function(torch.ops.aten.unsqueeze.default,
                                         (mask_node, 2))
                else:
                    mk = mask_node if with_mask else None
                if prefill:
                    # (… , block_size, is_bidirectional, mask=None, …)
                    extra = ((True, mk) if with_mask == "pos"
                             else (False, mk) if with_mask == "causal"
                             else (False,))
                else:
                    extra = (mk,) if with_mask else ()
                o5 = g.call_function(op, base + extra)
            else:
                o5 = g.call_function(paged_ref, base, {"prefill": prefill})
            o4 = g.call_function(torch.ops.aten.reshape.default,
                                 (o5, [batch, n_heads, -1, head_dim]))
        nd.replace_all_uses_with(o4)
        g.erase_node(nd)
        st.replaced += 1
        st.mask_dropped += int(had_mask)

    before = len(g.nodes)
    g.eliminate_dead_code()                   # ← repeat_kv disappears here
    st.repeat_kv_removed = (before - len(g.nodes))
    g.lint()
    gm.recompile()
    if verbose:
        print(f"[resident] {st}", flush=True)
    return st


def check_prefill_len(seq: int, block_size: int = BLOCK) -> None:
    """Reject prompts longer than one block.

    This path supports a prefill of up to `block_size` tokens in a single call
    (chunked prefill is not supported), so **`block_size` is the prompt length
    limit**.  Raise it with `NPU_POC_BLOCK`.
    """
    if seq > block_size:
        raise SystemExit(
            f"[resident] prompt {seq} tokens > block {block_size} — rejected.\n"
            f"  This path supports prompts of up to one block.\n"
            f"  Raise NPU_POC_BLOCK or keep seq <= {block_size}.")


def cache_tensors(n_layers: int, kvh: int, head_dim: int, n_blocks: int,
                  dtype=torch.float32, block_size: int = BLOCK):
    """One (K, V) pair per layer, built **5-D** in the vendor layout."""
    return [torch.zeros(n_blocks, kvh, 1, block_size, head_dim, dtype=dtype)
            for _ in range(2 * n_layers)]


# ─────────────────────────────────────────────────────────────────────
# 3. Self-check -- step (1) (rewriter CPU equivalence)
# ─────────────────────────────────────────────────────────────────────

def selfcheck(split_gm, stage_names, cfg, *, batch: int, seq: int,
              driver: dict, block_size: int = BLOCK, verbose: bool = True):
    """Compare **before ↔ after the rewrite** on CPU.  `impl="ref"` only.

    The vendor kernel has no CPU implementation (see module docstring), so this
    checks one thing only: **is the rewrite correct**.  Whether the vendor kernel
    is correct is checked on the NPU.

    Warning: modifies `split_gm` **in place** -- the caller must not use that
      object for the default path afterwards (`poc.py` returns right after the check).
    """
    import torch.nn as nn

    NH = cfg.num_attention_heads
    KVH = cfg.num_key_value_heads
    HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    check_prefill_len(seq, block_size)        # prompts longer than a block are not supported
    nblk = (seq + block_size - 1) // block_size
    args = [driver[k] for k in driver]

    with torch.no_grad():
        ref = split_gm(*args)
    ref = ref[0] if isinstance(ref, (list, tuple)) else ref

    total, caches_by = 0, {}
    for nm in stage_names:
        sub = split_gm.get_submodule(nm)
        if not any(n.op == "call_function"
                   and "scaled_dot_product_attention" in str(n.target)
                   for n in sub.graph.nodes):
            continue
        st = rewrite(sub, kvh=KVH, n_heads=NH, head_dim=HD, batch=batch,
                     prefill=True, impl="ref", block_size=block_size,
                     verbose=False)
        if verbose:
            print(f"[resident]   {nm}: {st}", flush=True)
        if st.skipped:
            return False, f"could not unwind SDPA: {st.skipped}"
        total += st.replaced
        caches_by[nm] = cache_tensors(st.replaced, KVH, HD, nblk,
                                      block_size=block_size)

    if total == 0:
        return False, "no SDPA found"

    seq0 = torch.tensor([[0]], dtype=torch.int32)
    bt = torch.arange(nblk, dtype=torch.int32).reshape(batch, nblk)

    class _Bound(nn.Module):
        """Binds the inputs added by the rewrite outside the stage boundary."""

        def __init__(self, inner, extra):
            super().__init__()
            self.inner, self.extra = inner, extra

        def forward(self, *a):
            return self.inner(*a, *self.extra)

    for nm, cs in caches_by.items():
        setattr(split_gm, nm,
                _Bound(split_gm.get_submodule(nm), [seq0, bt, *cs]))

    with torch.no_grad():
        got = split_gm(*args)
    got = got[0] if isinstance(got, (list, tuple)) else got

    x, y = got.flatten().double(), ref.flatten().double()
    cos = (x @ y / (x.norm() * y.norm() + 1e-30)).item()
    dmax = (got - ref).abs().max().item()
    rel = dmax / ref.abs().max().item()
    used = sum(float(c.abs().sum()) for cs in caches_by.values() for c in cs)
    ok = cos >= 0.999 and rel <= 2e-2 and used > 0
    msg = (f"rewrote {total} · cos={cos:.6f} |Δ|max={dmax:.4e} rel={rel:.4e} · "
           f"cache usage {used:.1f}")
    return ok, msg


def prune_kv_outputs(gm, keep: int = 1) -> int:
    """**Strip K/V from the graph outputs** -- keep only the first `keep` (usually logits).

    Why -- chunked prefill needs `position_ids` (from the second chunk on,
    positions do not start at 0), and the engine adds that input **only when
    `use_kv_cache=True`**.  But that mode **also emits K/V as outputs**, which this
    path must avoid to work with TP>1 on rebel-compiler 0.10.x.

    With the resident cache those outputs are **useless** -- the values are
    already in the device cache.  So strip them.  **This is the only way to get
    position_ids without losing TP.**

    Returns: the number of outputs removed.
    """
    out = next(n for n in gm.graph.nodes if n.op == "output")
    args = out.args[0]
    if not isinstance(args, (list, tuple)) or len(args) <= keep:
        return 0
    dropped = len(args) - keep
    out.args = (tuple(args[:keep]),)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return dropped


def prune_outputs_at(gm, drop: list) -> int:
    """Strip **only the given indices** from the graph outputs (for PP).

    `prune_kv_outputs` keeps the first `keep` outputs, so it **cannot be used for
    PP stages** -- an intermediate stage's outputs are mixed, like
    `[boundary values..., K/V...]`, and the next stage needs the boundary values.
    So only the **K/V positions** are removed.

    Warning: **the top-level FX graph hard-codes indices via `getitem(submod_k, i)`** --
      removing outputs here shifts those indices.  So the caller's wrapper
      **puts dummies back into the removed positions** to restore the count
      (`restore_dummies`).  This is safe because K/V only go to the top-level
      output and no later stage consumes them.
    """
    out = next(n for n in gm.graph.nodes if n.op == "output")
    args = list(out.args[0])
    keep = [a for i, a in enumerate(args) if i not in set(drop)]
    out.args = (tuple(keep),)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return len(args) - len(keep)


def restore_dummies(vals, drop: list, total: int):
    """Re-insert `None` at the positions removed by `prune_outputs_at` to restore the count."""
    it = iter(vals)
    return tuple(None if i in set(drop) else next(it) for i in range(total))


def drop_unused_placeholders(gm) -> list:
    """Remove **inputs that are no longer used** from the graph and return their original indices.

    When the rewrite drops the mask, that boundary input remains as a
    **placeholder nobody uses**.  For NPUs that remove unused graph inputs at
    compile time, a caller that still passes it fails with an input-count
    mismatch (`Number of inputs (8) exceeds the expected number (7)`).

    → We **remove them first**, and the caller **skips** the returned indices.
    """
    ph = [n for n in gm.graph.nodes if n.op == "placeholder"]
    dead = [i for i, n in enumerate(ph) if not n.users]
    for i in reversed(dead):
        gm.graph.erase_node(ph[i])
    if dead:
        gm.graph.lint()
        gm.recompile()
    return dead
