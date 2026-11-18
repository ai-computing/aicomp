"""Model patches — integration item #5 "per-backend patch sets".

The backend decides whether a patch is needed; the transformers version decides its contents.

    ATOM+ × tf 4.46.2 :  0   stock HF Llama-3.2-1B (1.24B) compiles as a whole
    ATOM+ × tf 5.x    :  1   packed_seq — needed for submod_0 to compile
    RNGD  × tf 5.1    :  4   (3 effective) — ported in the next step (as of SDK 2026.2.0.
                               SDK 2026.4.0 requires transformers ≥5.8, and the needed
                               set shrinks to two: rotary expand · skip mask construction)

Warning: do not reuse a patch set across backends — a patch for one backend can
  backfire on the other.  RNGD's "skip mask" patch removes the mask on ATOM+, which
  changes the attention graph and turns 3/4 compiled stages into 0/4.  What decides
  pass/fail is not what is computed but in what form it is written.

Warning: always verify and report whether a patch actually took effect.  The old
  PoC's patch_transformers_masking_for_fx() printed success even though its target
  symbol exists in neither tf 4.46.2 nor tf 5.x — it was creating a new attribute
  nobody reads.  → Here we return the number of rebound references, and 0 is
  reported as a failed application.
"""

from __future__ import annotations

import functools
import sys

import torch.nn as _nn


def _rebind(attr: str, old, new) -> int:
    """Replacing the module attribute alone is not enough — rebind already-imported namespaces.

    `modeling_llama.py` binds via `from ...masking_utils import create_causal_mask`
    into its own namespace, so patching only the source module does not change
    the call path.
    """
    n = 0
    for m in list(sys.modules.values()):
        if m is not None and getattr(m, attr, None) is old:
            setattr(m, attr, new)
            n += 1
    return n


# ─────────────────────────────────────────────────────────────────────

def packed_seq() -> tuple[bool, str]:
    """C5 — disable `find_packed_sequence_indices` (minimal patch for tf 5.x × ATOM+).

    For a single sequence this function normally returns None, but during export
    the `is_tracing()` guard prevents the early return, leaving diff/cumsum nodes
    in the graph, and submod_0 does not compile on ATOM+ with them.
    Mask construction itself is left untouched — unlike RNGD's skip-mask patch.
    """
    try:
        import transformers.masking_utils as mu
    except Exception:
        return False, "transformers.masking_utils not found (tf < 5.0) — not applied"
    old = getattr(mu, "find_packed_sequence_indices", None)
    if old is None:
        return False, "symbol find_packed_sequence_indices not found — not applied"
    new = lambda position_ids: None                       # noqa: E731
    mu.find_packed_sequence_indices = new
    n = _rebind("find_packed_sequence_indices", old, new)
    return True, f"find_packed_sequence_indices → None ({n} references rebound)"


def float_causal_mask() -> tuple[bool, str]:
    """C6 — build the causal mask as float additive instead of bool.

    HF's `create_causal_mask` passes `allow_is_causal_skip` for sdpa and `dtype`
    for eager, so the two paths produce different masks.  The sdpa path emits bool
    during export (and for padded batches).  On backends that do not take a bool
    attn_mask as a stage input, the boundary mask is made float additive instead.
    """
    import torch
    try:
        import transformers.masking_utils as mu
    except Exception:
        return False, "transformers.masking_utils not found (tf < 5.0) — not applied"
    old = getattr(mu, "create_causal_mask", None)
    if old is None:
        return False, "symbol create_causal_mask not found — not applied"

    return _float_mask(mu, old, follow_dtype=False)


def float_causal_mask_native() -> tuple[bool, str]:
    """C6 variant — float additive mask in the model's compute dtype (RNGD set).

    Same as `float_mask`, but the fill value is the minimum of the dtype of the
    input embeddings instead of float32, so a bf16 model gets a bf16 mask whose
    fill value stays within the bf16 range.
    """
    try:
        import transformers.masking_utils as mu
    except Exception:
        return False, "transformers.masking_utils not found (tf < 5.0) — not applied"
    old = getattr(mu, "create_causal_mask", None)
    if old is None:
        return False, "symbol create_causal_mask not found — not applied"
    return _float_mask(mu, old, follow_dtype=True)


def _float_mask(mu, old, follow_dtype: bool) -> tuple[bool, str]:
    import inspect
    import torch
    names = list(inspect.signature(old).parameters)

    def new(*a, **kw):
        m = old(*a, **kw)
        if m is not None and m.dtype == torch.bool:
            dt = torch.float32
            if follow_dtype:
                emb = None
                for key in ("inputs_embeds", "input_embeds"):    # name differs across versions
                    if emb is None:
                        emb = kw.get(key)
                    if emb is None and key in names[:len(a)]:
                        emb = a[names.index(key)]
                if emb is not None and emb.is_floating_point():
                    dt = emb.dtype
            m = torch.where(m, torch.zeros((), dtype=dt),
                            torch.full((), torch.finfo(dt).min, dtype=dt))
        return m

    new = functools.wraps(old)(new)          # keep the original signature visible
    mu.create_causal_mask = new
    n = _rebind("create_causal_mask", old, new)
    which = "model dtype" if follow_dtype else "float32"
    return True, (f"create_causal_mask: bool → float additive ({which}) "
                  f"({n} references rebound)")


def decode_mask_arith() -> tuple[bool, str]:
    """Build the decode-step attention mask with arithmetic ops (RNGD set).

    With a KV cache and a 2-D padding mask, the mask is causal (key position
    <= query position) intersected with padding.  It is built here from integer
    positions, a select and a scale in float32, then cast to the model dtype —
    the same 0 / dtype-minimum additive mask, without boolean gather or AND.
    Applies to the transformers 5.19-style mask arguments; any other call is
    delegated to the original function.
    """
    import inspect
    import torch
    try:
        import transformers.masking_utils as mu
    except Exception:
        return False, "transformers.masking_utils not found (tf < 5.0) — not applied"
    old = getattr(mu, "create_causal_mask", None)
    prep = getattr(mu, "_preprocess_mask_arguments", None)
    if old is None or prep is None:
        return False, "mask helpers not found — not applied"
    names = list(inspect.signature(old).parameters)
    if "inputs_embeds" not in names or "cache_position" in names:
        return False, "mask arguments of another transformers version — not applied"

    def new(*a, **kw):
        b = dict(zip(names, a))
        b.update(kw)
        cfg, emb, am = b.get("config"), b.get("inputs_embeds"), b.get("attention_mask")
        pkv = b.get("past_key_values")
        if (pkv is None or not isinstance(am, torch.Tensor) or am.dim() != 2
                or b.get("or_mask_function") is not None
                or b.get("and_mask_function") is not None
                or b.get("block_sequence_ids") is not None
                or not getattr(cfg, "is_causal", True)
                or getattr(cfg, "sliding_window", None) is not None):
            return old(*a, **kw)
        layer_idx = b.get("layer_idx")
        if layer_idx is None:
            sl = getattr(pkv, "is_sliding", None)
            layer_idx = sl.index(False) if sl is not None and False in sl else 0
        out = prep(cfg, emb, am, pkv, b.get("position_ids"), layer_idx)
        if out[0] or out[2] is not None:
            return old(*a, **kw)
        _, _, _, q_len, kv_len, q_off, kv_off = out
        pad = am
        if not (kv_off == 0 and am.shape[-1] == kv_len):
            pad = am[:, kv_off:kv_off + kv_len]
        q_idx = torch.arange(q_len, device=am.device) + q_off
        kv_idx = torch.arange(kv_len, device=am.device) + kv_off
        causal = (kv_idx[None, :] - q_idx[:, None]) <= 0                 # (q, kv)
        mn = torch.finfo(emb.dtype).min
        pad_add = pad.to(torch.float32) * (-mn) + mn                      # 0 or min
        pad_add = pad_add[:, None, None, :]                               # (B, 1, 1, kv)
        masked = pad_add * 0 + mn
        full = torch.where(causal[None, None], pad_add, masked)           # (B, 1, q, kv)
        return full.to(emb.dtype)

    new = functools.wraps(old)(new)          # keep the original signature visible
    mu.create_causal_mask = new
    n = _rebind("create_causal_mask", old, new)
    return True, f"create_causal_mask: arithmetic decode mask ({n} references rebound)"


# ─────────────────────────────────────────────────────────────────────

def skip_causal_mask() -> tuple[bool, str]:
    """Do not build a causal mask for unpadded prefill (RNGD set).

    When there is no padding mask, no KV cache and no extra mask function,
    `create_causal_mask` returns None and SDPA handles causality with
    `is_causal`.  Every other call (padded batches, decode with a cache) is
    delegated to the original function, so generation is unaffected.
    """
    import inspect
    try:
        import transformers.masking_utils as mu
    except Exception:
        return False, "transformers.masking_utils not found (tf < 5.0) — not applied"
    old = getattr(mu, "create_causal_mask", None)
    if old is None:
        return False, "symbol create_causal_mask not found — not applied"
    names = list(inspect.signature(old).parameters)

    def new(*a, **kw):
        bound = dict(zip(names, a))
        bound.update(kw)
        if (bound.get("attention_mask") is None
                and bound.get("past_key_values") is None
                and bound.get("or_mask_function") is None
                and bound.get("and_mask_function") is None):
            return None
        return old(*a, **kw)

    new = functools.wraps(old)(new)          # keep the original signature visible
    mu.create_causal_mask = new
    n = _rebind("create_causal_mask", old, new)
    return True, f"create_causal_mask: None for unpadded prefill ({n} references rebound)"


# ─────────────────────────────────────────────────────────────────────
# Model-instance patches — applied after the model is loaded (apply_model)
# ─────────────────────────────────────────────────────────────────────

def rotary_expand(model) -> tuple[bool, str]:
    """Materialize RoPE cos/sin to the hidden batch size (RNGD set).

    HF rotary embeddings return cos/sin of shape (1, S, D) that broadcast over
    the batch inside every layer.  A forward hook expands them to (B, S, D) at
    the rotary module, i.e. before they cross a stage boundary.  Values are only
    replicated, so the output is bit-identical.  With batch 1 it is a no-op.
    A hook is used instead of wrapping forward so that the engine's
    reconstructed call signature does not matter.
    """
    import torch
    n = 0
    for mod in model.modules():
        if not type(mod).__name__.endswith("RotaryEmbedding"):
            continue
        if getattr(mod, "_npu_poc_batch_expand", False):
            continue

        def hook(_m, args, out):
            if not (isinstance(out, tuple) and len(out) == 2
                    and all(isinstance(t, torch.Tensor) for t in out)):
                return out
            b = next((x.shape[0] for x in args
                      if isinstance(x, torch.Tensor) and x.dim() >= 1), None)
            cos, sin = out
            if b is None or b <= 1 or cos.dim() < 2 or cos.shape[0] != 1:
                return out
            rest = [-1] * (cos.dim() - 1)
            return (cos.expand(b, *rest).contiguous(), sin.expand(b, *rest).contiguous())

        mod.register_forward_hook(hook)
        mod._npu_poc_batch_expand = True
        n += 1
    if not n:
        return False, "no RotaryEmbedding module — not applied"
    return True, f"{n} RotaryEmbedding module(s) expand cos/sin to the batch size"


class _ExplicitLayerNorm(_nn.Module):
    """LayerNorm computed as mean((x - mu)^2) — numerically equal to nn.LayerNorm
    (biased variance), without the fused var_mean op."""

    def __init__(self, src):
        super().__init__()
        self.normalized_shape = tuple(src.normalized_shape)
        self.eps = src.eps
        self.weight = src.weight if src.elementwise_affine else None
        self.bias = getattr(src, "bias", None) if src.elementwise_affine else None

    def forward(self, x):
        import torch
        dims = tuple(range(-len(self.normalized_shape), 0))
        d = x - x.mean(dims, keepdim=True)
        y = d * torch.rsqrt((d * d).mean(dims, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


def layernorm(model) -> tuple[bool, str]:
    """Replace nn.LayerNorm with an explicit mean/variance form (RNGD set).

    For backends that do not take the fused `var_mean` op.  Parameters are
    shared with the original module (same dtype/device).  Llama uses RMSNorm,
    so this is a no-op there; GPT-2 is affected.
    """
    n = 0

    def walk(mod):
        nonlocal n
        for name, child in list(mod.named_children()):
            if isinstance(child, _nn.LayerNorm):
                setattr(mod, name, _ExplicitLayerNorm(child))
                n += 1
            else:
                walk(child)

    walk(model)
    if not n:
        return False, "no nn.LayerNorm — not applied"
    return True, f"{n} nn.LayerNorm → explicit mean/variance"


_REGISTRY = {
    "packed_seq": packed_seq,
    "float_mask": float_causal_mask,
    "float_mask_native": float_causal_mask_native,
    "skip_causal_mask": skip_causal_mask,
    "decode_mask_arith": decode_mask_arith,
}
_MODEL_REGISTRY = {
    "rotary_expand": rotary_expand,
    "layernorm": layernorm,
}


def apply(names) -> list[str]:
    """Apply the named patches and return only those that actually took effect."""
    applied = []
    for nm in names:
        if nm in _MODEL_REGISTRY:
            continue                     # applied after model load (apply_model)
        fn = _REGISTRY.get(nm)
        if fn is None:
            raise SystemExit(f"[patches] unknown patch: {nm!r} "
                             f"(available: {', '.join([*_REGISTRY, *_MODEL_REGISTRY])})")
        ok, msg = fn()
        print(f"[patches] {nm}: {'applied' if ok else 'NOT applied'} — {msg}", flush=True)
        if ok:
            applied.append(nm)
    if not names:
        print("[patches] 0 patches", flush=True)
    return applied


def apply_model(model, names) -> list[str]:
    """Apply the model-instance patches among `names`.  Prints nothing if there are none."""
    applied = []
    for nm in names:
        fn = _MODEL_REGISTRY.get(nm)
        if fn is None:
            continue
        ok, msg = fn(model)
        print(f"[patches] {nm}: {'applied' if ok else 'NOT applied'} — {msg}", flush=True)
        if ok:
            applied.append(nm)
    return applied
