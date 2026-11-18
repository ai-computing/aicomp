#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""PP + TP PoC — runs opt_prime's capture + split on a grid of Korean NPUs.

    cd opt_prime/examples            # from the repository root (the directory containing npu_poc/)
    python3 -m npu_poc.poc --model synthetic --pp 2
    python3 -m npu_poc.poc --model meta-llama/Llama-3.2-1B --pp 2 --tp 2

Before running, set up the NPU software stack (driver, firmware, runtime and the
vendor's Python compiler package) following the vendor documentation, and run with
a Python environment that holds the matching compiler package:

    ATOM+ (--backend rbln, default)  rebel-compiler matching the installed ATOM+ driver
                                     https://docs.rbln.ai
    RNGD  (--backend rngd)           furiosa-torch      https://developer.furiosa.ai

A compiler package that does not match the installed driver stops while loading the
runtime (e.g. "librbln-thunk.so library version is not in the supported range").
A separate virtual environment per backend avoids picking up another installation
(e.g. one in the user site-packages).  Versions validated: README.md "Requirements".

Parallelism ownership (keep it this way)
    PP  owned by opt_prime — IR capture · split · stage wiring (FX forward)
    TP  owned by the backend — inside the rebel binary on ATOM+, PE fusion on RNGD

The engine (`opt_prime/`) is not modified by a single line.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# This file is examples/npu_poc/poc.py, so opt_prime/ is two levels up.
# It must be on sys.path for `opt_prime.IR` to import (the engine is not pip-installed).
_HERE = os.path.dirname(os.path.abspath(__file__))          # examples/npu_poc
_ROOT = os.path.dirname(os.path.dirname(_HERE))             # opt_prime
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("RANK", "0")           # silence the engine's rank-based prints

import torch                                                      # noqa: E402

from . import compat, equivalence, models, patches, placement, runner  # noqa: E402
from .backends.base import get_backend                            # noqa: E402
from .capture import capture_and_split                            # noqa: E402
from . import stage_io                                            # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="npu_poc.poc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    g = p.add_argument_group("target")
    g.add_argument("--model", default="synthetic",
                   help="'synthetic' · 'gpt2' · HF model id "
                        "(e.g. meta-llama/Llama-3.2-1B)")
    g.add_argument("--batch", type=int, default=1)
    g.add_argument("--seq-len", type=int, default=16)
    g.add_argument("--seed", type=int, default=0,
                   help="input seed.  Without a seed, rel varies 2.7x even with "
                        "the same binary and PASS/FAIL flips (measured) — "
                        "do not disable")

    g = p.add_argument_group("grid")
    g.add_argument("--pp", type=int, default=2, help="number of pipeline stages")
    g.add_argument("--tp", type=int, default=1,
                   help="backend-managed tensor-parallel size.  On ATOM+, "
                        "rebel bakes the collectives into the binary "
                        "(opt_prime's DTensor TP is not used)")
    g.add_argument("--partitioner", default="simple",
                   choices=["simple", "llama-tp-split", "milp", "hierarchical",
                            "balanced", "balanced-layer"],
                   help="how to cut PP stages.  The first four are engine "
                        "partitioners; 'balanced' is the PoC's compute-balanced "
                        "split — none of the engine partitioners optimizes "
                        "compute balance, yet pipeline throughput is set by "
                        "max(t_stage)")
    g.add_argument("--generate", type=int, default=0, metavar="N",
                   help="generate N tokens with 1 prefill + N-1 decode steps.  "
                        "All earlier measurements were a single pass — this "
                        "path is the first to produce a second token.  The "
                        "cache is made functional (graph inputs/outputs), so "
                        "there are zero model modifications and no "
                        "vendor-specific ops")
    g.add_argument("--buckets", default=None,
                   help="decode buckets (e.g. 128,256,512).  Binaries have a "
                        "fixed shape, so each past length needs its own; "
                        "buckets reduce that to a few.  With --inplace-cache "
                        "the smallest bucket that holds prompt + N is the "
                        "cache capacity")
    g.add_argument("--inplace-cache", action="store_true",
                   help="with --generate: keep each stage's KV cache on its "
                        "device and update it in place (index_copy_) instead "
                        "of sending the cache both ways every step.  Standard "
                        "PyTorch ops only; the captured graphs are rewritten "
                        "before the split and split with llama-tp-split.  "
                        "Llama-family models · batch 1 · validated on rngd "
                        "(inplace_cache.py)")
    g.add_argument("--use-kv-cache", action="store_true",
                   help="capture with the KV cache enabled.  HF returns "
                        "past_key_values as outputs, so 32 layers put 64 (K,V) "
                        "tensors outside the graph — the question is whether "
                        "PP split and the vendor compiler accept them.  All "
                        "earlier measurements were a single cache-less prefill")
    g.add_argument("--cost-alpha", type=float, default=0.0, metavar="A",
                   help="cost model for the balanced partitioners: cost = "
                        "#params + A × #output elements.  0 means params only — "
                        "breaks at long seq (lm_head is large in its output, "
                        "not its params)")
    g.add_argument("--device-ids", default=None,
                   help="devices to use (e.g. 0,1,2,3).  Length must be pp × tp. "
                        "This is a shared machine, so take only what you need")
    g.add_argument("--cpu-stages", default="",
                   help="stage indices to keep on the CPU (e.g. 0).  Those stages are "
                        "not compiled; the original GraphModule runs on the host and "
                        "the remaining stages run on the NPU.  --device-ids then "
                        "covers only the NPU stages (heterogeneous placement, "
                        "integration item #4).  Default: none")

    g = p.add_argument_group("capture · version compatibility")
    g.add_argument("--backend", default="rbln", choices=["rbln", "rngd"])
    g.add_argument("--hf-tracer", action="store_true",
                   help="capture with HFTracer (default is torch.export).  "
                        "Unavailable on transformers >= 5.0, which has no "
                        "`transformers.utils.fx`")
    g.add_argument("--dynamic-capture", action="store_true",
                   help="capture with the engine default (dynamic batch+seq).  "
                        "Default is static — static capture leaves zero "
                        "boundary scalars, so no freezing wrapper is needed, "
                        "and NPU binaries have fixed shapes anyway")
    g.add_argument("--dtype", choices=["float32", "bfloat16"], default=None,
                   help="compute dtype.  If the backend forces one, it wins")
    g.add_argument("--attn", choices=["eager", "sdpa"], default=None,
                   help="attn_implementation.  Default sdpa (= transformers "
                        "default).  A first-class variable that decides "
                        "pass/fail on ATOM+, and its direction differs by "
                        "model (e.g. GPT-2 is used with sdpa)")
    g.add_argument("--patch", action="append", default=[],
                   dest="patch_names", metavar="NAME",
                   help="model patches (packed_seq · float_mask · float_mask_native · decode_mask_arith).  If omitted, "
                        "chosen automatically from version and backend")
    g.add_argument("--no-auto-patch", action="store_true",
                   help="disable automatic selection; use only --patch")
    g.add_argument("--allow-bool-boundary", action="store_true",
                   help="proceed even with bool tensors on a stage boundary "
                        "(for control experiments)")

    g = p.add_argument_group("resident cache (experimental · off by default)")
    g.add_argument("--resident-cache", action="store_true",
                   help="keep the KV cache resident on the device instead of "
                        "as graph inputs/outputs.  Warning: uses vendor-specific "
                        "ops (paged_causal_attn_* · mark_static_address), so it "
                        "is NOT the default — it is a control that measures what "
                        "giving up the goal (avoiding vendor implementations) "
                        "buys (answer: 1.6x longer context; with rebel-compiler 0.10.x also TP>1).  "
                        "This flag is a rewriter self-check — it rewrites the "
                        "SDPA of the captured/split graph into the vendor form, "
                        "compares against the original on CPU, and exits.  For "
                        "NPU execution use "
                        "`python3 -m npu_poc.resident_generate` (TP) · "
                        "`npu_poc.resident_pp_generate` (PP×TP)")

    g = p.add_argument_group("measurement")
    g.add_argument("--input", choices=["random", "text"], default=None,
                   help="driver input.  Default: 'text' for HF models, 'random' "
                        "for --model synthetic (it has no tokenizer).  "
                        "'random' is random tokens from "
                        "torch.randint; logits come out flat and the top two "
                        "are often near-tied, so argmax drops because of the "
                        "input.  'text' tokenizes real sentences and truncates "
                        "to seq (no padding)")
    g.add_argument("--microbatches", type=int, default=1, metavar="M",
                   help="stream M requests pipelined to measure throughput.  "
                        "PP's real value is here, not in latency — compared "
                        "against a serial baseline using the same binaries")
    g.add_argument("--repeat", type=int, default=3,
                   help="number of runs.  Timings fluctuate on a shared "
                        "machine, so report a range")
    a = p.parse_args(argv)
    if a.input is None:
        # Real sentences by default: random tokens flatten the logits and make
        # argmax drop because of the input, not the NPU.  The synthetic model has
        # no tokenizer, so it keeps random tokens.
        a.input = "random" if a.model == "synthetic" else "text"
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    torch.manual_seed(a.seed)

    print("=" * 78, flush=True)
    print(f"[poc] PP={a.pp} × TP={a.tp} · model={a.model} · "
          f"batch={a.batch} seq={a.seq_len} seed={a.seed} input={a.input}",
          flush=True)
    print("=" * 78, flush=True)

    # 1) Backend and its capability declaration ─────────────────────
    kw = {"make_async": a.microbatches > 1}
    if a.backend == "rngd":
        kw["tp_size"] = a.tp          # RNGD fixes TP (PE fusion) when the backend is created
    backend = get_backend(a.backend, **kw)
    caps = backend.caps()
    placement.check_shared_machine()

    # 2) Version compatibility plan ─────────────────────────────────
    plan = compat.build_plan(
        caps,
        want_dynamo=not a.hf_tracer,
        want_static=not a.dynamic_capture,
        attn=a.attn,
        dtype=getattr(torch, a.dtype) if a.dtype else None,
        patches=tuple(a.patch_names),
        auto_patch=not a.no_auto_patch,
    )
    plan.log()

    # 3) Model patches — applied BEFORE capture ─────────────────────
    if a.model == "synthetic":
        plan.patch_set = ()
    patches.apply(plan.patch_set)

    # 4) Load the model ─────────────────────────────────────────────
    model, cls = models.load(a.model, dtype=plan.compute_dtype, attn=plan.attn,
                             use_cache=a.use_kv_cache)
    patches.apply_model(model, plan.patch_set)       # model-instance patches (none for ATOM+)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[poc] {cls} params={n_params:,} dtype={plan.compute_dtype}",
          flush=True)

    # 5) Generation path — prefill + decode ─────────────────────────
    #    Does not use the PP split here.  Making the cache functional
    #    (graph inputs/outputs) comes first; combining with PP is next.
    cpu_idx = sorted({int(x) for x in a.cpu_stages.split(",") if x.strip()})
    if a.microbatches > 1 and not caps.supports_async:
        raise SystemExit(f"[poc] {caps.name} does not support async submission "
                         f"(--microbatches)")
    if cpu_idx and a.microbatches > 1:
        raise SystemExit("[poc] --cpu-stages is not supported with --microbatches")
    if cpu_idx and a.generate > 1 and a.pp < 2:
        raise SystemExit("[poc] --cpu-stages with --generate needs --pp 2 or more")
    if a.inplace_cache:
        if a.generate < 2:
            raise SystemExit("[poc] --inplace-cache needs --generate 2 or more")
        if a.backend != "rngd":
            raise SystemExit("[poc] --inplace-cache is validated on --backend rngd only")
        if a.batch != 1:
            raise SystemExit("[poc] --inplace-cache supports --batch 1 only")
        if a.pp < 2:
            raise SystemExit("[poc] --inplace-cache needs --pp 2 or more (a single "
                             "Llama-3.2-1B stage did not compile on rngd)")
    if a.generate > 1:
        return _generate(a, model, backend, cpu_idx)

    # 5) Engine: capture + split (unmodified) ───────────────────────
    cap = capture_and_split(model, a.pp, plan, batch=a.batch, seq=a.seq_len,
                            partitioner=a.partitioner,
                            cost_alpha=a.cost_alpha,
                            use_kv_cache=a.use_kv_cache)
    split_gm, names = cap.split_gm, cap.stage_names

    # 5-b) Resident cache — optional (off by default).  If on, run the CPU
    #      rewrite check and exit.  NPU execution is handled by
    #      resident_generate.py · resident_pp_generate.py.
    if a.resident_cache:
        from . import resident
        print("[poc] Warning: resident cache path — uses vendor-specific ops "
              "(not the default)",
              flush=True)
        ok, msg = resident.selfcheck(
            split_gm, names, model.config, batch=a.batch, seq=a.seq_len,
            driver=_make_inputs(a, model)[0])
        print(f"[poc] rewritten vs original (CPU): {msg}", flush=True)
        print(f"[poc] RESULT {'PASS' if ok else 'FAIL'} resident-rewrite "
              f"model={a.model} pp={a.pp}", flush=True)
        return 0 if ok else 1

    # 6) CPU reference + boundary contract ──────────────────────────
    driver = _augment(_make_inputs(a, model)[0], split_gm, a)
    specs, cpu_ref, cpu_split = stage_io.collect(split_gm, names, model, driver)

    # Wiring check — does the split match the original?  Passing in CPU
    # eager alone is not enough (eager silently accepts a misaligned
    # position_ids), but if it fails here there is no point looking at the NPU.
    v = equivalence.judge(stage_io._logits_of(cpu_split), cpu_ref)
    print(f"[poc] split ↔ original (CPU): {v}", flush=True)
    if not v.ok:
        print("[poc] FAIL split output differs from the original — fix this "
              "before going to the NPU", flush=True)
        return 2

    print(f"[poc] boundary contract:", flush=True)
    for nm in names:
        print(f"[poc]   {nm}: {specs[nm].summary()}", flush=True)
    elems, kib = stage_io.traffic(specs, names)
    print(f"[poc] inter-stage transfer (via host): {elems:,} elements · {kib:.1f} KiB · "
          f"{max(len(names) - 1, 0)} boundaries", flush=True)

    # Heterogeneous placement — stages in --cpu-stages stay on the host as the
    # original GraphModule; only the rest are placed, compiled and swapped.
    bad = [i for i in cpu_idx if i >= len(names)]
    if bad:
        raise SystemExit(f"[poc] --cpu-stages {bad} out of range (stages: {len(names)})")
    npu_names = [nm for i, nm in enumerate(names) if i not in cpu_idx]
    if not npu_names:
        raise SystemExit("[poc] --cpu-stages leaves no stage for the NPU")
    stage_io.check_boundary_dtypes({nm: specs[nm] for nm in npu_names}, caps,
                                   strict=not a.allow_bool_boundary)

    # 7) Placement ──────────────────────────────────────────────────
    dev_ids = [int(x) for x in a.device_ids.split(",")] if a.device_ids else None
    stage_devs = placement.assign(len(npu_names), a.tp, dev_ids, caps)
    print(f"[poc] {placement.describe(npu_names, stage_devs, a.tp, backend.name)}"
          + (f" · on CPU: {', '.join(names[i] for i in cpu_idx)}" if cpu_idx else ""),
          flush=True)

    # 8) Compile ────────────────────────────────────────────────────
    compiled = {}
    total_compile = 0.0
    for nm, devs in zip(npu_names, stage_devs):
        spec = specs[nm]
        target = stage_io.specialize(split_gm.get_submodule(nm), spec)
        tensors = [torch.zeros(s, dtype=getattr(torch, d) if d != "bool"
                               else torch.bool)
                   for (_, s, d) in spec.tensors]
        cs = backend.compile(target, tensors, name=nm, devices=devs,
                             tp_size=a.tp,
                             input_names=[n for (n, _, _) in spec.tensors])
        compiled[nm] = cs
        total_compile += cs.stats.total_s
    print(f"[poc] total compile {total_compile:.1f}s", flush=True)

    # 9) Micro-batch path — throughput ──────────────────────────────
    if a.microbatches > 1:
        return _throughput(a, model, split_gm, names, compiled, specs,
                           driver, total_compile)

    # 9) Run — swap submods for NPU wrappers and let FX forward drive ──
    wrappers = runner.attach(split_gm, compiled, specs)
    out, t_min, t_max, _ = runner.run(split_gm, driver, repeat=a.repeat)
    print(f"[poc] pipeline {t_min:.1f}~{t_max:.1f}ms ({a.repeat} runs) · "
          f"{runner.stage_times(wrappers)}", flush=True)
    if a.repeat > 1:
        print(f"[poc] repeat check: {a.repeat} runs "
              + ("bit-identical" if runner.last_repeat_identical
                 else "NOT identical — outputs changed between runs of the same input"),
              flush=True)

    # 10) Verdict ───────────────────────────────────────────────────
    npu_ref = stage_io._logits_of(out)
    want_argmax = npu_ref.dim() >= 2 and a.model != "synthetic"
    v = equivalence.judge(npu_ref, cpu_ref, want_argmax=want_argmax)

    # ── With multiple outputs, check ALL of them ──────────────────
    # With the KV cache on, every layer also emits (K,V) besides logits
    # (33 outputs for 16 layers).  Checking only out[0] supports "it compiled
    # and the logits are right" and leaves the cache tensors unverified —
    # in decode those tensors are the next step's input, so if they are
    # wrong everything is wrong.
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    refs = list(cpu_split) if isinstance(cpu_split, (list, tuple)) else [cpu_split]
    if len(outs) > 1 and len(outs) == len(refs):
        vs = [equivalence.judge(o, r) for o, r in zip(outs, refs)
              if isinstance(o, torch.Tensor) and isinstance(r, torch.Tensor)]
        n_ok = sum(1 for x in vs if x.ok)
        worst_i = min(range(len(vs)), key=lambda i: (vs[i].ok, vs[i].cos))
        print(f"[poc] all outputs ({len(vs)}): {n_ok}/{len(vs)} PASS · "
              f"cos {min(x.cos for x in vs):.6f}~{max(x.cos for x in vs):.6f} · "
              f"rel {min(x.rel for x in vs):.3e}~{max(x.rel for x in vs):.3e} · "
              f"worst out[{worst_i}] {vs[worst_i]}", flush=True)
    print("-" * 78, flush=True)
    print(f"[poc] {equivalence.thresholds_line()}", flush=True)
    print(f"[poc] NPU ↔ CPU : {v}", flush=True)
    print(f"[poc] RESULT {'PASS' if v.ok else 'FAIL'} "
          f"pp={a.pp} tp={a.tp} model={a.model} "
          f"compile={total_compile:.1f}s run={t_min:.1f}ms "
          f"cos={v.cos:.6f} rel={v.rel:.4e}"
          + (f" argmax={v.argmax * 100:.2f}%" if v.argmax is not None else ""),
          flush=True)
    return 0 if v.ok else 1


def _augment(d: dict, split_gm, a) -> dict:
    """Fill in the placeholders the top-level graph requires.

    With the KV cache on, the engine adds `position_ids` as an input
    (`IR._get_export_input_names`).  Reading what the graph actually
    requires is safer than listing inputs by hand per model.
    """
    want = [n.target for n in split_gm.graph.nodes if n.op == "placeholder"]
    missing = [w for w in want if w not in d]
    if not missing:
        return d
    import torch as _t
    out = dict(d)
    b, s_ = a.batch, a.seq_len
    for w in missing:
        if w == "position_ids":
            out[w] = _t.arange(s_, dtype=_t.long).unsqueeze(0).expand(b, -1).contiguous()
        elif w == "attention_mask":
            out[w] = _t.ones(b, s_, dtype=_t.long)
        else:
            raise SystemExit(f"[poc] don't know how to build graph input '{w}' "
                             f"(required: {want})")
    print(f"[poc] added driver inputs: {missing} (read from graph placeholders)", flush=True)
    # Order by graph placeholder order — the graph is called as split_gm(*values).
    return {w: out[w] for w in want}


def _make_inputs(a, model, n: int = 1) -> list:
    """n driver inputs.  --input selects the distribution.

    Note: measure with the distribution the model actually sees.  Random
      tokens overestimate near-tie (boundary) effects, and torch.randn
      activations underestimate error.  Same trap, opposite directions.
    """
    if a.input == "text":
        if a.model == "synthetic":
            raise SystemExit("[poc] --input text is only for HF models")
        return models.text_inputs(a.model, batch=a.batch, seq=a.seq_len, n=n)
    return [models.driver_inputs(model, batch=a.batch, seq=a.seq_len)
            for _ in range(n)]


def classify_step(npu_logits, cpu_logits, ref_tok: int, npu_tok: int,
                  cos_min: float = 0.999):
    """Teacher-forced step whose NPU token differs from the CPU token: tie or miss?

    The step's error e is the NPU↔CPU max|Δ| over the vocabulary EXCLUDING the
    two disputed tokens — otherwise a single wrong logit inflates e enough to
    excuse itself.  Tie only if both backends see the two tokens within e of
    each other (CPU gap ≤ e and NPU gap ≤ e) and the logits' cosine is ≥ cos_min.
    Returns (is_tie, description).
    """
    nl, c = npu_logits.float(), cpu_logits.float()
    d = (nl - c).abs()
    d[[ref_tok, npu_tok]] = 0
    e = float(d.max())
    cos = float(torch.nn.functional.cosine_similarity(nl.double(), c.double(), dim=0))
    cgap = float(c[ref_tok] - c[npu_tok])
    ngap = float(nl[npu_tok] - nl[ref_tok])
    info = (f"CPU {ref_tok} ↔ NPU {npu_tok} · CPU gap {cgap:.4f} · NPU gap {ngap:.4f} "
            f"(NPU logits {float(nl[ref_tok]):.4f} / {float(nl[npu_tok]):.4f}) · "
            f"error (other tokens) {e:.4f} · cos {cos:.6f}")
    return (cgap <= e and ngap <= e and cos >= cos_min), info


def _generate(a, model, backend, cpu_idx=()) -> int:
    """Generate N tokens and compare the token sequence with CPU greedy.

    Stricter than logits equivalence — once one step's argmax diverges,
    everything after it differs, so matching token sequences means the
    accumulated error also passed.  The verdict is teacher-forced and
    tie-aware: a differing step whose two candidates are within the step's
    measured NPU↔CPU logits error on both backends (and cos ≥ 0.999) counts
    as a tie, not a miss — see classify_step.
    """
    from . import generate as G

    cfg = model.config
    ids = _make_inputs(a, model)[0]["input_ids"]
    bl = tuple(int(x) for x in a.buckets.split(",")) if a.buckets else None

    # ── Grid placement — generation is PP × TP too ────────────────
    # Note: an earlier version hardcoded tp_size=1 on this path, so PP×TP
    #   grid verification covered only prefill equivalence and generation
    #   covered only the PP axis.  Placement is computed with the same
    #   function as the main path — computing it in two places lets them
    #   drift apart unnoticed.
    devs = [int(x) for x in a.device_ids.split(",")] if a.device_ids else None
    bad = [i for i in cpu_idx if i >= a.pp]
    if bad:
        raise SystemExit(f"[poc] --cpu-stages {bad} out of range (stages: {a.pp})")
    if len(cpu_idx) >= max(a.pp, 1):
        raise SystemExit("[poc] --cpu-stages leaves no stage for the NPU")
    stage_devs = placement.assign(a.pp - len(cpu_idx), a.tp, devs, backend.caps())

    print(f"[poc] generating {a.generate} tokens · prompt {ids.shape[1]} tokens · "
          f"KV {G.cache_bytes_per_token(cfg, model.dtype)/1024:.0f} KiB/token",
          flush=True)
    names = [f"submod_{i}" for i in range(a.pp)] if a.pp > 1 else ["whole"]
    npu_names = [nm for i, nm in enumerate(names) if i not in cpu_idx]
    print(f"[poc] {placement.describe(npu_names, stage_devs, a.tp, backend.name)}"
          + (f" · on CPU: {', '.join(names[i] for i in cpu_idx)}" if cpu_idx else ""),
          flush=True)

    def _gen(**kw):
        if a.inplace_cache:
            # cache kept on the stage devices, updated in place (inplace_cache.py)
            from . import buckets as BK, inplace_cache as IC
            cap = BK.pick(ids.shape[1] + a.generate, bl or BK.DEFAULT)
            return IC.run_pp(model, cfg, backend, ids, a.generate, pp=a.pp,
                             stage_devs=stage_devs, tp_size=a.tp, capacity=cap,
                             cpu_stages=cpu_idx, **kw)
        if a.pp > 1:
            # PP × TP × generation — both prefill and decode run on the grid.
            from . import buckets as BK
            bk = BK.pick(ids.shape[1] + a.generate, bl or BK.DEFAULT)
            return G.run_pp(model, cfg, backend, ids, a.generate, pp=a.pp,
                            stage_devs=stage_devs, tp_size=a.tp,
                            dtype=model.dtype, bucket=bk, cpu_stages=cpu_idx,
                            **kw)
        return G.run(model, cfg, backend, ids, a.generate,
                     devices=stage_devs[0], tp_size=a.tp,
                     dtype=model.dtype, bucket_list=bl, **kw)

    gen, st = _gen()

    # CPU reference — greedy with the same model and prompt
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=a.generate, do_sample=False,
                             use_cache=True,
                             pad_token_id=getattr(cfg, "eos_token_id", None))
    ref_new = ref[0, ids.shape[1]:].tolist()

    # ── Second, teacher-forced run — removes amplification, isolates accuracy ──
    tf_gen, tf_st = _gen(verbose=False, force=ref_new)

    n = min(len(gen), len(ref_new))
    same = sum(1 for i in range(n) if gen[i] == ref_new[i])
    first_diff = next((i for i in range(n) if gen[i] != ref_new[i]), None)
    d = st["decode_ms"]
    print("-" * 78, flush=True)
    print(f"[poc] NPU : {gen[:n]}", flush=True)
    print(f"[poc] CPU : {ref_new[:n]}", flush=True)
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.model)
        print(f"[poc] NPU text: {tk.decode(gen[:n])!r}", flush=True)
    except Exception:
        pass
    if "binaries" not in st:
        # PP path — binaries are (phase × stage)
        st["binaries"] = {f"{k}×{v} stages": float("nan")
                          for k, v in st["stages"].items()}
        nb = sum(st["stages"].values())
        print(f"[poc] binaries: {nb} "
              f"({' + '.join(f'{k} {v}' for k, v in st['stages'].items())}) · "
              f"total compile {st['compile_s']:.1f}s", flush=True)
        st["binaries"] = {}
    if st["binaries"]:
        print(f"[poc] binaries: {len(st['binaries'])} · total compile "
              f"{st['compile_s']:.1f}s · " +
              " · ".join(f"{k} {v:.1f}s" for k, v in st["binaries"].items()),
              flush=True)
    if d:
        print(f"[poc] prefill {st['prefill_ms']:.1f}ms · decode "
              f"{min(d):.1f}~{max(d):.1f}ms ({len(d)} steps) · "
              f"{1000/ (sum(d)/len(d)):.0f} tok/s", flush=True)
    # ── Diagnose the divergence point — error or near-tie? ────────
    #    Greedy decoding is chaotic.  Once one step's argmax flips, everything
    #    after it differs, so the token match rate alone cannot tell whether
    #    the computation is wrong.  So also look at the CPU top-2 gap there.
    if first_diff is not None:
        pre = torch.cat([ids, torch.tensor([gen[:first_diff]], dtype=torch.long)], 1)
        with torch.no_grad():
            lg = model(input_ids=pre, use_cache=False).logits[0, -1].float()
        top2 = lg.topk(2)
        margin = (top2.values[0] - top2.values[1]).item()
        allm = lg.topk(2, dim=-1).values
        print(f"[poc] diagnosis — diverged at step {first_diff}: "
              f"CPU chose {ref_new[first_diff]} ↔ NPU chose {gen[first_diff]} · "
              f"top-2 gap there {margin:.4f} "
              f"(for reference: typical gap is 1~3)", flush=True)
        verdict = ("near-tie — the top two are close, so a small error flips them"
                   if margin < 0.5
                   else "gap is wide — not explained by a simple near-tie")
        print(f"[poc]   → {verdict}", flush=True)
    nt = min(len(tf_gen), len(ref_new))
    same_tf = sum(1 for i in range(nt) if tf_gen[i] == ref_new[i])
    # ── Tie-aware teacher-forced verdict ──────────────────────────
    #    bf16 logits have a resolution of ~0.06 near typical logit values, so two
    #    candidates closer than the step's measured NPU↔CPU error can round to the
    #    same value and argmax then picks the lower index.  A mismatched step is
    #    counted as a "tie" (not a miss) only by classify_step's rule.
    ties, misses = [], []
    npu_lg = tf_st.get("logits") or []
    for i in range(nt):
        if tf_gen[i] == ref_new[i]:
            continue
        if i >= len(npu_lg):
            misses.append((i, "no NPU logits recorded"))
            continue
        ctx = torch.cat([ids, torch.tensor([ref_new[:i]], dtype=torch.long)], 1) if i else ids
        with torch.no_grad():
            c = model(input_ids=ctx, use_cache=False).logits[0, -1].float()
        is_tie, info = classify_step(npu_lg[i], c, ref_new[i], tf_gen[i])
        (ties if is_tie else misses).append((i, f"step {i}: {info}"))
    for _, info in ties:
        print(f"[poc]   tie  — {info}", flush=True)
    for _, info in misses:
        print(f"[poc]   MISS — {info}", flush=True)
    print(f"[poc] teacher-forced match: {same_tf}/{nt} exact"
          + (f" + {len(ties)} tie" if ties else "")
          + (f" · {len(misses)} miss" if misses else "")
          + f"  (free-running {same}/{n} — a gap there is amplification, not inaccuracy)",
          flush=True)
    ok = not misses
    print(f"[poc] RESULT {'PASS' if ok else 'FAIL'} generate "
          f"model={a.model} teacher-forced {same_tf}/{nt} exact"
          + (f" + {len(ties)} tie" if ties else "")
          + f" · free-running {same}/{n}"
          + (f" (first mismatch at step {first_diff})" if first_diff is not None else ""),
          flush=True)
    return 0 if ok else 1


def _throughput(a, model, split_gm, names, compiled, specs, driver,
                total_compile) -> int:
    """Stream M requests pipelined and compare against a serial baseline."""
    M = a.microbatches
    dag = runner.build_dag(split_gm, names)
    print(f"[poc] stage dependencies (read from the top-level FX graph):", flush=True)
    for nm in names:
        d = ", ".join(r if k == "input" else f"{r[0]}[{r[1]}]"
                      for k, r in dag[nm])
        print(f"[poc]   {nm} ← {d}", flush=True)

    # M distinct requests.  The seed is already fixed, so draw them in order.
    key = next(iter(driver))
    xs = _make_inputs(a, model, M)
    micro = {key: [x[key] for x in xs]}

    # CPU reference — per request.  Must use the same inputs as the NPU.
    cpu = []
    with torch.no_grad():
        for m in range(M):
            cpu.append(stage_io._logits_of(model(**{key: micro[key][m]})))

    # Pipelined vs serial.  Measure each `repeat` times with the same binaries
    # and use the minimum.
    pipe_t, ser_t, outs, souts = [], [], None, None
    for _ in range(max(1, a.repeat)):
        o, t = runner.pipelined(compiled, specs, names, dag, micro, M)
        pipe_t.append(t); outs = o
        o2, t2 = runner.serial(compiled, specs, names, dag, micro, M)
        ser_t.append(t2); souts = o2
    tp, ts = min(pipe_t), min(ser_t)

    # ── Deterministic control: pipelined vs serial ────────────────
    # Both run the same binaries on the same inputs, so they must be bit-exact.
    # A mismatch means the scheduler mixed up micro-batches; a match means the
    # scheduler is sound and any remaining error is input × compiler.
    # Without this control a FAIL cannot be attributed to our code vs the input.
    bad = [m for m in range(M)
           if not torch.equal(outs[m].float(), souts[m].float())]
    if bad:
        d = max((outs[m].float() - souts[m].float()).abs().max().item()
                for m in bad)
        print(f"[poc] FAIL pipelined vs serial mismatch — requests {bad} · |Δ|max={d:.4e}  "
              f"-> scheduler bug", flush=True)
    else:
        print(f"[poc] OK pipelined vs serial bit-exact (all M={M}) "
              f"— the scheduler does not affect numerics", flush=True)

    # ── Per-request verdict — so one worst case doesn't hide the rest ──
    per = [equivalence.judge(outs[m], cpu[m], want_argmax=True) for m in range(M)]
    n_ok = sum(1 for v in per if v.ok)
    worst = min(per, key=lambda v: (v.ok, v.cos))

    # ── Diagnose failed requests — near-tie effect or computation error? ──
    # argmax can go wrong in two ways:
    #   (1) the computation is badly wrong  → |Δ| is much larger than the top-2 gap
    #   (2) the top two were nearly tied    → flips even with small |Δ| (near-tie)
    # Random-token inputs produce flat distributions and tend to trigger (2),
    # since they are random tokens, not real sentences.  Without separating
    # the two, this reads wrongly as "ATOM+ loses accuracy at M=8".
    for m in range(M):
        if per[m].ok:
            continue
        r = cpu[m].detach().float()
        g = outs[m].detach().float()
        mism = (g.argmax(-1) != r.argmax(-1)).nonzero(as_tuple=False)
        top2 = r.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1])
        d = (g - r).abs()
        for i in mism:
            ix = tuple(i.tolist())
            gap = margin[ix].item()
            # How much error actually landed on the top two at that position.
            k2 = r[ix].topk(2).indices
            dd = max(d[ix][k2[0]].item(), d[ix][k2[1]].item())
            print(f"[poc] diagnosis request#{m} token{ix}: CPU top-2 gap {gap:.4f} "
                  f"(overall median {margin.median().item():.4f}) · "
                  f"|Δ| on the top-2 there {dd:.4f} "
                  f"→ {'larger than the gap — can flip' if dd > gap else 'smaller than the gap — unexplained'}",
                  flush=True)
        print(f"[poc] diagnosis request#{m}: mismatch {len(mism)}/{r.shape[-2]} tokens · "
              f"request |Δ|max={per[m].d_max:.4f} (for reference: other requests "
              f"{min(v.d_max for v in per):.4f}~{max(v.d_max for v in per):.4f})",
              flush=True)
    print(f"[poc] per-request verdict: {n_ok}/{M} PASS · "
          f"cos {min(v.cos for v in per):.6f}~{max(v.cos for v in per):.6f} · "
          f"rel {min(v.rel for v in per):.3e}~{max(v.rel for v in per):.3e} · "
          f"argmax {min(v.argmax for v in per)*100:.2f}~"
          f"{max(v.argmax for v in per)*100:.2f}%", flush=True)

    print("-" * 78, flush=True)
    print(f"[poc] serial {ts:.1f}ms ({min(ser_t):.1f}~{max(ser_t):.1f}) · "
          f"pipelined {tp:.1f}ms ({min(pipe_t):.1f}~{max(pipe_t):.1f}) · "
          f"{a.repeat} runs", flush=True)
    print(f"[poc] throughput {ts / tp:.2f}× (ideal upper bound "
          f"{M * len(names) / (M + len(names) - 1):.2f}× — "
          f"M={M}, {len(names)} stages)", flush=True)
    print(f"[poc] {equivalence.thresholds_line()}", flush=True)
    print(f"[poc] NPU ↔ CPU (worst of M): {worst}", flush=True)
    print(f"[poc] RESULT {'PASS' if worst.ok else 'FAIL'} "
          f"pp={a.pp} tp={a.tp} M={M} model={a.model} "
          f"compile={total_compile:.1f}s serial={ts:.1f}ms pipe={tp:.1f}ms "
          f"speedup={ts / tp:.2f}x cos={worst.cos:.6f}", flush=True)
    return 0 if worst.ok else 1


if __name__ == "__main__":
    sys.exit(main())
