#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Greedy generation with MoE expert offloading through Optimus_Inference.
#
# Usage:
#   # tiny random model of a family, checked against HF generate (no download)
#   torchrun --nproc_per_node=1 moe_offload_generate.py --tiny qwen3_moe --check
#   # a real checkpoint: profile routing on the host, keep the hottest experts within the budget on the GPU
#   torchrun --nproc_per_node=1 moe_offload_generate.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#       --gpu-budget-gb 20 --profile-prompt "Explain mixture of experts." --check
#
import argparse, copy, os, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.inference import Optimus_Inference  # noqa: E402
from opt_prime.accel.moe import MoEOffloadConfig, ExpertProfiler  # noqa: E402


def parse_tiers(text):
    """'backend[:device[:budget_gb]][:key=value...]', comma separated; e.g. 'gpu:15,rngd-opt:0:40:slots=64,cpu'."""
    from opt_prime.accel.moe import TierSpec
    out = []
    for item in text.split(","):
        f = item.split(":")
        opts = dict(x.split("=", 1) for x in f[1:] if "=" in x)
        opts = {k: int(v) if v.isdigit() else v for k, v in opts.items()}
        f = [f[0]] + [x for x in f[1:] if "=" not in x]
        if len(f) == 1: out.append(TierSpec(f[0], options=opts))
        elif len(f) == 2: out.append(TierSpec(f[0], budget_gb=float(f[1]), options=opts))
        else: out.append(TierSpec(f[0], device=int(f[1]), budget_gb=float(f[2]), options=opts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id or path")
    ap.add_argument("--tiny", default=None, help="random tiny model of this family (see moe_offload_check.py)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--profile-prompt", default=None, help="calibration text for the routing profile (default: --prompt)")
    ap.add_argument("--gpu-budget-gb", type=float, default=0.0, help="GPU memory for expert weights")
    ap.add_argument("--hot-fraction", type=float, default=None, help="instead of a budget: fraction of experts per layer")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv-cache", action="store_true", help="CachedSDPA KV cache (default: full recompute)")
    ap.add_argument("--pp-size", type=int, default=1)
    ap.add_argument("--adapt-every", type=int, default=0, help="adaptive re-placement every N layer calls")
    ap.add_argument("--adapt-swaps", type=int, default=2)
    ap.add_argument("--warmup-tokens", type=int, default=4)
    ap.add_argument("--tiers", default=None,
                    help="heterogeneous tiers, fastest first, host last: e.g. 'gpu:15,rngd:0:40,cpu' "
                         "(backend[:device]:budget_gb[:key=value]; the last tier takes the rest)")
    ap.add_argument("--compile", default=None, help="torch.compile the stage module with this mode "
                    "(default | reduce-overhead | max-autotune-no-cudagraphs); experimental")
    ap.add_argument("--compile-skip-kv", action="store_true", help="with --compile: run the KV-cache attention "
                    "eagerly (graph break) instead of recompiling for every decode position")
    ap.add_argument("--check", action="store_true", help="compare with HF generate on the original model")
    ap.add_argument("--no-offload", action="store_true", help="plain Optimus_Inference (baseline)")
    ap.add_argument("--experts-impl", default=None, help="transformers experts implementation of the model/reference "
                    "(eager | grouped_mm | batched_mm); offloaded experts compute like 'eager'")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)

    if a.tiny:
        from opt_prime.accel.testing import build_tiny
        model, tok = build_tiny(a.tiny, dt), None
        ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))
        prof_ids = [ids]
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).eval()
        ids = tok(a.prompt, return_tensors="pt").input_ids
        prof_ids = [tok(t, return_tensors="pt").input_ids for t in (a.profile_prompt or a.prompt).split("|||")]

    if a.experts_impl:
        if a.experts_impl == "opt_batched_mm":       # export-safe batched experts (opt_prime.accel.moe.train)
            from opt_prime.accel.moe.train import _register
            _register()
        for c in {id(c): c for c in (model.config, getattr(model.config, "get_text_config", lambda: None)())
                  if c is not None}.values():
            c._experts_implementation = a.experts_impl
    ref, ref_logits = None, None
    if a.check:
        t0 = time.time()
        with torch.no_grad():
            dev = "cuda" if a.tiny else "cpu"     # the full model need not fit on the GPU
            r = copy.deepcopy(model).to(dev) if a.tiny else model
            if a.tiny: r.generation_config.eos_token_id = None     # the engine is not told about eos either
            g = r.generate(ids.to(dev), attention_mask=torch.ones_like(ids).to(dev),   # no pad masking: ids are raw
                           max_new_tokens=a.max_new_tokens, do_sample=False, output_logits=True,
                           return_dict_in_generate=True)
            ref = g.sequences[0, ids.shape[1]:].tolist()
            ref_logits = [l[0].float().cpu() for l in g.logits]      # per step, along the reference path
            del r
        print(f"[check] HF reference ({dev}) in {time.time() - t0:.1f}s")

    with ExpertProfiler(model) as prof, torch.no_grad():
        for p in prof_ids:          # several calibration texts, separated by '|||' on the command line
            model(p)
    counts = prof.as_lists()
    if a.hot_fraction is not None:
        E = len(counts[0])
        cfg = MoEOffloadConfig(hot=[sorted(range(E), key=lambda e: -c[e])[:int(E * a.hot_fraction)] for c in counts])
    else:
        cfg = MoEOffloadConfig(gpu_budget_gb=a.gpu_budget_gb, profile=counts, adapt_every=a.adapt_every,
                               adapt_swaps=a.adapt_swaps, tiers=parse_tiers(a.tiers) if a.tiers else None)

    engine = Optimus_Inference(model, use_gpu=True, dtype=dt, use_kv_cache=a.kv_cache, dynamo_capture=True, pp_size=a.pp_size,
                               moe_offload=None if a.no_offload else cfg)
    engine.eval()
    if a.compile:
        if a.compile_skip_kv:      # keep the stateful KV-cache attention out of the compiled regions
            from opt_prime import kv_cache
            kv_cache.CachedScaledDotProductAttention.forward = torch.compiler.disable(
                kv_cache.CachedScaledDotProductAttention.forward)
        engine.run_info.submod = torch.compile(engine.run_info.submod, mode=None if a.compile == "default" else a.compile,
                                               dynamic=os.environ.get("MOE_OFFLOAD_COMPILE_DYNAMIC", "1") == "1")
    # warm-up: first calls upload the hot experts and initialize kernels; not part of the measurement
    engine.generate(ids.cuda() if engine.is_first_stage() else None, max_new_tokens=a.warmup_tokens, do_sample=False,
                    verbose=False)
    if not a.no_offload: cfg.stats.reset()
    torch.cuda.synchronize()
    t0 = time.time()
    out = engine.generate(ids.cuda() if engine.is_first_stage() else None, max_new_tokens=a.max_new_tokens, do_sample=False, verbose=False)
    torch.cuda.synchronize()
    dt_s = time.time() - t0
    if engine.is_output_rank():
        gen = out[0, ids.shape[1]:].tolist()
        frac = 1.0 if a.no_offload else cfg.stats.device_fraction
        print(f"[moe_offload] {engine.moe_offload_report}")
        if not a.no_offload:
            print(f"[moe_offload] pairs per tier: {cfg.stats.flush().tier_pairs}")
        print(f"[moe_offload] generated {len(gen)} tokens in {dt_s:.1f}s ({len(gen) / dt_s:.2f} tok/s incl. prefill); "
              f"routed pairs on GPU {frac:.3f}")
        if not a.no_offload:
            st = cfg.stats
            print(f"[moe_offload] host compute {st.host_s:.2f}s, waiting for host {st.wait_s:.2f}s over {st.calls} layer calls"
                  f", {st.swaps} expert swaps")
            if st.phases:
                print("[moe_offload] per layer call (us): " + ", ".join(
                    f"{k} {v / st.calls * 1e6:.0f}" for k, v in sorted(st.phases.items(), key=lambda kv: -kv[1])))
        if tok: print("[moe_offload] text:", repr(tok.decode(gen)))
        if ref is not None:
            n = next((i for i, (x, y) in enumerate(zip(gen, ref)) if x != y), min(len(gen), len(ref)))
            verdict = "PASS" if n == len(ref) else "DIFF"
            if verdict == "DIFF" and ref_logits is not None and n < len(ref_logits):
                # same prefix up to n: compare the two candidate tokens in the reference's own step-n logits.
                # A gap within bf16 rounding of the logits (1 ulp at their magnitude) is a tie two valid fp32
                # summation orders can break either way; anything larger is a real numerical error.
                lg = ref_logits[n]
                gap = float(lg[ref[n]] - lg[gen[n]])
                ulp = 2.0 ** (torch.floor(torch.log2(lg[ref[n]].abs().clamp_min(1e-6))).item() - 7)
                verdict = (f"DIFF at a bf16 tie (reference logit gap {gap:.4f} <= 1 ulp {ulp:.4f})" if gap <= ulp
                           else f"DIFF (reference logit gap {gap:.4f} > 1 ulp {ulp:.4f}: numerical error)")
            print(f"[check] first {n}/{len(ref)} tokens identical to HF generate -> {verdict}")


if __name__ == "__main__":
    main()
