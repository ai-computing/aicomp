#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# Pick an execution engine for a model from its block spec, generate, and optionally compare with HF.
#
# Usage:
#   python3 auto_generate.py --model upstage/Solar-Open-100B \
#       --rngd-engine <solar_engine binary> --rngd-data <its data dir with engine_manifest.json> --check
#   python3 auto_generate.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tiers gpu:15,rngd-opt:0:40,cpu
#
# The RNGD layer engine is chosen only when the model's geometry matches its manifest; otherwise Optimus (GPU path
# with MoE expert tiers) runs it. The table printed first shows every candidate and why it was or was not chosen.
import argparse, os, sys, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.exec import OptimusEngine, RngdLayerEngine, select_engine  # noqa: E402
from opt_prime.model import model_spec_from_pretrained  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain mixture of experts in two sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--rngd-engine", default=os.environ.get("OPT_PRIME_RNGD_ENGINE"))
    ap.add_argument("--rngd-data", default=os.environ.get("OPT_PRIME_RNGD_DATA"))
    ap.add_argument("--tiers", default=None, help="MoE expert tiers for the Optimus engine (see moe_offload_generate)")
    ap.add_argument("--engine", default="auto", help="auto | optimus | rngd-layer")
    ap.add_argument("--check", action="store_true", help="compare with HF generate on the host (eager experts)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids          # BatchEncoding in transformers 5
    ids = [int(t) for t in (ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)]

    t0 = time.time()
    spec = model_spec_from_pretrained(a.model)
    print(f"[auto] block spec in {time.time() - t0:.1f}s: {len(spec.layers)} layers, {len(spec.moe_layers)} MoE, "
          f"custom blocks {len(spec.custom_blocks)}, kv id {spec.kv_semantic_id()[:24]}")

    moe_cfg = None
    if a.tiers:
        from moe_offload_generate import parse_tiers
        from opt_prime.moe_offload import MoEOffloadConfig
        moe_cfg = MoEOffloadConfig(tiers=parse_tiers(a.tiers))
    candidates = [OptimusEngine(moe_offload=moe_cfg)]
    if a.rngd_engine and a.rngd_data:
        candidates.append(RngdLayerEngine(a.rngd_engine, a.rngd_data))
    if a.engine != "auto":
        candidates = [c for c in candidates if c.name == a.engine]
    engine, table = select_engine(spec, candidates)
    for name, reasons, tps in table:
        mark = "->" if name == engine.name else "  "
        print(f"[auto] {mark} {name:10s} expected {tps if tps is not None else '?'} tok/s; "
              f"{'supported' if not reasons else 'not supported: ' + '; '.join(reasons[:3])}")

    ref = None
    if a.check:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
        m.config._experts_implementation = "eager"
        with torch.no_grad():
            t = torch.tensor([ids])
            ref = m.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=a.max_new_tokens,
                             do_sample=False)[0, len(ids):].tolist()
        del m

    t0 = time.time()
    engine.start(a.model)
    print(f"[auto] {engine.name} started in {time.time() - t0:.1f}s")
    try:
        engine.generate(ids, max_new=4)                     # warm-up
        t0 = time.time()
        toks, info = engine.generate(ids, max_new=a.max_new_tokens)
        dt = time.time() - t0
    finally:
        engine.close()
    print(f"[auto] {engine.name}: {len(toks)} tokens in {dt:.2f}s ({len(toks) / dt:.2f} tok/s incl. prefill)"
          + (f"; engine decode {1000 / info['decode_ms_per_tok']:.2f} tok/s" if info.get("decode_ms_per_tok") else ""))
    print("[auto] text:", repr(tok.decode(toks)))
    if ref is not None:
        n = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), min(len(toks), len(ref)))
        print(f"[check] first {n}/{len(ref)} tokens identical to HF generate -> {'PASS' if n == len(ref) else 'DIFF'}")


if __name__ == "__main__":
    main()
