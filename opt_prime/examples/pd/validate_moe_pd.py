"""PD on a real MoE checkpoint, optionally with expert offloading (#56): PD greedy tokens == single-engine tokens.

The prefill and decode TorchCausalLMEngines share one model object (no second copy of the weights) and keep separate
KV caches; TorchCacheTransport copies the KV between them. The reference is Hugging Face `generate` (greedy, no EOS
stop) on the same model. With --offload-gpu-gb, the experts are offloaded first (hot experts on the GPU within the
budget, the rest on the host CPU), so both the reference and PD run the offloaded model.

    python validate_moe_pd.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --out results/qwen3-30b
    python validate_moe_pd.py --model upstage/Solar-Open-100B --offload-gpu-gb 40 --out results/solar-100b
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opt_prime.accel.pd import PDRequest, PDRuntime
from opt_prime.accel.pd.torch_cache import TorchCacheTransport
from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine

PROMPTS = ("The capital of France is",
           "Explain why the sky appears blue in one sentence.",
           "한국의 수도는 어디인가요? 간단히 답해주세요.",
           "def fibonacci(n):",
           "List three prime numbers greater than 100:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--offload-gpu-gb", type=float, default=None,
                    help="offload experts: hot experts within this GPU budget, the rest computed on the host")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    model.generation_config.eos_token_id = None          # the PD engines run to max_new_tokens as well
    report = None
    if a.offload_gpu_gb is not None:
        from opt_prime.accel.moe import ExpertProfiler, MoEOffloadConfig, apply_moe_offload
        with ExpertProfiler(model) as prof, torch.no_grad():
            for p in PROMPTS:                             # routing profile on the host
                model(tok(p, return_tensors="pt").input_ids)
        report = apply_moe_offload(model, MoEOffloadConfig(gpu_budget_gb=a.offload_gpu_gb, profile=prof.as_lists()))
    model.to("cuda")
    print(f"loaded in {time.time() - t0:.0f}s; offload: {report}", flush=True)

    opts = dict(model_id=a.model, revision="local", max_context=a.max_context)
    source = TorchCausalLMEngine(model, endpoint="source", **opts)
    decode = TorchCausalLMEngine(model, endpoint="decode", **opts)
    runtime = PDRuntime(source, decode, TorchCacheTransport(source="source", destination="decode", spec=source.spec))

    rows = []
    for i, text in enumerate(PROMPTS):
        ids = tok(text, return_tensors="pt").input_ids
        with torch.inference_mode():
            t1 = time.time()
            ref = model.generate(ids.cuda(), attention_mask=torch.ones_like(ids).cuda(), do_sample=False,
                                 max_new_tokens=a.max_new_tokens)[0, ids.shape[1]:].tolist()
            t_ref = time.time() - t1
        t1 = time.time()
        pd = runtime.generate(PDRequest(f"p{i}", tuple(ids[0].tolist()), a.max_new_tokens))
        t_pd = time.time() - t1
        rows.append(dict(prompt=text, input_length=ids.shape[1], tokens_exact=pd == ref,
                         first_mismatch=next((j for j, (x, y) in enumerate(zip(pd, ref)) if x != y), None),
                         reference_s=round(t_ref, 2), pd_s=round(t_pd, 2), transfer_s=round(
                             runtime.last_timings.get("transfer_seconds", 0), 4),
                         text=tok.decode(pd)))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    result = dict(model=a.model, dtype="bfloat16", max_new_tokens=a.max_new_tokens,
                  offload_gpu_gb=a.offload_gpu_gb,
                  offload=None if report is None else [(t["tier"], t["experts"]) for t in report.tiers],
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  passed=all(r["tokens_exact"] for r in rows), rows=rows)
    (a.out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("PASS" if result["passed"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
