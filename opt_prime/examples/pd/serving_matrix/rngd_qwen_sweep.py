"""Qwen3 research prefill sweep; use PR200 prefill/spec and the isolated SDK launcher."""

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from inputs import exact_prompt
from prefill import Prefill
from safetensors import safe_open
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--output", required=True)
p.add_argument("--fixtures")
p.add_argument("--cases", nargs="+")
p.add_argument("--fp32-rms-weights", action="store_true")
p.add_argument("--experimental-long-width", action="store_true")
a = p.parse_args()
if a.experimental_long_width:
    if a.fp32_rms_weights:
        p.error("isolate width and RMS precision experiments")
    from long_width_variant import load_variant

    Prefill = load_variant()
if a.fp32_rms_weights:
    from rms_precision_variant import load_variant

    Prefill = load_variant()
path = Path(snapshot_download(a.model, local_files_only=True))
c = json.loads((path / "config.json").read_text())
weights = {}
for shard in path.glob("*.safetensors"):
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
            weights[key] = f.get_tensor(key).float()
tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
filler = tok.encode(
    "This is background context. The red box contains apples. The blue box contains books.\n",
    add_special_tokens=False,
)
suffix = tok.encode(
    "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
)
ids = exact_prompt(prefix, filler, suffix, a.width)
eos_ids = []
cases = [{"case": "fixed", "ids": ids}]
if a.fixtures:
    data = json.loads(Path(a.fixtures).read_text())
    eos_ids = data["eos_ids"]
    cases = data["cases"]
    if a.cases:
        cases = [r for r in cases if r["case"] in a.cases]
        assert {r["case"] for r in cases} == set(a.cases), "unknown case"
    assert cases and all(0 < len(r["ids"]) <= a.width for r in cases)
t = time.perf_counter()
engine = Prefill(c, weights, width=a.width, chip=0)
print(
    json.dumps(
        {"ready": True, "init_s": time.perf_counter() - t, "revision": path.name}
    ),
    flush=True,
)
with open(a.output + ".jsonl", "w", buffering=1) as f:
    for case_index, case in enumerate(cases):
        ids = case["ids"]
        previous = None
        for repeat in range(3):
            t = time.perf_counter()
            k, v, logits = engine.prefill(ids)
            elapsed = time.perf_counter() - t
            assert all(bool(torch.isfinite(x).all()) for x in (k, v, logits))
            exact = previous is None or all(
                torch.equal(x, y) for x, y in zip(previous, (k, v, logits))
            )
            assert exact, "repeat mismatch"
            previous = (k, v, logits)
            row = {
                "model": a.model,
                "fp32_rms_weights": a.fp32_rms_weights,
                "experimental_long_width": a.experimental_long_width,
                "revision": path.name,
                "width": a.width,
                "input_tokens": len(ids),
                "case": case["case"],
                "repeat": repeat,
                "prefill_s": elapsed,
                "repeat_exact": exact,
                "first_token": int(logits.argmax()),
                "kv_shape": list(k.shape),
                "dtype": str(k.dtype),
            }
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        torch.save(
            {
                "ids": ids,
                "keys": k,
                "values": v,
                "logits": logits,
                "metadata": row,
                "eos_ids": eos_ids,
            },
            a.output + (f"-{case_index}" if a.fixtures else "") + ".pt",
        )
