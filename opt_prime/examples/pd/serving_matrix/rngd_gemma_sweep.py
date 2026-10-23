"""Pinned Gemma3-1B full-prefix producer; run in an isolated RNGD SDK environment."""

import argparse
import json
import time
from pathlib import Path

import torch
from gemma_prefill import Prefill
from huggingface_hub import snapshot_download
from inputs import exact_prompt
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "unsloth/gemma-3-1b-it"
REVISION = "5b11413a10db4e486ef16a20101fd028f8f2499c"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--width", type=int, choices=[128, 512, 2048, 8192], required=True)
p.add_argument("--output", required=True)
p.add_argument("--fixtures")
p.add_argument("--cases", nargs="+")
a = p.parse_args()
path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
c = json.loads((path / "config.json").read_text())
weights = {}
for shard in sorted(path.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118
            weights[key] = f.get_tensor(key).to(torch.bfloat16)
tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
eos_ids = []
if a.fixtures:
    data = json.loads(Path(a.fixtures).read_text())
    eos_ids = data["eos_ids"]
    cases = data["cases"]
    if a.cases:
        cases = [r for r in cases if r["case"] in a.cases]
        assert {r["case"] for r in cases} == set(a.cases), "unknown case"
    assert cases and all(0 < len(r["ids"]) <= a.width for r in cases)
else:
    ids = exact_prompt(
        [tok.bos_token_id],
        tok.encode(
            "This is background context. The red box contains apples. The blue box contains books.\n",
            add_special_tokens=False,
        ),
        tok.encode(
            "\nQuestion: What is the capital of France?\nAnswer:",
            add_special_tokens=False,
        ),
        a.width,
    )
    cases = [{"case": "fixed", "ids": ids}]
start = time.perf_counter()
print(json.dumps({"compile_start": True, "width": a.width}), flush=True)
engine = Prefill(c, weights, width=a.width)
print(
    json.dumps(
        {"ready": True, "init_s": time.perf_counter() - start, "width": a.width}
    ),
    flush=True,
)
with torch.inference_mode(), open(a.output + ".jsonl", "w", buffering=1) as f:
    for case_index, case in enumerate(cases):
        ids = case["ids"]
        previous = None
        for repeat in range(3):
            start = time.perf_counter()
            k, v, logits = engine.prefill(ids)
            elapsed = time.perf_counter() - start
            assert k.shape == v.shape == (26, 1, len(ids), 256)
            assert all(bool(torch.isfinite(t).all()) for t in (k, v, logits))
            exact = previous is None or all(
                torch.equal(t, u) for t, u in zip(previous, (k, v, logits))
            )
            assert exact, "repeat mismatch"
            previous = (k, v, logits)
            row = {
                "model": MODEL,
                "revision": REVISION,
                "width": a.width,
                "input_tokens": len(ids),
                "case": case["case"],
                "repeat": repeat,
                "prefill_s": elapsed,
                "repeat_exact": exact,
                "first_token": int(logits.argmax()),
                "kv_shape": list(k.shape),
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
