"""Compare the validated Gemma layer equations on GPU with a live RNGD export."""

import argparse
import json
import time
from pathlib import Path

import torch
from gemma_parts import (
    GemmaLayer,
    attention_inputs,
    final_logits,
    initial_hidden,
    layer_parameters,
)
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--export", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--width", type=int, default=8192)
a = parser.parse_args()
root = Path(a.model)
r = torch.load(a.export, weights_only=True)
assert 0 < len(r["ids"]) <= a.width
assert r["keys"].shape == r["values"].shape and r["keys"].shape[2] == len(r["ids"])

c = json.loads((root / "config.json").read_text())
tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
weights = {}
for shard in root.glob("*.safetensors"):
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
            weights[k] = f.get_tensor(k)
rows = []
for dtype in [torch.bfloat16, torch.float32]:
    t0 = time.perf_counter()
    w = {k: v.to(device="cuda", dtype=dtype) for k, v in weights.items()}
    width = a.width
    positions = {
        i: tuple(
            t.to(device="cuda", dtype=dtype) for t in attention_inputs(c, width, i)
        )
        for i in [0, 5]
    }
    model = GemmaLayer(c).cuda()
    errors = []
    with torch.inference_mode():
        x = initial_hidden(
            w["model.embed_tokens.weight"],
            r["ids"] + [tok.pad_token_id] * (width - len(r["ids"])),
        )
        for i in range(c["num_hidden_layers"]):
            norms, mats = layer_parameters(w, i)
            x, k, v = model(x, norms, mats, *positions[5 if (i + 1) % 6 == 0 else 0])
            got = [
                k[: len(r["ids"])].permute(1, 0, 2),
                v[: len(r["ids"])].permute(1, 0, 2),
            ]
            rel = []
            for tensor, kind in zip(got, ["keys", "values"]):
                ref = r[kind][i].to(device="cuda", dtype=torch.float32)
                rel.append(
                    float(
                        (tensor.float() - ref).square().mean().sqrt()
                        / ref.square().mean().sqrt()
                    )
                )
            errors.append(rel)
        logits = final_logits(
            x[len(r["ids"]) - 1],
            w["model.norm.weight"],
            w.get("lm_head.weight", w["model.embed_tokens.weight"]),
            c["rms_norm_eps"],
        )
        assert torch.isfinite(logits).all()
        top = logits.topk(10)
        row = {
            "dtype": str(dtype),
            "width": width,
            "input_tokens": len(r["ids"]),
            "seconds": time.perf_counter() - t0,
            "first_token": int(logits.argmax()),
            "top10": [
                {"token": int(i), "text": tok.decode([int(i)]), "logit": float(z)}
                for i, z in zip(top.indices, top.values)
            ],
            "kv_layer_relative_rms_vs_rngd": errors,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    del w, positions, x, k, v, logits, model
    torch.cuda.empty_cache()
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
