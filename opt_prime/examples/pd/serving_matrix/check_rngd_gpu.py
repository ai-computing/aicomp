"""Validate trusted RNGD Qwen3 exports against GPU prefill and GPU decode."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
a = p.parse_args()
model = (
    AutoModelForCausalLM.from_pretrained(
        a.model,
        dtype=getattr(torch, a.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    .cuda()
    .eval()
)
rows = []


def decode(token, cache):
    tokens = [token]
    for _ in range(31):
        o = model(
            input_ids=torch.tensor([[tokens[-1]]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        assert bool(torch.isfinite(o.logits).all())
        tokens.append(int(o.logits[0, -1].argmax()))
        cache = o.past_key_values
    return tokens


def relative(a, b):
    return float(
        (a.float() - b.float()).square().mean().sqrt()
        / b.float().square().mean().sqrt()
    )


with torch.inference_mode():
    for path in a.exports:
        r = torch.load(path, weights_only=True)
        o = model(
            input_ids=torch.tensor([r["ids"]], device="cuda"),
            use_cache=True,
            logits_to_keep=1,
        )
        keys = torch.stack([l.keys[0].cpu() for l in o.past_key_values.layers])
        values = torch.stack([l.values[0].cpu() for l in o.past_key_values.layers])
        logits = o.logits[0, -1].cpu()
        native = decode(int(logits.argmax()), o.past_key_values)
        cache = DynamicCache(config=model.config)
        for i, (k, v) in enumerate(zip(r["keys"], r["values"])):
            dk, dv = [t[None].to(device="cuda", dtype=model.dtype) for t in (k, v)]
            assert torch.equal(dk.cpu()[0], k) and torch.equal(dv.cpu()[0], v)
            cache.update(dk, dv, i)
        transferred = decode(int(r["logits"].argmax()), cache)
        row = {
            "export": Path(path).name,
            "decode_dtype": a.dtype,
            "key_layer_relative_rms": [relative(x, y) for x, y in zip(r["keys"], keys)],
            "value_layer_relative_rms": [
                relative(x, y) for x, y in zip(r["values"], values)
            ],
            "input_tokens": len(r["ids"]),
            "key_relative_rms": relative(r["keys"], keys),
            "value_relative_rms": relative(r["values"], values),
            "logit_relative_rms": relative(r["logits"], logits),
            "first_equal": native[0] == transferred[0],
            "tokens_equal": native == transferred,
            "readback_exact_slots": 2 * len(cache.layers),
            "native_tokens": native,
            "pd_tokens": transferred,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
