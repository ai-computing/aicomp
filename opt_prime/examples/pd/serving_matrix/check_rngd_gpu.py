"""Validate trusted RNGD Qwen3 exports against GPU prefill and GPU decode."""

import argparse
import json
import math
from pathlib import Path

import torch
from cache_comparison import retained_export
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


def relative_many(actual, expected):
    numerator = sum(
        float((a.float() - b.float()).square().sum()) for a, b in zip(actual, expected)
    )
    denominator = sum(float(b.float().square().sum()) for b in expected)
    return math.sqrt(numerator / denominator)


with torch.inference_mode():
    for path in a.exports:
        r = torch.load(path, weights_only=True)
        o = model(
            input_ids=torch.tensor([r["ids"]], device="cuda"),
            use_cache=True,
            logits_to_keep=1,
        )
        layers = o.past_key_values.layers
        keys = [layer.keys[0].cpu() for layer in layers]
        values = [layer.values[0].cpu() for layer in layers]
        matched_k = [retained_export(k, layer) for k, layer in zip(r["keys"], layers)]
        matched_v = [retained_export(v, layer) for v, layer in zip(r["values"], layers)]
        logits = o.logits[0, -1].cpu()
        native = decode(int(logits.argmax()), o.past_key_values)
        cache = DynamicCache(config=model.config)
        for i, (k, v) in enumerate(zip(r["keys"], r["values"])):
            dk, dv = [t[None].to(device="cuda", dtype=model.dtype) for t in (k, v)]
            assert torch.equal(dk.cpu()[0], k) and torch.equal(dv.cpu()[0], v)
            cache.update(dk, dv, i)
            assert torch.equal(
                cache.layers[i].keys[0].cpu(), retained_export(k, cache.layers[i])
            )
            assert torch.equal(
                cache.layers[i].values[0].cpu(), retained_export(v, cache.layers[i])
            )
        transferred = decode(int(r["logits"].argmax()), cache)
        row = {
            "export": Path(path).name,
            "decode_dtype": a.dtype,
            "key_layer_relative_rms": [relative(x, y) for x, y in zip(matched_k, keys)],
            "value_layer_relative_rms": [
                relative(x, y) for x, y in zip(matched_v, values)
            ],
            "input_tokens": len(r["ids"]),
            "compared_cache_lengths": [key.shape[-2] for key in keys],
            "key_relative_rms": relative_many(matched_k, keys),
            "value_relative_rms": relative_many(matched_v, values),
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
