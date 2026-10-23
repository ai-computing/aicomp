"""Pinned Gemma3-1B full-prefix producer; run in an isolated RNGD SDK environment."""

import argparse
import json
import time
from pathlib import Path

import torch
from gemma_parts import (
    FlatGemmaLayer,
    attention_inputs,
    final_logits,
    initial_hidden,
    layer_parameters,
)
from huggingface_hub import snapshot_download
from inputs import exact_prompt
from rngd_pd.engines.fsdk import CompileModule, FCConfig, dev_str, hint, sync
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "unsloth/gemma-3-1b-it"
REVISION = "5b11413a10db4e486ef16a20101fd028f8f2499c"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--width", type=int, choices=[128, 512, 2048, 8192], required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
c = json.loads((path / "config.json").read_text())
module = FlatGemmaLayer(c)
weights = {}
for shard in sorted(path.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118
            weights[key] = f.get_tensor(key).to(torch.bfloat16)
tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
ids = exact_prompt(
    [tok.bos_token_id],
    tok.encode(
        "This is background context. The red box contains apples. The blue box contains books.\n",
        add_special_tokens=False,
    ),
    tok.encode(
        "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
    ),
    a.width,
)
dev = dev_str(0)
start = time.perf_counter()
params = [layer_parameters(weights, i) for i in range(c["num_hidden_layers"])]
rot, mask = attention_inputs(c, a.width, 0)
sample = (
    torch.zeros(a.width, c["hidden_size"], dtype=torch.bfloat16),
    *params[0][0],
    *params[0][1],
    rot.to(torch.bfloat16),
    mask.to(torch.bfloat16),
)
print(json.dumps({"compile_start": True, "width": a.width}), flush=True)
# Match the contiguous runtime tensors; zeros_like preserves transposed weight strides.
compiled = CompileModule.from_module(
    module,
    tuple(torch.zeros(t.shape, dtype=t.dtype) for t in sample),
    compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
)
compiled.to(dev)
params = [
    tuple(t.contiguous().to(dev) for t in (*norms, *mats)) for norms, mats in params
]
# Only two distinct positional/mask inputs, reused across the layer schedule.
positions = {}
for index in [0, 5]:
    positions[index] = tuple(
        t.to(torch.bfloat16).to(dev) for t in attention_inputs(c, a.width, index)
    )
sync(0)
print(
    json.dumps(
        {"ready": True, "init_s": time.perf_counter() - start, "width": a.width}
    ),
    flush=True,
)
previous = None
with torch.inference_mode(), open(a.output + ".jsonl", "w", buffering=1) as f:
    for repeat in range(3):
        start = time.perf_counter()
        x = initial_hidden(weights["model.embed_tokens.weight"], ids).to(dev)
        keys = []
        values = []
        for i, layer in enumerate(params):
            local = (i + 1) % c.get("sliding_window_pattern", 6) != 0
            x, k, v = compiled(x, *layer, *positions[0 if local else 5])
            keys.append(k)
            values.append(v)
        sync(0)
        k = torch.stack([t.cpu().permute(1, 0, 2) for t in keys])
        v = torch.stack([t.cpu().permute(1, 0, 2) for t in values])
        logits = final_logits(
            x.cpu()[-1],
            weights["model.norm.weight"],
            weights.get("lm_head.weight", weights["model.embed_tokens.weight"]),
            c["rms_norm_eps"],
        )
        elapsed = time.perf_counter() - start
        assert k.shape == v.shape == (26, 1, a.width, 256)
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
            "repeat": repeat,
            "prefill_s": elapsed,
            "repeat_exact": exact,
            "first_token": int(logits.argmax()),
            "kv_shape": list(k.shape),
        }
        f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
torch.save(
    {"ids": ids, "keys": k, "values": v, "logits": logits, "metadata": row},
    a.output + ".pt",
)
