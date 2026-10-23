"""Compile one checkpoint-sized Gemma layer and compare with CPU equations."""

import json
import time

import torch
from gemma_parts import FlatGemmaLayer, attention_inputs
from rngd_pd.engines.fsdk import CompileModule, FCConfig, dev_str, hint, sync

c = {
    "hidden_size": 1152,
    "intermediate_size": 6912,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "sliding_window": 512,
    "sliding_window_pattern": 6,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 256,
}
torch.manual_seed(223)
width = 128
h = c["hidden_size"]
d = c["head_dim"]
m = c["intermediate_size"]
q = 4 * d
rot, mask = attention_inputs(c, width, 0)
x = torch.randn(width, h, dtype=torch.bfloat16) * 0.1
norms = [torch.randn(n) * 0.1 for n in [h, h, h, h, d, d]]
mats = [
    torch.randn(a, b, dtype=torch.bfloat16) * 0.01
    for a, b in [(h, q), (h, d), (h, d), (q, h), (h, m), (h, m), (m, h)]
]
args = (x, *norms, *mats, rot.to(torch.bfloat16), mask.to(torch.bfloat16))
module = FlatGemmaLayer(c)
start = time.perf_counter()
print(json.dumps({"compile_start": True, "width": width}), flush=True)
compiled = CompileModule.from_module(
    module,
    tuple(torch.zeros_like(t) for t in args),
    compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
)
compiled.to(dev_str(0))
sync(0)
print(json.dumps({"compile_s": time.perf_counter() - start}), flush=True)
with torch.inference_mode():
    expected = module(*args)
    actual = compiled(*(t.to(dev_str(0)) for t in args))
    sync(0)
    for name, a, b in zip(["hidden", "keys", "values"], actual, expected):
        a = a.cpu()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        error = float(
            (a.float() - b.float()).square().mean().sqrt()
            / b.float().square().mean().sqrt()
        )
        assert bool(torch.isfinite(a).all())
        print(
            json.dumps({"tensor": name, "shape": list(a.shape), "relative_rms": error}),
            flush=True,
        )
