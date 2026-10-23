"""RNGD regression for attention output layout; run via the isolated SDK launcher."""

import json

import torch
from koni_parts import AttentionPart
from rngd_pd.engines.fsdk import (
    CompileModule,
    FCConfig,
    TacticKernelModule,
    dev_str,
    hint,
    sync,
)
from rngd_pd.kernels.prefill_chunk import softmax_dsl, zipadd_dsl

BF = torch.bfloat16
C, H, NH, NK, D = 32, 4096, 32, 8, 128


class Softmax(torch.nn.Module):
    def forward(self, x):
        return (x.float().softmax(-1).to(x.dtype),)


class Add(torch.nn.Module):
    def forward(self, x, y):
        return (x + y,)


torch.manual_seed(221)
x = torch.randn(C, H, dtype=BF) * 0.1
wq = torch.randn(H, NH * D, dtype=BF) * 0.01
wk = torch.randn(H, NK * D, dtype=BF) * 0.01
wv = torch.randn(H, NK * D, dtype=BF) * 0.01
wo = torch.randn(NH * D, H, dtype=BF) * 0.01
rot = torch.eye(D, dtype=BF).repeat(C, 1, 1)
mask = torch.triu(torch.full((C, C), -10000, dtype=BF), 1).repeat(4, 1)[None]
args = (x, x, wq, wk, wv, wo, rot, mask)
mod = AttentionPart(
    C,
    NH,
    NK,
    D,
    TacticKernelModule(softmax_dsl(NK, 4 * C, C)),
    TacticKernelModule(zipadd_dsl(C, H)),
)
m = CompileModule.from_module(
    mod,
    tuple(torch.zeros_like(t) for t in args),
    compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
)
m.to(dev_str(0))
with torch.inference_mode():
    out = m(*(t.to(dev_str(0)) for t in args))
    sync(0)
    host = [t.cpu() for t in out]
    ref = AttentionPart(C, NH, NK, D, Softmax(), Add())(*args)
    print(
        json.dumps(
            {
                "device_shapes": [list(t.shape) for t in out],
                "host_shapes": [list(t.shape) for t in host],
                "expected": [list(t.shape) for t in ref],
            }
        ),
        flush=True,
    )
    for name, a, b in zip(["hidden", "keys", "values"], host, ref):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        error = float(
            (a.float() - b.float()).square().mean().sqrt()
            / b.float().square().mean().sqrt()
        )
        assert bool(torch.isfinite(a).all()) and error < 0.001, (name, error)
        print(
            json.dumps({"tensor": name, "relative_rms": error, "finite": True}),
            flush=True,
        )
