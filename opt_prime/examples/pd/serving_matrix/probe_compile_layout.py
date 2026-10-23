"""Measure sample/runtime stride mismatch and verify the contiguous contract."""

import json

import torch
from rngd_pd.engines.fsdk import CompileModule, FCConfig, dev_str, hint, sync


class Matmul(torch.nn.Module):
    def forward(self, x, w):
        return x @ w


x = torch.randn(32, 128, dtype=torch.bfloat16)
w = torch.randn(64, 128, dtype=torch.bfloat16).T
with torch.inference_mode():
    for layout in ["preserve", "contiguous"]:
        sample = (
            torch.zeros_like(w)
            if layout == "preserve"
            else torch.zeros(w.shape, dtype=w.dtype)
        )
        m = CompileModule.from_module(
            Matmul(),
            (torch.zeros_like(x), sample),
            compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
        )
        m.to(dev_str(0))
        out = m(x.to(dev_str(0)), w.contiguous().to(dev_str(0)))
        sync(0)
        out = out.cpu()
        ref = x @ w
        print(
            json.dumps(
                {
                    "layout": layout,
                    "sample_stride": list(sample.stride()),
                    "runtime_stride": list(w.contiguous().stride()),
                    "relative_rms": float(
                        (out.float() - ref.float()).norm() / ref.float().norm()
                    ),
                }
            ),
            flush=True,
        )

        if layout == "contiguous":
            torch.testing.assert_close(out, ref, atol=0.05, rtol=0.01)
