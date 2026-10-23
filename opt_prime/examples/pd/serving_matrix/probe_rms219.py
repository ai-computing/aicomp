import json
import time

import torch
from rngd_pd.engines.fsdk import (
    CompileModule,
    FCConfig,
    TacticKernelModule,
    dev_str,
    hint,
    sync,
)
from rngd_pd.kernels.prefill_chunk import rms_dsl


class M(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.r = TacticKernelModule(rms_dsl(c, 4096))

    def forward(self, x, w, e, n):
        return self.r(x, w, e, n)[0]


for c in [512, 256]:
    t = time.perf_counter()
    try:
        sample = (
            torch.zeros(c, 4096, dtype=torch.bfloat16),
            torch.ones(c, 4096, dtype=torch.bfloat16),
            torch.ones(c, 4096),
            torch.full((c, 4096), -0.5),
        )
        m = CompileModule.from_module(
            M(c),
            sample,
            compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
        )
        m.to(dev_str(0))
        y = m(*(x.to(dev_str(0)) for x in sample))
        sync(0)
        print(
            json.dumps(
                {
                    "rows": c,
                    "passed": True,
                    "finite": bool(torch.isfinite(y.cpu()).all()),
                    "seconds": time.perf_counter() - t,
                }
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - preserve failed compiler probes
        print(
            json.dumps(
                {
                    "rows": c,
                    "passed": False,
                    "error": str(e)[:500],
                    "seconds": time.perf_counter() - t,
                }
            ),
            flush=True,
        )
