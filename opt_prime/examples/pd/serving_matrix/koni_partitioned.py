"""Research stage-only KONI engine: separate RMS, attention and FFN compiles."""

import json
import time

import numpy as np
import torch
from koni_parts import AttentionPart, FFNPart, Parts
from rngd_pd.engines.fsdk import CompileModule, FCConfig, TacticKernelModule, hint
from rngd_pd.engines.prefill_full import FullPrefillEngine, rot_matrices_fast
from rngd_pd.kernels.prefill_chunk import (
    HD,
    NH,
    NKV,
    G,
    H,
    M,
    rms_dsl,
    softmax_dsl,
    swiglu_dsl,
    zipadd_dsl,
)

BF = torch.bfloat16


class NormPart(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.kernel = TacticKernelModule(rms_dsl(rows, H))

    def forward(self, x, w, eps, exponent):
        return self.kernel(x, w, eps, exponent)[0]


class PartitionedPrefill(FullPrefillEngine):
    def __init__(self, cfg, weights, *, rms_tile=512, **kwargs):
        if kwargs.get("group") != 1 or rms_tile <= 0 or kwargs["C"] % rms_tile:
            raise ValueError(
                "partitioned engine requires group1 and divisible RMS tiles"
            )
        self.rms_tile = rms_tile
        super().__init__(cfg, weights, **kwargs)

    def _build_lmhead(self, cache_dir):
        # The existing host exporter owns final normalization and the output head.
        pass

    def add_bucket(self, width):
        if width in self.buckets:
            return self.buckets[width]
        rows = self.rms_tile
        if width % rows:
            raise ValueError("width must be divisible by RMS rows")

        def zeros(*shape, dtype=BF):
            return torch.zeros(shape, dtype=dtype)

        def compile_part(name, module, sample):
            start = time.perf_counter()
            print(json.dumps({"compile_start": name, "width": width}), flush=True)
            compiled = CompileModule.from_module(
                module,
                sample,
                compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
            )
            compiled.to(self.dev)
            print(
                json.dumps(
                    {"compile_done": name, "seconds": time.perf_counter() - start}
                ),
                flush=True,
            )
            return compiled

        norm = compile_part(
            "rms",
            NormPart(rows),
            (
                zeros(rows, H),
                zeros(rows, H),
                zeros(rows, H, dtype=torch.float32),
                zeros(rows, H, dtype=torch.float32),
            ),
        )
        attention = compile_part(
            "attention",
            AttentionPart(
                width,
                NH,
                NKV,
                HD,
                TacticKernelModule(softmax_dsl(NKV, G * width, width)),
                TacticKernelModule(zipadd_dsl(width, H)),
            ),
            (
                zeros(width, H),
                zeros(width, H),
                zeros(H, NH * HD),
                zeros(H, NKV * HD),
                zeros(H, NKV * HD),
                zeros(NH * HD, H),
                zeros(width, HD, HD),
                zeros(1, G * width, width),
            ),
        )
        ffn = compile_part(
            "ffn",
            FFNPart(
                TacticKernelModule(swiglu_dsl(width, M)),
                TacticKernelModule(zipadd_dsl(width, H)),
            ),
            (zeros(width, H), zeros(width, H), zeros(H, M), zeros(H, M), zeros(M, H)),
        )
        causal = np.triu(np.full((width, width), -1e4, np.float32), 1)
        self.buckets[width] = {
            "m": Parts(width, rows, norm, attention, ffn),
            "norms": [
                (
                    self._d(np.tile(self._n1[i], (rows, 1)))[None],
                    self._d(np.tile(self._n2[i], (rows, 1)))[None],
                )
                for i in range(self.nl)
            ],
            "eps": self._df(np.full((rows, H), self.eps * H, np.float32)),
            "nh": self._df(np.full((rows, H), -0.5, np.float32)),
            "rot": self._d(rot_matrices_fast(self._inv, width)),
            "mask": self._d(np.tile(causal, (G, 1)).reshape(1, G * width, width)),
        }
        self.C = max(self.buckets)
        return self.buckets[width]
