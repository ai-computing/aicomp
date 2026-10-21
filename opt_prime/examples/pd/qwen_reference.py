"""Test-only host admission and complete-logit oracle for the P2P worker."""

import numpy as np


class ReferenceValidator:
    def __init__(self):
        self.references = {}

    def admit(self, engine, spec, command):
        import torch

        with np.load(command["prefix"]) as data:
            kv = np.stack((data["keys"].transpose(0, 1, 3, 2), data["values"]), axis=1)
            kv = kv.reshape(spec.caches, spec.kv_heads, 512, spec.head_dim)
        for cache, host in zip(engine.caches, kv):
            cache.copy_(torch.from_numpy(host[None].copy()))
        torch.rbln.synchronize()

    def check(self, command, actual):
        key = tuple(command["prompt"])
        if command["cmd"] == "reference":
            self.references[key] = actual
        expected = self.references[key][: command["count"]]
        return dict(
            passed=bool(np.array_equal(actual, expected)),
            max_abs_diff=float(np.max(np.abs(actual - expected))),
        )
