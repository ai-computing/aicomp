"""Validate actual subprocess prefill results and recovery against trusted exports."""

import argparse
import json
import os
from pathlib import Path

import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--python", required=True)
p.add_argument("--model", required=True)
p.add_argument("--engine-factory", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
exports = [torch.load(path, weights_only=True) for path in a.exports]
layers, heads, _, dim = exports[0]["keys"].shape
rows = []
worker = SubprocessPrefillWorker(
    a.python,
    ["--model", a.model, "--width", str(a.width), "--engine-factory", a.engine_factory],
    slots=1,
    layers=layers,
    kv_heads=heads,
    head_dim=dim,
    max_context=a.width,
    env=dict(os.environ),
)
try:
    for i, record in enumerate(exports + exports[:1]):
        if i == len(exports):
            bad = worker.submit(
                PDRequest("oversize", tuple([1] * (a.width + 1)), 1), slot=0
            )
            try:
                bad.result(timeout=120)
            except RuntimeError as error:
                assert "capacity" in str(error)
            else:
                raise AssertionError("oversize request was accepted")
        request = PDRequest(str(i), tuple(record["ids"]), 1)
        out = worker.submit(request, slot=0).result(timeout=120)
        assert out.start == 0 and out.token == int(record["logits"].argmax())
        assert torch.equal(out.keys, record["keys"]) and torch.equal(
            out.values, record["values"]
        )
        rows.append(
            {
                "case": record["metadata"]["case"],
                "keys_exact": True,
                "values_exact": True,
                "first_token_exact": True,
                "prefill_s": out.prefill_s,
            }
        )
    result = {"worker": "live_rngd_subprocess", "rows": rows, "oversize_recovery": True}
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
finally:
    worker.close()
