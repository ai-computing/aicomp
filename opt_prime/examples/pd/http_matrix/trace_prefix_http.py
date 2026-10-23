"""Record actual HTTP owner logits at the two initial 2K positions (#212)."""

import hashlib
import json
import sys
from pathlib import Path

import validate
from opt_prime.accel.pd import batching

trace = {"samples": [], "prefills": []}
sample = batching.select_token
prefill = batching.ContinuousBatcher._prefill_chunk
build = validate.build


def select(logits, params, position):
    if position in (2048, 2049):
        top = logits.float().topk(5)
        trace["samples"].append(
            {
                "position": position,
                "ids": top.indices.tolist(),
                "logits": top.values.tolist(),
            }
        )
    return sample(logits, params, position)


def chunk(self, r):
    if len(r.request.input_ids) == 2048:
        trace["prefills"].append(
            {
                "request": r.request.request_id,
                "start": r.prefilled,
                "end": min(r.prefilled + self.chunk, 2048),
                "input_sha256": hashlib.sha256(
                    json.dumps([list(r.request.input_ids)]).encode()
                ).hexdigest(),
            }
        )
    return prefill(self, r)


def traced_build(args):
    owner = build(args)
    m = owner.service.batcher.model
    trace["attention"] = m.config._attn_implementation
    trace["dtype"] = str(next(m.parameters()).dtype)
    return owner


batching.select_token = select
batching.ContinuousBatcher._prefill_chunk = chunk
validate.build = traced_build
try:
    validate.main()
finally:
    batching.select_token = sample
    batching.ContinuousBatcher._prefill_chunk = prefill
    validate.build = build
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.with_suffix(".trace.json").write_text(json.dumps(trace, indent=2) + "\n")
