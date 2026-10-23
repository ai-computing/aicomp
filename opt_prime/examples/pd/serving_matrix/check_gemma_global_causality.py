"""Test native Gemma prefix causality by changing only the prompt suffix."""

import argparse
import json
from pathlib import Path

import gemma_causal_compat  # noqa: F401 - validated process-local SDK compatibility
import numpy as np
import torch
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from rebel.kv_cache import aligned_tensor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache-dir", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
p.add_argument("--global-causal", action="store_true")
a = p.parse_args()
if a.global_causal:
    import gemma_global_causal_compat  # noqa: F401
devices = list(Path("/dev").glob("rbln[0-9]*"))
assert len(devices) == 1
engine = AtomRblnEngine(
    devices=[int(devices[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=3,
    cache_dir=a.cache_dir,
)
rows = []
try:
    engine.start("unsloth/gemma-3-1b-it")
    b = engine.batcher(prefix_cache=False)
    model = b.model
    decoder = b.decoder
    rc = model.rbln_config
    assert rc.cache_impl == "hybrid" and rc.batch_size == 1 and rc.sliding_window == 512
    metas = rc.kvcache_metas
    assert all(isinstance(m, dict) and m["dtype"] == "float32" for m in metas)
    assert len(metas) == 52
    runtime = model.model[1]
    buffers = []
    for meta in metas:
        count = int(np.prod(meta["shape"]))
        size = runtime.get_kv_cache_size_for_layer(meta["name"])
        assert size == count * 2, (meta["name"], size, count)
        buffers.append(
            aligned_tensor(
                count, dtype=np.uint16, alignment=2 << 20, tensor_type="np"
            ).reshape(meta["shape"])
        )
    global_table = torch.zeros((1, 1), dtype=torch.int16)
    local_table = torch.zeros((1, 1), dtype=torch.int16)

    with torch.inference_mode():
        for path in a.exports:
            r = torch.load(path, weights_only=True)
            ids = r["ids"]
            n = len(ids)
            assert n == 128
            cutoff = n // 2
            changed = ids[:cutoff] + [
                (x + 1) % model.config.vocab_size for x in ids[cutoff:]
            ]
            states = []
            for prompt in [ids, changed]:
                model.prefill_decoder(
                    input_ids=torch.tensor([prompt]),
                    cache_position=torch.arange(n, dtype=torch.int32)[None],
                    batch_idx=0,
                    block_tables=global_table[0],
                    local_block_tables=local_table[0],
                )
                saved = []
                for meta, buf in zip(metas, buffers):
                    runtime.get_kv_cache_for_layer(meta["name"], 0, 0, n, buf)
                    saved.append(buf[0, :, :cutoff].copy())
                states.append(saved)
            changed_slots = []
            for i, (x, y) in enumerate(zip(*states)):
                changed_slots.append(
                    {
                        "slot": i,
                        "layer": i // 2,
                        "equal": bool(np.array_equal(x, y)),
                        "changed_fraction": float(np.mean(x != y)),
                    }
                )
            row = {
                "global_causal": a.global_causal,
                "input_tokens": n,
                "unchanged_prefix": cutoff,
                "slots": changed_slots,
                "prefix_exact": all(s["equal"] for s in changed_slots),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
finally:
    engine.close()
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
