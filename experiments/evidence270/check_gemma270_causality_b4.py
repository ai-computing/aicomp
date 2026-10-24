"""Test native Gemma prefix causality by changing only the prompt suffix."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from rebel.kv_cache import aligned_tensor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache-dir", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
p.add_argument("--kv-blocks", type=int, choices=(5,), required=True)
a = p.parse_args()
artifact = Path(a.cache_dir) / f"unsloth--gemma-3-1b-it-tp1-s9216-b4-k{a.kv_blocks}"
assert (artifact / "config.json").is_file(), "Existing compiled artifact required; never compile in this diagnostic"
metadata = json.loads((artifact / "rbln_config.json").read_text())
assert (metadata["batch_size"], metadata["kvcache_num_blocks"], metadata["max_seq_len"]) == (4, a.kv_blocks, 9216)
assert not Path(a.output).exists(), "Preserve existing evidence"
devices = list(Path("/dev").glob("rbln[0-9]*"))
assert len(devices) == 1
engine = AtomRblnEngine(
    devices=[int(devices[0].name[4:])],
    max_seq_len=9216,
    batch_size=4,
    kv_blocks=a.kv_blocks,
    cache_dir=a.cache_dir,
)
rows = []
try:
    engine.start("unsloth/gemma-3-1b-it")
    assert engine.compiled is False, "Diagnostic must load existing graph"
    b = engine.batcher(prefix_cache=False)
    model = b.model
    decoder = b.decoder
    rc = model.rbln_config
    assert rc.cache_impl == "hybrid" and rc.batch_size == 4 and rc.sliding_window == 512
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
        for lane, path in product(range(4), a.exports):
            global_table.fill_(lane)
            local_table.fill_(lane)
            r = torch.load(path, weights_only=True)
            ids = r["ids"]
            n = len(ids)
            assert n == 128
            cutoff = n // 2
            changed = ids[:cutoff] + [
                (x + 1) % model.config.vocab_size for x in ids[cutoff:]
            ]
            # A prefix-changing control detects wrong-lane or stale/zero-buffer reads.
            prefix_changed = [(x + 1) % model.config.vocab_size for x in ids[:cutoff]] + ids[cutoff:]
            assert prefix_changed[:cutoff] != ids[:cutoff] and prefix_changed[cutoff:] == ids[cutoff:]
            states = []
            for prompt in [ids, changed, prefix_changed]:
                model.prefill_decoder(
                    input_ids=torch.tensor([prompt]),
                    cache_position=torch.arange(n, dtype=torch.int32)[None],
                    batch_idx=lane,
                    block_tables=global_table[0],
                    local_block_tables=local_table[0],
                )
                saved = []
                for meta, buf in zip(metas, buffers):
                    runtime.get_kv_cache_for_layer(meta["name"], lane, 0, n, buf)
                    saved.append(buf[lane, :, :cutoff].copy())
                states.append(saved)
            changed_slots = []
            for i, (x, y) in enumerate(zip(states[0], states[1])):
                changed_slots.append(
                    {
                        "slot": i,
                        "layer": i // 2,
                        "equal": bool(np.array_equal(x, y)),
                        "changed_fraction": float(np.mean(x != y)),
                    }
                )
            control_changes = [not np.array_equal(x, y) for x, y in zip(states[0], states[2])]
            assert len(control_changes) == 52
            row = {
                "artifact": artifact.name,
                "lane": lane,
                "kv_blocks": a.kv_blocks,
                "input_tokens": n,
                "unchanged_prefix": cutoff,
                "slots": changed_slots,
                "prefix_exact": all(s["equal"] for s in changed_slots),
                "prefix_control_changed_slots": [i for i, changed in enumerate(control_changes) if changed],
                "prefix_control_passed": all(control_changes),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
finally:
    engine.close()
with Path(a.output).open("x") as output:
    output.write(json.dumps(rows, indent=2) + "\n")
assert all(row["prefix_exact"] for row in rows), "Causality violation: changing suffix changed prefix KV"

assert len(rows) == 4 * len(a.exports) and {row["lane"] for row in rows} == set(range(4))
assert all(row["prefix_control_passed"] for row in rows), "Prefix control failed: validate lane/cache readback before interpreting causality"
