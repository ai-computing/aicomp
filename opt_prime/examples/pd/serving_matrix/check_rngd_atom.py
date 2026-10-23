"""Check RNGD exports against ATOM standalone and both existing import paths."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from atom_single_block import SingleBlockTarget
from atom_stream_upload import import_atom_span
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--cache-dir", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
p.add_argument("--eos-aware", action="store_true")
p.add_argument("--max-output", type=int, default=32)
p.add_argument("--kv-blocks", type=int, default=3)
a = p.parse_args()
devices = list(Path("/dev").glob("rbln[0-9]*"))
assert len(devices) == 1
e = AtomRblnEngine(
    devices=[int(devices[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=a.kv_blocks,
    cache_dir=a.cache_dir,
)
rows = []
try:
    e.start(a.model)
    b = e.batcher(prefix_cache=False)
    blocks = b.blocks if b.blocks is not None else SingleBlockTarget(b.model)

    def decode(first, n):
        tokens = [first]
        vectors = []
        for j in range(a.max_output - 1):
            if tokens[-1] in eos:
                break
            z = (
                b.decoder(
                    input_ids=torch.tensor([[tokens[-1]]]),
                    cache_position=torch.tensor([[n + j]], dtype=torch.int32),
                    **(
                        {"block_tables": b.blocks.table([0])}
                        if b.blocks is not None
                        else {}
                    ),
                )
                .logits[0, -1]
                .float()
            )
            assert bool(torch.isfinite(z).all())
            tokens.append(int(z.argmax()))
            vectors.append(z.clone())
        return tokens, torch.stack(vectors) if vectors else torch.empty((0,))

    with torch.inference_mode():
        for path in a.exports:
            r = torch.load(path, weights_only=True)
            eos = set(r["eos_ids"]) if a.eos_aware else set()
            assert not a.eos_aware or eos
            assert a.max_output > 0
            ids = r["ids"]
            n = len(ids)
            if b.blocks is not None:
                b.blocks.admit(0, ids)
            block = blocks.lane_block[0]
            z = b._prefill(0, ids).logits[0, -1].float()
            native, native_logits = decode(int(z.argmax()), n)
            variants = {}
            vectors = []
            for name in ["original", "portable"]:
                t = time.perf_counter()
                if name == "original":
                    blocks.import_span(0, 0, r["keys"], r["values"])
                else:
                    import_atom_span(blocks, 0, 0, r["keys"], r["values"])
                elapsed = time.perf_counter() - t
                exact = 0
                for meta, buf in zip(blocks.metas, blocks._staging()):
                    expected = buf[block, :, :n].copy()
                    buf.fill(0)
                    blocks.io_rt.get_kv_cache_for_layer(meta["name"], block, 0, n, buf)
                    assert np.array_equal(expected, buf[block, :, :n])
                    exact += 1
                tokens, logits = decode(int(r["logits"].argmax()), n)
                vectors.append(logits)
                different = next(
                    (i for i, (x, y) in enumerate(zip(native, tokens)) if x != y), None
                )
                divergence = None
                if different is not None:
                    native_row = z if different == 0 else native_logits[different - 1]
                    imported_row = (
                        r["logits"] if different == 0 else logits[different - 1]
                    )
                    divergence = {
                        "index": different,
                        "native_top5": [
                            {"token": int(i), "logit": float(native_row[i])}
                            for i in native_row.topk(5).indices
                        ],
                        "imported_top5": [
                            {"token": int(i), "logit": float(imported_row[i])}
                            for i in imported_row.topk(5).indices
                        ],
                    }
                variants[name] = {
                    "first_divergence": divergence,
                    "tokens": tokens,
                    "first_decode_top5": [
                        {"token": int(i), "logit": float(logits[0, i])}
                        for i in (logits[0].topk(5).indices if len(logits) else [])
                    ],
                    "native_equal": tokens == native,
                    "readback_exact_slots": exact,
                    "import_s": elapsed,
                }
            row = {
                "export": Path(path).name,
                "case": r.get("metadata", {}).get("case"),
                "eos_ids": sorted(eos),
                "input_tokens": n,
                "native_tokens": native,
                "native_first_decode_top5": [
                    {"token": int(i), "logit": float(native_logits[0, i])}
                    for i in (
                        native_logits[0].topk(5).indices if len(native_logits) else []
                    )
                ],
                "variants": variants,
                "importer_logits_exact": torch.equal(*vectors),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if b.blocks is not None:
                b.blocks.release(0, None)
finally:
    e.close()
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
