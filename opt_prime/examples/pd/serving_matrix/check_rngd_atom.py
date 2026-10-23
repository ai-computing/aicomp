"""Check RNGD exports against ATOM standalone and both existing import paths."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from atom_stream_upload import import_atom_span
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--cache-dir", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
devices = list(Path("/dev").glob("rbln[0-9]*"))
assert len(devices) == 1
e = AtomRblnEngine(
    devices=[int(devices[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=3,
    cache_dir=a.cache_dir,
)
rows = []
try:
    e.start(a.model)
    b = e.batcher(prefix_cache=False)

    def decode(first, n):
        tokens = [first]
        vectors = []
        for j in range(31):
            z = (
                b.decoder(
                    input_ids=torch.tensor([[tokens[-1]]]),
                    cache_position=torch.tensor([[n + j]], dtype=torch.int32),
                    block_tables=b.blocks.table([0]),
                )
                .logits[0, -1]
                .float()
            )
            assert bool(torch.isfinite(z).all())
            tokens.append(int(z.argmax()))
            vectors.append(z.clone())
        return tokens, torch.stack(vectors)

    with torch.inference_mode():
        for path in a.exports:
            r = torch.load(path, weights_only=True)
            ids = r["ids"]
            n = len(ids)
            b.blocks.admit(0, ids)
            block = b.blocks.lane_block[0]
            z = b._prefill(0, ids).logits[0, -1].float()
            native, native_logits = decode(int(z.argmax()), n)
            variants = {}
            vectors = []
            for name in ["original", "portable"]:
                t = time.perf_counter()
                if name == "original":
                    b.blocks.import_span(0, 0, r["keys"], r["values"])
                else:
                    import_atom_span(b.blocks, 0, 0, r["keys"], r["values"])
                elapsed = time.perf_counter() - t
                exact = 0
                for meta, buf in zip(b.blocks.metas, b.blocks._staging()):
                    expected = buf[block, :, :n].copy()
                    buf.fill(0)
                    b.blocks.io_rt.get_kv_cache_for_layer(
                        meta["name"], block, 0, n, buf
                    )
                    assert np.array_equal(expected, buf[block, :, :n])
                    exact += 1
                tokens, logits = decode(int(r["logits"].argmax()), n)
                vectors.append(logits)
                variants[name] = {
                    "tokens": tokens,
                    "first_decode_top5": [
                        {"token": int(i), "logit": float(logits[0, i])}
                        for i in logits[0].topk(5).indices
                    ],
                    "native_equal": tokens == native,
                    "readback_exact_slots": exact,
                    "import_s": elapsed,
                }
            row = {
                "export": Path(path).name,
                "input_tokens": n,
                "native_tokens": native,
                "native_first_decode_top5": [
                    {"token": int(i), "logit": float(native_logits[0, i])}
                    for i in native_logits[0].topk(5).indices
                ],
                "variants": variants,
                "importer_logits_exact": torch.equal(*vectors),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            b.blocks.release(0, None)
finally:
    e.close()
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
