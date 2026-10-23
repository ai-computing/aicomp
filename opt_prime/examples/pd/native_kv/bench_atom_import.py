import json
import time
from pathlib import Path

import numpy as np
import torch
from atom_stream_upload import import_atom_span as candidate
from atom_stream_upload_baseline import import_atom_span as baseline
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

nodes = list(Path("/dev").glob("rbln[0-9]*"))
assert len(nodes) == 1
engine = AtomRblnEngine(
    devices=[int(nodes[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=3,
    cache_dir=str(Path("atom-artifacts").resolve()),
)
rows = []
try:
    engine.start("Qwen/Qwen3-4B")
    b = engine.batcher(prefix_cache=False)
    for n in [128, 512, 2048]:
        r = torch.load(f"rngd-qwen3-4b-{n}.pt", weights_only=True)
        b.blocks.admit(0, r["ids"])
        block = b.blocks.lane_block[0]
        saved = None
        for repeat in range(6):
            for name in (
                ["baseline", "candidate"]
                if repeat % 2 == 0
                else ["candidate", "baseline"]
            ):
                t = time.perf_counter()
                (baseline if name == "baseline" else candidate)(
                    b.blocks, 0, 0, r["keys"], r["values"]
                )
                elapsed = time.perf_counter() - t
                slots = 0
                # Check each variant once at each length; exclude readback and decode from import timing.
                if repeat == 0:
                    for meta, buf in zip(b.blocks.metas, b.blocks._staging()):
                        expected = buf[block, :, :n].copy()
                        buf.fill(0)
                        b.blocks.io_rt.get_kv_cache_for_layer(
                            meta["name"], block, 0, n, buf
                        )
                        assert np.array_equal(expected, buf[block, :, :n])
                        slots += 1
                    token = int(r["logits"].argmax())
                    vectors = []
                    with torch.inference_mode():
                        for j in range(8):
                            z = (
                                b.decoder(
                                    input_ids=torch.tensor([[token]]),
                                    cache_position=torch.tensor(
                                        [[n + j]], dtype=torch.int32
                                    ),
                                    block_tables=b.blocks.table([0]),
                                )
                                .logits[0, -1]
                                .float()
                            )
                            vectors.append(z.clone())
                            token = int(z.argmax())
                    vec = torch.stack(vectors)
                    if saved is None:
                        saved = vec
                    else:
                        assert torch.equal(saved, vec), "importer decode logits differ"
                row = {
                    "input_tokens": n,
                    "repeat": repeat,
                    "mode": name,
                    "import_s": elapsed,
                    "readback_exact_slots": slots,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        b.blocks.release(0, None)
finally:
    engine.close()
Path("atom-import205.json").write_text(json.dumps(rows, indent=2) + "\n")
