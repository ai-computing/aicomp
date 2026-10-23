"""ATOM B1 direct-batcher sweep; launch through the isolated allocation bootstrap."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from inputs import exact_prompt
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--cache-dir", required=True)
p.add_argument("--output", required=True)
p.add_argument("--max-context", type=int, default=9216)
p.add_argument("--experimental-hybrid-tables", action="store_true")
p.add_argument("--kv-blocks", type=int, default=3)
a = p.parse_args()
devices = list(Path("/dev").glob("rbln[0-9]*"))
assert len(devices) == 1, "requires isolated single-ATOM allocation"
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
filler = tok.encode(
    "This is background context. The red box contains apples. The blue box contains books.\n",
    add_special_tokens=False,
)
suffix = tok.encode(
    "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
)
e = AtomRblnEngine(
    devices=[int(devices[0].name[4:])],
    max_seq_len=a.max_context,
    batch_size=1,
    kv_blocks=a.kv_blocks,
    cache_dir=a.cache_dir,
)
try:
    t = time.perf_counter()
    e.start(a.model)
    print(
        json.dumps(
            {"ready": True, "load_s": time.perf_counter() - t, "compiled": e.compiled}
        ),
        flush=True,
    )
    b = e.batcher(prefix_cache=False)
    if a.experimental_hybrid_tables:
        from hybrid_tables import install

        install(b)
    with open(a.output, "w", buffering=1) as f:
        for n in [128, 512, 2048, 8192]:
            ids = exact_prompt(prefix, filler, suffix, n)
            for count in [32, 128]:
                previous = None
                for repeat in range(2):
                    clocks = []
                    start = time.perf_counter()
                    future = b.submit(
                        PDRequest(f"{n}-{count}-{repeat}", tuple(ids), count),
                        on_token=lambda _, clocks=clocks: clocks.append(
                            time.perf_counter()
                        ),
                    )
                    while not future.done():
                        b.step()
                    tokens = future.result()
                    end = time.perf_counter()
                    assert len(tokens) == count
                    assert previous is None or tokens == previous, "repeat mismatch"
                    previous = tokens
                    row = {
                        "input_tokens": n,
                        "output_tokens": count,
                        "repeat": repeat,
                        "batch": 1,
                        "input_sha256": hashlib.sha256(
                            json.dumps([ids]).encode()
                        ).hexdigest(),
                        "ttft_s": clocks[0] - start,
                        "e2e_s": end - start,
                        "tpot_s": (clocks[-1] - clocks[0]) / (count - 1),
                        "tokens": tokens,
                        "status": "passed",
                    }
                    f.write(json.dumps(row) + "\n")
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "tokens"}),
                        flush=True,
                    )
finally:
    e.close()
