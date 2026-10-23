import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hetero_p2p.adapters.atom_koni_external import ExternalCacheDecoder
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--artifact", required=True)
p.add_argument("--tables", required=True)
a = p.parse_args()
artifact, tables = a.artifact, a.tables
# This fixture owns exactly one device-plugin allocated ATOM, visible as index0.
import rebel

assert len(list(Path("/dev").glob("rbln[0-9]*"))) == 1
assert rebel.device_count() == 1
e = ExternalCacheDecoder(artifact, tables, device=0)
assert e.batch_size == 4
rows = json.loads(Path("koni196_prompts.json").read_text())
host = []
for i in range(4):
    with np.load(f"koni196_kv_{i}.npz") as z:
        host.append(
            np.stack((z["keys"].transpose(0, 1, 3, 2), z["values"]), axis=1)
            .reshape(64, 8, 512, 128)
            .copy()
        )
host = np.stack(host, axis=1)
runs = []
checks = []
readback = 0
for run in range(2):
    for i, c in enumerate(e.caches):
        c.copy_(torch.from_numpy(host[i]))
    torch.rbln.synchronize()
    if run == 0:
        for i, c in enumerate(e.caches):
            assert np.array_equal(c.cpu().float().numpy(), host[i]), i
            readback += 1
    current = [r["ids"][-1] for r in rows]
    tokens = [[] for _ in rows]
    vectors = []
    for j in range(16):
        out = e.step_batch(current, [len(r["ids"]) - 1 + j for r in rows])
        assert np.isfinite(out).all()
        vectors.append(out.copy())
        current = out.argmax(axis=1).tolist()
        for i, t in enumerate(current):
            tokens[i].append(t)
    runs.append(np.stack(vectors))
    checks.append(tokens)
tok = AutoTokenizer.from_pretrained("KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024")
result = {
    "readback_exact_slots": readback,
    "repeated_logits_exact": bool(np.array_equal(*runs)),
    "outputs": [
        {"prompt": r["prompt"], "tokens": t, "text": tok.decode(t)}
        for r, t in zip(rows, checks[0])
    ],
    "logits_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in runs],
    "scope": "RNGD prefix KV through host admission to existing ATOM graph; no DMA or native-prefill claim",
}
Path("koni196_decode.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(result, ensure_ascii=False), flush=True)
