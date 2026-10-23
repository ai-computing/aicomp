import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from prefill import Prefill
from safetensors import safe_open

path = Path(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
c = json.loads((path / "config.json").read_text())
w = {}
for shard in path.glob("*.safetensors"):
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open is not iterable
            w[k] = f.get_tensor(k).float()
records = torch.load("qwen3_197_kv.pt", weights_only=True)
from prefill_baseline import Prefill as Baseline

rows = []
e = Prefill(c, w, width=128, chip=0)
for repeat in range(4):
    for i, r in enumerate(records):
        outputs = {}
        times = {}
        for name in (
            ["baseline", "host_stack"]
            if repeat % 2 == 0
            else ["host_stack", "baseline"]
        ):
            t = time.perf_counter()
            outputs[name] = (
                Baseline.prefill(e, r["ids"])
                if name == "baseline"
                else e.prefill(r["ids"])
            )
            times[name] = time.perf_counter() - t
        exact = [
            torch.equal(a, b)
            for a, b in zip(outputs["baseline"], outputs["host_stack"])
        ]
        assert all(exact), exact
        row = {
            "repeat": repeat,
            "prompt": i,
            "n": len(r["ids"]),
            "seconds": times,
            "kv_logits_exact": exact,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
Path("qwen3_199_export_ab.json").write_text(json.dumps(rows, indent=2) + "\n")
