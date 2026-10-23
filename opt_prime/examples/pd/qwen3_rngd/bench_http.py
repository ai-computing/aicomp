import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
a = ap.parse_args()
base = "http://127.0.0.1:18197"
records = torch.load("qwen3_197_kv.pt", weights_only=True)
rows = []


def request(i, repeat, extra=None):
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "prompt": records[i]["ids"],
        "temperature": 0,
        "max_tokens": 16,
        "ignore_eos": True,
    }
    body.update(extra or {})
    req = urllib.request.Request(
        base + "/v1/completions",
        json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-request-id": f"{a.name}-{repeat}-{i}",
        },
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    row = {
        "repeat": repeat,
        "prompt": i,
        "seconds": time.perf_counter() - t,
        "response": out,
    }
    print(json.dumps(row), flush=True)
    return row


for repeat in range(4):
    for i in range(3):
        rows.append(request(i, repeat))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    rows.extend(pool.map(lambda i: request(i, 4), range(3)))
# Capacity must be rejected before submitting to the worker, then service must remain usable.
try:
    request(0, 5, {"prompt": [1] * 129})
    raise AssertionError("oversize request unexpectedly accepted")
except urllib.error.HTTPError as e:
    assert e.code == 400, e.code
    rejected = json.loads(e.read())
rows.append(request(0, 6))
with urllib.request.urlopen(base + "/metrics") as r:
    metrics = r.read().decode()
Path(f"qwen3_199_http_{a.name}.json").write_text(
    json.dumps({"rows": rows, "oversize": rejected, "metrics": metrics}, indent=2)
    + "\n"
)
print(metrics, flush=True)
