import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
a = ap.parse_args()
records = torch.load("qwen3_197_kv.pt", weights_only=True)
rows = []
for repeat in range(3):
    for i, r in enumerate(records):
        payload = {
            "model": "Qwen/Qwen3-0.6B",
            "prompt": r["ids"],
            "max_tokens": 16,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:18197/v1/completions",
            json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-request-id": f"{a.name}-stream-{repeat}-{i}",
            },
        )
        t = time.perf_counter()
        first = None
        done = False
        chunks = []
        with urllib.request.urlopen(req, timeout=120) as f:
            for line in f:
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    done = True
                    break
                chunk = json.loads(line[6:])
                chunks.append(chunk)
                if (
                    chunk.get("choices")
                    and chunk["choices"][0].get("text")
                    and first is None
                ):
                    first = time.perf_counter() - t
        assert done and first is not None
        row = {
            "repeat": repeat,
            "prompt": i,
            "ttft_s": first,
            "e2e_s": time.perf_counter() - t,
            "text": "".join(
                c["choices"][0].get("text", "") for c in chunks if c.get("choices")
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
payload.update(stream=False, ignore_eos=False, prompt=records[0]["ids"])
req = urllib.request.Request(
    "http://127.0.0.1:18197/v1/completions",
    json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "x-request-id": a.name + "-eos"},
)
with urllib.request.urlopen(req, timeout=120) as f:
    eos = json.load(f)
assert eos["choices"][0]["finish_reason"] == "stop", eos
Path(f"qwen3_199_stream_{a.name}.json").write_text(
    json.dumps({"rows": rows, "eos": eos}, indent=2) + "\n"
)
