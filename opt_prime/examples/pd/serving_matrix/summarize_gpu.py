"""Summarize observed runs; does not label numerical differences as transport failures."""

import argparse
import json
from pathlib import Path

from inputs import through_eos

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--results", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
eos = {
    "qwen3-06b": [151645],
    "qwen3-4b": [151645],
    "koni-8b": [128001],
    "gemma3-1b": [1, 106],
}
summary = {}
for name, stops in eos.items():
    rows = []
    for file in sorted(a.results.glob(f"gpu-{name}*.jsonl")):
        rows.extend(
            r
            for r in map(json.loads, file.read_text().splitlines())
            if "metadata" not in r
        )
    runs = {
        (r["input_tokens"], r["batch"], r.get("output_tokens"), r["repeat"]): r
        for r in rows
    }
    if len(runs) != len(rows):
        raise ValueError("duplicate run keys")
    comparisons = []
    for (n, b, count, repeat), r in runs.items():
        if b == 1 or repeat != 1 or r["status"] != "passed":
            continue
        base = runs.get((n, 1, count, repeat))
        if base is None or base["status"] != "passed":
            continue
        left, right = base["tokens"][0], r["tokens"][0]
        comparisons.append(
            {
                "input_tokens": n,
                "batch": b,
                "output_tokens": count,
                "full_tokens_equal": left == right,
                "through_eos_equal": through_eos(left, stops)
                == through_eos(right, stops),
            }
        )
    summary[name] = {
        "expected_runs": 48,
        "observed_runs": len(rows),
        "passed_runs": sum(r["status"] == "passed" for r in rows),
        "oom_runs": sum(r["status"] == "oom" for r in rows),
        "batch_comparisons": comparisons,
    }
a.output.write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            k: {s: v for s, v in row.items() if s != "batch_comparisons"}
            for k, row in summary.items()
        },
        indent=2,
    )
)
