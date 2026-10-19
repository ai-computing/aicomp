"""Summarize measured transport A/B/B/A rows; retain warmup parity counts."""

import argparse
import json
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]
    if not rows or not all(r["full_logits_exact"] for r in rows):
        raise ValueError("missing results or parity failed")
    groups = {}
    for mode in ("npz", "shared"):
        selected = [
            r for r in rows if r["transport"] == mode and not r.get("warmup", False)
        ]
        groups[mode] = dict(
            requests=len(selected),
            prefix_seconds=sum(r["prefix_seconds"] for r in selected),
            request_seconds=sum(r["request_seconds"] for r in selected),
        )
    if (
        groups["npz"]["requests"] != groups["shared"]["requests"]
        or not groups["npz"]["requests"]
    ):
        raise ValueError("unequal or empty comparison")
    return dict(
        all_requests=len(rows),
        all_tokens=sum(r["output_length"] for r in rows),
        measured=groups,
        reduction_percent={
            key: 100 * (1 - groups["shared"][key] / groups["npz"][key])
            for key in ("prefix_seconds", "request_seconds")
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    print(json.dumps(summarize(a.run), indent=2))
