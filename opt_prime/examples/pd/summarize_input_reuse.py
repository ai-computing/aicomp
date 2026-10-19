"""Summarize matched A/B/B/A runs; never discard a slow case or first-use cost."""

import argparse
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    runs = {}
    for name in ("a1", "b1", "b2", "a2"):
        result = json.loads((a.root / name / "result.json").read_text())
        if not result["passed"] or result["worker_exit"] != 0:
            raise ValueError("all runs must complete successfully")
        runs[name] = [
            json.loads(s)
            for s in (a.root / name / "runs.jsonl").read_text().splitlines()
        ]
        for row in runs[name]:
            if not row["reply"]["passed"] or row["reply"]["max_abs_diff"] != 0:
                raise ValueError("full-logit parity required")
    summary = {}
    for variant, names in [("baseline", ("a1", "a2")), ("reuse", ("b1", "b2"))]:
        rows = [r for n in names for r in runs[n]]
        steps = [s for r in rows for s in r["reply"]["step_timings"]]
        summary[variant] = dict(
            requests=len(rows),
            tokens=sum(r["output_length"] for r in rows),
            decode_seconds=sum(r["reply"]["seconds"] for r in rows),
            total_seconds=sum(r["stage_timings"]["total_seconds"] for r in rows),
            step_median_ms={k: 1000 * median(s[k] for s in steps) for k in steps[0]},
        )
    cases = []
    keys = {(tuple(r["prompt"]), r["output_length"]) for r in runs["a1"]}
    for prompt, count in sorted(keys):
        entry = dict(input_length=len(prompt), output_length=count)
        reference_tokens = None
        for label, names in [("baseline", ("a1", "a2")), ("reuse", ("b1", "b2"))]:
            rows = [
                r
                for n in names
                for r in runs[n]
                if tuple(r["prompt"]) == prompt and r["output_length"] == count
            ]
            if len(rows) != 4:
                raise ValueError("unbalanced matched cases")
            for row in rows:
                if reference_tokens is None:
                    reference_tokens = row["tokens"]
                if row["tokens"] != reference_tokens:
                    raise ValueError("cross-run token mismatch")
            entry[label] = dict(
                decode_ms=1000 * median(r["reply"]["seconds"] for r in rows),
                total_ms=1000
                * median(r["stage_timings"]["total_seconds"] for r in rows),
            )
        cases.append(entry)
    summary["cases"] = cases
    summary["decode_reduction_pct"] = 100 * (
        1 - summary["reuse"]["decode_seconds"] / summary["baseline"]["decode_seconds"]
    )
    summary["total_reduction_pct"] = 100 * (
        1 - summary["reuse"]["total_seconds"] / summary["baseline"]["total_seconds"]
    )
    summary["scope"] = (
        "finite B1 validation, full logits retained; no vendor-native baseline or saturation claim"
    )
    (a.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
