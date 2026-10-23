"""Recalculate compact/full prefix results without hiding first-use costs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    summary = {}
    expected = {}
    for label, names in [("full", ("a1", "a2")), ("compact", ("b1", "b2"))]:
        rows = []
        for name in names:
            result = json.loads((a.root / name / "result.json").read_text())
            if not result["passed"] or result["worker_exit"] != 0:
                raise ValueError("incomplete run")
            run = [
                json.loads(l)
                for l in (a.root / name / "runs.jsonl").read_text().splitlines()
            ]
            if len(run) != 16:
                raise ValueError("expected16requests per run")
            for r in run:
                if not r["reply"]["passed"] or r["reply"]["max_abs_diff"] != 0:
                    raise ValueError("parity failed")
                key = (tuple(r["prompt"]), r["output_length"])
                expected.setdefault(key, r["tokens"])
                if expected[key] != r["tokens"]:
                    raise ValueError("cross-run tokens differ")
            rows.extend(run)
        summary[label] = dict(
            requests=len(rows), tokens=sum(r["output_length"] for r in rows)
        )
        for subset, select in [
            ("", rows),
            ("repeat1_", [r for r in rows if r["repeat"] == 1]),
        ]:
            for stage in ("total_seconds", "prefill_seconds"):
                summary[label][subset + stage] = sum(
                    r["stage_timings"][stage] for r in select
                )
    for prefix in ("", "repeat1_"):
        key = prefix + "total_seconds"
        summary[prefix + "change_pct"] = 100 * (
            summary["compact"][key] / summary["full"][key] - 1
        )
    summary["scope"] = (
        "positive change is regression; repeat1 is separate warmed diagnostic, all requests retained"
    )
    (a.root / "compact-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
