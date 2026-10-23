"""Summarize warm paired HTTP runs and host-call overlap, including partial runs."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(data, producer):
    groups = {}
    for batch in data['batches']:
        key = (batch['input_tokens'], batch['output_tokens'], batch['concurrency'])
        groups.setdefault(key, {False: [], True: []})[batch['prefetch']].append(batch)
    rows = []
    for (n, out, c), modes in sorted(groups.items()):
        off = {b['repeat']: b for b in modes[False]}
        on = {b['repeat']: b for b in modes[True]}
        repeats = sorted(off.keys() & on.keys())
        if not repeats:
            continue
        baseline = statistics.median(off[r]['tok_s'] for r in repeats)
        candidate = statistics.median(on[r]['tok_s'] for r in repeats)
        row = dict(input_tokens=n, output_tokens=out, concurrency=c,
            paired_repeats=repeats, off_tok_s=baseline, on_tok_s=candidate,
            change_pct=100*(candidate/baseline-1),
            paired_changes_pct=[100*(on[r]['tok_s']/off[r]['tok_s']-1) for r in repeats])
        for field in ['median_first_token_s', 'median_e2e_s']:
            # Median of batch medians over matched repetitions, not a pooled percentile.
            row['off_'+field] = statistics.median(off[r][field] for r in repeats)
            row['on_'+field] = statistics.median(on[r][field] for r in repeats)
        rows.append(row)
    completed = {r['wire_id']: r for r in producer}
    overlap = {False: [], True: []}
    for submission in data['submissions']:
        if not submission['id'].startswith('r'):
            continue  # Exclude warmup, correctness and fault injection.
        interval = completed.get(submission['wire_id'])
        if interval is None:
            continue
        intersections = []
        for call in data['decode_calls']:
            if not call['ids'] or submission['id'] in call['ids']:
                continue
            start = max(interval['start_s'], call['start_s'])
            end = min(interval['end_s'], call['end_s'])
            if end > start:
                intersections.append((start, end))
        # Union intervals so overlapping instrumentation cannot double-count.
        total, end = 0., float('-inf')
        for start, stop in sorted(intersections):
            total += max(0., stop-max(start, end))
            end = max(end, stop)
        overlap[submission['prefetch']].append(total)
    return dict(model=data['model'], stage=data['stage'], passed=data['passed'],
        measured_batches=len(data['batches']), rows=rows,
        latency_kind='median of matched batch medians; first token is owner emission, not client SSE',
        overlap_kind='producer prefill+export host-call versus other-request decode host-call; not device-kernel proof',
        overlap={str(mode): dict(completed_measured_producers=len(values),
            with_overlap=sum(v > 0 for v in values), total_overlap_s=sum(values))
            for mode, values in overlap.items()}, lifecycle=data['lifecycle'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    producer = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    result = summarize(data, producer)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
