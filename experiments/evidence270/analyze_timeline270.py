"""Summarize host observations without inferring device utilization or DMA overlap."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def distribution(values):
    return dict(count=len(values),sum=sum(values),min=min(values) if values else None,
                median=statistics.median(values) if values else None,max=max(values) if values else None)


def summarize(trace):
    remote,decode=trace['remote'],trace['decode']
    for row in remote+decode:
        if row['end']<row['start']:
            raise ValueError('negative host interval')
    completed=sorted((r for r in remote if not r['error']),key=lambda r:r['end'])
    ends=[r['end'] for r in completed]
    backlogged_gaps=[b['end']-a['end'] for a,b in zip(completed,completed[1:]) if b['start']<=a['end']]
    lane_hist=Counter(str(r['lanes']) for r in decode if not r['error'])
    return dict(remote=dict(completed=len(remote),errors=sum(r['error'] for r in remote),
                submit_to_completion_s=distribution([r['end']-r['start'] for r in remote]),
                completion_gap_s=distribution([b-a for a,b in zip(ends,ends[1:])]),
                backlogged_completion_gap_s=distribution(backlogged_gaps)),
                decode=dict(calls=len(decode),errors=sum(r['error'] for r in decode),
                lane_histogram=dict(sorted(lane_hist.items())),max_lanes=max((int(k) for k in lane_hist),default=0),
                calls_with_ready_prefill=sum(r['ready']>0 for r in decode),
                calls_with_waiting=sum(r['waiting']>0 for r in decode),
                calls_with_prefilling=sum(r['prefilling']>0 for r in decode),
                host_call_s=distribution([r['end']-r['start'] for r in decode])),
                limits='All harness phases included. Completion gaps include client/phase idle periods; submit-to-completion includes producer queueing. Queue states are snapshots at decode entry, not continuous ready-time measurements. Python intervals do not prove DMA overlap; overlapping sums are not wall-clock decomposition.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('server_log',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.server_log.read_bytes()
    rows=[json.loads(l.split(' ',1)[1]) for l in raw.decode().splitlines() if l.startswith('ATOM270_HOST_TIMELINE ')]
    if len(rows)!=1:raise ValueError('Expected one completed host timeline')
    out=summarize(rows[0]);out['source_sha256']=hashlib.sha256(raw).hexdigest()
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
