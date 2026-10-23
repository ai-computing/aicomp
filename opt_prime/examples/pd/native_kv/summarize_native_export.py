"""Summarize built-in native profiler spans without publishing raw traces."""
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--arrivals',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    events=[e for e in json.loads(a.trace.read_text())['traceEvents'] if e.get('ph')=='X']
    groups=sorted((e for e in events if e['name']=='DMAConnector::dma_read_payload'),key=lambda e:e['ts'])
    arrivals=json.loads(a.arrivals.read_text())['rows']
    arrivals=sorted(arrivals,key=lambda x:x['index'])
    assert len(groups)==18*(len(arrivals)+1),'expected one excluded warmup and18 groups per request'
    rows=[]
    for i in range(len(arrivals)+1):
        gs=groups[18*i:18*(i+1)]
        assert [int(g['args']['group']) for g in gs]==list(range(18))
        row={'warmup':i==0,'input_tokens':128 if i==0 else arrivals[i-1]['input_tokens'],
             'groups':[],'read_span_ms':(gs[-1]['ts']+gs[-1]['dur']-gs[0]['ts'])/1000}
        for g in gs:
            children=[e for e in events if e['tid']==g['tid'] and g['ts']<=e['ts']<g['ts']+g['dur']]
            scatter=[e for e in children if e['name']=='Flow::compute_kv_cache_scatter_fragment']
            d2h=[e for e in children if e['name']=='Stream::d2h']
            assert len(d2h)==4,'expected four D2H submissions per group'
            enqueues=[e for e in children if e['name']=='Stream::enqueue']
            last_submit=max(e['ts']+e['dur'] for e in enqueues+d2h)
            row['groups'].append({'group':int(g['args']['group']),'duration_ms':g['dur']/1000,
                'scatter_call_ms':sum(e['dur'] for e in scatter)/1000,
                'd2h_call_ms':sum(e['dur'] for e in d2h)/1000,
                'after_last_submission_ms':max(0,g['ts']+g['dur']-last_submit)/1000})
        rows.append(row)
    summary={}
    for n in (128,512,1024):
        # First request of each length can still be a cold compiled bucket.
        rs=[r for r in rows if not r['warmup'] and r['input_tokens']==n]
        summary[n]={'requests':len(rs),'median_read_span_ms':statistics.median(r['read_span_ms'] for r in rs),
            'median_group_duration_ms':statistics.median(g['duration_ms'] for r in rs for g in r['groups']),
            'median_scatter_call_ms':statistics.median(g['scatter_call_ms'] for r in rs for g in r['groups']),
            'median_d2h_call_ms':statistics.median(g['d2h_call_ms'] for r in rs for g in r['groups']),
            'median_after_submission_ms':statistics.median(g['after_last_submission_ms'] for r in rs for g in r['groups'])}
    a.out.write_text(json.dumps({'summary':summary,'requests':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
