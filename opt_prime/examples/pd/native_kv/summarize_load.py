"""Keep load, client lateness and rejected requests visible in finite-run summaries."""
import argparse,json
from collections import Counter
from pathlib import Path


def percentile(xs,p):
    xs=sorted(xs)
    return xs[min(len(xs)-1,int((len(xs)-1)*p))] if xs else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();rows=[]
    for path in a.inputs:
        for raw in json.loads(Path(path).read_text()):
            row={k:v for k,v in raw.items() if k!='results'}
            row['source']=Path(path).name
            results=raw['results'];assert len(results)==raw['requests']
            row['client_start_delay_p95_ms']=percentile([x['start_delay_ms'] for x in results],.95)
            row['client_start_delay_max_ms']=max(x['start_delay_ms'] for x in results)
            row['error_counts']=dict(Counter(str(x.get('status'))+':'+str(x.get('error')) for x in results if not x['ok']))
            row['quarters']=[]
            for i in range(4):
                xs=[x for x in results if i*len(results)//4<=x['index']<(i+1)*len(results)//4]
                good=[x for x in xs if x['ok']]
                row['quarters'].append(dict(requests=len(xs),successes=len(good),errors=len(xs)-len(good),
                    ttft_p50_ms=percentile([x['scheduled_ttft_ms'] for x in good],.5),
                    ttft_p95_ms=percentile([x['scheduled_ttft_ms'] for x in good],.95),
                    total_p95_ms=percentile([x['scheduled_total_ms'] for x in good],.95)))
            rows.append(row)
    Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(r['source'],r['rate'],r['mode'],r['successes'],r['errors'],round(r['completed_tok_s'],1),round(r['ttft_p95_ms'],1) if r['ttft_p95_ms'] else None,r['error_counts'])


if __name__=='__main__':main()
