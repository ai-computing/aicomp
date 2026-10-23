"""Summarize instrumented GPU phases and check first-token timing accounting."""
import argparse,json,statistics
from pathlib import Path


def stats(values):
    xs=sorted(values)
    return {'p50':statistics.median(xs),'p95':xs[int((len(xs)-1)*.95)],'max':xs[-1]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--arrivals',nargs='+',required=True)
    p.add_argument('--stages',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    stages=[json.loads(s) for s in Path(a.stages).read_text().splitlines()]
    waves=[json.loads(s) for s in Path(a.stages+'.waves.jsonl').read_text().splitlines()]
    out=[];offset=0
    for path in a.arrivals:
        arrivals=json.loads(Path(path).read_text())
        for r in arrivals:
            if r['mode']!='gpu':continue
            chunk=stages[offset:offset+1+r['successes']];offset+=1+r['successes']
            assert len(chunk)==1+r['successes']
            rows=chunk[1:];ids={x['native_wave_id'] for x in rows}
            ws=[w for w in waves if w['native_wave_id'] in ids]
            item={'source':Path(path).name,'rate':r['rate'],'requests':r['requests'],
                  'successes':r['successes'],'errors':r['errors'],'ttft_p95_ms':r['ttft_p95_ms'],
                  'completed_tok_s':r['completed_tok_s'],'metrics':{},'waves':{}}
            for key in ('admission_wait_ms','gather_validate_ms','native_complete_ms','prepared_wait_ms',
                        'gpu_import_ms','decode_start_ms','first_gpu_token_ms','first_text_ready_ms',
                        'admitted_at_native_start','pending_at_native_start','decode_active_at_import'):
                values=[x[key] for x in rows if key in x]
                if values:item['metrics'][key]=stats(values)
            residual=[]
            for x in rows:
                accounted=sum(x[k] for k in ('admission_wait_ms','gather_validate_ms','native_complete_ms',
                    'cpu_pack_queue_ms','cpu_pack_ms','prepared_wait_ms','gpu_import_ms'))
                accounted+=x['first_gpu_token_ms']-x['decode_start_ms']
                residual.append(x['first_gpu_token_ms']-accounted)
            item['accounting_residual_ms']=stats(residual)
            item['accounting_absolute_max_ms']=max(abs(x) for x in residual)
            for key in ('requests','wave_ms','credit_wait_ms','pending_before','pending_after','admitted_before'):
                item['waves'][key]=stats([x[key] for x in ws])
            item['waves']['count']=len(ws)
            item['waves']['all16_admitted']=sum(x['admitted_before']==16 for x in ws)
            out.append(item)
    assert offset==len(stages),(offset,len(stages))
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n')
    for r in out:print(json.dumps(r))


if __name__=='__main__':main()
