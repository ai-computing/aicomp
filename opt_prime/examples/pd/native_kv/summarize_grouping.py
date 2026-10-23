"""Reduce completed grouping experiments without publishing generated text."""
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.input_dir)
    states=json.loads((root/'placement_runstate.json').read_text())
    assert len(states)==3 and all(x['state']=='complete' and x['gpu_exit']==0 and x['native_exit'] in (0,-15) for x in states)
    result=[]
    for state in states:
        label=state['label'];row={'label':label,'group':state['group'],'max_batch_slots':state['max_batch_slots']}
        arrivals=json.loads((root/f'{label}_arrival.json').read_text())
        assert len(arrivals)==4 and all(x['successes']==96 and not x['errors'] for x in arrivals)
        row['arrivals']=[{k:v for k,v in x.items() if k!='results'} for x in arrivals]
        stages=[json.loads(s) for s in (root/f'{label}_stages.jsonl').read_text().splitlines()]
        receives=[json.loads(s) for s in (root/f'{label}_receive.jsonl').read_text().splitlines()]
        row['stages']=[]
        for n in (128,512,1024):
            xs=[x for x in stages if x['input_tokens']==n];rs=[x for x in receives if x['positions']==n-1]
            item={'input_tokens':n,'samples':len(xs),'receive_samples':len(rs),'batch_counts':sorted({len(x['events']) for x in rs})}
            for key in ('native_ready_ms','native_kv_ms','gpu_import_ms'):item[key]=statistics.median(x[key] for x in xs)
            for key,fn in (
                ('receive_wall_ms',lambda x:sum(e[1]-e[0] for e in x['events'])*1000),
                ('receive_cpu_ms',lambda x:sum(e[2] for e in x['events'])*1000),
                ('receive_span_ms',lambda x:(x['events'][-1][1]-x['events'][0][0])*1000)):
                item[key]=statistics.median(fn(x) for x in rs)
            row['stages'].append(item)
        qpath=root/f'{label}_quality.json'
        if qpath.exists():
            q=json.loads(qpath.read_text())
            row['quality']={'cases':len(q),'equal':sum(x['texts_equal'] for x in q),'exact':{mode:sum(x[mode]['exact'] for x in q) for mode in ('native','solo','concurrent')}}
        pause=root/f'{label}_pause.json'
        if pause.exists():row['pause']=json.loads(pause.read_text())
        result.append(row)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    for row in result:print(row['label'],[(x['mode'],round(x['ttft_p95_ms'],2)) for x in row['arrivals']])


if __name__=='__main__':main()
