"""Summarize complete placement experiments without publishing generated text."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.input_dir)
    states=json.loads((root/'placement_runstate.json').read_text())
    producer = 'native_exit' in states[0]
    assert len(states)==3 and all(x['state']=='complete' and
        (x['gpu_exit']==0 and x['native_exit'] in (0,-15) if producer else x['exit_code']==0) for x in states)
    def memory(snapshot):
        item={k:v for k,v in snapshot.items() if k!='numastat'}
        totals=next(line for line in snapshot['numastat'].splitlines() if line.startswith('Total')).split()[1:]
        item['memory_mb_by_node']=list(map(float,totals[:-1]));item['memory_mb_total']=float(totals[-1])
        return item
    result=[]
    for state in states:
        label=state['label']
        arrivals=json.loads((root/f'{label}_arrival.json').read_text())
        assert len(arrivals)==4 and all(x['successes']==96 and x['errors']==0 for x in arrivals)
        if producer:
            row={k:v for k,v in state.items() if k not in ('native_pid','gpu_pid','gpu_snapshot','native_snapshots')}
            row['gpu_snapshot']=memory(state['gpu_snapshot'])
            row['native_snapshots']=[memory(x) for x in state['native_snapshots']]
        else:
            row=memory({k:v for k,v in state.items() if k!='pid'})
        row['arrivals']=[{k:v for k,v in x.items() if k!='results'} for x in arrivals]
        stages=[json.loads(s) for s in (root/f'{label}_stages.jsonl').read_text().splitlines()]
        receives=[json.loads(s) for s in (root/f'{label}_receive.jsonl').read_text().splitlines()]
        row['stages']=[]
        for n in (128,512,1024):
            xs=[x for x in stages if x['input_tokens']==n]
            rs=[x for x in receives if x['positions']==n-1]
            item={'input_tokens':n,'samples':len(xs),'receive_samples':len(rs)}
            for key in ('native_ready_ms','native_kv_ms','gpu_import_ms'):
                item[key]=statistics.median(x[key] for x in xs)
            for key,fn in (
                ('receive_wall_ms',lambda x:sum(e[1]-e[0] for e in x['events'])*1000),
                ('receive_cpu_ms',lambda x:sum(e[2] for e in x['events'])*1000),
                ('receive_span_ms',lambda x:(x['events'][-1][1]-x['events'][0][0])*1000)):
                item[key]=statistics.median(fn(x) for x in rs)
            row['stages'].append(item)
        quality=root/f'{label}_quality.json'
        if quality.exists():
            q=json.loads(quality.read_text())
            row['quality']={'cases':len(q),'equal':sum(x['texts_equal'] for x in q),
                'exact':{mode:sum(x[mode]['exact'] for x in q) for mode in ('native','solo','concurrent')}}
        result.append(row)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    for row in result:
        print(row['label'],[(x['mode'],round(x['ttft_p95_ms'],2)) for x in row['arrivals']])


if __name__=='__main__':main()
