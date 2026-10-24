"""Audit completed same-Pod ATOM width ABBA; occupancy is reported independently."""
import argparse,copy,hashlib,json,statistics
from pathlib import Path
from audit_http265 import audit_result


def record(log,prefix):
    found=[json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    assert len(found)==1, f'Expected one {prefix}'
    return found[0]


def audit(queue,runs,logs):
    assert queue['state']=='complete' and queue['cleanup_exit_code']==0,'Incomplete or failed queue'
    rows=queue['runs'];assert len(rows)==len(runs)==len(logs)==4,'Incomplete ABBA'
    assert [q['width'] for q in rows]==[8192,1024,1024,8192]
    assert len({q['stem'] for q in rows})==4 and all(Path(q['stem']).name==q['stem'] for q in rows)
    configs=[];bindings=[];histograms=[]
    for q,d,log in zip(rows,runs,logs):
        assert q['exit_code']==0,'Failed child'
        audit_result(d,'atom','whole',lengths=(128,512),expected_slots=4,expected_atom_devices=1)
        assert d['producer_chunk']==q['width'] and d['kv_blocks']==5 and d['tensor_parallel_size']==1
        assert d['prompts']==runs[0]['prompts'] and d['references']==runs[0]['references']
        assert all(d[k]==runs[0][k] for k in ['model','dtype','torch','transformers','harness_sha256'])
        cfg=copy.deepcopy(d['server_config']);assert len(cfg['servers'])==1
        argv=cfg['servers'][0];pos=argv.index('--width')+1
        assert argv[pos]==str(q['width']) and cfg['metadata']['producer_chunk']==q['width']
        argv[pos]='WIDTH';cfg['metadata']['producer_chunk']='WIDTH';configs.append(cfg)
        assert record(log,'ATOM270_COMPILED ')==dict(batch_size=4,kv_blocks=5,context=9216,tensor_parallel_size=1)
        binding=record(log,'ATOM270_BINDING ');assert binding==q['binding']
        assert binding['visible_count']==1 and len(binding['physical_to_visible'])==1
        assert binding['opened']==['/dev/rbln'+next(iter(binding['physical_to_visible']))]
        bindings.append(binding)
        hist=record(log,'ATOM270_ACTIVE_LANES ')
        assert len(hist)==5 and all(type(n) is int and n>=0 for n in hist) and sum(hist)>0
        histograms.append(hist)
    assert all(c==configs[0] for c in configs),'Other config changed'
    assert all(b==bindings[0] for b in bindings),'Physical binding changed'
    comparisons=[]
    for n in (128,512):
        for o in (32,128):
            samples={}
            for label,indices in [('wide',(0,3)),('narrow',(1,2))]:
                samples[label]=[next(b['summary'] for b in runs[i]['batches'] if (b['input_tokens'],b['output_tokens_per_request'])==(n,o)) for i in indices]
            ratio=statistics.median(s['output_tok_s'] for s in samples['narrow'])/statistics.median(s['output_tok_s'] for s in samples['wide'])
            comparisons.append(dict(input_tokens=n,output_tokens=o,samples=samples,throughput_ratio=ratio))
    return dict(stage='passed',conditions=16,warmups=16,correctness_responses=256,timed_text_responses=256,binding=bindings[0],active_lane_histograms=histograms,full_batch_observed=[h[4]>0 for h in histograms],rows=comparisons,limits='Two timed batches per width per condition, whole route, short inputs only. No confidence interval, heldout quality acceptance, other-model generalization or device overlap proof. Occupancy failures remain separate from ABBA correctness.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--queue',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    q=json.loads(a.queue.read_text())
    if q['state']=='failed' or any(r['exit_code']!=0 for r in q['runs']):
        print('failed: queue/child failure');raise SystemExit(1)
    if q['state']!='complete':
        print(f"pending: {len(q['runs'])}/4 recorded rows");raise SystemExit(2)
    paths=[a.queue];data=[];logs=[]
    for row in q['runs']:
        assert Path(row['stem']).name==row['stem']
        rp=a.queue.parent/(row['stem']+'.json');lp=a.queue.parent/(row['stem']+'.server0.log')
        data.append(json.loads(rp.read_text()));logs.append(lp.read_text());paths.extend([rp,lp])
    out=audit(q,data,logs);out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
