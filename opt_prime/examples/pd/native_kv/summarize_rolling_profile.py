import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description='Summarize rolling stage and receive timing by prompt length')
p.add_argument('--root',type=Path,required=True);p.add_argument('--prefix',required=True)
a=p.parse_args();prefix=a.prefix;root=a.root
stages=[json.loads(x) for x in (root/(prefix+'_stage_profile.jsonl')).read_text().splitlines()]
receive=[json.loads(x) for x in (root/(prefix+'_receive_profile.jsonl')).read_text().splitlines()]
# The arrival benchmark sends one warmup first; remove it from both logs.
stages=stages[1:]
receive=receive[1:]
out={}
def stats(xs):
 xs=sorted(xs);return {'median':statistics.median(xs),'p95':xs[int((len(xs)-1)*.95)]}
for n in [128,512,1024]:
 rows=[x for x in stages if x['input_tokens']==n]
 rx=[x for x in receive if x['positions']==n-1]
 if not rows:continue
 cols={k:stats([x[k] for x in rows]) for k in ['admission_wait_ms','native_ready_ms','native_kv_ms','native_complete_ms','prepared_wait_ms','gpu_import_ms','first_gpu_token_ms']}
 cols['kv_after_ready_ms']=stats([x['native_kv_ms']-x['native_ready_ms'] for x in rows])
 cols['http_after_kv_ms']=stats([x['native_complete_ms']-x['native_kv_ms'] for x in rows])
 cols['receive_wall_ms']=stats([sum(e[1]-e[0] for e in x['events'])*1000 for x in rx])
 cols['receive_cpu_ms']=stats([sum(e[2] for e in x['events'])*1000 for x in rx])
 cols['receive_span_ms']=stats([(x['events'][-1][1]-x['events'][0][0])*1000 for x in rx])
 cols['between_receive_ms']=stats([(b[0]-a[1])*1000 for x in rx for a,b in zip(x['events'],x['events'][1:])])
 out[n]={'samples':len(rows),'metrics':cols}
(root/(prefix+'_profile_summary.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
