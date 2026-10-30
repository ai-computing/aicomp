import argparse,pathlib,tempfile,json,subprocess,sys
parser=argparse.ArgumentParser(description="Negative controls for native GPU short/long HTTP audit")
parser.add_argument("artifacts",type=pathlib.Path)
args=parser.parse_args()
b=args.artifacts.resolve();auditor=pathlib.Path(__file__).with_name("audit_gpu_edge201.py");q=json.loads((b/'edge201-gpu-queue.json').read_text());q['runs']=[r for r in q['runs'] if r.get('exit_code')==0];q['state']='running';assert q['runs'];failures=[]
with tempfile.TemporaryDirectory(prefix='edge201-check-') as td:
 t=pathlib.Path(td)
 for r in q['runs']:
  m=r['model']
  for name in [f'edge201-{m}-gpu-native.json',f'stream265-{m}-http-c16-native.json',f'default201-{m}-gpu-config.json']:(t/name).symlink_to(b/name)
 state=t/'edge201-gpu-queue.json';state.write_text(json.dumps(q))
 def check(label,success):
  result=subprocess.run([sys.executable,str(auditor),str(t)],capture_output=True,text=True)
  if (result.returncode==0)!=success:failures.append(label);print('FAIL',label,result.stderr[-500:])
  else:print('PASS',label)
 check('valid completed subset',True)
 summary=json.loads((t/'edge201-gpu-summary.json').read_text());assert all(x['output_tokens']==x['concurrency']*x['output_tokens_per_request'] for d in summary['models'].values() for x in d['measurements'])
 duplicate=dict(q);duplicate['state']='complete';duplicate['runs']=[q['runs'][0]]*4;state.write_text(json.dumps(duplicate));check('reject duplicate complete arms',False);state.write_text(json.dumps(q))
 m=q['runs'][0]['model'];p=t/f'edge201-{m}-gpu-native.json';original=p.read_text();p.unlink()
 for label,mutate in [('warmup token corruption',lambda d:d['warmups'][0]['tokens'].__setitem__(0,-1)),('token corruption',lambda d:d['correctness_batches'][0]['rows'][0]['tokens'].__setitem__(0,-1)),('SSE corruption',lambda d:d['batches'][0]['rows'][0].__setitem__('text','CORRUPTED')),('throughput corruption',lambda d:d['batches'][0]['summary'].__setitem__('output_tok_s',-1)),('missing condition',lambda d:d['batches'].pop())]:
  d=json.loads(original);mutate(d);p.write_text(json.dumps(d));check('reject '+label,False)
 p.unlink();check('reject missing raw',False)
assert not failures,failures
