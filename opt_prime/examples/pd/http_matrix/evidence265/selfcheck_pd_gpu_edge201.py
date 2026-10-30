"""Exercise raw-result rejection without changing the measured artifacts."""
import argparse,json,pathlib,subprocess,sys,tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifacts',type=pathlib.Path)
b=parser.parse_args().artifacts.resolve()
auditor=pathlib.Path(__file__).with_name('audit_pd_gpu_edge201.py')
q=json.loads((b/'pdedge201-gpu-queue.json').read_text())
q['runs']=[r for r in q['runs'] if r.get('exit_code')==0]
q['state']='running'
assert q['runs']
with tempfile.TemporaryDirectory(prefix='pdedge201-check-') as directory:
 t=pathlib.Path(directory)
 names=set()
 for r in q['runs']:
  m,mode=r['model'],r['mode']
  names.update([f'pdedge201-{m}-{mode}.json',f'stream265-{m}-http-c16-{mode}.json',f'stream265-{m}-http-c16-native.json',f'edge201-{m}-gpu-native.json'])
 for name in names:(t/name).symlink_to(b/name)
 state=t/'pdedge201-gpu-queue.json';state.write_text(json.dumps(q))
 def check(label,success):
  result=subprocess.run([sys.executable,str(auditor),str(t)],capture_output=True,text=True)
  assert (result.returncode==0)==success,(label,result.stderr)
  print('PASS',label)
 check('valid completed subset',True)
 summary=json.loads((t/'pdedge201-gpu-summary.json').read_text())
 assert all(x['output_tokens']==x['output_tokens_per_request']*x['concurrency'] for v in summary['models'].values() for x in v['measurements'])
 print('PASS per-request and aggregate output lengths remain distinct')
 duplicate=dict(q);duplicate['state']='complete';duplicate['runs']=[q['runs'][0]]*8
 state.write_text(json.dumps(duplicate));check('reject duplicate complete arms',False);state.write_text(json.dumps(q))
 r=q['runs'][0];p=t/f"pdedge201-{r['model']}-{r['mode']}.json";original=p.read_text();p.unlink()
 for label,mutate in [
  ('warmup corruption',lambda d:d['warmups'][0]['tokens'].__setitem__(0,-1)),
  ('token corruption',lambda d:d['correctness_batches'][0]['rows'][0]['tokens'].__setitem__(0,-1)),
  ('SSE corruption',lambda d:d['batches'][0]['rows'][0].__setitem__('text','CORRUPTED')),
  ('throughput corruption',lambda d:d['batches'][0]['summary'].__setitem__('output_tok_s',-1)),
  ('resource corruption',lambda d:d.__setitem__('physical_rngd',0)),
  ('missing condition',lambda d:d['batches'].pop())]:
  d=json.loads(original);mutate(d);p.write_text(json.dumps(d));check('reject '+label,False)
 p.unlink();check('reject missing raw result',False)
