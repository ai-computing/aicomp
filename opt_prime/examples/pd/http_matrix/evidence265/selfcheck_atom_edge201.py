import argparse,pathlib,tempfile,json,subprocess,sys
parser=argparse.ArgumentParser(description="Negative controls for the ATOM short/long raw-result auditor")
parser.add_argument("artifacts",type=pathlib.Path)
args=parser.parse_args()
auditor=pathlib.Path(__file__).with_name("audit_atom_edge201.py");b=args.artifacts.resolve();q=json.loads((b/'edge201-atom1030-queue.json').read_text());q['runs']=[r for r in q['runs'] if r.get('exit_code')==0 and r.get('passed') and r.get('parity_passed')];q['state']='running'
assert len(q['runs'])>=4
with tempfile.TemporaryDirectory(prefix='mid201-audit-test-') as td:
 t=pathlib.Path(td)
 names=['mid201-atom-final-artifact-metadata.json','mid201-atom-final-koni-eos.json']
 for r in q['runs']:
  m,mode=r['model'],r['mode'];names += [f'edge201-atom-final-{m}-{mode}.json',f'stream265-{m}-http-atom-c16-native.json',f'stream265-{m}-http-atom-c16-{mode}.json']
 for name in set(names):
  if (b/name).exists():(t/name).symlink_to(b/name)
 (t/'stream265-qwen4b-http-atom-c16-native.json').unlink(missing_ok=True)
 (t/'edge201-atom1030-queue.json').write_text(json.dumps(q))
 def run(expect_success):
  r=subprocess.run([sys.executable,str(auditor),str(t)],capture_output=True,text=True)
  assert (r.returncode==0)==expect_success,r.stderr
  return r
 run(True)
 summary=json.loads((t/'edge201-atom-final-summary.json').read_text())
 for key,result in summary['models'].items():
  model,mode=key.split('/')
  raw=json.loads((t/f'edge201-atom-final-{model}-{mode}.json').read_text())
  for row,batch in zip(result['measurements'],raw['batches'],strict=True):
   assert row['output_tokens_per_request']==batch['output_tokens_per_request']
   assert row['output_tokens']==row['concurrency']*row['output_tokens_per_request']
 print('PASS per-request output length preserved separately from aggregate output tokens')
 duplicate=dict(q);duplicate['state']='complete';duplicate['runs']=[q['runs'][0]]*12
 (t/'edge201-atom1030-queue.json').write_text(json.dumps(duplicate))
 run(False);print('PASS duplicate arms cannot certify full matrix')
 (t/'edge201-atom1030-queue.json').write_text(json.dumps(q))
 run(True)
 s=json.loads((t/'edge201-atom-final-summary.json').read_text());assert s['models']['qwen4b/native']['historical_c16_native_reference_tokens_equal'] is None
 print('PASS missing historical control stays unavailable while current raw is checked')
 name='edge201-atom-final-qwen4b-native.json';p=t/name;original=(b/name).read_text();p.unlink()
 for label,mutate in [
  ('duplicate warmup hides missing pair',lambda d:d['warmups'].__setitem__(-1,d['warmups'][0])),
  ('wrong warmup hidden by green flag',lambda d:d['warmups'][0]['tokens'].__setitem__(0,-1)),
  ('wrong token hidden by green flag',lambda d:d['correctness_batches'][0]['rows'][0]['tokens'].__setitem__(0,-1)),
  ('wrong SSE hidden by green flag',lambda d:d['batches'][0]['rows'][0].__setitem__('text','CORRUPTED')),
  ('wrong resource metadata',lambda d:d.__setitem__('physical_atom',0)),
  ('missing condition',lambda d:d['batches'].pop()),
  ('incorrect throughput summary',lambda d:d['batches'][0]['summary'].__setitem__('output_tok_s',-1))]:
  d=json.loads(original);mutate(d);p.write_text(json.dumps(d));run(False);print('PASS rejected',label)
 p.unlink();run(False);print('PASS missing current artifact rejected')
