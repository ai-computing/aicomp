import copy,json,pathlib,subprocess,sys,tempfile
b=pathlib.Path(sys.argv[1]);audit=str(pathlib.Path(__file__).with_name('audit_full_grid.py'))
q=json.loads((b/'cachefull212r1-gpu-queue.json').read_text());q['runs']=[r for r in q['runs'] if r['model']=='qwen06b'];assert len(q['runs'])==2 and all(r['exit_code']==0 for r in q['runs']);q['state']='running'
name='cachefull212r1-qwen06b-off.json';d=json.loads((b/name).read_text())
def run(label,change,ok=False):
 with tempfile.TemporaryDirectory() as tmp:
  p=pathlib.Path(tmp);qq=copy.deepcopy(q);dd=copy.deepcopy(d);change(qq,dd)
  (p/'cachefull212r1-gpu-queue.json').write_text(json.dumps(qq));(p/name).write_text(json.dumps(dd))
  cfg='cachefull212r1-qwen06b-off-config.json';(p/cfg).write_bytes((b/cfg).read_bytes())
  for extra in ['cachefull212r1-qwen06b-on.json','cachefull212r1-qwen06b-on-config.json']:(p/extra).write_bytes((b/extra).read_bytes())
  # Serialize the unmodified control identically so the recorded reference hash remains valid.
  if dd==d:(p/name).write_bytes((b/name).read_bytes())
  r=subprocess.run([sys.executable,audit,str(p)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(label,r.stdout,r.stderr)
  if ok:
   summary=json.loads((p/'cachefull212r1-summary.json').read_text())['arms']['qwen06b/off'];x=next(x for x in summary['measurements'] if x['concurrency']==16);assert x['output_tokens']==16*x['output_tokens_per_request']
  print('PASS',label)
run('real raw and per-request vs aggregate count',lambda q,d:None,True)
run('duplicate arm',lambda q,d:q['runs'].append(copy.deepcopy(q['runs'][0])))
run('incomplete queue cannot claim complete',lambda q,d:q.update(state='complete'))
run('wrong warmup under green flag',lambda q,d:d['warmups'][0]['tokens'].__setitem__(0,-1))
run('wrong token under green flag',lambda q,d:d['correctness_batches'][0]['rows'][0]['tokens'].__setitem__(0,-1))
run('wrong SSE under green flag',lambda q,d:d['batches'][0]['rows'][0].update(text='corrupt'))
run('missing condition',lambda q,d:d['batches'].pop())
run('incorrect throughput',lambda q,d:d['batches'][0]['summary'].update(output_tok_s=1e15))
run('wrong resources',lambda q,d:d.update(physical_atom=1))
run('wrong config fingerprint',lambda q,d:q['runs'][0].update(config_sha256='incorrect'))

run('wrong cache-on reference fingerprint',lambda q,d:q['runs'][1].update(reference_sha256='incorrect'))
def duplicate_warmup(q,d):
 q['runs']=q['runs'][:1]
 d['warmups'][-1]=copy.deepcopy(d['warmups'][0])
run('duplicate warmup cannot conceal missing reference',duplicate_warmup)
