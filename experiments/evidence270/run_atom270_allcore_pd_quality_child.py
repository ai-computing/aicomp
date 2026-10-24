"""Frozen concurrent quality gate for the all-core RNGD→ATOM routes."""
import hashlib,json,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
A=Path(__file__).resolve().parent
import argparse
p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['whole','stream']);a=p.parse_args()
stem='atom270-qwen4b-allcore-pd-quality-'+a.mode
config=json.loads((A/f'atom270-qwen4b-allcore-pd-quality-{a.mode}-config.json').read_text())
fixture=A/'quality270-qwen4b-mapped-fixtures.json'
output=A/(stem+'.jsonl');statepath=A/(stem+'-run.json')
assert not output.exists() and not statepath.exists()
f=json.loads(fixture.read_text());assert len(f['cases'])==36
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(f['model'],local_files_only=True)
for c in f['cases']:
 ids=tok.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
 assert list(ids)==c['ids']
 assert hashlib.sha256(json.dumps(c['ids'],separators=(',',':')).encode()).hexdigest()==c['input_sha256']
with socket.socket() as sock:
 sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
argv=[str(port) if x=='{port}' else x for x in config['servers'][0]]
endpoint=f'http://127.0.0.1:{port}'
state={'state':'running','argv':argv,'fixtures_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'sequential':False,'concurrency':12}
statepath.write_text(json.dumps(state,indent=2)+'\n')
with (A/(stem+'.server0.log')).open('x') as log:
 proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  deadline=time.monotonic()+600
  while True:
   if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}')
   try:
    with urllib.request.urlopen(endpoint+'/ready',timeout=2) as response:
     if response.status==200:break
   except OSError:pass
   if time.monotonic()>deadline:raise TimeoutError('readiness timeout')
   time.sleep(2)
  subprocess.run([sys.executable,str(A/'heldout_quality270_concurrent.py'),'--fixtures',str(fixture),'--endpoint',endpoint,'--output',str(output)],check=True)
  from heldout_quality270 import score
  rows=[json.loads(l) for l in output.read_text().splitlines()]
  assert len(rows)==36 and len({r['case'] for r in rows})==36
  by_case={r['case']:r for r in rows}
  for case in f['cases']:
   row=by_case[case['case']]
   assert row['case']==case['case'] and row['input_sha256']==case['input_sha256'] and 'error' not in row
   assert all(row[k]==v for k,v in score(row['text'],case['expected']).items())
  state.update(state='complete',cases=36,answer_correct=sum(r['answer_correct'] for r in rows),format_correct=sum(r['format_correct'] for r in rows))
 except BaseException as e:
  state.update(state='failed',error=repr(e));raise
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state['server_exit']=proc.returncode
  statepath.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
