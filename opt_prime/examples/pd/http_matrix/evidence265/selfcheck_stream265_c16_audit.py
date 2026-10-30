import json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
audit=root/'audit_stream265_c16.py'
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp); labels=['qwen06b','qwen4b','gemma','koni']
 for label in labels:
  d=json.loads((root/f'stream265-{label}-c4-gpu.json').read_text())
  d.update(concurrency=16,decoder_slots=4,producer_slots=4,input_lengths=[2048,8192]*8)
  for r in d['runs']:
   for key in ['tokens','token_times_s','ttft_s']:r[key]*=4
  for mode in d['summary'].values():mode['median_ttft_s']*=4
  (p/f'stream265-{label}-c16-gpu.json').write_text(json.dumps(d))
 queue=p/'stream265-c16-queue.json';queue.write_text(json.dumps([dict(label=l,exit_code=0) for l in labels]))
 def run():return subprocess.run(['python3',str(audit),str(p)],capture_output=True).returncode
 assert run()==0
 f=p/'stream265-qwen06b-c16-gpu.json';original=f.read_text();d=json.loads(original)
 d['runs'][2]['tokens'][15][0]+=1;f.write_text(json.dumps(d));assert run()!=0;f.write_text(original)
 d=json.loads(original);d['runs'][1]['ttft_s'].pop();f.write_text(json.dumps(d));assert run()!=0;f.write_text(original)
 d=json.loads(original);d['producer_slots']=16;f.write_text(json.dumps(d));assert run()!=0;f.write_text(original)
 queue.write_text(json.dumps([dict(label=l,exit_code=1 if l=='gemma' else 0) for l in labels]));assert run()!=0
 queue.write_text(json.dumps([dict(label=l,exit_code=0) for l in labels]));f.unlink();assert run()==2
print('PASS: complete fixture accepted; token corruption, missing TTFT, changed slots, failed child rejected; missing model exits2')
