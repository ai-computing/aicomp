"""Validate all-core decode with RNGD whole and streamed KV, retaining failed evidence."""
import json,subprocess
from pathlib import Path
from audit_http265 import audit_result
A=Path(__file__).resolve().parent
queue=A/'atom270-allcore-pd-arms.json';assert not queue.exists();state=[]
for mode in ['whole','stream']:
 stem=f'atom270-qwen4b-allcore-pd-{mode}'
 assert not (A/(stem+'.json')).exists()
 argv=['bash',str(A/'run-http270-atom.sh'),'--model','Qwen/Qwen3-4B','--mode','pd','--output',stem+'.json','--server-config',f'atom270-qwen4b-allcore-{mode}-config.json','--lengths','128','512','2048','8192','--outputs','32','128','--concurrency','16','--repeats','1','--slots','4','--startup-timeout','1800','--request-timeout','600']
 if mode=='stream':argv+=['--reference','atom270-qwen4b-allcore-pd-whole.json']
 print('PD_START',mode,flush=True);child=subprocess.run(argv,cwd=A)
 record={'mode':mode,'exit_code':child.returncode,'argv':argv};state.append(record);queue.write_text(json.dumps(state,indent=2)+'\n')
 assert child.returncode==0,'Failed raw data retained; dependent mode not run'
 d=json.loads((A/(stem+'.json')).read_text());record['audit']=audit_result(d,'atom',mode,expected_slots=4,expected_atom_devices=1)
 queue.write_text(json.dumps(state,indent=2)+'\n');print('PD_END',mode,json.dumps(record),flush=True)
