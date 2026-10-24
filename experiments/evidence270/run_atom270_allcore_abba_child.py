"""Same-Pod fresh-process ABBA; parity failures remain failures in preserved raw data."""
import json,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent
queue=A/'atom270-allcore-abba-arms.json'
assert not queue.exists()
state=[]
for arm,config in [('a1','atom270-qwen4b-tp1-b4-native-config.json'),('b1','atom270-qwen4b-allcore-http-config.json'),('b2','atom270-qwen4b-allcore-http-config.json'),('a2','atom270-qwen4b-tp1-b4-native-config.json')]:
 name=f'atom270-qwen4b-allcore-abba-{arm}.json';assert not (A/name).exists()
 argv=['bash',str(A/'run-http270-atom.sh'),'--model','Qwen/Qwen3-4B','--mode','native','--output',name,'--server-config',config,'--lengths','512','8192','--outputs','128','--concurrency','16','--repeats','1','--slots','4','--startup-timeout','600','--request-timeout','600']
 print('ABBA_START',arm,flush=True)
 child=subprocess.run(argv,cwd=A)
 record={'arm':arm,'config':config,'exit_code':child.returncode,'argv':argv}
 if (A/name).exists():
  d=json.loads((A/name).read_text());record.update(runtime_complete=d.get('passed',False),parity_passed=d.get('parity_passed',False),batches=len(d.get('batches',[])))
 state.append(record);queue.write_text(json.dumps(state,indent=2)+'\n')
 # Continue expected parity failures; stop actual runtime failures, preserving the failed arm.
 assert record.get('runtime_complete') and record.get('batches')==2,record
 assert child.returncode==(0 if record['parity_passed'] else 1),record
 print('ABBA_END',arm,json.dumps(record),flush=True)
