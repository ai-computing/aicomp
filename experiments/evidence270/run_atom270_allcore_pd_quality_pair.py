"""Run both frozen quality routes once; retain every answer including wrong ones."""
import json,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent
for mode in ['whole','stream']:
 print('QUALITY_START',mode,flush=True)
 subprocess.run(['bash',str(A/'run-quality270-pd-atom.sh'),'--mode',mode],check=True)
 d=json.loads((A/f'atom270-qwen4b-allcore-pd-quality-{mode}-run.json').read_text())
 assert d['state']=='complete' and d['cases']==36 and d['server_exit']==0
 print('QUALITY_END',mode,d['answer_correct'],d['format_correct'],flush=True)
