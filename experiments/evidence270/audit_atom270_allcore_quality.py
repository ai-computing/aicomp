"""Independent frozen-fixture, EOS, raw-text scoring and baseline comparison."""
import hashlib,json
from pathlib import Path
from heldout_quality270 import score,compare
A=Path(__file__).resolve().parent
p=A/'atom270-qwen4b-allcore-quality-retry1.jsonl'
rows=[json.loads(l) for l in p.read_text().splitlines()]
f=json.loads((A/'quality263-qwen4b-fixtures.json').read_text())
mapped=json.loads((A/'quality270-qwen4b-mapped-fixtures.json').read_text())
assert f['cases']==mapped['cases'] and mapped['model']=='Qwen/Qwen3-4B'
base=[json.loads(l) for l in (A/'quality263-qwen4b-native-atom.jsonl').read_text().splitlines()]
assert len(rows)==len(f['cases'])==36 and len({r['case'] for r in rows})==36
for r,c in zip(rows,f['cases']):
 assert r['case']==c['case'] and r['input_sha256']==c['input_sha256'] and r['expected']==c['expected'] and 'error' not in r
 assert all(r[k]==v for k,v in score(r['text'],c['expected']).items())
 assert r['finish_reason']=='stop' and r['tokens'][-1]==151645
log=(A/'atom270-qwen4b-allcore-quality-retry1.server0.log').read_text()
def value(marker):
 lines=[l.split(marker,1)[1] for l in log.splitlines() if marker in l];assert len(lines)==1
 return json.loads(lines[0])
assert value('ATOM270_QUALITY_EOS ')==[151643,151645]
outer=json.loads((A/'atom270-allcore-quality-retry1-outer.json').read_text())
assert outer['state']=='complete' and outer['exit_code']==outer['cleanup_exit_code']==0
r={'cases':36,'answer_correct':sum(x['answer_correct'] for x in rows),'format_correct':sum(x['format_correct'] for x in rows),'baseline_comparison':compare(base,rows),'eos_stopped':36,'active_lanes':value('ATOM270_ACTIVE_LANES '),'geometry':value('ATOM270_COMPILED '),'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cleanup_exit_code':0,'scope':'Sequential fixed36cases, short/2K/8K; no concurrent B4 quality, general quality or speedup claim.'}
(A/'atom270-allcore-quality-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
