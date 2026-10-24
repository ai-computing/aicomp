"""Re-score frozen raw answers; do not conflate completed execution with quality passing."""
import hashlib,json
from pathlib import Path
from heldout_quality270 import score,compare
A=Path(__file__).resolve().parent

def audit_rows(rows,fixture):
 cases={c['case']:c for c in fixture['cases']};assert len(rows)==len({r['case'] for r in rows})==len(cases)==36
 for r in rows:
  c=cases[r['case']]
  assert 'error' not in r and r['input_sha256']==c['input_sha256'] and r['expected']==c['expected'] and r['input_tokens']==len(c['ids'])
  assert hashlib.sha256(json.dumps(c['ids'],separators=(',',':')).encode()).hexdigest()==c['input_sha256']
  choice=r['response']['choices'][0];assert r['text']==choice['text'] and r['tokens']==choice['token_ids'] and r['finish_reason']==choice['finish_reason']
  assert all(r[k]==v for k,v in score(r['text'],c['expected']).items())
  assert 0<len(r['tokens'])<=128 and r['finish_reason'] in ['stop','length']
  if r['finish_reason']=='stop':assert r['tokens'][-1] in [151643,151645]
  else:assert len(r['tokens'])==128
 for n in [0,2048,8192]:
  cohort=[r for r in rows if r['cohort']==n];assert len(cohort)==12
  assert max(r['request_start_monotonic'] for r in cohort)<min(r['request_end_monotonic'] for r in cohort)
 return {'cases':36,'answer_correct':sum(r['answer_correct'] for r in rows),'format_correct':sum(r['format_correct'] for r in rows),'eos_stopped':sum(r['finish_reason']=='stop' for r in rows),'length_limited':sum(r['finish_reason']=='length' for r in rows)}

def main():
 outer=json.loads((A/'atom270-allcore-pd-quality-outer.json').read_text());assert outer['state']=='complete' and outer['exit_code']==outer['cleanup_exit_code']==0
 fixture=json.loads((A/'quality263-qwen4b-fixtures.json').read_text());mapped=json.loads((A/'quality270-qwen4b-mapped-fixtures.json').read_text());assert fixture['cases']==mapped['cases']
 baseline=[json.loads(l) for l in (A/'quality263-qwen4b-rngd-atom.jsonl').read_text().splitlines()]
 native=[json.loads(l) for l in (A/'atom270-qwen4b-allcore-quality-c12.jsonl').read_text().splitlines()]
 result={'modes':{},'limits':'Frozen36cases per route, short/2K/8K,c12. Not general quality or performance; actual occupancy is separate from submitted concurrency.'};records={};bindings=[]
 for mode in ['whole','stream']:
  stem=f'atom270-qwen4b-allcore-pd-quality-{mode}';p=A/(stem+'.jsonl');rows=[json.loads(l) for l in p.read_text().splitlines()];records[mode]=rows;r=audit_rows(rows,fixture)
  state=json.loads((A/(stem+'-run.json')).read_text());assert state['state']=='complete' and state['server_exit']==0
  assert all(state[k]==r[k] for k in ['cases','answer_correct','format_correct'])
  log=(A/(stem+'.server0.log')).read_text()
  def one(marker):
   values=[json.loads(l[len(marker):]) for l in log.splitlines() if l.startswith(marker)];assert len(values)==1;return values[0]
  assert one('ATOM270_COMPILED ')==dict(batch_size=4,kv_blocks=5,context=9216,tensor_parallel_size=1)
  assert one('ATOM270_QUALITY_EOS ')==[151643,151645]
  bindings.append(one('ATOM270_BINDING '));hist=one('ATOM270_ACTIVE_LANES ');assert len(hist)==5 and sum(hist)>0
  r.update(active_lanes=hist,full_batch_observed=hist[4]>0,historical_pd_comparison=compare(baseline,rows),native_allcore_comparison=compare(native,rows),raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  result['modes'][mode]=r
 assert bindings[0]==bindings[1];result['whole_stream_comparison']=compare(records['whole'],records['stream'])
 result['all_answers_and_formats_pass']=all(r['answer_correct']==r['format_correct']==36 for r in result['modes'].values())
 (A/'atom270-allcore-pd-quality-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
