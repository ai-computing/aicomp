"""Post-hoc diagnostic only: preserve frozen primary scores, separate wrapper/type failures."""
import json,re,sys,itertools
from pathlib import Path

def content(row):
 text=row['text'].strip();fenced=False
 match=re.fullmatch(r'```(?:json)?[ \t]*\n(.*?)\n```',text,flags=re.S)
 if match:text=match[1];fenced=True
 def unique(pairs):
  if len(dict(pairs))!=len(pairs):raise ValueError("duplicate keys")
  return dict(pairs)
 try:obj=json.loads(text,object_pairs_hook=unique)
 except (ValueError,TypeError):return None,fenced,False
 if not isinstance(obj,dict) or set(obj)!={'answer'}:return None,fenced,False
 answer=obj['answer'];numeric=type(answer) is int and row['case'].startswith('arithmetic-')
 if numeric:answer=str(answer)
 if not isinstance(answer,str):answer=None
 return answer,fenced,numeric

def check():
 def row(text,case='arithmetic-0-0'):return dict(text=text,case=case)
 assert content(row('```json\n{"answer": 96}\n```'))==('96',True,True)
 assert content(row('{"answer": true}'))[0] is None
 assert content(row('{"answer": 96.0}'))[0] is None
 assert content(row('{"answer": 96}','passkey-0-0'))[0] is None
 assert content(row('prefix ```json\n{"answer":"96"}\n```'))[0] is None
 assert content(row('{"answer":"096"}'))[0]=='096'
 assert content(row('{"answer":null}'))[0] is None
 assert content(row('{"answer":"wrong","answer":"96"}'))[0] is None
 assert content(row('{"answer":"96","extra":1}'))[0] is None

check()
if __name__=='__main__':
 root=Path(sys.argv[1]);out={'scope':'Post-hoc exploratory diagnostic. Frozen primary scores unchanged; this is not a replacement acceptance criterion. Only an outer JSON/unlabelled fence and integer-to-string arithmetic answers are relaxed.','models':{}}
 routes=('native-gpu','rngd-gpu','native-atom','rngd-atom')
 for model in ('qwen06b','qwen4b','koni','gemma'):
  rows={route:[json.loads(x) for x in (root/f'quality263-{model}-{route}.jsonl').read_text().splitlines()] for route in routes}
  assert all(len(v)==36 and len({r['case'] for r in v})==36 for v in rows.values())
  stats={};parsed={}
  for route,records in rows.items():
   parsed[route]={r['case']:content(r)[0] for r in records}
   stats[route]=dict(cases=36,primary_correct=sum(r['answer_correct'] for r in records),content_correct=sum(content(r)[0]==r['expected'] for r in records),outer_fence=sum(content(r)[1] for r in records),arithmetic_integer=sum(content(r)[2] for r in records),content_errors=[dict(case=r['case'],expected=r['expected'],answer=content(r)[0],text=r['text']) for r in records if content(r)[0]!=r['expected']])
  pairs={}
  for a,b in itertools.combinations(routes,2):
   aa={r['case']:r for r in rows[a]};bb={r['case']:r for r in rows[b]}
   assert aa.keys()==bb.keys()
   assert all(aa[k]['input_sha256']==bb[k]['input_sha256'] and aa[k]['expected']==bb[k]['expected'] for k in aa)
   pairs[a+' vs '+b]=[dict(case=k,left=parsed[a][k],right=parsed[b][k],expected=aa[k]['expected']) for k in aa if parsed[a][k]!=parsed[b][k]]
  out['models'][model]=dict(routes=stats,content_differences=pairs)
 print(json.dumps(out,ensure_ascii=False,indent=2))
