"""Frozen sequential quality checks on native HTTP; reuse the unchanged strict scorer."""
import argparse,hashlib,json,time,urllib.request
from pathlib import Path
from heldout_quality270 import score

def run(fixtures,endpoint,model,output,eos_token_ids):
 if not eos_token_ids or any(type(x) is not int or x<0 for x in eos_token_ids):
  raise ValueError("explicit nonempty EOS token IDs required")
 data=json.loads(Path(fixtures).read_text());cases=data['cases']
 if len(cases)!=36 or len({c['case'] for c in cases})!=36:
  raise ValueError('expected36 unique frozen cases')
 for c in cases:
  if hashlib.sha256(json.dumps(c['ids'],separators=(',',':')).encode()).hexdigest()!=c['input_sha256']:
   raise ValueError('frozen input hash mismatch')
 with Path(output).open('x') as out:
  for c in cases:
   body=dict(model=model,prompt=c['ids'],temperature=0,max_tokens=data['max_tokens'],ignore_eos=False,stream=False,return_token_ids=True,stop_token_ids=sorted(set(eos_token_ids)))
   req=urllib.request.Request(endpoint.rstrip('/')+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':'native201-quality-'+c['case']})
   row=dict(case=c['case'],route='native-rngd',input_sha256=c['input_sha256'],input_tokens=len(c['ids']),expected=c['expected'])
   start=time.monotonic()
   try:
    with urllib.request.urlopen(req,timeout=600) as response:payload=json.load(response)
    row['response']=payload
    choice=payload['choices'][0]
    assert choice['prompt_token_ids']==c['ids'], 'server prompt IDs differ'
    assert isinstance(choice['token_ids'],list) and len(choice['token_ids'])<=data['max_tokens']
    assert choice['finish_reason'] in ['stop','length']
    row.update(text=choice['text'],tokens=choice['token_ids'],finish_reason=choice['finish_reason'],stop_reason=choice.get('stop_reason'),**score(choice['text'],c['expected']))
   except Exception as e:
    row.update(error=repr(e),answer_correct=False,format_correct=False)
    raise
   finally:
    row['elapsed_seconds']=time.monotonic()-start
    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['fixtures','endpoint','model','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--eos-token-ids',type=int,nargs='+',required=True)
 a=p.parse_args();run(a.fixtures,a.endpoint,a.model,a.output,a.eos_token_ids)
