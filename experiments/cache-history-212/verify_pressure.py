import argparse
import json
from pathlib import Path

def audit_run(d,mode):
 assert d['passed'] and d['remaining_requests']==0 and d['capacity']['device_pages']==36 and d['capacity']['page_size']==256
 assert d['sequence']==['A','B','A','A','B','A'] and len(d['rows'])==len(d['allocations'])==6
 assert all(len(p)==8192 for p in d['prompts'].values()) and d['prompts']['A'][:256]!=d['prompts']['B'][:256]
 matches=[]
 for i,r in enumerate(d['rows']):
  assert r['index']==i and r['prompt_key']==d['sequence'][i] and r['input_tokens']==8192 and len(r['tokens'])==32
  eq=r['tokens']==d['references'][r['prompt_key']];assert eq==r['tokens_equal'];matches.append(eq)
  assert len(r['allocations'])==1 and r['allocations'][0]==d['allocations'][i]
  a=r['allocations'][0];assert a['slot']==0 and a['reserved_tokens']==8224 and a['prompt_tokens']==8192 and a['reused_tokens'] is not None
  assert r['after']['live_refs']==0 and r['after']['query_tokens']-r['before']['query_tokens']==8192
  assert r['after']['hit_tokens']-r['before']['hit_tokens']==a['reused_tokens']
 assert all(matches)==d['parity_passed']
 if mode=='off':assert d['final']['hit_tokens']==0 and d['final']['device_removed']==0
 else:
  assert d['rows'][1]['after']['device_removed']>d['rows'][1]['before']['device_removed']
  assert d['rows'][3]['allocations'][0]['reused_tokens']==7936
  if mode=='device':assert d['rows'][2]['allocations'][0]['reused_tokens']==0 and d['final']['host'] is None
  else:
   r=d['rows'][2];before,after=r['before']['host'],r['after']['host'];reload=after['reloaded']-before['reloaded'];assert reload>0 and after['hit_tokens']-before['hit_tokens']==reload*256
   assert r['allocations'][0]['reused_tokens']==7936 and d['capacity']['host_pages']>=64
 return matches

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--mode",choices=["off","device","host"],required=True)
    a=p.parse_args()
    matches=audit_run(json.loads(Path(a.input).read_text()),a.mode)
    print(json.dumps({"matches":sum(matches),"checks":len(matches)}))
