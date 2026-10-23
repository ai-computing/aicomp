"""Long-context exact retrieval through native/solo/concurrent HTTP paths."""
import argparse,concurrent.futures,json
from pathlib import Path
import requests
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True);cases=[]
for count in (40,80,120):
 for target in (0,count//2,count-1):
  records=[f'Record item{i:03d}: code Z{(i*7919+count*31)%100000:05d}.' for i in range(count)]
  expected=f'Z{(target*7919+count*31)%100000:05d}'
  text='Read these records.\n'+'\n'.join(records)+f'\nWhat is the code for item{target:03d}? Reply with only the code.'
  tokens=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
  assert len(tokens)+64<=2048,len(tokens)
  cases.append(dict(records=count,target=target,expected=expected,body=dict(model='Qwen/Qwen3-8B-FP8',prompt=tokens,max_tokens=64,temperature=0)))
def run(port,case):
 r=requests.post(f'http://127.0.0.1:{port}/v1/completions',json=case['body'],timeout=120);r.raise_for_status();data=r.json();text=data['choices'][0]['text']
 return dict(text=text,exact=text.strip()==case['expected'],finish_reason=data['choices'][0]['finish_reason'])
rows=[]
for case in cases:
 row={k:v for k,v in case.items() if k!='body'};row['input_tokens']=len(case['body']['prompt'])
 row['native']=run(19101,case);row['solo']=run(19111,case);rows.append(row)
with concurrent.futures.ThreadPoolExecutor(8) as pool:concurrent=list(pool.map(lambda c:run(19111,c),cases))
for row,c in zip(rows,concurrent):
 row['concurrent']=c;row['texts_equal']=row['native']['text']==row['solo']['text']==c['text']
Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'cases':len(rows),'lengths':sorted(set(x['input_tokens'] for x in rows)),'equal':sum(x['texts_equal'] for x in rows),'exact':{mode:sum(x[mode]['exact'] for x in rows) for mode in ('native','solo','concurrent')}}),flush=True)
