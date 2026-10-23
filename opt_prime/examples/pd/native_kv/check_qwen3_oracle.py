import json,torch
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
mid='Qwen/Qwen3-0.6B';tok=AutoTokenizer.from_pretrained(mid)
m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float32).eval()
rows=json.load(open('qwen3_no_thinking.json'))[mid]['atom_standalone']['outputs'];out=[]
with torch.inference_mode():
 for r in rows:
  ids=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False,enable_thinking=False)
  x=torch.tensor([ids]);y=m.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=32,do_sample=False)[0,len(ids):].tolist()
  d=dict(prompt=r['prompt'],tokens=y,text=tok.decode(y),atom_tokens=r['tokens'],first_token_equal=y[0]==r['tokens'][0],all_equal=y==r['tokens']);out.append(d);print(json.dumps(d,ensure_ascii=False),flush=True)
Path('qwen3_cpu_oracle.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
