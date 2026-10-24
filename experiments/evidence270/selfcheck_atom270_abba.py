"""Derived fixtures test audit logic only; never reported as hardware results."""
import copy,json
from pathlib import Path
from audit_atom270_abba import audit
A=Path(__file__).resolve().parent
wide=json.loads((A/'atom270-timeline-gemma-b4-whole.json').read_text())
narrow=json.loads((A/'atom270-shortwidth-gemma-b4-whole.json').read_text())
wide['prompts']={k:v for k,v in wide['prompts'].items() if k in ('128','512')}
wide['references']={k:v for k,v in wide['references'].items() if k.split(':')[0] in ('128','512')}
wide['warmups']=[r for r in wide['warmups'] if int(r['id'].split('-')[2]) in (128,512)]
for key in ['correctness_batches','batches']:wide[key]=[r for r in wide[key] if r['input_tokens'] in (128,512)]
binding=dict(visible_count=1,physical_to_visible={'2':0},opened=['/dev/rbln2'])
log='\n'.join(prefix+json.dumps(value) for prefix,value in [('ATOM270_COMPILED ',dict(batch_size=4,kv_blocks=5,context=9216,tensor_parallel_size=1)),('ATOM270_BINDING ',binding),('ATOM270_ACTIVE_LANES ',[0,10,5,1,0])])
queue=dict(state='complete',cleanup_exit_code=0,runs=[dict(stem=f'r{i}',width=w,exit_code=0,binding=binding) for i,w in enumerate([8192,1024,1024,8192])])
data=[wide,narrow,copy.deepcopy(narrow),copy.deepcopy(wide)];logs=[log]*4
result=audit(queue,data,logs);assert result['correctness_responses']==256 and len(result['rows'])==4
for mutation in ['incomplete','child','order','duplicate','binding','config','tokens','metrics']:
 q=copy.deepcopy(queue);d=copy.deepcopy(data);ll=list(logs)
 if mutation=='incomplete':q['state']='running'
 elif mutation=='child':q['runs'][1]['exit_code']=1
 elif mutation=='order':q['runs'][0]['width']=1024
 elif mutation=='duplicate':q['runs'][1]['stem']='r0'
 elif mutation=='binding':ll[1]=ll[1].replace('/dev/rbln2','/dev/rbln3')
 elif mutation=='config':d[1]['server_config']['servers'][0].append('--unrelated')
 elif mutation=='tokens':d[1]['references']['128:32']['tokens'][0]+=1
 elif mutation=='metrics':d[1]['batches'][0]['summary']['output_tok_s']*=2
 try:audit(q,d,ll)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Accepted corrupt '+mutation)
print('PASS derived ABBA fixture and eight corruption rejections; not hardware results')
