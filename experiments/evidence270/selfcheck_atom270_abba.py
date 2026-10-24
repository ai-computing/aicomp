"""Synthetic fixtures test audit logic without archived hardware artifacts."""
import copy,json
from pathlib import Path
from audit_atom270_abba import audit
def fixture(width):
    d=dict(passed=True,parity_passed=True,slots_per_decoder=4,physical_atom=1,physical_gpus=0,physical_rngd=1,
           context=9216,kv_blocks=5,tensor_parallel_size=1,decoder_cache=False,producer_cache=False,
           producer_chunk=width,model='synthetic',dtype='bfloat16',torch='fixture',transformers='fixture',
           harness_sha256='fixture',server_config={'servers':[['server.py','--width',str(width)]],
           'metadata':{'producer_chunk':width}},prompts={},references={},warmups=[],correctness_batches=[],batches=[])
    for n in (128,512):
        d['prompts'][str(n)]=[3]*n
        for o in (32,128):
            tokens=[7]*o;ref=dict(tokens=tokens,text='fixture');d['references'][f'{n}:{o}']=ref
            d['warmups'].append(dict(id=f'w-0-{n}-{o}',tokens_equal=True,tokens=tokens))
            d['correctness_batches'].append(dict(input_tokens=n,output_tokens=o,concurrency=16,
                rows=[dict(id=str(i),tokens_equal=True,tokens=tokens) for i in range(16)]))
            rows=[dict(id=str(i),input_tokens=n,output_tokens=o,text_equal=True,text='fixture',
                       first_sse_s=.25,e2e_s=.5,tpot_est_s=.25/(o-1)) for i in range(16)]
            summary=dict(requests=16,output_tokens=16*o,all_text_equal=True,wall_s=1.,output_tok_s=16.*o,
                         median_first_sse_s=.25,median_e2e_s=.5,median_tpot_est_s=.25/(o-1))
            d['batches'].append(dict(input_tokens=n,output_tokens_per_request=o,concurrency=16,repeat=0,rows=rows,summary=summary))
    return d
wide=fixture(8192);narrow=fixture(1024)
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
print('PASS synthetic ABBA fixture and eight corruption rejections; not hardware results')
