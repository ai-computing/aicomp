"""Check unforced128-token generation after changing only output projection precision."""
import gc
import json
from pathlib import Path
from concurrent.futures import Future
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillOut
root=Path(__file__).resolve().parent
output=root/'head-fp32-free-generation.json'
if output.exists(): raise FileExistsError(output)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
rows=[]
for label, model_path, cases in [
    ('koni','/tmp/serving-matrix-models/KONI-8B',[(128,256,'koni261-short-kv.pt'),(8192,8192,'rngd-koni-8b-8192-parts221-layout.pt')]),
    ('qwen4b','/tmp/serving-matrix-models/Qwen3-4B',[(8192,8192,'rngd-qwen3-4b-8192.pt')])]:
    data=json.loads((root/f'local201-{label}-matrix-a.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    head=model.get_output_embeddings()
    original_forward=head.forward
    weight_fp32=head.weight.detach().float()
    bias_fp32=None if head.bias is None else head.bias.detach().float()
    calls=[0]
    def project(hidden):
        calls[0]+=1
        return torch.nn.functional.linear(hidden.float(),weight_fp32,bias_fp32)
    for n,chunk,name in cases:
        export=torch.load(root/name,weights_only=True,mmap=True)
        assert export['ids']==data['prompts'][str(n)]
        live=next(b['rows'][0]['tokens'] for b in data['correctness_batches'] if b['input_tokens']==n and b['output_tokens']==128)
        if label=='qwen4b':
            expected=json.loads((root/'local201-qwen4b-native-full.json').read_text())['references']['8192:128']['tokens']
        elif n==8192:
            expected=next(r['tokens'] for r in json.loads((root/'local201-koni-diagnostic.json').read_text()) if r['mode']=='native8192')
        else: expected=data['references']['128:128']['tokens']
        class Replay:
            def submit(self,request,send_from=0,slot=None):
                assert list(request.input_ids)==export['ids'] and send_from==0
                f=Future()
                token=export['token'] if 'token' in export else int(export['logits'].argmax())
                f.set_result(PrefillOut(token,0,export['keys'],export['values']))
                return f
        result=dict(model=label,input_tokens=n,native_chunk=chunk,rows=[])
        for precision in ['bf16','fp32_head']:
            head.forward=original_forward if precision=='bf16' else project
            for mode in ['native','retained_rngd_kv']:
                kw=dict(slots=4,max_context=9216,kv='paged',page_size=64,prefix_cache=False)
                b=ContinuousBatcher(model,prefill_chunk=chunk,**kw) if mode=='native' else PDBatcher(model,Replay(),**kw)
                f=b.submit(PDRequest(mode,tuple(export['ids']),128));b.run_until_idle();tokens=f.result()
                if precision=='bf16': assert tokens==(expected if mode=='native' else live), 'baseline reproduction failed'
                else: assert calls[0]>0 and next(model.parameters()).dtype==torch.bfloat16
                result['rows'].append(dict(precision=precision,mode=mode,tokens=tokens))
                del b,f
                gc.collect();torch.cuda.empty_cache()
        a,b=[r['tokens'] for r in result['rows'] if r['precision']=='fp32_head']
        result.update(head_fp32_equal=a==b,first_difference=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None))
        rows.append(result)
        output.write_text(json.dumps(dict(complete=False,rows=rows),indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    head.forward=original_forward
    del model,head,original_forward,weight_fp32,bias_fp32
    gc.collect();torch.cuda.empty_cache()
output.write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')
