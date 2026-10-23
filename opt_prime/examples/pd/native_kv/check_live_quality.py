"""Small deterministic real-prompt smoke check; not a general quality benchmark.

Native respects EOS. GPU computes a fixed budget then truncates at the checkpoint's
first EOS for scoring. GPU timing therefore must not be compared with native here.
"""
import argparse,json,time
from pathlib import Path
from gpu_decoder import load_model,GraphDecoder
from gpu_norms import install_fused_norms
from memory_receiver import NativeKVClient

CASES=[
 ('multiply','What is 17 times 19? Reply with only the integer.','323'),
 ('factorial','What is 6 factorial? Reply with only the integer.','720'),
 ('binary','Convert the binary number 101101 to decimal. Reply with only the integer.','45'),
 ('sort','Sort [9, 1, 7, 3, 2] in ascending order. Reply with only a JSON array.',[1,2,3,7,9]),
 ('deduplicate','Remove duplicates from [3, 1, 3, 2, 1] and sort ascending. Reply with only a JSON array.',[1,2,3]),
 ('extract','From {"name":"Mina","age":27,"city":"Busan"}, return the age. Reply with only the integer.','27'),
 ('logic','All A are B. No B are C. Can any A be C? Reply only yes or no.','no'),
 ('reverse','Reverse the string abcde. Reply with only the reversed string.','edcba'),
 ('korean','대한민국의 수도 이름만 한국어로 답하세요.','서울'),
 ('capital','What is the capital of France? Reply with only the city name.','Paris'),
 ('units','Convert 3.5 kilometers to meters. Reply with only the integer.','3500'),
 ('subtract','A box has 48 pencils. 17 are removed and then 9 are added. How many remain? Reply with only the integer.','40'),
]
for count in (24,64,128):
    target=count//2+7
    records='\n'.join(f'item_{i:03d} = {1000+i}' for i in range(count))
    CASES.append((f'context-{count}',f'Read these records:\n{records}\nWhat is the value of item_{target:03d}? Reply with only the integer.',str(1000+target)))


def grade(text,expected):
    if isinstance(expected,list):
        try:
            actual=json.loads(text.strip())
            return (isinstance(actual,list) and len(actual)==len(expected) and
                    all(type(x) is type(y) and x==y for x,y in zip(actual,expected)))
        except (ValueError,TypeError):return False
    return text.strip().casefold()==expected.casefold()


def main(a):
    import torch,requests
    from transformers import AutoTokenizer
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    eos=json.loads((Path(a.model_dir)/'generation_config.json').read_text())['eos_token_id']
    eos={eos} if isinstance(eos,int) else set(eos)
    model=load_model(a.model_dir,a.device,'marlin',fuse_mlp=a.fuse_mlp,fuse_qkv=a.fuse_qkv)
    graphs={'unfused':GraphDecoder(model,2048,'flash')}
    install_fused_norms(model,'torch-stats');graphs['torch-stats']=GraphDecoder(model,2048,'flash')
    install_fused_norms(model,'fused');graphs['fused']=GraphDecoder(model,2048,'flash')
    rows=[];session=requests.Session()
    client=NativeKVClient(a.http,19201,19202,json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
    try:
        for name,prompt,expected in CASES:
            if a.long_only and not name.startswith('context-'):continue
            if a.short_only and name.startswith('context-'):continue
            tokens=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,
                                                add_generation_prompt=True,enable_thinking=False)
            response=session.post(a.http+'/v1/completions',json={'model':a.model,'prompt':tokens,'temperature':0,
                'max_tokens':a.tokens,'min_tokens':0,'ignore_eos':False,'stream':False},timeout=120)
            response.raise_for_status();body=response.json();native=body['choices'][0]['text']
            row={'case':name,'prompt':prompt,'input_tokens':len(tokens),'expected':expected,'fuse_mlp':a.fuse_mlp,'fuse_qkv':a.fuse_qkv,
                 'native':{'text':native,'passed':grade(native,expected),'usage':body.get('usage'),
                           'finish_reason':body['choices'][0].get('finish_reason')}}
            views,receive=client.prefill(a.model,tokens,'views')
            row['receive_ms']=receive['complete_ms']
            for label,graph in graphs.items():
                graph.load_prefix(views.torch_layers(),tokens[-1],packed_import=True)
                generated,metrics=graph.generate(a.tokens)
                end=next((i for i,t in enumerate(generated) if t in eos),len(generated))
                text=tokenizer.decode(generated[:end],skip_special_tokens=True)
                row[label]={'text':text,'passed':grade(text,expected),'ended_with_eos':end<len(generated),
                            'tokens_before_eos':end,'generated_until_eos':generated[:end],
                            'matches_native_text':text==native,'fixed_budget_decode_ms':metrics['decode_ms']}
            rows.append(row);Path(a.out).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('prompt',)},ensure_ascii=False),flush=True)
    finally:
        client.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--http',default='http://127.0.0.1:19101');p.add_argument('--model',default='Qwen/Qwen3-8B-FP8')
    p.add_argument('--tokens',type=int,default=128);p.add_argument('--out',required=True)
    p.add_argument('--fuse-mlp',action='store_true')
    p.add_argument('--fuse-qkv',action='store_true')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--long-only',action='store_true')
    selection.add_argument('--short-only',action='store_true')
    main(p.parse_args())
