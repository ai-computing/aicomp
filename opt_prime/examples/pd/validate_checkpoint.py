"""Local pretrained-checkpoint parity. Never downloads weights or executes remote code."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from opt_prime.accel.pd import PDRequest, PDRuntime
from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine
from opt_prime.accel.pd.torch_cache import TorchCacheTransport


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.float32).eval()
    print('MODEL_LOADED',args.model_id,flush=True)
    options=dict(model_id=args.model_id,revision=args.model_dir.name,max_context=128,capture_logits=True)
    source=TorchCausalLMEngine(model,endpoint='source',**options)
    decoder=TorchCausalLMEngine(copy.deepcopy(model),endpoint='decode',**options)
    transport=TorchCacheTransport(source='source',destination='decode',spec=source.spec)
    runtime=PDRuntime(source,decoder,transport)
    rows=[]
    for text in ('한국의 수도는 어디인가요? 간단히 답해주세요.',
                 'Explain why the sky appears blue in one sentence.'):
        prompt=tokenizer.encode(text,add_special_tokens=True)
        full=list(prompt);expected=[]
        with torch.inference_mode():
            for _ in range(8):
                row=model(torch.tensor([full]),use_cache=False).logits[0,-1].clone()
                expected.append(row);full.append(int(row.argmax()))
        oracle=torch.stack(expected)
        for prefix in (False,True):
            source.prefix_only=prefix
            actual=runtime.generate(PDRequest(f'case-{len(rows)}',tuple(prompt),8))
            logits=torch.stack(source.last_logits+decoder.last_logits)
            maximum=float((logits-oracle).abs().max())
            record=dict(prompt=text,input_length=len(prompt),prefix_only=prefix,
                        output_length=8,tokens_exact=actual==full[len(prompt):],
                        max_abs_diff=maximum,tokens=actual,text=tokenizer.decode(actual))
            rows.append(record)
            (args.out/'progress.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
            if actual!=full[len(prompt):]:raise RuntimeError('pretrained token parity failed')
            torch.testing.assert_close(logits,oracle,atol=2e-4,rtol=2e-4)
            print('PASS',record['input_length'],prefix,maximum,flush=True)
    hashes={}
    for path in sorted(args.model_dir.glob('*.safetensors')):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        hashes[path.name]=h.hexdigest()
    (args.out/'result.json').write_text(json.dumps(dict(passed=True,model_id=args.model_id,
        revision=args.model_dir.name,device='cpu',dtype='float32',threads=8,
        atol=2e-4,rtol=2e-4,weights_sha256=hashes,cases=rows),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
