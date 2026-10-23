"""Teacher-forced kernel comparison using the exact same BF16 prefix and tokens.

This isolates decode arithmetic from prefill differences and autoregressive drift.
It is a correctness diagnostic, not a throughput benchmark.
"""
import argparse,gc,json,random
from pathlib import Path
from gpu_decoder import load_model,GraphDecoder


def main(a):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rng=random.Random(139)
    cases=[(f'synthetic-{n}',[rng.randrange(1000,30000) for _ in range(n)]) for n in (128,512,1024)]
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    for i,prompt in enumerate(('Explain why the sky looks blue in simple terms.',
                              'Write a Python function that merges two sorted lists.',
                              'GPU와 NPU를 함께 사용해 언어 모델을 실행할 때 고려할 점을 설명해 주세요.')):
        cases.append((f'text-{i}',tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)))
    reference=load_model(a.model_dir,a.device,'bf16')
    evidence=[]
    with torch.inference_mode():
        for label,tokens in cases:
            prefix=reference(torch.tensor([tokens[:-1]],device=a.device,dtype=torch.long),use_cache=True,logits_to_keep=1).past_key_values
            pairs=[(layer.keys.cpu().clone(),layer.values.cpu().clone()) for layer in prefix.layers]
            current=tokens[-1];logits=[];generated=[]
            for _ in range(a.tokens):
                result=reference(torch.tensor([[current]],device=a.device),past_key_values=prefix,use_cache=True,logits_to_keep=1)
                scores=result.logits[0,-1].float().cpu();current=int(scores.argmax())
                logits.append(scores);generated.append(current)
            evidence.append((label,tokens,pairs,torch.stack(logits),generated))
            print('REFERENCE',label,flush=True)
    del reference,prefix,result;gc.collect();torch.cuda.empty_cache()
    rows=[]
    for weight,attention,norm in [('bf16','sdpa',False),('marlin','sdpa',False),('marlin','flash',False),('marlin','flash',True)]:
        model=load_model(a.model_dir,a.device,weight)
        if norm:
            from gpu_norms import install_fused_norms
            install_fused_norms(model)
        graph=GraphDecoder(model,2048,attention)
        with torch.inference_mode():
            for label,tokens,pairs,ref,generated in evidence:
                graph.load_prefix(pairs,tokens[-1],packed_import=True)
                predicted=[];deltas=[];cosines=[];mismatches=[]
                for step in range(a.tokens):
                    # Force the same history in every implementation.
                    if step:graph.input_id.fill_(generated[step-1])
                    graph.graph.replay();torch.cuda.synchronize()
                    scores=graph.last_logits[0,-1].float().cpu()
                    token=int(scores.argmax());predicted.append(token)
                    deltas.append(float((scores-ref[step]).abs().max()))
                    cosines.append(float(torch.nn.functional.cosine_similarity(scores,ref[step],dim=0)))
                    if token!=generated[step]:
                        top=ref[step].topk(2)
                        mismatches.append({'step':step,'reference_token':generated[step],'candidate_token':token,
                            'reference_top2_margin':float(top.values[0]-top.values[1]),
                            'candidate_advantage':float(scores[token]-scores[generated[step]])})
                row={'case':label,'input_tokens':len(tokens),'output_tokens':a.tokens,'weight':weight,'attention':attention,
                     'fused_norms':norm,'matching_top1':sum(x==y for x,y in zip(predicted,generated)),
                     'max_abs_logit_delta':max(deltas),'min_logit_cosine':min(cosines),'mismatches':mismatches}
                rows.append(row);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(row),flush=True)
        del graph,model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--tokens',type=int,default=128);p.add_argument('--out',required=True)
    main(p.parse_args())
