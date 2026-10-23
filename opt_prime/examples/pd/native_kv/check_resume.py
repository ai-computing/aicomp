"""Diagnostic torch resume with native physical slots; Qwen3-8B-FP8 only.

Loads the same FP8 checkpoint but dequantizes weights to BF16 for CPU execution.
This changes arithmetic; a successful run is NOT proof of native FP8 parity.
"""
import argparse
import json
from pathlib import Path
import time


def main(args):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root=Path(args.model_dir)
    cfg=Qwen3Config.from_pretrained(root,local_files_only=True)
    if (cfg.num_hidden_layers,cfg.num_key_value_heads,cfg.head_dim)!=(36,8,128):
        raise ValueError('only measured Qwen3 geometry accepted')
    cfg._attn_implementation='sdpa'
    with torch.device('meta'):
        model=Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    state={}
    expected=set(model.state_dict())
    started=time.monotonic()
    for shard in sorted(set(index[name] for name in expected)):
        with safe_open(root/shard,framework='pt',device='cpu') as f:
            for name in sorted(expected):
                if index[name]!=shard:continue
                w=f.get_tensor(name)
                scale_name=name.replace('.weight','.weight_scale_inv')
                if scale_name in index:
                    # Current checkpoint stores scales beside their weight shard.
                    scale=f.get_tensor(scale_name).float()
                    w=w.float()*scale.repeat_interleave(128,0).repeat_interleave(128,1)[:w.shape[0],:w.shape[1]]
                state[name]=w.to(torch.bfloat16)
        print('Loaded weight shard',shard,flush=True)
    model.load_state_dict(state,assign=True,strict=True)
    del state
    model.model.rotary_emb=Qwen3RotaryEmbedding(cfg)
    model.eval()
    def check_case(slots_dir,n):
        ids=torch.arange(1000,1000+n+1).reshape(1,-1)
        native=DynamicCache(config=cfg)
        slots=[]
        for i in range(72):
            raw=bytearray((Path(slots_dir)/f'slot{i:02d}.bf16').read_bytes())
            slots.append(torch.frombuffer(raw,dtype=torch.bfloat16).reshape(n,8,128).transpose(0,1).unsqueeze(0).contiguous())
        for layer in range(36):native.update(slots[2*layer],slots[2*layer+1],layer)
        with torch.inference_mode():
            reference=model(ids[:,:-1],use_cache=True,logits_to_keep=1)
            metrics=[]
            for layer in range(36):
                for kv,target in [('key',reference.past_key_values.layers[layer].keys),('value',reference.past_key_values.layers[layer].values)]:
                    physical=slots[2*layer+(kv=='value')].float();target=target.float()
                    metrics.append({'layer':layer,'kind':kv,
                        'cosine':torch.nn.functional.cosine_similarity(physical.flatten(),target.flatten(),dim=0).item(),
                        'relative_rmse':((physical-target).square().mean()/target.square().mean()).sqrt().item()})
            expected_logits=model(ids[:,-1:],past_key_values=reference.past_key_values,use_cache=True,logits_to_keep=1).logits[0,-1].float()
            resumed_logits=model(ids[:,-1:],past_key_values=native,use_cache=True,logits_to_keep=1).logits[0,-1].float()
            full_logits=model(ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            native_tokens=[resumed_logits.argmax().item()]
            torch_tokens=[expected_logits.argmax().item()]
            for _ in range(args.decode_steps-1):
                a=model(torch.tensor([[native_tokens[-1]]]),past_key_values=native,use_cache=True,logits_to_keep=1).logits[0,-1]
                b=model(torch.tensor([[torch_tokens[-1]]]),past_key_values=reference.past_key_values,use_cache=True,logits_to_keep=1).logits[0,-1]
                native_tokens.append(a.argmax().item());torch_tokens.append(b.argmax().item())
        def comparison(a,b):
            return {'argmax_a':a.argmax().item(),'argmax_b':b.argmax().item(),'argmax_equal':bool(a.argmax()==b.argmax()),
                    'rmse':(a-b).square().mean().sqrt().item(),'max_abs':(a-b).abs().max().item(),
                    'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item(),
                    'top5_a':torch.topk(a,5).indices.tolist(),'top5_b':torch.topk(b,5).indices.tolist()}
        result={'positions':n,'native_tokens':native_tokens,'torch_tokens':torch_tokens,'generated_tokens_equal':native_tokens==torch_tokens,'reference':'same FP8 weights dequantized to BF16, SDPA CPU; native arithmetic parity not established',
                'layer_metrics':metrics,'native_resume_vs_torch_cache':comparison(resumed_logits,expected_logits),
                'torch_cache_vs_full_prompt':comparison(expected_logits,full_logits),
                'wall_seconds_diagnostic_not_benchmark':time.monotonic()-started}
        return result
    if args.capture_dir:
        cases=json.loads((Path(args.capture_dir)/'results.json').read_text())
        results=[]
        for case in cases:
            result=check_case(Path(args.capture_dir)/case['directory'],case['transferred_positions'])
            results.append(result)
            Path(args.out).write_text(json.dumps(results,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='layer_metrics'},indent=2),flush=True)
    else:
        if args.positions is None or args.slots is None:raise ValueError('supply --capture-dir or both --slots and --positions')
        result=check_case(args.slots,args.positions)
        Path(args.out).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='layer_metrics'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--slots')
    p.add_argument('--positions',type=int);p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--capture-dir');p.add_argument('--decode-steps',type=int,default=4)
    main(p.parse_args())
