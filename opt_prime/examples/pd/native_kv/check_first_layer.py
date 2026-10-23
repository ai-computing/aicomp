"""Diagnostic only: compare physical slots to dequantized first-layer Qwen3 K/V.

This is not a matching native FP8 execution oracle or a full decode parity test.
"""
import argparse
import json
from pathlib import Path


def main(args):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

    torch.set_num_threads(args.threads)
    root = Path(args.model_dir)
    cfg = Qwen3Config.from_pretrained(root, local_files_only=True)
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        with safe_open(root/index[name], framework='pt', device='cpu') as f:
            value = f.get_tensor(name).float()
        scale_name = name.replace('.weight', '.weight_scale_inv')
        if scale_name in index:
            with safe_open(root/index[scale_name], framework='pt', device='cpu') as f:
                scale = f.get_tensor(scale_name).float()
            value *= scale.repeat_interleave(128,0).repeat_interleave(128,1)[:value.shape[0],:value.shape[1]]
        return value
    def norm(x, w):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+cfg.rms_norm_eps)).to(x.dtype)*w.to(x.dtype)
    n = args.positions
    ids = list(range(1000,1000+n))
    name = 'model.embed_tokens.weight'
    with safe_open(root/index[name],framework='pt',device='cpu') as f:
        embeddings = f.get_slice(name)[1000:1000+n].clone()
    prefix='model.layers.0.'
    x=norm(embeddings,weight(prefix+'input_layernorm.weight'))
    k=torch.nn.functional.linear(x,weight(prefix+'self_attn.k_proj.weight').to(x.dtype)).reshape(1,n,8,128)
    k=norm(k,weight(prefix+'self_attn.k_norm.weight')).transpose(1,2)
    v=torch.nn.functional.linear(x,weight(prefix+'self_attn.v_proj.weight').to(x.dtype)).reshape(n,8,128)
    rope=Qwen3RotaryEmbedding(cfg)
    cos,sin=rope(x,torch.arange(n).reshape(1,n))
    _,k=apply_rotary_pos_emb(k,k,cos,sin)
    refs={'key':k[0].transpose(0,1).float(),'value':v.float()}
    slots=[]
    for i in range(72):
        raw=bytearray((Path(args.slots)/f'slot{i:02d}.bf16').read_bytes())
        slots.append(torch.frombuffer(raw,dtype=torch.bfloat16).reshape(n,8,128).float())
    result={'positions':n,'reference':'FP8 weights dequantized to BF16; no native dynamic activation quantization','matches':{}}
    for name,ref in refs.items():
        scores=[]
        for i,value in enumerate(slots):
            diff=value-ref
            scores.append({'slot':i,'cosine':torch.nn.functional.cosine_similarity(value.flatten(),ref.flatten(),dim=0).item(),
                           'rmse':diff.square().mean().sqrt().item(),'max_abs':diff.abs().max().item(),
                           'reference_rms':ref.square().mean().sqrt().item()})
        result['matches'][name]=sorted(scores,key=lambda x:x['rmse'])[:4]
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--slots',required=True)
    p.add_argument('--positions',required=True,type=int);p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int,default=4)
    main(p.parse_args())
