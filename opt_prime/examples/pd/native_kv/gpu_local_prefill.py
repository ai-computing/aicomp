"""GPU-local BF16 prefill of FP8-checkpoint weights, with packed ragged FA2 attention."""
import types
import torch
from gpu_decoder import load_model
from gpu_norms import install_fused_norms
from gpu_activation import install_fused_post_attention


class LocalPrefiller:
    def __init__(self,model_dir,device='cuda:0'):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        self.model=load_model(model_dir,device,weight_mode='bf16')
        install_fused_norms(self.model,'fused');install_fused_post_attention(self.model)
        self.device=torch.device(device)
        owner=self
        for layer in self.model.model.layers:
            module=layer.self_attn;original=module.forward
            def forward(attn,hidden_states,position_embeddings,attention_mask,past_key_values=None,
                        cache_position=None,_original=original,**kwargs):
                if past_key_values is not owner:
                    return _original(hidden_states,position_embeddings,attention_mask,past_key_values,cache_position,**kwargs)
                total=hidden_states.shape[1]
                q=attn.q_norm(attn.q_proj(hidden_states).view(1,total,32,128)).transpose(1,2)
                k=attn.k_norm(attn.k_proj(hidden_states).view(1,total,8,128)).transpose(1,2)
                v=attn.v_proj(hidden_states).view(total,8,128)
                q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
                q=q.transpose(1,2).reshape(total,32,128).contiguous()
                k=k.transpose(1,2).reshape(total,8,128).contiguous()
                owner.layers[attn.layer_idx]=(k,v)
                if owner.attention_reference:
                    # Diagnostic oracle: independent causal attention per request,
                    # retaining exactly the same packed projections and RoPE.
                    parts=[]
                    for begin,end in zip(owner.offsets,owner.offsets[1:]):
                        qq=q[begin:end].transpose(0,1)[None]
                        kk=k[begin:end].transpose(0,1)[None].repeat_interleave(4,dim=1)
                        vv=v[begin:end].transpose(0,1)[None].repeat_interleave(4,dim=1)
                        part=torch.nn.functional.scaled_dot_product_attention(
                            qq,kk,vv,is_causal=True,scale=attn.scaling)
                        parts.append(part[0].transpose(0,1))
                    out=torch.cat(parts,dim=0)
                else:
                    out=flash_attn_varlen_func(q,k,v,max_seqlen_q=owner.max_length,cu_seqlens_q=owner.cu,
                        max_seqlen_k=owner.max_length,cu_seqlens_k=owner.cu,softmax_scale=attn.scaling,
                        causal=True,fa_version=2)
                return attn.o_proj(out.reshape(1,total,-1)),None
            module.forward=types.MethodType(forward,module)

    @torch.inference_mode()
    def prefill(self,prompts,*,attention_reference=False):
        if not 1<=len(prompts)<=16 or any(not 1<=len(p)<=2048 for p in prompts):
            raise ValueError('unsupported prefill shape')
        if any(type(t) is not int or not 0<=t<self.model.config.vocab_size for p in prompts for t in p):
            raise ValueError('invalid token')
        lengths=[len(p) for p in prompts];offsets=[0]
        for n in lengths:offsets.append(offsets[-1]+n)
        self.offsets=offsets
        self.attention_reference=attention_reference
        self.cu=torch.tensor(offsets,dtype=torch.int32,device=self.device)
        self.max_length=max(lengths);self.layers=[None]*36
        ids=torch.tensor([sum(prompts,[])],dtype=torch.long,device=self.device)
        positions=torch.tensor([sum([list(range(n)) for n in lengths],[])],dtype=torch.long,device=self.device)
        hidden=self.model.model(ids,past_key_values=self,cache_position=torch.arange(offsets[-1],device=self.device),
            position_ids=positions,attention_mask={'full_attention':None},use_cache=True).last_hidden_state[0]
        last=hidden.index_select(0,self.cu[1:].long()-1)
        logits=self.model.lm_head(last)
        tokens=logits.argmax(-1).tolist()
        layers=self.layers
        self.layers=None
        return tokens,layers,offsets,logits
