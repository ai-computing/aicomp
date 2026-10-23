"""Experimental Qwen3 BF16 / Marlin W8A16 loader and fixed-capacity GPU decode.

Uses installed vLLM's Marlin kernels, not native FP8 tensor-core arithmetic.
No model weights or vendor binaries are distributed with this example.
"""
import json
from pathlib import Path


def load_model(model_dir, device='cuda:0', weight_mode='marlin', fuse_mlp=False, fuse_qkv=False):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    if weight_mode not in ('marlin','bf16'):raise ValueError('unknown weight mode')
    if (fuse_mlp or fuse_qkv) and weight_mode!='marlin':raise ValueError('projection fusion requires Marlin')
    root=Path(model_dir)
    cfg=Qwen3Config.from_pretrained(root,local_files_only=True)
    if (cfg.num_hidden_layers,cfg.num_key_value_heads,cfg.head_dim)!=(36,8,128):
        raise ValueError('only measured Qwen3-8B geometry admitted')
    cfg._attn_implementation='sdpa'
    with torch.device('meta'):
        model=Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    expected=list(model.state_dict())
    if weight_mode=='marlin':
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,apply_fp8_marlin_linear)
        class MarlinLinear(torch.nn.Module):
            def __init__(self,weight,scale):
                super().__init__()
                self.output_size_per_partition,self.input_size_per_partition=weight.shape
                self.weight_block_size=[128,128]
                self.orig_dtype=torch.bfloat16
                self.register_parameter('weight',torch.nn.Parameter(weight,requires_grad=False))
                self.register_parameter('weight_scale_inv',torch.nn.Parameter(scale,requires_grad=False))
                prepare_fp8_layer_for_marlin(self,size_k_first=False)
            def forward(self,x):
                return apply_fp8_marlin_linear(x,self.weight,self.weight_scale_inv,self.workspace,
                    self.output_size_per_partition,self.input_size_per_partition,None)
    for shard in sorted(set(index[name] for name in expected)):
        with safe_open(root/shard,framework='pt',device='cpu') as f:
            for name in expected:
                if index[name]!=shard:continue
                tensor=f.get_tensor(name).to(device)
                scale_name=name.replace('.weight','.weight_scale_inv')
                path,param=name.rsplit('.',1)
                if scale_name in index:
                    if index[scale_name]!=shard:raise ValueError('scale stored in another shard')
                    scale=f.get_tensor(scale_name).to(device)
                    if weight_mode=='marlin':
                        parent,attr=path.rsplit('.',1)
                        setattr(model.get_submodule(parent),attr,MarlinLinear(tensor,scale))
                        continue
                    tensor=tensor.float()*scale.float().repeat_interleave(128,0).repeat_interleave(128,1)[:tensor.shape[0],:tensor.shape[1]]
                model.get_submodule(path).register_parameter(param,torch.nn.Parameter(tensor.to(torch.bfloat16),requires_grad=False))
        print('Loaded',shard,weight_mode,flush=True)
    def combined_projection(names):
        weights=[];scales=[]
        for name in names:
            scale_name=name.replace('.weight','.weight_scale_inv')
            with safe_open(root/index[name],framework='pt',device='cpu') as f:
                weights.append(f.get_tensor(name).to(device))
            with safe_open(root/index[scale_name],framework='pt',device='cpu') as f:
                scales.append(f.get_tensor(scale_name).to(device))
        if any(w.shape[0]%128 for w in weights):raise ValueError('unaligned projection widths')
        return MarlinLinear(torch.cat(weights,dim=0),torch.cat(scales,dim=0))
    if fuse_qkv:
        for i,layer in enumerate(model.model.layers):
            attention=layer.self_attn
            attention.qkv_proj=combined_projection([f'model.layers.{i}.self_attn.{p}_proj.weight' for p in ('q','k','v')])
            attention.qkv_sizes=(cfg.num_attention_heads*cfg.head_dim,cfg.num_key_value_heads*cfg.head_dim,cfg.num_key_value_heads*cfg.head_dim)
        # Original projections remain for ordinary HF prefill; the extra copy costs ~0.9GB.
        print('Fused decode QKV projections',len(model.model.layers),flush=True)
    if fuse_mlp:
        class FusedMLP(torch.nn.Module):
            def __init__(self,old,projection):
                super().__init__()
                self.gate_up_proj=projection
                self.down_proj=old.down_proj
                self.act_fn=old.act_fn
            def forward(self,x):
                gate,up=self.gate_up_proj(x).chunk(2,dim=-1)
                return self.down_proj(self.act_fn(gate)*up)
        for i,layer in enumerate(model.model.layers):
            # Output widths are block-aligned: concatenation preserves every FP8 block/scale.
            combined=combined_projection([f'model.layers.{i}.mlp.{p}_proj.weight' for p in ('gate','up')])
            layer.mlp=FusedMLP(layer.mlp,combined)
        print('Fused gate/up projections',len(model.model.layers),flush=True)
    model.model.rotary_emb=Qwen3RotaryEmbedding(cfg).to(device)
    model.eval()
    if any(p.is_meta for p in model.parameters()):raise ValueError('unloaded model parameter')
    return model


class GraphDecoder:
    """One owned lane. Copy all prefix KV before replay; valid positions mask stale cache."""
    def __init__(self,model,capacity=2048,attention='sdpa'):
        import torch
        from transformers import StaticCache
        if not 4<=capacity<=2048:raise ValueError('unsupported capacity')
        self.model,self.capacity=model,capacity
        self.device=next(model.parameters()).device
        cfg=model.config
        self.cache=StaticCache(config=cfg,max_cache_len=capacity)
        self.cache.early_initialization(1,8,128,torch.bfloat16,self.device)
        self.input_id=torch.tensor([[1000]],device=self.device)
        self.position=torch.zeros(1,device=self.device,dtype=torch.long)
        self.all_positions=torch.arange(capacity,device=self.device)
        self.history=torch.zeros((1,capacity),device=self.device,dtype=torch.long)
        self.last_logits=None
        if attention not in ('sdpa','flash'):raise ValueError('unknown attention mode')
        cu_q=torch.tensor([0,1],device=self.device,dtype=torch.int32)
        cu_k=torch.zeros(2,device=self.device,dtype=torch.int32)
        if attention=='flash':install_flash_decode(model,cu_q,cu_k,capacity,self.cache)
        def step():
            cu_k[1:].copy_(self.position+1)
            mask=(self.all_positions<=self.position).reshape(1,1,1,-1)
            logits=model(self.input_id,past_key_values=self.cache,cache_position=self.position,
                         position_ids=self.position.unsqueeze(0),attention_mask={'full_attention':mask},
                         use_cache=True,logits_to_keep=1).logits
            self.last_logits=logits
            next_id=logits[:,-1].argmax(-1,keepdim=True)
            self.history.index_copy_(1,self.position,next_id)
            self.input_id.copy_(next_id)
            self.position.add_(1)
        with torch.inference_mode():
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):step()
        self.loaded_positions=None

    def begin_prefix(self,n):
        """Prepare an owned upload; no decode is admitted until all slots arrive."""
        import torch
        if not 1<=n<self.capacity:raise ValueError('prefix outside capacity')
        torch.cuda.synchronize(self.device)
        self.loaded_positions=None
        self.streaming_positions=n
        self.streaming_seen=set()
        shape=(72,n,8,128)
        if not hasattr(self,'host_staging') or self.host_staging.shape!=shape:
            self.host_staging=torch.empty(shape,dtype=torch.bfloat16,pin_memory=True)
            self.device_staging=torch.empty(shape,dtype=torch.bfloat16,device=self.device)

    def stage_prefix_slots(self,slots,values):
        """Copy validated CPU [slots,heads,positions,dim] while later batches arrive."""
        import numpy as np
        import torch
        n=getattr(self,'streaming_positions',None)
        indices=[int(i) for i in slots]
        if (n is None or not indices or len(set(indices))!=len(indices) or
            any(i<0 or i>=72 or i in self.streaming_seen for i in indices) or
            values.shape!=(len(indices),8,n,128) or values.dtype!=np.uint16):
            raise ValueError('invalid streaming KV batch')
        source=torch.from_numpy(values).view(torch.bfloat16).transpose(1,2)
        with torch.inference_mode():
            if indices==list(range(indices[0],indices[0]+len(indices))):
                section=slice(indices[0],indices[-1]+1)
                self.host_staging[section].copy_(source)
                self.device_staging[section].copy_(self.host_staging[section],non_blocking=True)
            else:
                for j,i in enumerate(indices):
                    self.host_staging[i].copy_(source[j])
                    self.device_staging[i].copy_(self.host_staging[i],non_blocking=True)
            for i in indices:
                layer=self.cache.layers[i//2]
                dest=layer.keys if i%2==0 else layer.values
                dest[:,:,:n].copy_(self.device_staging[i].transpose(0,1).unsqueeze(0))
        self.streaming_seen.update(indices)

    def finish_prefix(self,last_token):
        import torch
        if getattr(self,'streaming_positions',None) is None or len(self.streaming_seen)!=72:
            raise ValueError('incomplete streaming prefix')
        if not 0<=last_token<self.model.config.vocab_size:raise ValueError('invalid final prompt token')
        with torch.inference_mode():
            self.input_id.fill_(last_token)
            self.position.fill_(self.streaming_positions)
        torch.cuda.synchronize(self.device)
        self.loaded_positions=self.streaming_positions
        self.streaming_positions=None

    def load_prefix(self,layers,last_token,packed_import=False):
        import torch
        if len(layers)!=36:raise ValueError('expected 36 layer pairs')
        n=layers[0][0].shape[2]
        if not 1<=n<self.capacity:raise ValueError('prefix outside capacity')
        for k,v in layers:
            if k.shape!=(1,8,n,128) or v.shape!=k.shape or k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16:
                raise ValueError('invalid canonical KV')
        if not 0<=last_token<self.model.config.vocab_size:raise ValueError('invalid final prompt token')
        with torch.inference_mode():
            if packed_import:
                if any(t.device.type!='cpu' for pair in layers for t in pair):
                    raise ValueError('packed import requires CPU KV')
                # One pinned, contiguous upload avoids 72 pageable strided CUDA copies.
                shape=(72,n,8,128)
                if not hasattr(self,'host_staging') or self.host_staging.shape!=shape:
                    self.host_staging=torch.empty(shape,dtype=torch.bfloat16,pin_memory=True)
                    self.device_staging=torch.empty(shape,dtype=torch.bfloat16,device=self.device)
                for i,pair in enumerate(layers):
                    for j,t in enumerate(pair):
                        self.host_staging[2*i+j].copy_(t[0].transpose(0,1))
                self.device_staging.copy_(self.host_staging,non_blocking=True)
                layers=[(self.device_staging[2*i].transpose(0,1).unsqueeze(0),
                         self.device_staging[2*i+1].transpose(0,1).unsqueeze(0)) for i in range(36)]
            for i,(k,v) in enumerate(layers):
                self.cache.layers[i].keys[:,:,:n].copy_(k)
                self.cache.layers[i].values[:,:,:n].copy_(v)
            self.input_id.fill_(last_token)
            self.position.fill_(n)
        self.loaded_positions=n
        # Completion fence makes the import lifetime explicit for callers.
        torch.cuda.synchronize(self.device)

    def iter_tokens(self,count,eos_ids=(),ignore_eos=False,cancel=None):
        """Deliver each token to the CPU before advancing; consumes the owned prefix."""
        n=self.loaded_positions
        if n is None or not 1<=count<=self.capacity-n:raise ValueError('load prefix or reduce output length')
        self.loaded_positions=None
        for _ in range(count):
            if cancel is not None and cancel.is_set():return
            self.graph.replay()
            token=int(self.input_id.item())
            yield token
            if not ignore_eos and token in eos_ids:return

    def generate(self,count):
        import torch,time
        n=self.loaded_positions
        if n is None or not 1<=count<=self.capacity-n:raise ValueError('load prefix or reduce output length')
        self.loaded_positions=None
        torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        self.graph.replay();torch.cuda.synchronize(self.device)
        first=time.perf_counter()
        for _ in range(count-1):self.graph.replay()
        torch.cuda.synchronize(self.device)
        end=time.perf_counter()
        return self.history[0,n:n+count].tolist(),{'first_decode_ms':(first-start)*1000,
            'tpot_ms':(end-first)*1000/(count-1) if count>1 else None,
            'decode_ms':(end-start)*1000,'decode_tok_s':count/(end-start)}


def install_flash_decode(model, cu_q, cu_k, capacity, owned_cache):
    """Use vLLM's FA2 GQA kernel for B1/Q1 while keeping ordinary prefill intact.

    cu_k is GPU state updated once per captured step. No CPU sequence-length read.
    """
    import types
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
    for layer in model.model.layers:
        module=layer.self_attn
        original=module.forward
        def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,
                    cache_position=None,_original=original,**kwargs):
            if hidden_states.shape[:2]!=(1,1) or past_key_values is not owned_cache:
                return _original(hidden_states,position_embeddings,attention_mask,past_key_values,
                                 cache_position,**kwargs)
            if hasattr(self,'qkv_proj'):
                q,k,v=self.qkv_proj(hidden_states).split(self.qkv_sizes,dim=-1)
            else:
                q,k,v=self.q_proj(hidden_states),self.k_proj(hidden_states),self.v_proj(hidden_states)
            q=self.q_norm(q.view(1,1,-1,self.head_dim)).transpose(1,2)
            k=self.k_norm(k.view(1,1,-1,self.head_dim)).transpose(1,2)
            v=v.view(1,1,-1,self.head_dim).transpose(1,2)
            cos,sin=position_embeddings
            q,k=apply_rotary_pos_emb(q,k,cos,sin)
            k,v=past_key_values.update(k,v,self.layer_idx,{'cache_position':cache_position})
            out=flash_attn_varlen_func(q[0].transpose(0,1),k[0].transpose(0,1),v[0].transpose(0,1),
                max_seqlen_q=1,cu_seqlens_q=cu_q,max_seqlen_k=capacity,cu_seqlens_k=cu_k,
                softmax_scale=self.scaling,causal=False,fa_version=2)
            return self.o_proj(out.reshape(1,1,-1)),None
        module.forward=types.MethodType(forward,module)
