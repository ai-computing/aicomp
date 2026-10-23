"""Optional BF16 SwiGLU fusion with the existing intermediate BF16 rounding."""
import types
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu(X,Y,N:tl.constexpr,TOTAL:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row,col=offsets//N,offsets%N
    gate=tl.load(X+row*(2*N)+col,offsets<TOTAL,other=0).to(tl.float32)
    up=tl.load(X+row*(2*N)+N+col,offsets<TOTAL,other=0).to(tl.float32)
    # Preserve the separate SiLU output cast before multiplication by up.
    silu=tl.div_rn(gate,1.0+libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+offsets,(silu*up).to(tl.bfloat16),offsets<TOTAL)


def fused_swiglu(gate_up):
    if not gate_up.is_cuda or gate_up.dtype!=torch.bfloat16 or not gate_up.is_contiguous():
        raise ValueError('expected contiguous CUDA BF16 gate/up')
    n=gate_up.shape[-1]//2
    if gate_up.shape[-1]!=2*n or n!=12288: raise ValueError('only Qwen3-8B width admitted')
    out=torch.empty((*gate_up.shape[:-1],n),dtype=gate_up.dtype,device=gate_up.device)
    _swiglu[(triton.cdiv(out.numel(),256),)](gate_up,out,n,out.numel(),256,
        num_warps=4,enable_fp_fusion=False)
    return out


def install_fused_swiglu(model):
    count=0
    for layer in model.model.layers:
        module=layer.mlp
        if not hasattr(module,'gate_up_proj'): raise ValueError('SwiGLU fusion requires fused gate/up projection')
        original=module.forward
        def forward(self,x,_original=original):
            if x.is_cuda and x.dtype==torch.bfloat16:
                return self.down_proj(fused_swiglu(self.gate_up_proj(x)))
            return _original(x)
        module.forward=types.MethodType(forward,module)
        count+=1
    return count


@triton.jit
def _add_norm(A,B,W,S,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    a=tl.load(A+row*N+col,col<N,other=0).to(tl.float32)
    b=tl.load(B+row*N+col,col<N,other=0).to(tl.float32)
    summed=(a+b).to(tl.bfloat16)
    x=summed.to(tl.float32)
    variance=tl.sum(x*x,0)/N
    weight=tl.load(W+col,col<N,other=0).to(tl.float32)
    normalized=(x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
    tl.store(S+row*N+col,summed,col<N)
    tl.store(Y+row*N+col,(normalized*weight).to(tl.bfloat16),col<N)


def fused_add_norm(a,b,weight,epsilon):
    if (not a.is_cuda or a.dtype!=torch.bfloat16 or not a.is_contiguous()
        or b.shape!=a.shape or b.dtype!=a.dtype or b.device!=a.device or not b.is_contiguous()
        or a.shape[-1]!=4096 or weight.shape!=(4096,) or weight.dtype!=a.dtype
        or weight.device!=a.device or not weight.is_contiguous()):
        raise ValueError('expected contiguous Qwen3 CUDA BF16 residual/norm inputs')
    summed=torch.empty_like(a);normalized=torch.empty_like(a)
    _add_norm[(a.numel()//4096,)](a,b,weight,summed,normalized,4096,epsilon,4096,
        num_warps=4,enable_fp_fusion=False)
    return summed,normalized


def install_fused_post_attention(model):
    """Equivalent to residual add then the existing fully fused norm; opt-in only."""
    for layer in model.model.layers:
        original=layer.forward
        def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_values=None,
                    use_cache=False,cache_position=None,position_embeddings=None,_original=original,**kwargs):
            if not hidden_states.is_cuda or hidden_states.dtype!=torch.bfloat16:
                return _original(hidden_states,attention_mask,position_ids,past_key_values,use_cache,
                                 cache_position,position_embeddings,**kwargs)
            attention,_=self.self_attn(hidden_states=self.input_layernorm(hidden_states),
                attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,
                use_cache=use_cache,cache_position=cache_position,position_embeddings=position_embeddings,**kwargs)
            residual,normalized=fused_add_norm(hidden_states,attention,
                self.post_attention_layernorm.weight,self.post_attention_layernorm.variance_epsilon)
            return residual+self.mlp(normalized)
        layer.forward=types.MethodType(forward,layer)
    return len(model.model.layers)
