"""Experimental fused Qwen3 RMSNorm preserving the intermediate BF16 rounding."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X,W,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+col,col<N,other=0).to(tl.float32)
    w=tl.load(W+col,col<N,other=0).to(tl.float32)
    variance=tl.sum(x*x,0)/N
    # HF casts normalized activation to BF16 before multiplying its BF16 weight.
    normalized=(x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
    y=(normalized*w).to(tl.bfloat16)
    tl.store(Y+row*N+col,y,col<N)


def fused_rms_norm(x,weight,epsilon):
    if not x.is_cuda or x.dtype!=torch.bfloat16 or not x.is_contiguous():
        raise ValueError('fused norm requires contiguous CUDA BF16 input')
    n=x.shape[-1]
    if weight.shape!=(n,) or weight.dtype!=x.dtype or weight.device!=x.device or not weight.is_contiguous():
        raise ValueError('invalid norm weight')
    if n not in (128,4096):raise ValueError('only Qwen3-8B norm widths admitted')
    out=torch.empty_like(x)
    _norm[(x.numel()//n,)](x,weight,out,n,epsilon,triton.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def _apply_norm(X,W,S,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+col,col<N,other=0).to(tl.float32)
    w=tl.load(W+col,col<N,other=0).to(tl.float32)
    scale=tl.load(S+row)
    normalized=(x*scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row*N+col,(normalized*w).to(tl.bfloat16),col<N)


def torch_stats_rms_norm(x,weight,epsilon):
    """Keep HF's variance/rsqrt operations; fuse only the final BF16 multiplications."""
    if not x.is_cuda or x.dtype!=torch.bfloat16 or not x.is_contiguous():
        raise ValueError('norm requires contiguous CUDA BF16 input')
    n=x.shape[-1]
    if n not in (128,4096) or weight.shape!=(n,) or weight.dtype!=x.dtype or weight.device!=x.device or not weight.is_contiguous():
        raise ValueError('invalid Qwen3 norm weight')
    variance=x.float().pow(2).mean(-1,keepdim=True)
    scale=torch.rsqrt(variance+epsilon)
    out=torch.empty_like(x)
    _apply_norm[(x.numel()//n,)](x,weight,scale,out,n,triton.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return out


def install_fused_norms(model,mode='fused'):
    import types
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    if mode not in ('fused','torch-stats'):raise ValueError('unknown norm mode')
    implementation=fused_rms_norm if mode=='fused' else torch_stats_rms_norm
    count=0
    for module in model.modules():
        if isinstance(module,Qwen3RMSNorm):
            original=module.forward
            def forward(self,x,_original=original):
                if x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous():
                    return implementation(x,self.weight,self.variance_epsilon)
                return _original(x)
            module.forward=types.MethodType(forward,module)
            count+=1
    return count
