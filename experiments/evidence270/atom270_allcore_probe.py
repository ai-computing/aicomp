"""Task-only diagnostic: append all core-attention outputs; retain all hidden states."""
from optimum.rbln.transformers.models.qwen3.qwen3_architecture import Qwen3Wrapper
class CoreProbeWrapper(Qwen3Wrapper):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._observed=[]
        def record(module,args,output):self._observed.append(output)
        for layer in self.model.model.layers:
            layer.self_attn.get_attention_op().register_forward_hook(record)
    def forward(self,*args):
        self._observed=[]
        logits,states=super().forward(*args)
        assert len(self._observed)==36 and len(states)==37
        if self.phase!='decode':return logits,states
        return logits,(*states,*self._observed)
