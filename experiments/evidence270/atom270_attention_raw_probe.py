"""Task-only export instrumentation; installed SDK and production engine unchanged."""
from optimum.rbln.transformers.models.qwen3.qwen3_architecture import Qwen3Wrapper
LABELS=['embedding','q_proj','k_proj','v_proj','q_norm','k_norm','rotary_q','rotary_k','attention_v','attention_result','o_proj']
class AttentionProbeWrapper(Qwen3Wrapper):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        a=self.model.model.layers[0].self_attn
        self._observed=[]
        def record(module,args,output):
            self._observed.append(output)
        def inputs(module,args):
            for value in args[:3]:self._observed.append(value)
        for module in [a.q_proj,a.k_proj,a.v_proj,a.q_norm,a.k_norm]:module.register_forward_hook(record)
        a.get_attention_op().register_forward_pre_hook(inputs)
        a.get_attention_op().register_forward_hook(record)
        a.o_proj.register_forward_hook(record)
    def forward(self,*args):
        self._observed=[]
        logits,states=super().forward(*args)
        assert len(self._observed)==10 and len(states)==37
        if self.phase!='decode':return logits,states
        return logits,(states[0],*self._observed,*states[11:])
