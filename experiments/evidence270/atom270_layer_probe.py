"""Isolated export instrumentation: original arithmetic, selected first-layer outputs.

This module is imported only by the task-owned diagnostic compiler process.
It never modifies installed SDK files or a serving engine.
"""
from optimum.rbln.transformers.models.qwen3.qwen3_architecture import Qwen3Wrapper

class LayerProbeWrapper(Qwen3Wrapper):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        layer=self.model.model.layers[0]
        self._observed=[]
        def record(module,args,output):
            self._observed.append(output)
        for module in [layer.get_pre_attention_layernorm(),layer.self_attn,
                       layer.get_post_attention_layernorm(),layer.get_mlp(),layer]:
            module.register_forward_hook(record)

    def forward(self,*args):
        self._observed=[]
        logits,states=super().forward(*args)
        assert len(self._observed)==5 and len(states)==37
        # Keep the SDK output count/shapes; slots0..5 are diagnostic, not layer indices.
        return logits,(states[0],*self._observed,*states[6:])
