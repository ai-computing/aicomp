#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""The Optimus engine: Optimus_Inference on GPUs, optionally with MoE expert tiers."""
import time
from typing import Optional

from opt_prime.accel.engines.base import ExecutionEngine


class OptimusEngine(ExecutionEngine):
    """Optimus_Inference on the GPU, optionally with MoE expert tiers (moe_offload). Supports any model torch runs;
    its speed depends on the tiers, so it reports no fixed estimate unless given one."""
    name = "optimus"

    def __init__(self, moe_offload=None, tok_s: Optional[float] = None, **inference_kw):
        self.moe_offload, self.tok_s, self.kw = moe_offload, tok_s, inference_kw
        self.engine = None

    def expected_tok_s(self, spec):
        return self.tok_s

    def start(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM
        from opt_prime.inference import Optimus_Inference
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        kw = dict(use_gpu=True, dtype=torch.bfloat16, use_kv_cache=True, dynamo_capture=True)
        kw.update(self.kw)
        self.engine = Optimus_Inference(model, moe_offload=self.moe_offload, **kw)
        self.engine.eval()

    def generate(self, ids, max_new=64):
        import torch
        t = torch.tensor([list(ids)]).cuda()
        t0 = time.time()
        out = self.engine.generate(t, max_new_tokens=max_new, do_sample=False, verbose=False)
        dt = time.time() - t0
        return out[0, len(ids):].tolist(), dict(seconds=dt)
