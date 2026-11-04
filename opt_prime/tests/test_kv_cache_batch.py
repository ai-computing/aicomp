#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.inference: greedy generation with the KV cache matches HF at batch size 1 and 2, on the torch.export path
# and on the HFTracer path (transformers 4.x only, #191).
import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.IR import _HF_FX_AVAILABLE, IR  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Optimus_Inference needs a GPU")
needs_hf_fx = pytest.mark.skipif(not _HF_FX_AVAILABLE, reason="HFTracer needs transformers < 5")


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(0)
    cfg = AutoConfig.for_model("llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                               attn_implementation="sdpa")
    return AutoModelForCausalLM.from_config(cfg).eval()


@needs_hf_fx
def test_hftracer_kv_cache_trace(tiny):
    # The model's DynamicCache output used to stop HFTracer (ProxyableDynamicCache); the graph must trace and
    # compute the model's logits.
    ir = IR(tiny, SimpleNamespace(model2type={"hf": 50, "sy": 51, "vt": 52}))
    ir.retrieve_IR(tiny, use_kv_cache=True, dynamo_capture=False)
    ids = torch.randint(0, 128, (2, 6), generator=torch.Generator().manual_seed(0))
    pos = torch.arange(6).expand(2, -1)
    with torch.no_grad():
        out = ir.gm(input_ids=ids, position_ids=pos)
        ref = tiny(ids).logits
    torch.testing.assert_close(out["logits"], ref)


def test_export_kv_cache_trace(tiny):
    # torch 2.10 + transformers 4.46: a stale hook signature left layer 0's input_layernorm indexed as a 2-tuple
    # (#292); the exported graph must compute the model's logits.
    ir = IR(tiny, SimpleNamespace(model2type={"hf": 50, "sy": 51, "vt": 52}))
    ir.retrieve_IR(tiny, use_kv_cache=True, dynamo_capture=True)
    ids = torch.randint(0, 128, (2, 6), generator=torch.Generator().manual_seed(0))
    pos = torch.arange(6).expand(2, -1)
    with torch.no_grad():
        out = ir.gm(input_ids=ids, position_ids=pos)[0]   # (logits, *present key/values)
        ref = tiny(ids).logits
    torch.testing.assert_close(out, ref)


@needs_gpu
@pytest.mark.parametrize("dynamo_capture", [True, pytest.param(False, marks=needs_hf_fx)])
@pytest.mark.parametrize("batch", [1, 2])
def test_kv_cache_generation_matches_hf(tiny, batch, dynamo_capture):
    from opt_prime.inference import Optimus_Inference
    ids = torch.randint(0, 128, (batch, 6), generator=torch.Generator().manual_seed(batch))
    with torch.no_grad():
        ref = copy.deepcopy(tiny).cuda().generate(ids.cuda(), max_new_tokens=5, do_sample=False,
                                                  pad_token_id=0).cpu()
    eng = Optimus_Inference(copy.deepcopy(tiny), use_kv_cache=True, dynamo_capture=dynamo_capture,
                            dtype=torch.float32)
    eng.eval()
    out = eng.generate(ids, max_new_tokens=5, do_sample=False, verbose=False).cpu()
    assert torch.equal(out, ref)
