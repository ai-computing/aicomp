#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.inference: greedy generation with the KV cache (torch.export path) matches HF at batch size 1 and 2.
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Optimus_Inference needs a GPU")


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(0)
    cfg = AutoConfig.for_model("llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                               attn_implementation="sdpa")
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.mark.parametrize("batch", [1, 2])
def test_kv_cache_generation_matches_hf(tiny, batch):
    from opt_prime.inference import Optimus_Inference
    ids = torch.randint(0, 128, (batch, 6), generator=torch.Generator().manual_seed(batch))
    with torch.no_grad():
        ref = copy.deepcopy(tiny).cuda().generate(ids.cuda(), max_new_tokens=5, do_sample=False,
                                                  pad_token_id=0).cpu()
    eng = Optimus_Inference(copy.deepcopy(tiny), use_kv_cache=True, dynamo_capture=True, dtype=torch.float32)
    eng.eval()
    out = eng.generate(ids, max_new_tokens=5, do_sample=False, verbose=False).cpu()
    assert torch.equal(out, ref)
