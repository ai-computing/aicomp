#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# The export-based IR (dynamo_capture=True) on transformers >= 5: the module-level graph rebuilt from torch.export
# computes the same logits as the model, for dense families with plain, q/k-norm and hybrid sliding attention.
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.IR import IR  # noqa: E402


def _tiny(kind):
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        cfg = AutoConfig.for_model(kind, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
                                   max_position_embeddings=128)
    except Exception as e:
        pytest.skip(f"{kind}: {type(e).__name__}")
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3", "mistral", "gemma3_text"])
def test_export_graph_matches_model(kind, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    m = _tiny(kind)
    ir = IR(m, SimpleNamespace(model2type={"hf": 50, "sy": 51, "vt": 52}))
    ir.retrieve_IR(m, dynamo_capture=True)
    ids = torch.randint(3, 120, (1, 9), generator=torch.Generator().manual_seed(1))
    feed = {"input_ids": ids, "position_ids": torch.arange(9).unsqueeze(0), "attention_mask": torch.ones_like(ids)}
    names = [n.name for n in ir.gm.graph.nodes if n.op == "placeholder"]
    out = ir.gm(*[feed[n] for n in names])
    out = out[0] if isinstance(out, (tuple, list)) else out
    out = getattr(out, "logits", out)
    with torch.no_grad():
        ref = m(ids).logits
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-4)
