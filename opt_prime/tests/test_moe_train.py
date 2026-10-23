#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_batched_mm (opt_prime.accel.moe.train): same outputs and gradients as transformers' eager experts, exportable with
# torch.export in fp32 and bf16, and selected by prepare_moe_for_training for models with experts.
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.accel.spec.discover import find_moe_experts  # noqa: E402
from opt_prime.accel.testing import build_tiny  # noqa: E402
from opt_prime.accel.moe.train import NAME, prepare_moe_for_training  # noqa: E402


def _tiny(family, dtype=torch.float32):
    try:
        return build_tiny(family, dtype)
    except Exception as e:
        pytest.skip(f"{family}: {type(e).__name__}")


@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "mixtral", "gemma4_text"])
def test_matches_eager_outputs_and_grads(family):
    ref = _tiny(family)
    for c in {id(c): c for c in (ref.config, ref.config.get_text_config())}.values():
        c._experts_implementation = "eager"
    m = copy.deepcopy(ref)
    assert prepare_moe_for_training(m) and m.config.get_text_config()._experts_implementation == NAME
    ids = torch.randint(3, 120, (2, 11), generator=torch.Generator().manual_seed(0))
    outs = []
    for model in (ref, m):
        model.zero_grad()
        logits = model(ids).logits
        logits.float().pow(2).mean().backward()
        e = find_moe_experts(model)[0].module
        outs.append((logits.detach(), e.gate_up_proj.grad.clone(), e.down_proj.grad.clone()))
    for a, b in zip(*outs):
        torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exportable(dtype):
    m = _tiny("qwen3_moe", dtype)
    m.config.use_cache = False
    prepare_moe_for_training(m)
    ids = torch.randint(3, 120, (1, 9))
    ep = torch.export.export(m, (ids,), strict=False)
    torch.testing.assert_close(ep.module()(ids).logits, m(ids).logits, atol=1e-2 if dtype == torch.bfloat16 else 1e-5,
                               rtol=1e-2 if dtype == torch.bfloat16 else 1e-4)


def test_dense_models_untouched():
    from transformers import AutoConfig, AutoModelForCausalLM
    m = AutoModelForCausalLM.from_config(AutoConfig.for_model("llama", hidden_size=32, intermediate_size=64,
                                                              num_hidden_layers=1, num_attention_heads=2,
                                                              num_key_value_heads=1, vocab_size=64))
    assert prepare_moe_for_training(m) is False


@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "gpt_oss"])
def test_grouped_matches_batched_with_grads(family):
    """opt_grouped_mm (one expert at a time, used by the offloaded experts' backward) == opt_batched_mm."""
    from opt_prime.accel.moe.batched import opt_batched_mm, opt_grouped_mm
    try:
        m = build_tiny(family, torch.float32)
    except Exception as e:
        pytest.skip(f"{family}: {type(e).__name__}")
    ex = find_moe_experts(m)[0].module
    H = ex.gate_up_proj.shape[1] if getattr(ex, "is_transposed", False) else ex.gate_up_proj.shape[2]
    torch.manual_seed(0)
    x = torch.randn(7, H, requires_grad=True)
    idx = torch.stack([torch.randperm(ex.num_experts)[:2] for _ in range(7)])
    w = torch.rand(7, 2, requires_grad=True)
    g = torch.randn(7, H)
    a = opt_batched_mm(ex, x, idx, w)
    ga = torch.autograd.grad(a, (x, w), g)
    b = opt_grouped_mm(ex, x, idx, w)
    gb = torch.autograd.grad(b, (x, w), g)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    for p, q in zip(ga, gb):
        assert torch.allclose(p, q, atol=1e-5, rtol=1e-4)
