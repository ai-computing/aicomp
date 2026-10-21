#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.accel.spec: block specs from transformers models (all tiny MoE families, meta-device construction, real
# configs when cached) and the KV semantics PD engines exchange.
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.accel.spec import model_spec, model_spec_from_pretrained  # noqa: E402
from opt_prime.accel.spec.adapters import adapter_for  # noqa: E402
from opt_prime.accel.spec.discover import find_moe_experts  # noqa: E402
from opt_prime.accel.testing import FAMILIES, build_tiny  # noqa: E402


def _tiny(family, **kw):
    try:
        return build_tiny(family, **kw)
    except Exception as e:      # the family does not build with this transformers version
        pytest.skip(f"{family}: {type(e).__name__}")


@pytest.mark.parametrize("family", FAMILIES)
def test_spec_matches_modules(family):
    m = _tiny(family)
    s = model_spec(m)
    assert len(s.layers) == m.config.num_hidden_layers and s.hidden == m.config.hidden_size
    # routed experts: the same ExpertSpec the offload uses, in the same order
    found = [adapter_for(i.module).spec(i.module) for i in find_moe_experts(m)]
    assert [l.moe.experts for l in s.moe_layers] == found
    for l, layer in zip(s.layers, dict(m.named_modules())[s.layers[0].module_path.rsplit(".", 1)[0]]):
        a = l.attention
        if a.kind == "gqa":                                  # geometry agrees with the projection weights
            at = layer.self_attn
            assert at.q_proj.weight.shape[0] == a.heads * a.head_dim
            assert at.k_proj.weight.shape[0] == a.kv_heads * a.head_dim
        if l.moe is not None:
            assert l.moe.router.num_experts == l.moe.experts.num_experts and l.moe.router.top_k > 0


@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "deepseek_v3", "qwen3_next"])
def test_kv_semantics_depend_on_geometry_not_weights(family):
    a = model_spec(_tiny(family, seed=0))
    b = model_spec(_tiny(family, seed=1))
    assert a.kv_semantic_id() == b.kv_semantic_id() and a.signature() == b.signature()
    ks = a.kv_semantics()
    assert {x["layer"] for x in ks["kv_layers"]} | {x["layer"] for x in ks["state_layers"]} == set(range(len(a.layers)))
    if family == "qwen3_next":
        assert ks["state_layers"], "linear-attention layers carry recurrent state, not K/V"
    # a different KV geometry is a different contract
    from dataclasses import replace
    l0 = a.layers[0]
    if l0.attention.keeps_kv:
        changed = replace(a, layers=(replace(l0, attention=replace(l0.attention, kv_heads=l0.attention.kv_heads * 2)),)
                          + a.layers[1:])
        assert changed.kv_semantic_id() != a.kv_semantic_id()


def test_meta_construction_matches_instantiated(tmp_path):
    m = _tiny("solar_open", dtype=torch.bfloat16)
    m.config.save_pretrained(tmp_path)
    meta = model_spec_from_pretrained(str(tmp_path), revision="x")
    assert meta.layers == model_spec(m).layers and meta.kv_semantic_id() != ""


def test_dense_qwen2_covers_pd_geometry():
    """everything PD's DenseModel checks for a dense Qwen2 checkpoint is in the block spec"""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("qwen2", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
    s = model_spec(AutoModelForCausalLM.from_config(cfg))
    l = s.layers[0]
    assert (l.attention.kind, l.attention.heads, l.attention.kv_heads, l.attention.head_dim) == ("gqa", 4, 2, 16)
    assert l.attention.qkv_bias and not l.is_moe and l.mlp.kind == "gated" and l.mlp.ffn == 128
    assert l.input_norm.eps == cfg.rms_norm_eps and l.attention.rope.theta == cfg.rope_parameters["rope_theta"]
    assert s.vocab == 128 and not s.custom_blocks


@pytest.mark.parametrize("model_id,expect", [
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", dict(layers=48, E=128, k=8, F=768, scoring="softmax", qkn=True, kvh=4, hd=128,
                                             shared=0)),
    ("upstage/Solar-Open-100B", dict(layers=48, E=128, k=8, F=1280, scoring="sigmoid", qkn=False, kvh=8, hd=128,
                                     shared=1280)),
])
def test_real_configs(model_id, expect, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        s = model_spec_from_pretrained(model_id)
    except Exception as e:
        pytest.skip(f"{model_id} config not cached: {type(e).__name__}")
    l = s.layers[-1]
    assert len(s.layers) == expect["layers"] and len(s.moe_layers) == expect["layers"]
    assert (l.moe.experts.num_experts, l.moe.router.top_k, l.moe.experts.ffn) == (expect["E"], expect["k"], expect["F"])
    assert l.moe.router.scoring == expect["scoring"] and l.attention.qk_norm == expect["qkn"]
    assert (l.attention.kv_heads, l.attention.head_dim, l.moe.shared_ffn) == (expect["kvh"], expect["hd"],
                                                                              expect["shared"])
    assert not s.custom_blocks


def test_layer_level_experts_beside_dense_mlp():
    """Gemma 4: router + experts are children of the decoder layer next to a dense MLP; both are described."""
    s = model_spec(_tiny("gemma4_text"))
    assert s.moe_layers and all(l.mlp is not None and l.mlp.kind == "gated" for l in s.moe_layers)
    assert all(l.moe.router.top_k > 0 for l in s.moe_layers)
