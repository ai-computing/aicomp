import sys
import types

import pytest
import torch
from gemma_prefill import Prefill
from transformers import Gemma3ForCausalLM, Gemma3TextConfig


@pytest.fixture
def cpu_sdk(monkeypatch):
    sdk = types.ModuleType("rngd_pd.engines.fsdk")

    class Compile:
        @staticmethod
        def from_module(module, sample, compiler_config):
            return module

    sdk.CompileModule = Compile
    sdk.FCConfig = lambda **kw: kw
    sdk.dev_str = lambda chip: "cpu"
    sdk.hint = lambda name: name
    sdk.sync = lambda chip: None
    for name in ["rngd_pd", "rngd_pd.engines"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "rngd_pd.engines.fsdk", sdk)


def test_reusable_gemma_prefill_matches_hf_and_reuses_without_stale_prefix(cpu_sdk):
    torch.manual_seed(234)
    config = Gemma3TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=4,
        sliding_window_pattern=6,
        query_pre_attn_scalar=4,
        vocab_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    ref = Gemma3ForCausalLM(config).to(torch.bfloat16).eval()
    engine = Prefill(config.to_dict(), ref.state_dict(), width=12)
    first = None
    with torch.inference_mode():
        for ids in [[1, 4, 5], [1, 7, 8, 9, 10, 11, 12], [1, 4, 5]]:
            k, v, logits = engine.prefill(ids)
            out = ref(input_ids=torch.tensor([ids]), use_cache=True)
            assert k.shape == v.shape == (6, 1, len(ids), 4)
            torch.testing.assert_close(
                logits, out.logits[0, -1].float(), atol=0.02, rtol=0.08
            )
            for i, layer in enumerate(out.past_key_values.layers):
                kept = layer.keys.shape[2]
                for actual, expected in [
                    (k[i, :, -kept:], layer.keys[0]),
                    (v[i, :, -kept:], layer.values[0]),
                ]:
                    relative = (
                        actual.float() - expected.float()
                    ).square().mean().sqrt() / expected.float().square().mean().sqrt()
                    assert float(relative) < 0.03, (i, float(relative))
            if len(ids) == 3:
                if first is None:
                    first = (k.clone(), v.clone(), logits.clone())
                else:
                    assert all(torch.equal(a, b) for a, b in zip(first, (k, v, logits)))
        for ids in [[], list(range(13)), [-1], [32]]:
            with pytest.raises(ValueError):
                engine.prefill(ids)
