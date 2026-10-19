import unittest
from types import SimpleNamespace
from opt_prime.pd.torch_models import describe_model


def config(**changes):
    values=dict(model_type='qwen2',num_hidden_layers=2,num_key_value_heads=2,
                num_attention_heads=4,hidden_size=32,sliding_window=None)
    values.update(changes)
    return SimpleNamespace(**values)


class ModelCapabilityTests(unittest.TestCase):
    def test_unsupported_cache_and_attention_fail_closed(self):
        for changes in ({'model_type':'unknown'}, {'sliding_window':4096},
                        {'layer_types':['full_attention','sliding_attention']},
                        {'is_encoder_decoder':True}, {'add_cross_attention':True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                describe_model(config(**changes))

    def test_disabled_window_is_not_sliding_attention(self):
        geometry=describe_model(config(sliding_window=4096,use_sliding_window=False))
        self.assertEqual((geometry.kv_heads,geometry.head_dim),(2,8))

    def test_historical_contract_imports_are_same_types(self):
        from opt_prime.pd import KVSpec, PrefillEngine, DecodeEngine, Transport
        from opt_prime.pd.runtime import KVSpec as legacy
        self.assertIs(KVSpec,legacy)
        self.assertTrue(all(x is not None for x in (PrefillEngine,DecodeEngine,Transport)))
