import unittest

from spec import validate


class SpecTests(unittest.TestCase):
    def config(self, **kw):
        c = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 28,
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_theta": 1e6,
        }
        c.update(kw)
        return c

    def test_query_width_is_independent(self):
        s = validate(self.config())
        self.assertEqual(s["query_width"], 2048)
        self.assertEqual(s["hidden_size"], 1024)
        self.assertEqual(s["slots"], 56)

    def test_rejects_other_semantics(self):
        for kw in [
            {"model_type": "qwen2"},
            {"quantization_config": {"quant_method": "fp8"}},
            {"use_sliding_window": True},
            {"rope_scaling": {"rope_type": "linear"}},
            {"attention_bias": True},
            {"num_key_value_heads": 3},
            {"head_dim": 64},
        ]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                validate(self.config(**kw))


if __name__ == "__main__":
    unittest.main()
