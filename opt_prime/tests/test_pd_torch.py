"""Offline real-model parity through Optimus KVCacheManager, not fake logits."""
import copy
import json
import os
from pathlib import Path
import unittest
try:
    import torch
    from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
                              Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                              MistralConfig, MistralForCausalLM)
except ImportError as error:
    raise unittest.SkipTest("optional torch-pd dependencies missing") from error
from opt_prime.pd import PDRequest, PDRuntime
from opt_prime.pd.torch_engine import TorchCausalLMEngine, TorchCacheTransport


class TorchParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rows = []

    def test_dense_models_full_and_prefix_prefill(self):
        for kind in ('gpt2', 'llama', 'qwen2', 'qwen3', 'mistral'):
            torch.manual_seed(17)
            if kind == 'gpt2':
                model = GPT2LMHeadModel(GPT2Config(vocab_size=41, n_positions=128,
                    n_embd=32, n_layer=2, n_head=4, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0))
            else:
                config_cls, model_cls = {
                    'llama': (LlamaConfig, LlamaForCausalLM),
                    'qwen2': (Qwen2Config, Qwen2ForCausalLM),
                    'qwen3': (Qwen3Config, Qwen3ForCausalLM),
                    'mistral': (MistralConfig, MistralForCausalLM),
                }[kind]
                model = model_cls(config_cls(vocab_size=41, max_position_embeddings=128,
                    hidden_size=32, intermediate_size=64, num_hidden_layers=2, head_dim=8,
                    num_attention_heads=4, num_key_value_heads=2, attention_dropout=0,
                    sliding_window=None, use_sliding_window=False))
            model.eval()
            for prefix in (False, True):
                source = TorchCausalLMEngine(copy.deepcopy(model), model_id=kind,
                    revision='random-seed17', max_context=128, endpoint='source', prefix_only=prefix, capture_logits=True)
                dest = TorchCausalLMEngine(copy.deepcopy(model), model_id=kind,
                    revision='random-seed17', max_context=128, endpoint='destination', capture_logits=True)
                runtime = PDRuntime(source, dest, TorchCacheTransport(source='source',
                    destination='destination', spec=source.spec))
                for length in (2, 7, 19, 63):
                    for count in (1, 6, 16):
                        with self.subTest(model=kind, prefix=prefix, length=length, count=count):
                            prompt = tuple(1 + i % 39 for i in range(length))
                            full = list(prompt)
                            logits = []
                            with torch.inference_mode():
                                for _ in range(count):
                                    row = model(torch.tensor([full]), use_cache=False).logits[0, -1]
                                    logits.append(row)
                                    full.append(int(row.argmax()))
                            actual = runtime.generate(PDRequest(f'{length}-{count}', prompt, count))
                            self.assertEqual(actual, full[length:])
                            measured = source.last_logits + dest.last_logits
                            self.assertEqual(len(measured), count)
                            torch.testing.assert_close(torch.stack(measured), torch.stack(logits),
                                                       atol=1e-6, rtol=1e-5)
                            self.assertGreater(runtime.last_receipt['bytes'], 0)
                            self.assertEqual(runtime.state, 'idle')
                            self.rows.append(dict(model=kind, prefix_only=prefix,
                                input_length=length, output_length=count, tokens_exact=True,
                                max_abs_diff=float((torch.stack(measured)-torch.stack(logits)).abs().max()),
                                receipt=runtime.last_receipt))

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('PD_VALIDATION_REPORT'):
            import transformers
            Path(os.environ['PD_VALIDATION_REPORT']).write_text(json.dumps(dict(
                torch=torch.__version__, transformers=transformers.__version__,
                cases=cls.rows, seed=17, atol=1e-6, rtol=1e-5), indent=2))

    def test_context_rejection_before_model_execution(self):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=41, n_positions=16,
                                          n_embd=16, n_layer=1, n_head=2))
        engine = TorchCausalLMEngine(model, model_id='tiny', revision='seed',
                                    max_context=16, endpoint='cpu')
        with self.assertRaises(ValueError):
            engine.prefill(PDRequest('overflow', (1,) * 15, 2))


if __name__ == '__main__':
    unittest.main()
