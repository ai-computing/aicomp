"""Offline usage example: two independent tiny Llama engines, explicit KV copy."""
import copy
import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from opt_prime.accel.pd import PDRequest, PDRuntime
from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine, TorchCacheTransport


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=41, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64)).eval()
    options = dict(model_id='offline-tiny-llama', revision='seed17', max_context=64)
    prefill = TorchCausalLMEngine(model, endpoint='prefill', **options)
    decode = TorchCausalLMEngine(copy.deepcopy(model), endpoint='decode', **options)
    transport = TorchCacheTransport(source='prefill', destination='decode', spec=prefill.spec)
    runtime = PDRuntime(prefill, decode, transport)
    tokens = runtime.generate(PDRequest('example', (1, 2, 3, 4), 8))
    print(json.dumps(dict(token_ids=tokens, transfer=runtime.last_receipt, state=runtime.state)))


if __name__ == '__main__':
    main()
