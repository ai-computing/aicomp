"""Offline real Llama through the local upstream llm-d sidecar."""
import argparse
import copy
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from opt_prime.accel.pd import PDRequest, PDRuntime
from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine, TorchCacheTransport
from sidecar_validation import validate_sidecar


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sidecar',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(17);torch.set_num_threads(2)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=41,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)).eval()
    options=dict(model_id='tiny-llama',revision='seed17',max_context=64)
    source=TorchCausalLMEngine(model,endpoint='source',**options)
    decoder=TorchCausalLMEngine(copy.deepcopy(model),endpoint='decode',**options)
    transport=TorchCacheTransport(source='source',destination='decode',spec=source.spec)
    runtime=PDRuntime(source,decoder,transport)
    cases=[]
    for length in (2,7,19):
        prompt=tuple(1+i%39 for i in range(length))
        expected=runtime.generate(PDRequest(f'direct-{length}',prompt,6))
        cases.append(dict(prompt=prompt,count=6,expected=expected))
    print(validate_sidecar(source,decoder,transport,cases,binary=args.sidecar,output=args.out,
        model='tiny-llama',vocab_size=41,context=64,max_output=32,render=lambda tokens:str(tokens)))


if __name__=='__main__':
    main()
