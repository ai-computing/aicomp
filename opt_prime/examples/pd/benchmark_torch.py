"""Offline, order-balanced CPU ablation; never a claim about NPU throughput."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time
import torch
import transformers
from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                          MistralConfig, MistralForCausalLM)
from opt_prime.accel.pd import PDRequest, PDRuntime
from opt_prime.accel.pd.torch_engine import TorchCausalLMEngine
from opt_prime.accel.pd.torch_cache import TorchCacheTransport


class Timed:
    def __init__(self, engine, samples):
        self.engine, self.samples = engine, samples
        self.spec = getattr(engine, 'spec', None)
    def _call(self, name, *args):
        start = time.perf_counter()
        value = getattr(self.engine, name)(*args)
        self.samples[name] = time.perf_counter() - start
        return value
    def prefill(self, *args): return self._call('prefill', *args)
    def reserve(self, *args): return self._call('reserve', *args)
    def decode(self, *args): return self._call('decode', *args)
    def supports(self, *args): return self.engine.supports(*args)
    def transfer(self, *args): return self._call('transfer', *args)
    def finish(self, *args): return self._call('finish', *args)


def model_for(kind):
    if kind == 'gpt2':
        return GPT2LMHeadModel(GPT2Config(vocab_size=8192, n_positions=384,
            n_embd=128, n_layer=2, n_head=4, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)).eval()
    config, model = {'llama':(LlamaConfig,LlamaForCausalLM),
                     'qwen2':(Qwen2Config,Qwen2ForCausalLM),
                     'qwen3':(Qwen3Config,Qwen3ForCausalLM),
                     'mistral':(MistralConfig,MistralForCausalLM)}[kind]
    return model(config(vocab_size=8192, max_position_embeddings=384, hidden_size=128,
        intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, sliding_window=None,
        use_sliding_window=False, attention_dropout=0)).eval()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=9)
    args=parser.parse_args()
    if args.repeats < 3 or args.repeats % 3:
        raise ValueError('repeats must be a positive multiple of three for balanced order')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    variants=[('baseline',False,True),('last_logits',True,True),('fast',True,False)]
    rows=[]
    for kind in ('gpt2','llama','qwen2','qwen3','mistral'):
        torch.manual_seed(23)
        model=model_for(kind)
        engines={}
        for name,optimize,capture in variants:
            options=dict(model_id=kind,revision='random-seed23',max_context=384,
                         optimize_prefill=optimize,capture_logits=capture)
            source=TorchCausalLMEngine(copy.deepcopy(model),endpoint='source',**options)
            dest=TorchCausalLMEngine(copy.deepcopy(model),endpoint='decode',**options)
            times={}
            transport=TorchCacheTransport(source='source',destination='decode',spec=source.spec)
            engines[name]=(PDRuntime(Timed(source,times),Timed(dest,times),Timed(transport,times)),times)
        for length in (32,128,256):
            request=PDRequest(f'{kind}-{length}',tuple(1+i*13%8191 for i in range(length)),16)
            expected=None
            for name,_,_ in variants:
                for _ in range(2):
                    actual=engines[name][0].generate(request)
                    if expected is None:expected=actual
                    if actual!=expected:raise RuntimeError('warmup token parity failure')
            for repeat in range(args.repeats):
                order=variants[repeat%3:]+variants[:repeat%3]
                for name,_,_ in order:
                    runtime,times=engines[name]
                    start=time.perf_counter();actual=runtime.generate(request);elapsed=time.perf_counter()-start
                    if actual!=expected:raise RuntimeError('measured token parity failure')
                    row=dict(model=kind,prompt=length,output=16,repeat=repeat,variant=name,
                             total=elapsed,**times,tokens_exact=True)
                    rows.append(row)
                    with (args.out/'runs.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('DONE',kind,length,flush=True)
    summaries=[]
    for kind in ('gpt2','llama','qwen2','qwen3','mistral'):
        for length in (32,128,256):
            medians={name:{metric:statistics.median(r[metric] for r in rows
                if r['model']==kind and r['prompt']==length and r['variant']==name)
                for metric in ('total','prefill','reserve','transfer','decode')}
                for name,_,_ in variants}
            summaries.append(dict(model=kind,prompt=length,medians_seconds=medians,
                total_speedup=medians['baseline']['total']/medians['fast']['total'],
                prefill_speedup=medians['baseline']['prefill']/medians['last_logits']['prefill']))
    (args.out/'summary.json').write_text(json.dumps(dict(seed=23,threads=2,
        torch=torch.__version__,transformers=transformers.__version__,repeats=args.repeats,
        warmups=2,order='rotating baseline/last_logits/fast',precision='float32',device='cpu',
        note='Random small models; correctness and CPU ablation only, not NPU speed or task quality',
        results=summaries),indent=2))


if __name__=='__main__':
    main()
