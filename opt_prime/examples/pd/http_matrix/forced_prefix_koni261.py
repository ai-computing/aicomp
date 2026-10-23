"""Verify short-KV replay, then compare KONI logits under an identical64-token prefix."""
import argparse
import gc
import json
from pathlib import Path
from concurrent.futures import Future
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillOut

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
for name in ['export', 'live', 'output']:
    parser.add_argument('--'+name, type=Path, required=True)
args = parser.parse_args()
output = args.output
if output.exists():
    raise FileExistsError(output)
export = torch.load(args.export, weights_only=True, mmap=True)
data = json.loads(args.live.read_text())
assert export['ids'] == data['prompts']['128']
native = data['references']['128:128']['tokens']
live = next(b['rows'][0]['tokens'] for b in data['correctness_batches']
            if b['input_tokens']==128 and b['output_tokens']==128)
forced = native[:64]
assert forced == live[:64] and native[64] != live[64]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
result = dict(complete=False, replay_verified=False, rows=[])

class Replay:
    def submit(self, request, send_from=0, slot=None):
        assert list(request.input_ids) == export['ids'] and send_from == 0
        future = Future()
        future.set_result(PrefillOut(export['token'], 0, export['keys'], export['values']))
        return future

def make(model, mode):
    kw = dict(slots=4, max_context=9216, kv='paged', page_size=64, prefix_cache=False)
    return PDBatcher(model, Replay(), **kw) if mode == 'retained_rngd_kv' else ContinuousBatcher(model, prefill_chunk=256, **kw)

for dtype in [torch.bfloat16, torch.float32]:
    model = AutoModelForCausalLM.from_pretrained(args.model,
        dtype=dtype, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    if dtype == torch.bfloat16:
        for mode, expected in [('native128',native),('retained_rngd_kv',live)]:
            batcher = make(model, mode)
            future = batcher.submit(PDRequest(mode, tuple(export['ids']),128))
            batcher.run_until_idle()
            assert future.result() == expected, 'Original output not reproduced: '+mode
            del batcher, future
            gc.collect()
            torch.cuda.empty_cache()
        result['replay_verified'] = True
        output.write_text(json.dumps(result,indent=2)+'\n')
    for mode in ['native128','retained_rngd_kv']:
        batcher = make(model, mode)
        emitted, decoded = batcher._emit, batcher._decoded
        probes, kv_error = [], []

        def emit(request, token):
            index = len(request.tokens)
            if index == 0 and mode == 'native128':
                keys, values = batcher.kv.read_span(request.slot, 0, len(export['ids']))
                for layer in range(keys.shape[0]):
                    row = dict(layer=layer)
                    for name, actual in [('keys',keys),('values',values)]:
                        ref = actual[layer].float()
                        diff = export[name][layer].to(ref.device).float() - ref
                        row[name+'_relative_l2'] = float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref).clamp_min(1e-20))
                        row[name+'_max_abs'] = float(diff.abs().max())
                    kv_error.append(row)
            return emitted(request, forced[index] if index < 64 else token)

        def decode(requests, lengths, logits):
            for i, request in enumerate(requests):
                index = len(request.tokens)
                if index in [63,64]:
                    assert request.tokens == forced[:index]
                    values, tokens = logits[i].float().topk(8)
                    probes.append(dict(output_index=index,finite=bool(torch.isfinite(logits[i]).all()),
                        top_tokens=tokens.tolist(),top_logits=values.tolist(),native_branch_token=native[index],
                        live_branch_token=live[index],native_branch_logit=float(logits[i,native[index]]),
                        live_branch_logit=float(logits[i,live[index]])))
            return decoded(requests,lengths,logits)

        batcher._emit, batcher._decoded = emit, decode
        future = batcher.submit(PDRequest(mode,tuple(export['ids']),65))
        batcher.run_until_idle()
        tokens = future.result()
        assert tokens[:64] == forced and len(probes)==2 and all(p['finite'] for p in probes)
        result['rows'].append(dict(dtype=str(dtype),mode=mode,selected_token65=tokens[-1],
                                   probes=probes,kv_difference_from_native=kv_error))
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(dtype=str(dtype),mode=mode,token65=tokens[-1],probe=probes[-1])),flush=True)
        batcher._emit, batcher._decoded = emitted, decoded
        del batcher, emitted, decoded, future
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
result['complete'] = True
output.write_text(json.dumps(result,indent=2)+'\n')
