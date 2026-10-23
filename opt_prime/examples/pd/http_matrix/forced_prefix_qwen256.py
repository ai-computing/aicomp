"""Issue256: compare logits under an identical forced prefix, not generated-answer quality."""
import argparse
import json
import gc
from pathlib import Path
from concurrent.futures import Future
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillOut

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
for name in ['export', 'control', 'live', 'output']:
    parser.add_argument('--'+name, type=Path, required=True)
args = parser.parse_args()
output = args.output
if output.exists():
    raise FileExistsError(output)
export = torch.load(args.export, weights_only=True, mmap=True)
control = json.loads(args.control.read_text())
live = json.loads(args.live.read_text())
assert export['ids'] == control['prompts']['8192'] == live['prompts']['8192']
forced = control['references']['8192:128']['tokens'][:107]
actual = next(b['rows'][0]['tokens'] for b in live['correctness_batches']
              if b['input_tokens'] == 8192 and b['output_tokens'] == 128)
assert forced == actual[:107]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
rows = []

class Replay:
    def submit(self, request, send_from=0, slot=None):
        assert list(request.input_ids) == export['ids']
        future = Future()
        future.set_result(PrefillOut(int(export['logits'].argmax()), send_from,
            export['keys'][:, :, send_from:], export['values'][:, :, send_from:]))
        return future

for dtype in [torch.bfloat16, torch.float32]:
    model = AutoModelForCausalLM.from_pretrained(args.model,
        dtype=dtype, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    for mode in ['native8192', 'retained_rngd_kv']:
        kw = dict(slots=4, max_context=9216, kv='paged', page_size=64, prefix_cache=False)
        batcher = (PDBatcher(model, Replay(), **kw) if mode == 'retained_rngd_kv'
                   else ContinuousBatcher(model, prefill_chunk=8192, **kw))
        emitted, decoded = batcher._emit, batcher._decoded
        probes = []

        def emit(request, token):
            index = len(request.tokens)
            if index < len(forced):
                token = forced[index]
            return emitted(request, token)

        def decode(requests, lengths, logits):
            for i, request in enumerate(requests):
                index = len(request.tokens)
                if index in [12, 106, 107]:
                    assert request.tokens == forced[:index]
                    values, tokens = logits[i].float().topk(8)
                    probes.append(dict(output_index=index, finite=bool(torch.isfinite(logits[i]).all()),
                        top_tokens=tokens.tolist(), top_logits=values.tolist(),
                        native_branch_token=control['references']['8192:128']['tokens'][index],
                        live_branch_token=actual[index],
                        native_branch_logit=float(logits[i, control['references']['8192:128']['tokens'][index]]),
                        live_branch_logit=float(logits[i, actual[index]])))
            return decoded(requests, lengths, logits)

        batcher._emit, batcher._decoded = emit, decode
        future = batcher.submit(PDRequest(mode, tuple(export['ids']), 108))
        batcher.run_until_idle()
        tokens = future.result()
        assert tokens[:107] == forced and len(probes) == 3 and all(p['finite'] for p in probes)
        row = dict(dtype=str(dtype), mode=mode, forced_prefix_tokens=107,
                   selected_token108=tokens[-1], probes=probes)
        rows.append(row)
        output.write_text(json.dumps(dict(complete=False, rows=rows), indent=2)+'\n')
        print(json.dumps(row), flush=True)
        batcher._emit, batcher._decoded = emitted, decoded
        del batcher, emitted, decoded
        gc.collect()
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
output.write_text(json.dumps(dict(complete=True, rows=rows), indent=2)+'\n')
