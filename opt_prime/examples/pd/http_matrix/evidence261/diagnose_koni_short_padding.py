"""Compare real-length and RNGD-shaped padded GPU prefill without changing production sampling."""
import json
from pathlib import Path
from concurrent.futures import Future
import torch
from transformers import AutoModelForCausalLM
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillOut

root = Path(__file__).resolve().parent
output = root/'koni-short-padding-diagnostic.json'
if output.exists():
    raise FileExistsError(output)
data = json.loads((root/'local201-koni-matrix-a.json').read_text())
ids = data['prompts']['128']
assert len(ids) == 128
native = data['references']['128:128']['tokens']
live = next(b['rows'][0]['tokens'] for b in data['correctness_batches']
            if b['input_tokens'] == 128 and b['output_tokens'] == 128)
torch.set_num_threads(4)
model = AutoModelForCausalLM.from_pretrained('/tmp/serving-matrix-models/KONI-8B',
    dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
rows = []

class PaddedGPU:
    def __init__(self, width):
        self.width = width

    @torch.inference_mode()
    def submit(self, request, send_from=0, slot=None):
        assert list(request.input_ids) == ids and send_from == 0
        # The research RNGD _DevicePrefillStage repeats the last token to its bucket width.
        padded = ids + [ids[-1]] * (self.width - len(ids))
        out = model(torch.tensor([padded], device='cuda:0'), use_cache=True,
                    logits_to_keep=torch.tensor([len(ids)-1], device='cuda:0'))
        keys = torch.stack([layer.keys[0, :, :len(ids)].contiguous() for layer in out.past_key_values.layers])
        values = torch.stack([layer.values[0, :, :len(ids)].contiguous() for layer in out.past_key_values.layers])
        result = PrefillOut(int(out.logits[0, 0].argmax()), 0, keys, values)
        future = Future()
        future.set_result(result)
        return future

for mode, width in [('native128', 128), ('hf128_import', 128), ('hf8192_padded_import', 8192)]:
    kw = dict(slots=4, max_context=9216, kv='paged', page_size=64, prefix_cache=False)
    batcher = ContinuousBatcher(model, prefill_chunk=256, **kw) if mode == 'native128' else PDBatcher(model, PaddedGPU(width), **kw)
    future = batcher.submit(PDRequest(mode, tuple(ids), 128))
    batcher.run_until_idle()
    tokens = future.result()
    assert len(tokens) == 128
    if mode == 'native128':
        assert tokens == native, 'native baseline changed; stop diagnostic'
    rows.append(dict(mode=mode, physical_prefill_tokens=width, logical_prompt_tokens=128,
                     tokens=tokens, equals_native=tokens == native, equals_live=tokens == live,
                     first_diff_native=next((i for i,(a,b) in enumerate(zip(tokens,native)) if a != b), None)))
    output.write_text(json.dumps(dict(complete=False, rows=rows), indent=2)+'\n')
    print(json.dumps({k:v for k,v in rows[-1].items() if k != 'tokens'}), flush=True)
    del batcher, future
    torch.cuda.empty_cache()
output.write_text(json.dumps(dict(complete=True, rows=rows), indent=2)+'\n')
