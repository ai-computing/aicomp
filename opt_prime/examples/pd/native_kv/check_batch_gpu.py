"""Hardware check: mixed prefix lengths, retirement, cancellation and lane reuse."""
import argparse
import json
import time
import torch
from gpu_decoder import load_model, GraphDecoder
from gpu_batch_decoder import BatchGraphDecoder
from gpu_norms import install_fused_norms

p = argparse.ArgumentParser()
p.add_argument('--model-dir', required=True)
p.add_argument('--output', required=True)
p.add_argument('--count', type=int, default=128)
a = p.parse_args()
torch.set_num_threads(1)
model = load_model(a.model_dir, fuse_mlp=True, fuse_qkv=True)
install_fused_norms(model, 'fused')
single = GraphDecoder(model, attention='flash')
batch = BatchGraphDecoder(model, batch_size=4)
records = []
prefixes = []
with torch.inference_mode():
    for n in (16, 128, 512, 1024):
        gen = torch.Generator().manual_seed(139+n)
        ids = torch.randint(1000, 20000, (1, n), generator=gen).cuda()
        cache = model(ids[:, :-1], use_cache=True).past_key_values
        layers = [(layer.keys.cpu(), layer.values.cpu()) for layer in cache.layers]
        last = int(ids[0, -1])
        single.load_prefix(layers, last, packed_import=True)
        expected, timing = single.generate(a.count)
        prefixes.append((layers, last, expected))
        records.append({'input_tokens': n, 'single': timing})
    for lane, (layers, last, _) in enumerate(prefixes): batch.load_lane(lane, layers, last, a.count)
    got = [[] for _ in prefixes]
    start = time.perf_counter()
    while any(batch.remaining):
        for lane, token in batch.step(ignore_eos=True).items(): got[lane].append(token)
    elapsed = time.perf_counter()-start
    for lane, (_, _, expected) in enumerate(prefixes):
        records[lane]['equal_positions'] = sum(x == y for x,y in zip(expected, got[lane]))
        records[lane]['expected'] = expected
        records[lane]['actual'] = got[lane]
    # A cancelled lane must not emit; a fresh request must not consume old KV.
    layers, last, expected = prefixes[0]
    batch.load_lane(0, layers, last, 3)
    batch.cancel_lane(0)
    batch.load_lane(2, layers, last, 3)
    reused = []
    while any(batch.remaining):
        step = batch.step(ignore_eos=True)
        assert set(step) == {2}, step
        reused.append(step[2])
    assert reused == expected[:3], (reused, expected[:3])
    # Refill a cancelled slot while another lane continues at a different position.
    other_layers, other_last, other_expected = prefixes[2]
    batch.load_lane(0, layers, last, 8)
    batch.load_lane(3, other_layers, other_last, 8)
    first = batch.step(ignore_eos=True)
    assert first == {0: expected[0], 3: other_expected[0]}
    batch.cancel_lane(0)
    batch.load_lane(0, prefixes[1][0], prefixes[1][1], 3)
    live = {0: [], 3: [first[3]]}
    while any(batch.remaining):
        for lane, token in batch.step(ignore_eos=True).items(): live[lane].append(token)
    assert live[0] == prefixes[1][2][:3]
    assert live[3] == other_expected[:8]
    # EOS retirement (use the known next token as the stop token).
    batch.load_lane(1, layers, last, 3)
    assert batch.step(eos_ids={expected[0]}) == {1: expected[0]}
    assert not any(batch.remaining)
result = {'cases': records, 'output_tokens': a.count, 'batch_tok_s': 4*a.count/elapsed, 'batch_ms_per_step': elapsed*1000/a.count,
          'cancel_reuse_eos': 'pass'}
open(a.output, 'w').write(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
