"""Teacher-forced width comparison: identical prefix/inputs, independent caches."""
import argparse
import json
from pathlib import Path
import torch
from gpu_decoder import load_model
from gpu_batch_decoder import BatchGraphDecoder
from gpu_norms import install_fused_norms

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir', required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=128)
a = p.parse_args()
torch.set_num_threads(1)
model = load_model(a.model_dir, fuse_mlp=True, fuse_qkv=True)
install_fused_norms(model, 'fused')
full = BatchGraphDecoder(model, 4)
one = BatchGraphDecoder(model, 1)
switch = BatchGraphDecoder(model, 4, adaptive_single=True)
rows = []
with torch.inference_mode():
    ids = torch.full((1, 127), 1000, dtype=torch.long, device='cuda:0')
    cache = model(ids, use_cache=True).past_key_values
    layers = [(x.keys.cpu(), x.values.cpu()) for x in cache.layers]
    for decoder in (full, one, switch): decoder.load_lane(0, layers, 1000, a.steps)
    token = 1000
    for step in range(a.steps):
        width = 1 if step < 16 or step >= 64 else 4
        for decoder, graph in ((full, full.graph), (one, one.graph),
                               (switch, switch.single_graph if width == 1 else switch.graph)):
            decoder.input_id[0].fill_(token)
            graph.replay()
        ref = full.logits_by_width[4][0, 0].float()
        top = ref.topk(2)
        row = {'step':step, 'input':token, 'switch_width':width,
               'reference_token':int(top.indices[0]), 'reference_margin':float(top.values[0]-top.values[1])}
        for name, decoder, w in [('one',one,1), ('switch',switch,width)]:
            actual = decoder.logits_by_width[w][0, 0].float()
            pos = 127+step
            diffs = [(x[0,pos].float()-y[0,pos].float()).abs().max()
                     for xs,ys in ((decoder.keys,full.keys),(decoder.values,full.values))
                     for x,y in zip(xs,ys)]
            row[name] = {'token':int(actual.argmax()), 'logit_max_abs':float((actual-ref).abs().max()),
                         'new_kv_max_abs':float(torch.stack(diffs).max()), 'position':int(decoder.position[0])}
            assert int(decoder.position[0]) == 128+step
        rows.append(row)
        token = row['reference_token']
    # Replaying inactive lanes must not alter any earlier prefix position.
    prefix_intact = all(torch.equal(decoder.keys[i][0,:127].cpu(), layers[i][0][0].transpose(0,1))
        and torch.equal(decoder.values[i][0,:127].cpu(), layers[i][1][0].transpose(0,1))
        for decoder in (full,one,switch) for i in range(36))
    assert prefix_intact
result = {'prefix_intact':prefix_intact, 'rows':rows,
          'one_argmax_matches':sum(r['one']['token']==r['reference_token'] for r in rows),
          'switch_argmax_matches':sum(r['switch']['token']==r['reference_token'] for r in rows)}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
