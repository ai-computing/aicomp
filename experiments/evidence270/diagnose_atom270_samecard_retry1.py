"""Isolate ATOM B4 lane versus KV block under a fixed 2048-token prompt (#270)."""
import json
from pathlib import Path


def select_idle(batcher, lane, block):
    assert batcher.idle and sorted(batcher.free) == list(range(4))
    assert sorted(batcher.blocks.free) == list(range(4))
    assert batcher.blocks.lane_block == [None] * 4
    assert lane in batcher.free and block in batcher.blocks.free
    batcher.free[:] = [lane] + [x for x in batcher.free if x != lane]
    batcher.blocks.free[:] = [block] + [x for x in batcher.blocks.free if x != block]


def main(fixed_prefill_lane=None, run_label="single"):
    import torch
    from opt_prime.accel.pd import PDRequest
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    from atom_server270_http import validate_compiled_geometry
    root = Path(__file__).resolve().parent
    suffix = '' if fixed_prefill_lane is None else '-prefill0'
    output = root / f'atom270-qwen4b-samecard-retry1-{run_label}{suffix}.json'
    if output.exists():
        raise FileExistsError(output)
    data = json.loads((root / 'atom270-qwen4b-tp1-b4-native.json').read_text())
    prompt = data['prompts']['2048']
    reference = data['references']['2048:32']['tokens']
    torch.set_num_threads(4)
    nodes = list(Path('/dev').glob('rbln[0-9]*'))
    assert len(nodes) == 1
    engine = AtomRblnEngine(devices=[int(nodes[0].name[4:])], max_seq_len=9216,
                           batch_size=4, kv_blocks=5, cache_dir=str(root/'atom270-artifacts'))
    result = dict(complete=False, model='Qwen/Qwen3-4B', batch_size=4, kv_blocks=5,
                  input_tokens=2048, fixed_prefill_lane=fixed_prefill_lane, rows=[])
    try:
        engine.start(result['model'])
        validate_compiled_geometry(engine.model.rbln_config, 4, 5,
            artifact_config=json.loads((Path(engine.dir)/'rbln_config.json').read_text()))
        batcher = engine.batcher(prefix_cache=False, local_prefill=True)
        if fixed_prefill_lane is not None:
            assert fixed_prefill_lane == 0 and not batcher.hybrid_cache
            def prefill(lane, prompt, start=0):
                assert start == 0
                return batcher.model.prefill_decoder(
                    input_ids=torch.tensor([list(prompt)], dtype=torch.int64),
                    cache_position=torch.arange(len(prompt), dtype=torch.int32)[None],
                    batch_idx=fixed_prefill_lane,
                    block_tables=torch.tensor([batcher.blocks.lane_block[lane]], dtype=torch.int16))
            batcher._prefill = prefill
        emit, decoded = batcher._emit, batcher._decoded
        for forced in (False, True):
            for lane in range(4):
                for block in range(4):
                    select_idle(batcher, lane, block)
                    probes, placements = [], []
                    def on_emit(request, token):
                        assert request.slot == lane and batcher.blocks.lane_block[lane] == block
                        placements.append([request.slot, batcher.blocks.lane_block[lane]])
                        index = len(request.tokens)
                        return emit(request, reference[index] if forced and index < 13 else token)
                    def on_decoded(rows, lengths, logits):
                        for i, request in enumerate(rows):
                            index = len(request.tokens)
                            if index in (12,13):
                                if forced:
                                    assert request.tokens == reference[:index]
                                values, tokens = logits[i].float().topk(8)
                                probes.append(dict(index=index, top_tokens=tokens.tolist(), top_logits=values.tolist(),
                                    logit369=float(logits[i,369]), logit264=float(logits[i,264])))
                        return decoded(rows, lengths, logits)
                    batcher._emit, batcher._decoded = on_emit, on_decoded
                    future = batcher.submit(PDRequest(f'{forced}-{lane}-{block}', tuple(prompt), 14 if forced else 32))
                    batcher.run_until_idle()
                    tokens = future.result()
                    assert len(tokens) == (14 if forced else 32) and len(probes) == 2
                    row = dict(forced_prefix=forced, lane=lane, block=block, tokens=tokens,
                               matches_reference=tokens == reference[:len(tokens)], probes=probes,
                               checked_placements=len(placements))
                    result['rows'].append(row)
                    output.write_text(json.dumps(result, indent=2)+'\n')
                    print(json.dumps(row), flush=True)
        batcher._emit, batcher._decoded = emit, decoded
        result['complete'] = True
        output.write_text(json.dumps(result, indent=2)+'\n')
    finally:
        engine.close()


if __name__ == '__main__':
    import sys
    assert sys.argv[1] in ('a1','b1','b2','a2')
    main(0 if sys.argv[1].startswith('b') else None, sys.argv[1])
