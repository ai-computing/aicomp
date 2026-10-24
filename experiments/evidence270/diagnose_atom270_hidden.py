"""SDK decode replay without AtomBatcher: identical forced tokens, positions and prefill."""
import json
from pathlib import Path


def decode_inputs(active, reference, length, index):
    assert 1 <= index <= len(reference)
    assert active and len(set(active)) == len(active) and all(0 <= x < 4 for x in active)
    return ([reference[index-1] if i in active else 0 for i in range(4)],
            [length+index-1 if i in active else 0 for i in range(4)],
            [[i if i in active else 4] for i in range(4)])


def main():
    import torch
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    from atom_server270_http import validate_compiled_geometry
    root=Path(__file__).resolve().parent
    output=root/'atom270-qwen4b-hidden-sdk.json'
    if output.exists():raise FileExistsError(output)
    data=json.loads((root/'atom270-qwen4b-tp1-b4-native.json').read_text())
    nodes=list(Path('/dev').glob('rbln[0-9]*'));assert len(nodes)==1
    torch.set_num_threads(4)
    engine=AtomRblnEngine(devices=[int(nodes[0].name[4:])],max_seq_len=9216,
                         batch_size=4,kv_blocks=5,cache_dir=str(root/'atom270-artifacts-hidden'))
    result=dict(complete=False,model='Qwen/Qwen3-4B',rows=[])
    try:
        engine.start(result['model'])
        model=engine.model;decoder=model.decoders[4]
        validate_compiled_geometry(model.rbln_config,4,5,
            artifact_config=json.loads((Path(engine.dir)/'rbln_config.json').read_text()))
        result['compiled_dir_name']=Path(engine.dir).name
        result['decoder_class']=type(decoder).__name__
        with torch.inference_mode():
            for length,target in [(2048,13),(512,50)]:
                prompt=data['prompts'][str(length)]
                reference=data['references'][f'{length}:128']['tokens']
                for active in [[0,1,2,3]]:
                    prefill=[]
                    for block in range(5):
                        ids=prompt if block<4 else [0]
                        out=model.prefill_decoder(input_ids=torch.tensor([ids],dtype=torch.int64),
                            cache_position=torch.arange(len(ids),dtype=torch.int32)[None],batch_idx=0,
                            block_tables=torch.tensor([block],dtype=torch.int16))
                        assert int(out.padded_cache_lengths or 0)==0
                        prefill.append(int(out.logits[0,-1].argmax()))
                    assert len(set(prefill[:4]))==1
                    probes=[]
                    for index in range(1,target+1):
                        tokens,positions,blocks=decode_inputs(active,reference,length,index)
                        out=decoder(input_ids=torch.tensor(tokens,dtype=torch.int64)[:,None].contiguous(),
                            cache_position=torch.tensor(positions,dtype=torch.int32)[:,None].contiguous(),
                            block_tables=torch.tensor(blocks,dtype=torch.int16))
                        logits=out.logits[:,-1].float()
                        assert torch.isfinite(logits).all()
                        if index in [target-1,target]:
                            rows=[]
                            for lane in active:
                                values,ids=logits[lane].topk(8)
                                rows.append(dict(lane=lane,token=int(logits[lane].argmax()),
                                    top_tokens=ids.tolist(),top_logits=values.tolist(),
                                    reference_token=reference[index],reference_logit=float(logits[lane,reference[index]]),
                                    max_abs_from_lane0=float((logits[lane]-logits[0]).abs().max()) if 0 in active else None))
                            hidden=out.hidden_states
                            assert hidden is not None and len(hidden)==37
                            hidden_stats=[]
                            for boundary,h in enumerate(hidden):
                                assert h.shape[0]==4 and torch.isfinite(h).all()
                                hidden_stats.append(dict(boundary=boundary,shape=list(h.shape),dtype=str(h.dtype),max_abs_from_lane0=[float((h[i].float()-h[0].float()).abs().max()) for i in range(4)]))
                            probes.append(dict(index=index,logits_dtype=str(out.logits.dtype),rows=rows,hidden=hidden_stats))
                    row=dict(input_tokens=length,active_lanes=active,prefill_tokens=prefill[:4],probes=probes)
                    result['rows'].append(row)
                    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
        result['complete']=True;output.write_text(json.dumps(result,indent=2)+'\n')
    finally:engine.close()


if __name__=='__main__':main()
