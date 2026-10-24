"""CPU scheduler control, not Gemma/SDK numerics or device-local sliding KV proof."""
import copy
import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.platforms.atom.batching import AtomBatcher
from test_atom_batching import FakeBlocks, _llama, _prompts, fake_rebel
from test_atom_stream_progress import Producer

@pytest.mark.parametrize('hybrid_admission',[False,True])
def test_four_ready_requests_decode_together_but_serial_readiness_does_not(fake_rebel,hybrid_admission):
    model=_llama(); prompts=_prompts(4,20,35); results=[]; histograms=[]
    for simultaneous in [True,False]:
        fake=FakeBlocks(copy.deepcopy(model),batch=4,blocks=5)
        if hybrid_admission:
            # Exercise the real hybrid admission branch with all-global synthetic KV.
            # This fixture intentionally does not pretend to emulate sliding-window hardware.
            fake.rbln_config.cache_impl='hybrid'
            for meta in fake.rbln_config.kvcache_metas:meta['layer_type']='full_attention'
            fake.dec_attn_mask=torch.ones(4,128)
            def decode(*,local_block_tables,**kw):
                assert local_block_tables.tolist()==[[0],[1],[2],[3]]
                return fake._decode(**kw)
            decode.dec_attn_mask=fake.dec_attn_mask
            fake.decoders={4:decode}
        fake_rebel['model']=fake
        worker=Producer(fake.m)
        batcher=AtomBatcher(fake,prefill_worker=worker,local_prefill=False)
        seen=[]; original=batcher._decode_step
        def observed():
            seen.append(len(batcher.decoding));return original()
        batcher._decode_step=observed
        futures=[batcher.submit(PDRequest(str(i),p,8)) for i,p in enumerate(prompts)]
        batcher.step();assert len(worker.jobs)==4 and not seen
        if simultaneous:
            for i in range(4):worker.finish(i)
            batcher.run_until_idle()
        else:
            for i in range(4):
                worker.finish(i)
                for _ in range(16):
                    if futures[i].done():break
                    batcher.step()
                assert futures[i].done()
        assert batcher.idle and not batcher.poisoned and sorted(batcher.free)==[0,1,2,3]
        results.append([f.result() for f in futures]);histograms.append(seen)
    assert max(histograms[0])==4 and max(histograms[1])==1
    assert results[0]==results[1]
