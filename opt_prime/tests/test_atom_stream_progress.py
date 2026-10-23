"""Partial ATOM cache state stays private and never overwrites an active lane."""
import copy
from concurrent.futures import Future
from dataclasses import replace
import numpy as np
import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import TorchPrefillEngine,_KVProgress
from opt_prime.accel.platforms.atom.batching import AtomBatcher,InvalidKVImport
from test_atom_batching import FakeBlocks,_llama,_prompts,fake_rebel


class Producer:
    def __init__(self,model):
        self.engine=TorchPrefillEngine(copy.deepcopy(model),max_context=128,chunk=16,page_size=16)
        self.jobs=[]
    def submit(self,request,send_from=0,slot=None):
        out=self.engine.prefill(request,send_from)
        k,v=[(t if torch.is_tensor(t) else torch.cat(t)).to(torch.bfloat16) for t in (out.keys,out.values)]
        slab=torch.stack((k,v)).view(torch.int16).numpy().view(np.uint16)
        progress=_KVProgress(slab,send_from,len(request.input_ids));f=Future();f.kv_progress=progress
        progress.publish(0,1,send_from,len(request.input_ids))
        self.jobs.append((f,progress,replace(out,keys=k,values=v)))
        return f
    def finish(self,index):
        f,p,out=self.jobs[index]
        p.publish(1,out.keys.shape[0],p.coverage.start,p.coverage.end);p.coverage.finish();f.set_result(out)


def setup(holder,*,cache=False,prefetch=False):
    model=_llama();fake=FakeBlocks(model,batch=1,blocks=4 if cache else 1)
    holder['model']=fake;worker=Producer(model)
    return fake,worker,AtomBatcher(fake,prefix_cache=cache,prefill_worker=worker,local_prefill=False,remote_prefetch=prefetch)


def test_partial_import_before_completion_does_not_admit_decode(fake_rebel):
    fake,w,b=setup(fake_rebel);prompt=_prompts(1,20,35)[0]
    f=b.submit(PDRequest('a',prompt,3));b.step()
    torch.testing.assert_close(fake.kv[0][0,:,:len(prompt)],w.jobs[0][2].keys[0].float())
    assert not fake.kv[2].any() and b.kv.length[0]==0 and not b.decoding and not f.done()
    w.finish(0);b.run_until_idle();assert len(f.result())==3 and b.idle


def test_prefetch_progress_never_overwrites_active_decode(fake_rebel):
    fake,w,b=setup(fake_rebel,prefetch=True);p,q=_prompts(2,20,35)
    first=b.submit(PDRequest('first',p,12));second=b.submit(PDRequest('second',q,3))
    b.step();w.finish(0);b.step()
    assert len(w.jobs)==2 and b._prefetched.slot==-1 and len(b.decoding)==1
    block=b.blocks.lane_block[0];snapshot=[k[block,:,:len(p)].clone() for k in fake.kv]
    b.step()
    for k,old in zip(fake.kv,snapshot):torch.testing.assert_close(k[block,:,:len(p)],old)
    w.finish(1);b.run_until_idle();assert len(first.result())==12 and len(second.result())==3
    assert b.idle


def test_failed_prefill_does_not_publish_prefix(fake_rebel):
    fake,w,b=setup(fake_rebel,cache=True);p=_prompts(1,32,33)[0]
    f=b.submit(PDRequest('bad',p,3));b.step()
    w.jobs[0][0].set_exception(RuntimeError('producer failed'))
    b.run_until_idle()
    with pytest.raises(RuntimeError):f.result()
    assert not b.blocks.index and not b.blocks.entries, 'incomplete prompt published as reusable prefix'
    assert not b.blocks.drain_events()


def test_layer_span_rejects_bad_geometry_before_any_write(fake_rebel):
    fake,w,b=setup(fake_rebel);b.blocks.admit(0,(1,2))
    k=torch.ones((1,99,2,4))
    with pytest.raises(InvalidKVImport):b.blocks.import_layer_span(0,0,0,k,k)
    assert not any(t.any() for t in fake.kv)


def test_layer_span_only_touches_selected_layers(fake_rebel):
    fake,w,b=setup(fake_rebel);b.blocks.admit(0,(1,2))
    c=fake.config;k=torch.ones((1,c.num_key_value_heads,2,c.head_dim))
    b.blocks.import_layer_span(0,0,1,k,k*2)
    assert not fake.kv[0].any() and not fake.kv[1].any()
    assert torch.all(fake.kv[2][0,:,:2]==1) and torch.all(fake.kv[3][0,:,:2]==2)


def test_streamed_decode_matches_whole_import_of_identical_kv(fake_rebel):
    from types import SimpleNamespace
    fake,w,b=setup(fake_rebel);prompt=_prompts(1,20,35)[0]
    f=b.submit(PDRequest('stream',prompt,8));b.step();w.finish(0);b.run_until_idle()
    control=FakeBlocks(copy.deepcopy(fake.m),batch=1,blocks=1);fake_rebel['model']=control
    def submit(*args,**kw):
        result=Future();result.set_result(w.jobs[0][2]);return result
    baseline=AtomBatcher(control,prefix_cache=False,prefill_worker=SimpleNamespace(submit=submit),local_prefill=False)
    g=baseline.submit(PDRequest('whole',prompt,8));baseline.run_until_idle()
    assert f.result()==g.result()


def test_cancelled_partial_import_never_publishes_or_overwrites_reused_lane(fake_rebel):
    from opt_prime.accel.pd.runtime import RequestCancelled
    fake,w,b=setup(fake_rebel,cache=True);p,q=_prompts(2,32,33)
    old=b.submit(PDRequest('old',p,3));b.step();assert b.cancel('old');b.step()
    with pytest.raises(RequestCancelled):old.result()
    assert not b.blocks.index
    new=b.submit(PDRequest('new',q,3));b.step()
    snapshot=[t.clone() for t in fake.kv]
    w.finish(0);b.step()
    for t,s in zip(fake.kv,snapshot):torch.testing.assert_close(t,s)
    w.finish(1);b.run_until_idle();assert len(new.result())==3 and b.idle


def test_partial_sdk_failure_poisons_owner(fake_rebel):
    fake,w,b=setup(fake_rebel)
    def fail(*args):raise RuntimeError('SDK failed during partial write')
    b.blocks.io_rt.set_kv_cache_for_layer=fail
    b.submit(PDRequest('fail',_prompts(1,20,35)[0],3))
    with pytest.raises(RuntimeError,match='SDK failed'):b.step()
    assert b.poisoned and not b.blocks.index


def test_hybrid_layer_import_uses_lane_for_sliding_and_block_for_global(fake_rebel):
    from types import SimpleNamespace
    fake=FakeBlocks(_llama(),batch=2,blocks=4);fake.rbln_config.cache_impl='hybrid'
    for i,m in enumerate(fake.rbln_config.kvcache_metas):
        m['layer_type']='sliding_attention' if i<2 else 'full_attention'
        m['shape']=([2,m['shape'][1],4,m['shape'][3]] if i<2 else m['shape'])
    fake_rebel['model']=fake
    b=AtomBatcher(fake,prefix_cache=False,prefill_worker=SimpleNamespace(),local_prefill=False)
    # First free global block differs from lane1.
    b.blocks.admit(1,tuple(range(10)))
    c=fake.config;k=torch.arange(10).reshape(1,1,10,1).float().expand(1,c.num_key_value_heads,10,c.head_dim)
    writes=[]
    b.blocks.io_rt.get_kv_cache_size_for_layer=lambda name: int(np.prod(next(m['shape'] for m in b.blocks.metas if m['name']==name)))*4
    b.blocks.io_rt.set_kv_cache_for_layer=lambda name,block,offset,n,buf:writes.append((name,block,offset,n,buf[block,:,offset:offset+n].copy()))
    b.blocks.import_layer_span(1,0,0,k,k)
    b.blocks.import_layer_span(1,0,1,k,k)
    assert [row[1:4] for row in writes]==[(1,0,4),(1,0,4),(0,0,10),(0,0,10)]
    assert writes[0][4][0,:,0].tolist()==[6.,7.,8.,9.]
    assert writes[2][4][0,:,0].tolist()==list(map(float,range(10)))


def test_multilane_hybrid_defers_partial_cache_until_decode_admission(fake_rebel):
    from test_atom_hybrid import HybridFake
    fake=HybridFake();fake_rebel['model']=fake
    for meta in fake.rbln_config.kvcache_metas[:2]:meta['layer_type']='sliding_attention'
    w=Producer(fake.m);b=AtomBatcher(fake,prefill_worker=w,local_prefill=False)
    f=b.submit(PDRequest('hybrid',_prompts(1,20,35)[0],3));b.step()
    assert b.stats['progress_spans']==0, 'idle hybrid decode lanes can overwrite partial local KV'
    w.finish(0);b.run_until_idle();assert len(f.result())==3 and b.stats['progress_spans']==2


@pytest.mark.parametrize('outcome', ['error', 'cancel'])
def test_prefetched_partial_failure_preserves_active_decode_and_source_ownership(fake_rebel, outcome):
    from opt_prime.accel.pd.runtime import RequestCancelled
    fake, w, b = setup(fake_rebel, prefetch=True)
    p, q = _prompts(2, 20, 35)
    first = b.submit(PDRequest('first', p, 12))
    second = b.submit(PDRequest('second', q, 3))
    b.step(); w.finish(0); b.step()
    assert b.decoding and b._prefetched.slot == -1
    pending, progress, _ = w.jobs[1]
    assert progress.coverage.pending and not pending.done()
    imported = b.stats['progress_spans']
    if outcome == 'error':
        pending.set_exception(RuntimeError('partial source failed'))
    else:
        assert b.cancel('second')
    while not first.done():
        b.step()
        assert b.stats['progress_spans'] == imported
    if outcome == 'cancel':
        assert not pending.done() and not b.idle
        w.finish(1)
    b.run_until_idle()
    with pytest.raises(RuntimeError if outcome == 'error' else RequestCancelled):
        second.result()
    assert b.idle and not b.poisoned and b.free == [0]
    # Compare with the same BF16 source/cache path without the failed prefetch.
    _, control_worker, control = setup(fake_rebel)
    expected = control.submit(PDRequest('reference', p, 12))
    control.step(); control_worker.finish(0); control.run_until_idle()
    assert first.result() == expected.result()
