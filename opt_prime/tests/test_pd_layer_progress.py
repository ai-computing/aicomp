"""Partial KV must cover every layer/position without overlaps before publication."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).parents[1]/'opt_prime/accel/pd/layer_progress.py'

def tracker(*args):
    assert path.exists(), 'layer-progress validator is not implemented'
    spec=importlib.util.spec_from_file_location('layer_progress',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.LayerProgress(*args)


def test_partial_prefix_and_out_of_order_layers_require_complete_coverage():
    p=tracker(3,5,9)
    p.publish(1,3,5,9)
    p.publish(0,1,7,9)
    assert p.drain()==[(1,3,5,9),(0,1,7,9)]
    with pytest.raises(ValueError,match='incomplete'):p.finish()
    p.publish(0,1,5,7)
    p.finish()
    assert p.drain()==[(0,1,5,7)]
    assert p.drain()==[]
    with pytest.raises(ValueError,match='closed'):p.publish(0,1,5,7)


def test_overlap_rejection_is_atomic_across_layers():
    p=tracker(3,0,4)
    p.publish(1,2,0,2)
    with pytest.raises(ValueError,match='overlap'):p.publish(0,3,1,3)
    # If the rejected span mutated layer0, this valid replacement would fail.
    p.publish(0,1,0,4)
    p.publish(1,2,2,4)
    p.publish(2,3,0,4)
    p.finish()
    assert len(p.drain())==4


@pytest.mark.parametrize('span',[(-1,1,0,4),(0,4,0,4),(0,1,-1,4),
    (0,1,0,5),(0,0,0,4),(0,1,2,2),(True,2,0,4),(0,1,0.0,4)])
def test_invalid_progress_never_enters_queue(span):
    p=tracker(3,0,4)
    with pytest.raises(ValueError):p.publish(*span)
    assert p.drain()==[]


def test_empty_uncached_suffix_needs_no_transfer():
    p=tracker(3,8,8)
    p.finish()
    assert p.drain()==[]


def test_partial_hybrid_write_preserves_other_layers_and_positions():
    import torch
    from test_pd_dense_hybrid import pool
    cache=pool()
    assert hasattr(cache,'write_layer_span'), 'partial receiver import is not implemented'
    keys=torch.arange(20).reshape(2,1,5,2).float()+1
    cache.write_layer_span(0,3,1,keys,keys+100)
    idx=cache.span_index(0,3,8)
    for layer in [1,2]:
        k,v=cache.layer_kv(layer)
        torch.testing.assert_close(k[:,idx],keys[layer-1])
        torch.testing.assert_close(v[:,idx],keys[layer-1]+100)
    assert not cache.layer_kv(0)[0].any()
    assert cache.length[0]==0 and not cache.cached


@pytest.mark.parametrize('slot,start,layer',[(0,10,0),(0,-1,0),(1,0,0),(0,0,2)])
def test_partial_write_rejects_unreserved_positions_and_unknown_layers(slot,start,layer):
    import torch
    from test_pd_dense_hybrid import pool
    cache=pool()
    assert hasattr(cache,'write_layer_span'), 'partial receiver import is not implemented'
    keys=torch.ones(2,1,5,2)
    with pytest.raises(ValueError):cache.write_layer_span(slot,start,layer,keys,keys)
    assert all(not t.any() for t in cache.gk+cache.gv)


def test_partial_write_rejects_later_group_geometry_before_mutation():
    import torch
    from opt_prime.accel.pd.paged import PagedKV
    from opt_prime.accel.pd.torch_models import KVGroup
    cache=PagedKV(2,1,1,2,torch.float32,'cpu',pages=4,page_size=4,max_context=16,
        groups=(KVGroup('a',None,1,2,(0,)),KVGroup('b',None,2,2,(1,))))
    cache.alloc(0,12)
    assert hasattr(cache,'write_layer_span'), 'partial receiver import is not implemented'
    keys=torch.ones(2,1,5,2)
    with pytest.raises(ValueError):cache.write_layer_span(0,0,0,keys,keys)
    assert all(not t.any() for t in cache.gk+cache.gv)


def test_partial_write_cannot_overwrite_an_admitted_prefix():
    import torch
    from test_pd_dense_hybrid import pool
    cache=pool()
    cache.length[0]=4
    keys=torch.ones(1,1,5,2)
    with pytest.raises(ValueError,match='prefix'):cache.write_layer_span(0,3,0,keys,keys)
    assert all(not t.any() for t in cache.gk+cache.gv)
