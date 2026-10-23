"""Adapter ordering/layout checks without compiling an accelerator graph."""
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace,ModuleType
import pytest
import torch


def adapter():
    spec=importlib.util.spec_from_file_location('streaming_factories',Path(__file__).parents[1]/'examples/pd/streaming_factories.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_qwen_stream_exports_model_layer_order_before_next_call():
    m=adapter();spans=[]
    e=SimpleNamespace(width=4,embedding=torch.arange(20).reshape(10,2).float(),device='cpu',
                      layers=[(0,),(1,)],fixed=[],rotation=0,mask=0,chip=0,s={'rms_norm_eps':1e-6},
                      norm=torch.ones(2),head=torch.eye(2),sync=lambda _:None)
    def layer(x,i,*unused):
        assert len(spans)==i
        kv=torch.arange(16).reshape(2,1,2,4).to(torch.bfloat16)+i*16
        return x+1,kv
    e.module=layer
    def emit(i,start,k,v):
        assert k.shape==v.shape==(1,1,2,2)
        assert (i,start)==(len(spans),0)
        spans.append((k.clone(),v.clone()))
    logits=m.qwen_stream(e,[1,2],emit)
    z=e.embedding[2]+2
    torch.testing.assert_close(logits,z*torch.rsqrt(z.square().mean()+1e-6))
    assert spans[0][0][0,0].tolist()==[[0.,4.],[1.,5.]]
    assert spans[1][1][0,0].tolist()==[[24.,28.],[25.,29.]]


def test_koni_stream_uses_partitioned_stage_and_host_head():
    m=adapter();spans=[]
    class Stage:
        def __init__(self):self.Ks=[];self.Vs=[];self.x=torch.ones((4,2))
        def advance(self):
            assert len(spans)==len(self.Ks)
            self.Ks.append(torch.ones((1,1,2,4))*len(self.Ks));self.Vs.append(torch.ones((1,1,4,2))*len(self.Vs))
            return len(self.Ks)==2
    e=SimpleNamespace(width=4,config={'vocab_size':10,'rms_norm_eps':1e-6},engine=SimpleNamespace(begin_device_prefill=lambda _:Stage()),
                      chip=0,sync=lambda _:None,norm=torch.ones(2),head=torch.eye(2))
    def emit(i,start,k,v):
        assert k.shape==v.shape==(1,1,2,2)
        assert (i,start)==(len(spans),0);spans.append(i)
    torch.testing.assert_close(m.koni_stream(e,[1,2],emit),torch.ones(2)*torch.rsqrt(torch.tensor(1.+1e-6)))
    assert spans==[0,1]


def test_gemma_stream_keeps_padding_and_local_global_inputs(monkeypatch):
    m=adapter();spans=[];seen=[]
    helpers=ModuleType('gemma_parts')
    helpers.initial_hidden=lambda embedding,ids:embedding[ids]
    helpers.final_logits=lambda hidden,norm,head,eps:hidden.float() @ head.float().T
    monkeypatch.setitem(sys.modules,'gemma_parts',helpers)
    e=SimpleNamespace(width=4,config={'vocab_size':10,'rms_norm_eps':1e-6,'pad_token_id':0},device='cpu',
        weights={'model.embed_tokens.weight':torch.arange(20).reshape(10,2).float(),'model.norm.weight':torch.ones(2),'lm_head.weight':torch.eye(2)},
        params=[(0,),(1,)],positions={True:(11,),False:(22,)},local=[True,False],chip=0,sync=lambda _:None)
    def layer(x,i,position):
        assert len(spans)==i;seen.append(position)
        assert x[-1].tolist()==[float(i),float(i+1)]
        return x+1,torch.ones((4,1,2))*i,torch.ones((4,1,2))*(i+3)
    e.compiled=layer
    def emit(i,start,k,v):
        assert (i,start)==(len(spans),0) and k.shape==v.shape==(1,1,2,2)
        spans.append(i)
    assert m.gemma_stream(e,[1,2],emit).tolist()==[6.,7.]
    assert seen==[11,22]


@pytest.mark.parametrize('kind',['qwen','gemma','koni'])
def test_bad_prompt_rejected_before_computation(kind):
    m=adapter();e=SimpleNamespace(width=4)
    with pytest.raises(ValueError):getattr(m,kind+'_stream')(e,[],lambda *a:None)
