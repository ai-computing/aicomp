"""Explicit local factories preserve checkpoint inputs and validate slab geometry."""
import json,sys
from types import SimpleNamespace,ModuleType
import pytest
import torch
from safetensors.torch import save_file
from opt_prime.accel.pd import prefill_server as server


def setup(tmp_path,monkeypatch):
    config=dict(num_hidden_layers=2,num_key_value_heads=1,num_attention_heads=2,hidden_size=8)
    (tmp_path/'config.json').write_text(json.dumps(config))
    save_file({'w':torch.ones(2,dtype=torch.bfloat16)},tmp_path/'model.safetensors')
    module=ModuleType('test_local_prefill')
    calls=[]
    def factory(c,w,**kw):
        calls.append((c,w,kw));return SimpleNamespace(prefill=lambda tokens:None)
    module.make=factory;monkeypatch.setitem(sys.modules,module.__name__,module)
    args=SimpleNamespace(model=str(tmp_path),engine_factory='test_local_prefill:make',width='32',chip=0,
                         slots=1,layers=2,kv_heads=1,head_dim=4,max_context=64)
    return args,calls


def test_factory_receives_same_fp32_weights_and_config(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch)
    engine=server.factory_engine(args)
    assert callable(engine.prefill)
    assert len(calls)==1
    c,w,kw=calls[0]
    assert c['hidden_size']==8 and w['w'].dtype==torch.float32
    assert w['w'].tolist()==[1.,1.] and kw==dict(width=32,chip=0)


@pytest.mark.parametrize('field,value',[('head_dim',8),('layers',3),('width','0'),('width','128'),('engine_factory','bad')])
def test_factory_rejects_bad_setup_before_construction(tmp_path,monkeypatch,field,value):
    args,calls=setup(tmp_path,monkeypatch);setattr(args,field,value)
    with pytest.raises(ValueError):server.factory_engine(args)
    assert not calls


@pytest.mark.parametrize('stream',[False,True])
def test_production_server_loads_factory_over_real_pipes(children,stream):
    import time
    from opt_prime.accel.pd import PDRequest
    make,_,_,gate=children
    root=gate.parent
    (root/'config.json').write_text(json.dumps(dict(num_hidden_layers=1,num_key_value_heads=1,head_dim=2)))
    save_file({'w':torch.ones(1)},root/'model.safetensors')
    (root/'child').write_text(f'#!{sys.executable}\n'+'''
import os,sys,time
from pathlib import Path
from types import ModuleType
sys.path.insert(0,os.getcwd())
import torch
from opt_prime.accel.pd import prefill_server as server
root=Path(os.environ['PREFILL_TEST_GATE']).parent
module=ModuleType('local_fixture')
class Engine:
    def __init__(self,config,weights,**kw):
        assert weights['w'].tolist()==[1.] and kw['width']==8
    def prefill(self,tokens):
        assert isinstance(tokens,list), "factory expects token list"
        k=torch.tensor([1.,2.],dtype=torch.bfloat16).reshape(1,1,2,1).expand(1,1,2,2)
        return k,k,torch.tensor([0.,1.])
    def prefill_stream(self,tokens,emit):
        k,v,logits=self.prefill(tokens)
        emit(0,0,k[:,:,:1],v[:,:,:1])
        while not (root/'gate').exists():time.sleep(.005)
        emit(0,1,k[:,:,1:],v[:,:,1:])
        return logits
module.make=Engine;sys.modules[module.__name__]=module
sys.argv=[sys.argv[0]]+sys.argv[3:]+['--engine','factory','--engine-factory','local_fixture:make','--model',str(root),'--width','8']
server.main()
''')
    worker=make(stream_layers=stream)
    future=worker.submit(PDRequest('factory',(1,2),1),slot=0)
    if stream:
        deadline=time.monotonic()+5;rows=[]
        while not rows and time.monotonic()<deadline:
            rows=future.kv_progress.drain();time.sleep(.005)
        assert rows and not future.done()
        gate.touch()
    result=future.result(timeout=5)
    assert result.token==1 and result.keys[0,0,:,0].tolist()==[1.,2.]


from test_subprocess_prefill_failure import children
