"""Producer emits immutable CPU KV only after copying, before final logits."""
import numpy as np
import pytest
import torch
from opt_prime.accel.pd import prefill_server as server


def test_source_suffix_written_before_notification_and_complete_before_return():
    slab=np.zeros((1,2,2,1,8,2),dtype=np.uint16)
    messages=[]
    class Engine:
        def prefill_stream(self,tokens,emit):
            for layer in range(2):
                k=torch.full((1,1,len(tokens),2),layer+1,dtype=torch.bfloat16)
                emit(layer,0,k,k+2)
                assert len(messages)==layer+1
            return torch.tensor([1.,2.])
    def publish(msg):
        layer=msg['kv_span'][0]
        got=torch.from_numpy(slab[0,0,layer,:,:2].view(np.int16)).view(torch.bfloat16)
        assert torch.all(got==layer+1)
        messages.append(msg)
    logits=server.stream_result(Engine(),dict(id=7,tokens=[1,2,3,4],send_from=2,slot=0),slab,publish)
    assert logits.tolist()==[1.,2.]
    assert messages==[dict(id=7,kv_span=[i,i+1,2,4]) for i in range(2)]
    assert not slab[:,:,:,:,2:].any()


@pytest.mark.parametrize('mode',['overlap','hole','geometry'])
def test_invalid_source_never_completes(mode):
    slab=np.zeros((1,2,2,1,8,2),dtype=np.uint16);messages=[]
    class Engine:
        def prefill_stream(self,tokens,emit):
            k=torch.ones((1,1,2,2),dtype=torch.bfloat16)
            emit(0,0,k,k)
            if mode=='overlap':emit(0,0,k*3,k*3)
            if mode=='geometry':emit(1,0,k,torch.ones((1,2,2,2)))
            return torch.zeros(2)
    with pytest.raises(ValueError):
        server.stream_result(Engine(),dict(id=0,tokens=[1,2],send_from=0,slot=0),slab,messages.append)
    assert len(messages)==1
    assert not slab[:,:,1].any()


def test_rngd_full_emits_each_group_before_next_advance():
    emitted=[]
    class Stage:
        Ks=[];Vs=[];x=torch.tensor([1.])
        def advance(self,count):
            assert count==1
            assert len(emitted)==len(self.Ks)
            self.Ks.append(torch.ones((1,1,2,4))*len(self.Ks))
            self.Vs.append(torch.ones((1,1,4,2))*len(self.Vs))
            return len(self.Ks)==2
    class Engine:
        def begin_device_prefill(self,tokens):return Stage()
        def _logits(self,x,n):return np.array([1.,2.])
    wrapper=server._RngdFull.__new__(server._RngdFull)
    wrapper.engine=Engine();wrapper.widths=[4]
    def emit(layer,start,k,v):
        assert (layer,start)==(len(emitted),0)
        assert k.shape==v.shape==(1,1,3,2)
        assert k.device.type==v.device.type=='cpu'
        emitted.append(layer)
    assert wrapper.prefill_stream([1,2,3],emit).tolist()==[1.,2.]
    assert emitted==[0,1]


def test_real_server_and_receiver_stream_before_completion(children):
    import sys,time
    from opt_prime.accel.pd import PDRequest
    make,_,_,gate=children
    child=gate.parent/'child'
    child.write_text(f'#!{sys.executable}\n'+'''
import sys,os,time
sys.path.insert(0,os.getcwd())
from pathlib import Path
import torch
from opt_prime.accel.pd import prefill_server as server
class Engine:
    def __init__(self,args):pass
    def prefill_stream(self,tokens,emit):
        k=torch.ones((1,1,1,2),dtype=torch.bfloat16)
        emit(0,0,k,k)
        gate=Path(os.environ['PREFILL_TEST_GATE'])
        while not gate.exists():time.sleep(.005)
        emit(0,1,k*2,k*2)
        return torch.tensor([0.,1.])
server._RngdFull=Engine
sys.argv=[sys.argv[0]]+sys.argv[3:]+['--engine','rngd-full','--model',str(Path.cwd())]
server.main()
''')
    w=make(stream_layers=True)
    f=w.submit(PDRequest('source',(1,2),1),slot=0)
    deadline=time.monotonic()+5
    rows=[]
    while not rows and time.monotonic()<deadline:
        rows=f.kv_progress.drain();time.sleep(.005)
    assert rows and not f.done()
    torch.testing.assert_close(rows[0][2],torch.ones((1,1,1,2),dtype=torch.bfloat16))
    gate.touch();out=f.result(timeout=5)
    assert out.token==1
    assert out.keys[0,0,:,0].tolist()==[1.,2.]
    assert f.kv_progress.coverage.finished


from test_subprocess_prefill_failure import children


def test_fully_cached_suffix_publishes_no_spans():
    slab=np.zeros((1,2,2,1,8,2),dtype=np.uint16);messages=[]
    class Engine:
        def prefill_stream(self,tokens,emit):
            k=torch.ones((2,1,2,2),dtype=torch.bfloat16)
            emit(0,0,k,k)
            return torch.ones(2)
    server.stream_result(Engine(),dict(id=0,tokens=[1,2],send_from=2,slot=0),slab,messages.append)
    assert not messages and not slab.any()


@pytest.mark.parametrize('field,value',[('slot',True),('slot',-1),('send_from',True),('send_from',3),('id',True)])
def test_bad_reservation_rejected_before_engine(field,value):
    class Engine:
        def prefill_stream(self,*args):pytest.fail('invalid reservation reached engine')
    req=dict(id=0,tokens=[1,2],send_from=0,slot=0);req[field]=value
    with pytest.raises(ValueError):
        server.stream_result(Engine(),req,np.zeros((1,2,2,1,8,2),dtype=np.uint16),lambda _:None)


def test_rngd_full_legacy_head_exports_logits_without_argmax_loss():
    from types import SimpleNamespace
    calls=[]
    def head(*args):
        calls.append(args)
        return (torch.tensor([[1.,3.,2.,99.]],dtype=torch.bfloat16),)
    wrapper=server._RngdFull.__new__(server._RngdFull)
    wrapper.engine=SimpleNamespace(lmh=head,lm_norm_d=1,lm_eps_d=2,lm_nh_d=3,wlm_d=4,V=3)
    result=wrapper._last_logits(torch.tensor([[5.],[6.]]),2)
    assert result.tolist()==[1.,3.,2.] and result.dtype==torch.float32
    assert len(calls)==1 and calls[0][0].tolist()==[6.]
