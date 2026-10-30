"""Exercise real pipes/SHM: completed spans are observable before final logits."""
import sys
import time
import pytest
import torch
from opt_prime.accel.pd import PDRequest
from test_subprocess_prefill_failure import children

CHILD = '''
import json,os,sys,time
from pathlib import Path
import numpy as np
from multiprocessing import shared_memory,resource_tracker
mode=os.environ['PREFILL_TEST_MODE']
gate=Path(os.environ['PREFILL_TEST_GATE'])
def arg(name):return sys.argv[sys.argv.index(name)+1]
shm=shared_memory.SharedMemory(name=arg('--shm'))
resource_tracker.unregister(shm._name,'shared_memory')
shape=tuple(int(arg(x)) for x in ['--slots'])+(2,)+tuple(int(arg(x)) for x in ['--layers','--kv-heads','--max-context','--head-dim'])
slab=np.ndarray(shape,dtype=np.uint16,buffer=shm.buf)
print(json.dumps({'ready':True,'kv_progress':1}),flush=True)
for line in sys.stdin:
 msg=json.loads(line)
 assert msg['stream_layers'] is True
 start=msg['send_from'];n=len(msg['tokens'])-start;slot=msg['slot']
 slab[slot,:,:,:,:1]=0x3f80
 span=[0,1,start,start+1]
 if mode=='bad_id' and msg['id']==0:
  while not gate.exists():time.sleep(.005)
 print(json.dumps({'id':True if mode=='bad_id' and msg['id']==0 else msg['id'],'kv_span':span}),flush=True)
 if msg['id']==0:
  gate.with_suffix('.seen').touch()
  while not gate.exists():time.sleep(.005)
 if mode=='exit':sys.exit(8)
 if mode=='overlap':print(json.dumps({'id':msg['id'],'kv_span':span}),flush=True)
 if mode!='hole' and n>1:
  slab[slot,:,:,:,1:n]=0x4000
  print(json.dumps({'id':msg['id'],'kv_span':[0,1,start+1,start+n]}),flush=True)
 print(json.dumps({'id':msg['id'],'n':1 if mode=='short' else n,'token':17}),flush=True)
'''


def make_stream(children, mode='ok'):
    make,_,_,gate=children
    path=gate.parent/'child'
    path.write_text(f'#!{sys.executable}\n'+CHILD)
    return make(mode,stream_layers=True),gate


def first_span(future):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        rows=future.kv_progress.drain()
        if rows:return rows
        time.sleep(.005)
    pytest.fail('no partial KV notification before completion')


def test_progress_reads_actual_shm_before_final_completion(children):
    w,gate=make_stream(children)
    future=w.submit(PDRequest('stream',(1,2),1),slot=0)
    rows=first_span(future)
    assert not future.done()
    layer,start,keys,values=rows[0]
    assert (layer,start,tuple(keys.shape))==(0,0,(1,1,1,2))
    assert torch.equal(keys,torch.ones_like(keys)) and torch.equal(values,keys)
    gate.touch()
    out=future.result(timeout=3)
    assert out.token==17
    assert torch.equal(out.keys[0,0,:,0],torch.tensor([1,2],dtype=torch.bfloat16))
    assert len(future.kv_progress.drain())==1


@pytest.mark.parametrize('mode',['overlap','hole','short','exit'])
def test_bad_stream_never_completes_successfully(children,mode):
    w,gate=make_stream(children,mode)
    f=w.submit(PDRequest('bad',(1,2),1),slot=0)
    assert first_span(f)
    gate.touch()
    with pytest.raises(RuntimeError):f.result(timeout=3)
    assert not w.ready
    with pytest.raises(RuntimeError):w.submit(PDRequest('later',(1,2),1),slot=0).result(timeout=3)


def test_streaming_opt_in_requires_capable_producer(children):
    make,procs,_,_=children
    with pytest.raises(RuntimeError,match='progress'):
        make(stream_layers=True,startup_timeout=1)
    assert procs[-1].poll() is not None


def test_cancel_then_same_slot_reuse_keeps_new_stream_owned(children):
    w,gate=make_stream(children)
    first=w.submit(PDRequest('cancel',(1,2),1),slot=0)
    assert first_span(first)
    assert first.cancel()
    second=w.submit(PDRequest('reuse',(1,2),1),slot=0)
    gate.touch()
    out=second.result(timeout=3)
    assert first.cancelled() and out.token==17
    assert len(second.kv_progress.drain())==2
    assert torch.equal(out.values[0,0,:,0],torch.tensor([1,2],dtype=torch.bfloat16))


def test_progress_positions_are_absolute_for_cached_prefix(children):
    w,gate=make_stream(children)
    f=w.submit(PDRequest('suffix',(1,2),1),send_from=1,slot=0)
    rows=first_span(f)
    assert rows[0][1]==1 and rows[0][2].shape[2]==1
    gate.touch()
    assert f.result(timeout=3).start==1


def test_invalid_stream_capacity_fails_only_its_future(children):
    w,gate=make_stream(children)
    bad=w.submit(PDRequest('too-long',tuple(range(9)),1),slot=0)
    with pytest.raises(ValueError,match='suffix'):bad.result(timeout=1)
    assert w.ready
    good=w.submit(PDRequest('valid',(1,2),1),slot=0)
    assert first_span(good)
    gate.touch()
    assert good.result(timeout=3).token==17


def test_boolean_response_id_cannot_alias_another_inflight_request(children):
    w,gate=make_stream(children,'bad_id')
    first=w.submit(PDRequest('first',(1,2),1),slot=0)
    second=w.submit(PDRequest('second',(1,2),1),slot=1)
    gate.touch()
    for f in [first,second]:
        with pytest.raises(RuntimeError):f.result(timeout=3)
    assert second.kv_progress.drain()==[], 'boolean ID was routed to integer request1'
