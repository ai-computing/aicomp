"""Real model/cache, controlled producer completion: never publish partial state."""
import copy
from concurrent.futures import Future
from dataclasses import replace
import numpy as np
import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillWorker, TorchPrefillEngine, _KVProgress
from opt_prime.accel.pd.runtime import RequestCancelled
from test_pd_hybrid import _gemma3,_prompts


class ControlledProducer:
    def __init__(self,model):
        self.engine=TorchPrefillEngine(copy.deepcopy(model),max_context=128,chunk=16,page_size=8)
        self.jobs=[]

    def submit(self,request,send_from=0,slot=None):
        out=self.engine.prefill(request,send_from)
        def dense(groups):
            mapping={layer:t[i] for g,t in zip(self.engine.kv.groups,groups) for i,layer in enumerate(g.layers)}
            return torch.stack([mapping[i] for i in range(len(mapping))])
        keys,values=dense(out.keys),dense(out.values)
        slab=torch.stack((keys,values)).view(torch.int16).numpy().view(np.uint16)
        progress=_KVProgress(slab,send_from,len(request.input_ids))
        f=Future();f.kv_progress=progress
        progress.coverage.publish(0,1,send_from,len(request.input_ids))
        self.jobs.append((f,progress,replace(out,keys=keys,values=values)))
        return f

    def finish(self,index=0,complete=True):
        f,progress,out=self.jobs[index]
        progress.coverage.publish(1,out.keys.shape[0],progress.coverage.start,progress.coverage.end)
        if complete:progress.coverage.finish()
        f.set_result(out)


def setup(device="cpu"):
    # CUDA flex-attention dot products require a KV block dimension of at least16.
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model=_gemma3().to(torch.bfloat16)
    worker=ControlledProducer(model)
    batch=PDBatcher(copy.deepcopy(model).to(device),worker,slots=2,max_context=128,page_size=(16 if device == "cuda" else 8),prefix_cache=False)
    return model,worker,batch


def baseline(model,prompt,device="cpu"):
    worker=PrefillWorker(TorchPrefillEngine(copy.deepcopy(model),max_context=128,chunk=16,page_size=8))
    try:
        b=PDBatcher(copy.deepcopy(model).to(device),worker,slots=1,max_context=128,page_size=(16 if device == "cuda" else 8),prefix_cache=False)
        f=b.submit(PDRequest('whole',prompt,8));b.run_until_idle();return f.result()
    finally:worker.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hybrid_partial_import_happens_before_completion_without_admission(device):
    model,worker,b=setup(device);prompt=_prompts(1,25,26,265)[0]
    result=b.submit(PDRequest('stream',prompt,8));b.step()
    request=b.by_id['stream'];idx=b.kv.span_index(request.slot,0,len(prompt))
    keys,_=b.kv.layer_kv(0)
    torch.testing.assert_close(keys[:,idx].cpu(),worker.jobs[0][2].keys[0])
    assert not b.kv.layer_kv(1)[0].any()
    assert b.kv.length[request.slot]==0 and not b.decoding and not result.done()
    worker.finish();b.run_until_idle()
    assert result.result()==baseline(model,prompt,device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cancellation_and_late_old_completion_do_not_corrupt_reused_cache(device):
    model,worker,b=setup(device);p,q=_prompts(2,25,26,267)
    old=b.submit(PDRequest('old',p,8));b.step();assert b.cancel('old');b.step()
    with pytest.raises(RequestCancelled):old.result()
    new=b.submit(PDRequest('new',q,8));b.step()
    worker.finish(0);worker.finish(1);b.run_until_idle()
    assert new.result()==baseline(model,q,device)
    assert b.idle and not b.by_id and sorted(b.free)==[0,1]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partial_producer_failure_preserves_other_active_decode(device):
    model,worker,b=setup(device);p,q=_prompts(2,25,26,269)
    kept=b.submit(PDRequest('kept',p,8));b.step();worker.finish(0);b.step()
    bad=b.submit(PDRequest('bad',q,8));b.step()
    worker.jobs[1][0].set_exception(RuntimeError('producer failed'))
    b.run_until_idle()
    with pytest.raises(RuntimeError,match='producer failed'):bad.result()
    assert kept.result()==baseline(model,p,device)
    assert b.idle and not b.by_id


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_unvalidated_stream_completion_cannot_publish_cache(device):
    _,worker,b=setup(device);prompt=_prompts(1,25,26,271)[0]
    result=b.submit(PDRequest('incomplete',prompt,8));b.step();worker.finish(complete=False)
    b.run_until_idle()
    with pytest.raises(RuntimeError,match='coverage'):result.result()
    assert b.idle and not b.by_id and not b.kv.cached
