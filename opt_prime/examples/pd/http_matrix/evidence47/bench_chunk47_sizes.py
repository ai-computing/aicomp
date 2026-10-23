"""Whole-span versus bounded chunk export, including gather; no model compute to hide copies behind.
Run only after the owned GPU serving queue has finished and both devices are idle.
"""
import json, math, time, traceback
from pathlib import Path
import torch
from opt_prime.accel.pd.paged import PagedKV
from opt_prime.accel.pd.pd_batching import _ChunkCopy, move_kv

root=Path(__file__).resolve().parent
path=root/'chunk47-size-curve.json'
if path.exists():raise RuntimeError('Refusing overwrite '+str(path))
torch.set_num_threads(4)
result=dict(passed=False,torch=torch.__version__,runtime_commit='0870dd3',chunk=256,
            method='Export wall time includes source KV gather and both host-staged DMA legs; excludes validation and warmup. No concurrent model compute. Decimal GB, rounded up to a full 64-token page.',rows=[])

@torch.inference_mode()
def run(kv,n,overlap):
    for device in [0,1]:torch.cuda.synchronize(device)
    initial={d:torch.cuda.memory_allocated(d) for d in [0,1]}
    for device in [0,1]:torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    if overlap:
        copier=_ChunkCopy(torch.device('cuda:0'),torch.device('cuda:1'),0,n)
        try:
            for offset in range(0,n,256):copier.submit(kv,offset,min(n,offset+256))
            keys,values=copier.finish()
        finally:copier.close()
    else:
        keys,values=kv.read_span(0,0,n)
        torch.cuda.synchronize(0)
        keys,values=move_kv(keys,'cuda:1'),move_kv(values,'cuda:1')
        torch.cuda.synchronize(1)
    elapsed=time.perf_counter()-start
    peaks={str(d):torch.cuda.max_memory_allocated(d)-initial[d] for d in [0,1]}
    assert keys.shape==(32,8,n,128) and values.shape==keys.shape
    assert bool((keys==0.125).all()) and bool((values==0.5).all())
    return elapsed,peaks

try:
    for gb in [0.1,0.5,1,2,4]:
        n=math.ceil(gb*1e9/(2*32*8*128*2*64))*64
        kv=PagedKV(32,1,8,128,torch.bfloat16,'cuda:0',pages=n//64,page_size=64,max_context=n,prefix_cache=False)
        assert kv.alloc(0,n) is not None
        kv.k.fill_(0.125);kv.v.fill_(0.5)
        run(kv,n,False);run(kv,n,True)
        for repeat in range(3):
            for overlap in ([False,True] if repeat%2==0 else [True,False]):
                elapsed,peaks=run(kv,n,overlap)
                row=dict(requested_gb=gb,kv_bytes=2*32*8*n*128*2,tokens=n,repeat=repeat,copy_overlap=overlap,
                         elapsed_s=elapsed,incremental_peak_device_bytes=peaks,parity=True)
                result['rows'].append(row);print(json.dumps(row),flush=True)
                path.write_text(json.dumps(result,indent=2)+'\n')
        kv.release(0);del kv
    result['passed']=True
except BaseException:
    result['error']=traceback.format_exc()
    raise
finally:
    path.write_text(json.dumps(result,indent=2)+'\n')
