"""Task-owned hardware test: three active survivors, fourth-lane cancellation/reuse/death after observed device import.

Run through container_bootstrap270.py using the experiment's existing environment.
Only the producer subprocess created here may be killed. No device regrouping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ['model','family','cache-dir','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--worker-python',default='./furiosa_prefill_python.sh')
    parser.add_argument('--tensor-parallel-size',type=int,choices=[1,2],default=1)
    args=parser.parse_args();output=Path(args.output)
    assert args.family!='gemma', 'Hybrid B4 intentionally defers partial import'
    if output.exists():raise RuntimeError('Refuse output overwrite')
    from transformers import AutoTokenizer, AutoConfig
    from opt_prime.accel.pd import PDRequest
    from opt_prime.accel.pd.runtime import RequestCancelled
    from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    from atom_server270_http import check_shm_capacity,validate_compiled_geometry,observe_decode_batches
    c=AutoConfig.from_pretrained(args.model,local_files_only=True).get_text_config()
    check_shm_capacity(c,4,8192)
    nodes=list(Path('/dev').glob('rbln[0-9]*'));assert len(nodes)==args.tensor_parallel_size
    engine=AtomRblnEngine(devices=sorted(int(n.name[4:]) for n in nodes),max_seq_len=9216,batch_size=4,kv_blocks=5,cache_dir=args.cache_dir)
    artifact=Path(engine._dir(args.model))
    assert (artifact/'config.json').exists(),'Load-only test: existing compiled artifact required'
    if args.family=='gemma':
        provenance=json.loads((artifact/'causal_export_provenance.json').read_text())
        assert provenance['model']==args.model and provenance['batch']==4
    result=dict(model=args.model,batch=4,kv_blocks=5,width=8192,stage='starting',cases=[],
                scope='Direct batcher hardware fault test, not HTTP, throughput or semantic quality. Three survivors use512 output tokens; victim/replacement16.')
    result['source_sha256']={name:hashlib.sha256((Path('stream265-code/opt_prime')/name).read_bytes()).hexdigest() for name in ['accel/platforms/atom/batching.py','accel/pd/pd_batching.py','accel/pd/prefill_server.py']}
    worker=None
    def save():output.write_text(json.dumps(result,indent=2)+'\n')
    def step_until(batch,condition,timeout=600):
        deadline=time.monotonic()+timeout
        while not condition():
            assert time.monotonic()<deadline,'Owner deadline exceeded'
            batch.step()
    try:
        start=time.perf_counter();engine.start(args.model)
        result['decoder_load_s']=time.perf_counter()-start
        validate_compiled_geometry(engine.model.rbln_config,4,5,tensor_parallel_size=args.tensor_parallel_size,artifact_config=json.loads((artifact/'rbln_config.json').read_text()))
        assert not engine.compiled,'Fault test must not compile new model'
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        base=tok.encode('Seoul is the capital of South Korea. Count one two three.',add_special_tokens=True)
        def prompt(n):return tuple((base*((n+len(base)-1)//len(base)))[:n])
        short,long=prompt(512),prompt(8192)
        start=time.perf_counter()
        worker=SubprocessPrefillWorker(args.worker_python,['--engine','factory','--engine-factory',f'streaming_factories:{args.family}','--model',args.model,'--width','8192','--chip','0'],slots=4,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,max_context=8192,env=dict(os.environ),stream_layers=True)
        result['producer_startup_s']=time.perf_counter()-start
        save()
        def batcher():return engine.batcher(prefix_cache=False,prefill_worker=worker,local_prefill=False,remote_prefetch=False)
        def survivors(batch,prefix):return [batch.submit(PDRequest(f'{prefix}-{i}',short,512)) for i in range(3)]
        control=batcher();control_hist=observe_decode_batches(control)
        expected=survivors(control,'control');expected_reuse=control.submit(PDRequest('control-reuse',short,16))
        step_until(control,lambda:control.idle)
        reference=[f.result() for f in expected];replacement_reference=expected_reuse.result()
        assert all(len(t)==512 for t in reference) and len(replacement_reference)==16
        result['cases'].append(dict(case='control',histogram=control_hist,tokens=reference,replacement_tokens=replacement_reference));save()
        for outcome in ['cancel-reuse','producer-death']:
            batch=batcher();hist=observe_decode_batches(batch)
            alive=survivors(batch,outcome)
            victim=batch.submit(PDRequest('victim',long,16))
            trigger={}
            imported={'victim_spans':0}
            original_import=batch._import_progress
            def observe_import(r,**kwargs):
                before=batch.stats['progress_spans']
                original_import(r,**kwargs)
                if r.request.request_id=='victim':
                    imported['victim_spans']+=batch.stats['progress_spans']-before
            batch._import_progress=observe_import
            def ready_to_fault():
                r=batch.by_id.get('victim')
                if r is None or r.pending is None or r.pending.done():return False
                progress=getattr(r.pending,'kv_progress',None)
                if progress is None:return False
                host_pending=progress.coverage.pending
                if imported['victim_spans'] <= 0:return False
                active=[r for r in batch.decoding if r.request.request_id.startswith(outcome+'-')]
                if len(active)!=3:return False
                trigger.update(host_pending=host_pending,victim_imported_spans=imported['victim_spans'],slot=r.slot,active_tokens=[len(x.tokens) for x in active],progress_spans=batch.stats['progress_spans'],admitted=bool(getattr(r,'admitted',False)))
                return True
            deadline=time.monotonic()+600
            while not ready_to_fault():
                assert time.monotonic()<deadline and not victim.done(),'Failed to observe pending victim with three active survivors'
                batch.step()
            assert trigger['slot']==3 and not trigger['admitted']
            assert trigger['victim_imported_spans'] > 0
            if outcome=='cancel-reuse':
                assert batch.cancel('victim');batch.step()
                try:victim.result(timeout=0);raise AssertionError('Cancellation succeeded unexpectedly')
                except RequestCancelled:pass
                replacement=batch.submit(PDRequest('reuse',short,16));batch.step()
                assert batch.by_id['reuse'].slot==trigger['slot'],'Replacement did not reuse cancelled lane'
            else:
                assert worker._proc.poll() is None
                worker._proc.kill()
            step_until(batch,lambda:batch.idle)
            actual=[f.result() for f in alive]
            assert actual==reference,'Survivor output changed'
            assert not batch.poisoned and sorted(batch.free)==[0,1,2,3] and not batch.by_id
            row=dict(case=outcome,trigger=trigger,histogram=hist,survivor_tokens=actual,survivors_exact=True,idle=True,poisoned=False)
            if outcome=='cancel-reuse':
                assert replacement.result()==replacement_reference,'Reused lane output changed'
                row.update(replacement_tokens=replacement.result(),replacement_exact=True)
            else:
                assert victim.exception() is not None and worker._proc.wait(timeout=10)==-9 and not worker.ready
                row['victim_error']=repr(victim.exception())
            result['cases'].append(row);save()
        result['stage']='complete'
    except BaseException as error:
        result.update(stage='failed',error=repr(error));raise
    finally:
        if worker is not None:worker.close()
        engine.close();save()


if __name__=='__main__':main()
