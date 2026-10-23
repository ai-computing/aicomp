"""Warm-engine HTTP A/B and lifecycle validation for B1 ATOM remote prefetch (#253).
All configuration changes happen only between drained batches. No control endpoint is exposed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'cache-dir', 'batching', 'pd-module', 'service-module', 'http-module',
                 'worker-python', 'engine-factory', 'reference', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--kv-blocks', type=int, default=3)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--outputs', type=int, nargs='+', default=[32, 128])
    a = p.parse_args()
    output = Path(a.output)
    trace_path = output.with_suffix('.producer.jsonl')
    if output.exists() or trace_path.exists():
        raise RuntimeError('refusing to overwrite experiment artifacts')
    for name, path in [('opt_prime.accel.serve.llmd', a.http_module),
                       ('opt_prime.accel.pd.pd_batching', a.pd_module),
                       ('opt_prime.accel.platforms.atom.batching', a.batching),
                       ('opt_prime.accel.serve.batched', a.service_module)]:
        load_module(name, path)
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
    from opt_prime.accel.serve.llmd import serve_stage

    reference = json.loads(Path(a.reference).read_text())
    prompts = reference['prompts']
    result = dict(passed=False, stage='loading', model=a.model, runtime_commit='3e16c33',
                  torch=torch.__version__, physical_atom=1, physical_rngd=1, slots=1,
                  producer_width=8192, producer_cache=False, decoder_cache=False, prompts=prompts,
                  checks=[], normal_eos=[], batches=[], lifecycle={}, submissions=[], decode_calls=[])
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    nodes = list(Path('/dev').glob('rbln[0-9]*'))
    assert len(nodes) == 1, 'requires exactly one assigned ATOM'
    engine = AtomRblnEngine(devices=[int(nodes[0].name[4:])], max_seq_len=9216, batch_size=1,
                           kv_blocks=a.kv_blocks, cache_dir=a.cache_dir)
    worker = owner = server = http = client = None
    errors = []
    started = time.perf_counter()
    try:
        engine.start(a.model)
        c = engine.model.config.get_text_config()
        worker = SubprocessPrefillWorker(a.worker_python,
            ['--model', a.model, '--width', '8192', '--engine-factory', a.engine_factory],
            slots=1, layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
            head_dim=getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads,
            max_context=8192, env=dict(os.environ, PD_TRACE_PATH=str(trace_path.resolve())))
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        eos = GenerationConfig.from_pretrained(a.model, local_files_only=True).eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if tokenizer.eos_token_id is not None:
            eos = list(dict.fromkeys(eos + [tokenizer.eos_token_id]))
        b = engine.batcher(prefix_cache=False, prefill_worker=worker, local_prefill=False, remote_prefetch=False)
        service = BatchedService(b, model=a.model, vocab_size=c.vocab_size, max_output=max(a.outputs),
                                tokenizer=tokenizer, eos_token_ids=eos,
                                render=lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
        owner = BatchedOwner(service)
        emitted = {}
        original_emit, original_decode, original_submit = b._emit, b._decode_step, worker.submit
        def emit(r, token):
            emitted.setdefault(r.request.request_id, time.perf_counter())
            return original_emit(r, token)
        def decode():
            ids = [r.request.request_id for r in b.decoding]
            t0 = time.perf_counter()
            original_decode()
            result['decode_calls'].append(dict(ids=ids, start_s=t0, end_s=time.perf_counter()))
        def submit(request, send_from=0, slot=None):
            key = worker._next  # only the owner submits, so this matches the subprocess wire ID
            row = dict(wire_id=key, id=request.request_id, start_s=time.perf_counter(),
                       prefetch=b.remote_prefetch, decoding=[r.request.request_id for r in b.decoding])
            f = original_submit(request, send_from=send_from, slot=slot)
            result['submissions'].append(row)
            return f
        b._emit, b._decode_step, worker.submit = emit, decode, submit
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        server, http = serve_stage(owner, 'decode', port)
        result.update(load_s=time.perf_counter()-started, eos_token_ids=eos, stage='running')
        save()
        refs = {}

        def wait_for(predicate, description, timeout=240):
            end = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() >= end:
                    raise TimeoutError(description)
                time.sleep(.01)

        def mode(enabled):
            wait_for(lambda: b.idle, 'drain before mode switch')
            with b._lock:
                assert not (b.waiting or b.prefilling or b.decoding or b._prefetched)
                b.remote_prefetch = enabled

        def request(n, count, rid, ignore_eos=True, gate=None):
            body = dict(model=a.model, prompt=prompts[str(n)], max_tokens=count,
                        temperature=0, ignore_eos=ignore_eos, stream=False)
            req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',
                data=json.dumps(body).encode(), headers={'x-request-id': rid})
            if gate: gate.wait(timeout=30)
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=240) as response:
                    data = json.load(response)
            except urllib.error.HTTPError as error:
                return dict(id=rid, status=error.code, error=error.read().decode())
            elapsed = time.perf_counter()-t0
            choice = data['choices'][0]
            return dict(id=rid, status=200, tokens=choice['token_ids'], text=choice['text'],
                        finish=choice['finish_reason'], e2e_s=elapsed,
                        first_token_emitted_s=emitted[rid]-t0)

        def batch(n, count, concurrency, prefix, ignore_eos=True):
            gate = threading.Barrier(concurrency+1)
            with ThreadPoolExecutor(max_workers=concurrency) as clients:
                fs = [clients.submit(request,n,count,f'{prefix}-{i}',ignore_eos,gate) for i in range(concurrency)]
                t0=time.perf_counter(); gate.wait(timeout=30)
                rows=[f.result(timeout=260) for f in fs]
                wall=time.perf_counter()-t0
            assert all(r['status']==200 for r in rows), rows
            return rows,wall

        def cut(tokens):
            for i,t in enumerate(tokens):
                if t in eos: return tokens[:i+1]
            return tokens

        def exercise():
            try:
                wait_for(lambda: owner.started, 'owner startup')
                for n in [128,8192]:
                    for count in a.outputs:
                        r=request(n,count,f'warm-{n}-{count}')
                        assert r['status']==200 and len(r['tokens'])==count and r['finish']=='length',r
                        refs[f'{n}:{count}']=r['tokens']
                result['references']=refs
                for enabled in [False,True]:
                    mode(enabled)
                    for n in [128,8192]:
                        for count in a.outputs:
                            rows,_=batch(n,count,4,f'check-{enabled}-{n}-{count}')
                            assert all(r['tokens']==refs[f'{n}:{count}'] for r in rows),rows
                            result['checks'].append(dict(prefetch=enabled,input_tokens=n,output_tokens=count,rows=rows))
                        rows,_=batch(n,32,4,f'eos-{enabled}-{n}',False)
                        expected=cut(reference['references'][f'{n}:32']['tokens'])
                        assert all(r['tokens']==expected for r in rows),rows
                        assert all(r['finish']==('stop' if expected[-1] in eos else 'length') for r in rows)
                        result['normal_eos'].append(dict(prefetch=enabled,input_tokens=n,rows=rows))
                    save()
                for repeat in range(a.repeats):
                    for n in ([128,8192] if repeat%2==0 else [8192,128]):
                        for count in a.outputs:
                            for concurrency in [1,4]:
                                for enabled in ([False,True] if repeat%2==0 else [True,False]):
                                    mode(enabled)
                                    rows,wall=batch(n,count,concurrency,f'r{repeat}-{enabled}-{n}-{count}-{concurrency}')
                                    assert all(r['tokens']==refs[f'{n}:{count}'] for r in rows),rows
                                    result['batches'].append(dict(repeat=repeat,prefetch=enabled,input_tokens=n,
                                        output_tokens=count,concurrency=concurrency,wall_s=wall,
                                        tok_s=concurrency*count/wall,median_e2e_s=statistics.median(r['e2e_s'] for r in rows),
                                        median_first_token_s=statistics.median(r['first_token_emitted_s'] for r in rows),rows=rows))
                                    save()
                mode(True)
                for fault in ['cancel','worker_exit']:
                    with ThreadPoolExecutor(max_workers=3) as clients:
                        first=clients.submit(request,8192,max(a.outputs),f'{fault}-first')
                        wait_for(lambda:f'{fault}-first' in emitted,'first decode before fault')
                        victim=clients.submit(request,8192,32,f'{fault}-victim')
                        wait_for(lambda:any(r['id']==f'{fault}-victim' and r['decoding'] for r in result['submissions']),
                                 'prefetched victim during first decode')
                        if fault=='cancel':
                            req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/cancel',data=b'{}',headers={'x-request-id':f'{fault}-victim'})
                            with urllib.request.urlopen(req,timeout=10) as response:assert response.status in [200,202]
                            gone=victim.result(timeout=240);assert gone['status']==499,gone
                            recovery=clients.submit(request,8192,32,'cancel-recovery')
                            recovered=recovery.result(timeout=240)
                            assert recovered['status']==200 and recovered['tokens']==refs['8192:32']
                        else:
                            worker._proc.terminate()
                            gone=victim.result(timeout=240);assert gone['status'] in [500,503],gone
                            rejected=request(128,32,'after-worker-exit');assert rejected['status']==503,rejected
                            try:
                                urllib.request.urlopen(f'http://127.0.0.1:{port}/ready',timeout=10)
                            except urllib.error.HTTPError as error:
                                assert error.code==503
                            else:
                                raise AssertionError('failed producer must fail readiness')
                        kept=first.result(timeout=240)
                        assert kept['status']==200 and kept['tokens']==refs[f'8192:{max(a.outputs)}'],kept
                        wait_for(lambda:b.idle,'fault drain')
                        assert not b.poisoned and not b.by_id and b._prefetched is None
                        result['lifecycle'][fault]=dict(victim_status=gone['status'],first_tokens_equal=True,drained=True)
                        save()
                result.update(passed=True,stage='complete')
            except BaseException:
                result.update(error=traceback.format_exc(),stage='failed')
                errors.append(result['error'])
            finally:
                save()
                # Cancel only this harness's requests before graceful owner/worker shutdown.
                for rid in list(b.by_id): owner.cancel_request(rid)
                if errors and worker._proc.poll() is None:
                    worker._proc.terminate()  # settle pending futures before owner drain; keep slab mapped until it stops
                owner.submit('stop')
        client=threading.Thread(target=exercise,name='pipeline-clients',daemon=True)
        client.start()
        owner.run()
        client.join(10)
        assert not client.is_alive()
        if errors: raise RuntimeError(errors[0])
    except BaseException:
        result.setdefault('error',traceback.format_exc());result['stage']='failed';save()
        raise
    finally:
        if server:
            server.shutdown();server.server_close();http.join(5)
        if worker:worker.close()
        engine.close()


if __name__=='__main__':
    main()
