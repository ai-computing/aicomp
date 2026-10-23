"""Live RNGD/GPU HTTP lifecycle test; run separately from performance measurements."""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--factory', required=True)
    parser.add_argument('--worker-python', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--http-module', required=True)
    parser.add_argument('--markers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location('opt_prime.accel.serve.llmd', args.http_module)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from opt_prime.accel.pd.pd_batching import PDBatcher, SubprocessPrefillWorker
    from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
    from opt_prime.accel.serve.llmd import serve_stage

    reference = json.loads(args.reference.read_text())
    result = dict(passed=False, lifecycle_passed=False, stage='loading', model=args.model, faults={})
    worker = owner = server = http = client = None
    errors = []

    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')

    def wait_for(predicate, label, timeout=240):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            value = predicate()
            if value:
                return value
            time.sleep(.002)
        raise TimeoutError(label)

    save()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            local_files_only=True).to('cuda:0').eval()
        torch.cuda.synchronize()
        c = model.config.get_text_config()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

        def new_worker():
            return SubprocessPrefillWorker(args.worker_python,
                ['--model', args.model, '--width', '8192', '--engine-factory', 'observer_factory:make_prefill'],
                slots=4, layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
                head_dim=getattr(c, 'head_dim', None) or c.hidden_size//c.num_attention_heads,
                max_context=9216, env=dict(os.environ, PREFILL_OBSERVER_DIR=str(args.markers.resolve()),
                                         PREFILL_OBSERVER_FACTORY=args.factory))

        worker = new_worker()
        batcher = PDBatcher(model, worker, slots=4, max_context=9216, kv='paged', page_size=64, prefix_cache=False)
        service = BatchedService(batcher, model=args.model, vocab_size=c.vocab_size, max_output=32,
            tokenizer=tokenizer, render=lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
        owner = BatchedOwner(service)
        server, http = serve_stage(owner, 'decode', 0)
        result['stage'] = 'running'
        save()

        def request(path, body, rid):
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}{path}',
                data=None if body is None else json.dumps(body).encode(), headers={'x-request-id': rid})
            try:
                with urllib.request.urlopen(req, timeout=240) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, error.read().decode()

        def generate(rid, n=8192):
            return request('/v1/completions', dict(model=args.model, prompt=reference['prompts'][str(n)],
                temperature=0, max_tokens=32, ignore_eos=True, stream=False), rid)

        control_tokens = []

        def check_output(rid):
            status, response = generate(rid, 128)
            assert status == 200, (status, response)
            tokens = response['choices'][0]['token_ids']
            assert len(tokens) == 32
            if not control_tokens:
                control_tokens.extend(tokens)
                result['control_tokens'] = tokens
                result['native_tokens_equal'] = tokens == reference['references']['128:32']['tokens']
            else:
                assert tokens == control_tokens, 'fault changed output relative to the same PD route'

        def entered(rid):
            path = args.markers/f'{worker._proc.pid}.json'
            if not path.exists():
                return False
            event = json.loads(path.read_text())
            with batcher._lock:
                state = batcher.by_id.get(rid)
                pending = state is not None and state.pending is not None and not state.pending.done()
            return event if pending and event['phase'] == 'start' else False

        def exercise():
            nonlocal worker
            try:
                wait_for(lambda: owner.started, 'owner startup')
                check_output('before-faults')
                with ThreadPoolExecutor(max_workers=2) as clients:
                    future = clients.submit(generate, 'pending-cancel')
                    event = wait_for(lambda: entered('pending-cancel'), 'actual producer entry before cancel')
                    status, _ = request('/v1/cancel', {}, 'pending-cancel')
                    assert status in [200, 202], status
                    assert future.result(timeout=240)[0] == 499
                    wait_for(lambda: batcher.idle, 'cancel drain')
                    check_output('after-cancel')
                    result['faults']['cancel'] = dict(entry=event, status=499, reuse_exact=True)
                    save()
                    future = clients.submit(generate, 'pending-death')
                    event = wait_for(lambda: entered('pending-death'), 'actual producer entry before death')
                    worker._proc.terminate()  # Only this harness-owned producer.
                    status, _ = future.result(timeout=240)
                    assert status in [500, 503], status
                    wait_for(lambda: batcher.idle, 'failed producer drain')
                    assert generate('after-death', 128)[0] == 503
                    assert request('/ready', None, 'readiness-dead')[0] == 503
                    old_slab = worker._shm.name
                    worker.close()
                    try:
                        leaked = shared_memory.SharedMemory(name=old_slab)
                    except FileNotFoundError:
                        pass
                    else:
                        leaked.close()
                        raise AssertionError('worker slab still linked')
                    worker = new_worker()
                    with batcher._lock:
                        assert not (batcher.waiting or batcher.prefilling or batcher.decoding or batcher.by_id)
                        assert sorted(batcher.free) == [0, 1, 2, 3]
                        batcher.worker = worker
                    assert request('/ready', None, 'readiness-replaced')[0] == 200
                    check_output('after-replacement')
                    result['faults']['worker_exit'] = dict(entry=event, pending_status=status,
                        later_status=503, readiness=503, slab_unlinked=True, replacement_exact=True)
                result.update(lifecycle_passed=True, passed=result['native_tokens_equal'], stage='complete')
            except BaseException:
                errors.append(traceback.format_exc())
                result.update(stage='failed', error=errors[-1])
            finally:
                if errors:
                    worker.close()  # Settle futures before stopping/draining owner.
                save()
                owner.submit('stop')

        client = threading.Thread(target=exercise, daemon=True)
        client.start()
        owner.run()
        client.join(10)
        assert not client.is_alive()
        if errors:
            raise RuntimeError(errors[0])
        if not result['native_tokens_equal']:
            raise AssertionError('lifecycle passed, but native GPU token comparison failed; see recorded control')
    except BaseException:
        result.update(stage='failed', error=traceback.format_exc())
        save()
        raise
    finally:
        if server:
            server.shutdown()
            server.server_close()
            http.join(5)
        if worker:
            worker.close()


if __name__ == '__main__':
    main()
