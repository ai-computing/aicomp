"""Single assigned ATOM HTTP backend, optionally fed by the established RNGD subprocess adapter."""
import argparse
import importlib.util
import os
import json
import signal
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--batching', required=True, help='trusted ATOM batching module')
    p.add_argument('--http-module', required=True, help='trusted HTTP module')
    p.add_argument('--pd-module', required=True, help='trusted subprocess worker module')
    p.add_argument('--worker-python')
    p.add_argument('--stream-layers', action='store_true')
    p.add_argument('--engine-factory')
    p.add_argument('--width', type=int, default=8192)
    p.add_argument('--kv-blocks', type=int, default=3)
    p.add_argument('--batch-size', type=int, choices=(1,4), default=1)
    p.add_argument('--tensor-parallel-size', type=int, choices=(1,2), default=1)
    p.add_argument('--max-output', type=int, default=128)
    a = p.parse_args(argv)
    if a.kv_blocks < a.batch_size + 1:
        p.error('kv-blocks must reserve one scratch block beyond batch-size')
    if bool(a.worker_python) != bool(a.engine_factory):
        p.error('worker-python and engine-factory must be supplied together')
    if a.stream_layers and not a.worker_python:
        p.error('stream-layers requires a subprocess producer')
    return a


def check_shm_capacity(config, batch_size, width, *, free_bytes=None):
    if any(type(x) is not int or x <= 0 for x in (batch_size, width)):
        raise ValueError('batch size and width must be positive integers')
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    required = batch_size * 2 * config.num_hidden_layers * config.num_key_value_heads * width * head_dim * 2
    if free_bytes is None:
        stat = os.statvfs('/dev/shm')
        free_bytes = stat.f_bavail * stat.f_frsize
    if required > free_bytes:
        raise RuntimeError(f'Insufficient shared memory: KV slab needs {required} bytes, available {free_bytes}')
    return required


def validate_compiled_geometry(config, batch_size, blocks, *, tensor_parallel_size=1, artifact_config=None):
    counts = [getattr(c, 'num_devices', None) for c in getattr(config, 'compile_cfgs', [])]
    if artifact_config is not None:
        geometry = tuple(artifact_config.get(k) for k in ('batch_size', 'kvcache_num_blocks', 'max_seq_len'))
        if geometry != (batch_size, blocks, 9216) or any(n is not None and n != tensor_parallel_size for n in counts):
            raise ValueError('Runtime and saved artifact geometry disagree')
        # SDK loading may clear num_devices on config objects; saved compile metadata retains it.
        counts = [c.get('num_devices') for c in artifact_config.get('_compile_cfgs', [])]
    if ((config.batch_size, config.kvcache_num_blocks, config.max_seq_len) != (batch_size, blocks, 9216)
            or not counts or any(n != tensor_parallel_size for n in counts)):
        raise ValueError('Compiled batch/block/context geometry does not match experiment')


def observe_decode_batches(batcher):
    histogram = [0] * (batcher.lanes + 1)
    original = batcher._decode_step
    def step():
        count = len(batcher.decoding)
        result = original()
        histogram[count] += 1
        return result
    batcher._decode_step = step
    return histogram


def main():
    a = parse_args()
    for name, path in [('opt_prime.accel.pd.pd_batching', a.pd_module),
                       ('opt_prime.accel.platforms.atom.batching', a.batching),
                       ('opt_prime.accel.serve.llmd', a.http_module)]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
    from opt_prime.accel.serve.llmd import serve_stage
    from transformers import AutoTokenizer, AutoConfig

    if a.worker_python:
        config = AutoConfig.from_pretrained(a.model, local_files_only=True).get_text_config()
        required = check_shm_capacity(config, a.batch_size, a.width)
        print(f'PD_SHM_REQUIRED_BYTES {required}', flush=True)
    nodes = list(Path('/dev').glob('rbln[0-9]*'))
    if len(nodes) != a.tensor_parallel_size:
        raise RuntimeError('assigned ATOM count must equal requested tensor parallel size')
    engine = AtomRblnEngine(devices=sorted(int(n.name[4:]) for n in nodes), max_seq_len=9216,
                           batch_size=a.batch_size, kv_blocks=a.kv_blocks, cache_dir=a.cache_dir)
    worker = owner = server = http = None
    occupancy = None
    try:
        engine.start(a.model)
        validate_compiled_geometry(engine.model.rbln_config, a.batch_size, a.kv_blocks, tensor_parallel_size=a.tensor_parallel_size,
                                   artifact_config=json.loads((Path(engine.dir) / 'rbln_config.json').read_text()))
        print('ATOM270_COMPILED ' + json.dumps(dict(batch_size=a.batch_size, kv_blocks=a.kv_blocks, context=9216, tensor_parallel_size=a.tensor_parallel_size)), flush=True)
        c = engine.model.config.get_text_config()
        if a.worker_python:
            worker = SubprocessPrefillWorker(a.worker_python,
                ['--engine', 'factory', '--model', a.model, '--width', str(a.width), '--engine-factory', a.engine_factory],
                slots=a.batch_size, layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
                head_dim=getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads,
                max_context=a.width, stream_layers=a.stream_layers)
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        batcher = engine.batcher(prefix_cache=False, prefill_worker=worker, local_prefill=worker is None)
        occupancy = observe_decode_batches(batcher)
        from transformers import GenerationConfig
        eos=GenerationConfig.from_pretrained(a.model,local_files_only=True).eos_token_id
        eos=eos if isinstance(eos,list) else ([] if eos is None else [eos])
        eos=sorted(set(eos+([] if tok.eos_token_id is None else [tok.eos_token_id])))
        assert eos, 'Quality validation requires explicit EOS handling'
        print('ATOM270_QUALITY_EOS '+json.dumps(eos),flush=True)
        service = BatchedService(batcher, model=a.model, vocab_size=c.vocab_size,
                                 max_output=a.max_output, tokenizer=tok, eos_token_ids=eos,
                                 render=lambda ids: tok.decode(ids, skip_special_tokens=True))
        owner = BatchedOwner(service)
        # Quality requests use explicit model/tokenizer EOS; performance benchmark entrypoint is unchanged.
        signal.signal(signal.SIGTERM, lambda *_: owner.submit('stop'))
        server, http = serve_stage(owner, 'decode', a.port)
        owner.run()
    finally:
        if occupancy is not None:
            print('ATOM270_ACTIVE_LANES ' + json.dumps(occupancy), flush=True)
        if server:
            server.shutdown()
            server.server_close()
            http.join(5)
        if worker:
            worker.close()
        engine.close()


if __name__ == '__main__':
    main()
