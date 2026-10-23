"""Single assigned ATOM HTTP backend, optionally fed by the established RNGD subprocess adapter."""
import argparse
import importlib.util
import signal
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--batching', required=True, help='trusted ATOM batching module')
    p.add_argument('--http-module', required=True, help='trusted HTTP module')
    p.add_argument('--pd-module', required=True, help='trusted subprocess worker module')
    p.add_argument('--worker-python')
    p.add_argument('--engine-factory')
    p.add_argument('--width', type=int, default=8192)
    p.add_argument('--kv-blocks', type=int, default=3)
    p.add_argument('--max-output', type=int, default=128)
    a = p.parse_args()
    if bool(a.worker_python) != bool(a.engine_factory):
        p.error('worker-python and engine-factory must be supplied together')
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
    from transformers import AutoTokenizer

    nodes = list(Path('/dev').glob('rbln[0-9]*'))
    if len(nodes) != 1:
        raise RuntimeError('benchmark requires exactly one assigned ATOM device')
    engine = AtomRblnEngine(devices=[int(nodes[0].name[4:])], max_seq_len=9216,
                           batch_size=1, kv_blocks=a.kv_blocks, cache_dir=a.cache_dir)
    worker = owner = server = http = None
    try:
        engine.start(a.model)
        c = engine.model.config.get_text_config()
        if a.worker_python:
            worker = SubprocessPrefillWorker(a.worker_python,
                ['--model', a.model, '--width', str(a.width), '--engine-factory', a.engine_factory],
                slots=1, layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
                head_dim=getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads,
                max_context=a.width)
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        batcher = engine.batcher(prefix_cache=False, prefill_worker=worker, local_prefill=worker is None)
        service = BatchedService(batcher, model=a.model, vocab_size=c.vocab_size,
                                 max_output=a.max_output, tokenizer=tok,
                                 render=lambda ids: tok.decode(ids, skip_special_tokens=True))
        owner = BatchedOwner(service)
        # The benchmark uses fixed-length ignore_eos requests; no EOS-dependent timing comparisons.
        signal.signal(signal.SIGTERM, lambda *_: owner.submit('stop'))
        server, http = serve_stage(owner, 'decode', a.port)
        owner.run()
    finally:
        if server:
            server.shutdown()
            server.server_close()
            http.join(5)
        if worker:
            worker.close()
        engine.close()


if __name__ == '__main__':
    main()
