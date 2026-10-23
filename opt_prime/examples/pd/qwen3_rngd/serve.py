"""Experimental B1 Qwen3 HTTP service using existing optimus-rm ownership and batching."""

import argparse
import os
import signal
from pathlib import Path

from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import StageError, serve_stage
from transformers import AutoTokenizer


class BoundedService(BatchedService):
    def __init__(self, *args, prompt_capacity, **kwargs):
        self.prompt_capacity = prompt_capacity
        super().__init__(*args, **kwargs)

    def _request(self, rid, body, chat=False):
        request, stops = super()._request(rid, body, chat)
        if len(request.input_ids) > self.prompt_capacity:
            raise StageError(400, "prompt exceeds compiled Qwen3 prefill width")
        return request, stops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18197)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument(
        "--local-prefill",
        action="store_true",
        help="ATOM-only baseline using the same service",
    )
    a = ap.parse_args()
    devices = list(Path("/dev").glob("rbln[0-9]*"))
    if len(devices) != 1:
        raise ValueError("requires an isolated single-ATOM allocation")
    mid = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(mid)
    engine = AtomRblnEngine(
        devices=[int(devices[0].name[4:])],
        max_seq_len=512,
        batch_size=1,
        kv_blocks=3,
        cache_dir=a.cache_dir,
    )
    worker = server = None
    try:
        engine.start(mid)
        c = engine.model.config.get_text_config()
        if not a.local_prefill:
            worker = SubprocessPrefillWorker(
                str(Path(__file__).resolve().with_name("worker_python.sh")),
                ["--model", mid, "--width", str(a.width)],
                slots=1,
                layers=c.num_hidden_layers,
                kv_heads=c.num_key_value_heads,
                head_dim=c.head_dim,
                max_context=512,
                env=dict(os.environ),
            )
        b = engine.batcher(
            prefix_cache=False, prefill_worker=worker, local_prefill=a.local_prefill
        )
        eos = engine.model.generation_config.eos_token_id
        svc = BoundedService(
            b,
            prompt_capacity=a.width,
            model=mid,
            vocab_size=c.vocab_size,
            max_output=128,
            render=lambda t: tok.decode(t, skip_special_tokens=True),
            tokenizer=tok,
            eos_token_ids=tuple(eos) if isinstance(eos, (tuple, list)) else (eos,),
        )
        owner = BatchedOwner(svc)

        # Drain in-flight requests before releasing the shared-memory slab and devices.
        def stop(*_):
            owner.submit_async("stop")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server, _ = serve_stage(owner, "decode", a.port)
        print("HTTP_READY", flush=True)
        owner.run()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if worker is not None:
            worker.close()
        engine.close()


if __name__ == "__main__":
    main()
