"""Serve a Hugging Face causal LM with continuous batching and chunked prefill over the OpenAI-style HTTP API (#44).

    python examples/pd/serve_batched.py --model unsloth/Llama-3.2-1B-Instruct --port 8000 --slots 32 --kv paged
    # Rebellions ATOM, the optimum-rbln decoder compiled with batch 8 (one lane per request):
    python examples/pd/serve_batched.py --model unsloth/Llama-3.2-1B-Instruct --backend atom --devices 0 --slots 8
    # batched PD (#120): prefill on another GPU (or another venv via --pd-prefill-python), decode lanes batched:
    python examples/pd/serve_batched.py --model ... --pd-prefill-device cuda:0 --device cuda:1
    # + prefix caching with KV events for the llm-d EPP (topic kv@<pod-ip>:<port>@<model>):
    python examples/pd/serve_batched.py --model ... --prefix-cache --kv-events tcp://*:5557 --pod-ip 192.0.2.5
    # + a host-DRAM tier behind the prefix cache (#52):
    python examples/pd/serve_batched.py --model ... --prefix-cache --kv-host-memory-gb 16
    curl localhost:8000/v1/chat/completions -H 'x-request-id: a' -d '{"model": "...", "messages": [...], "stream": true}'
"""
import argparse, threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import serve_stage


def prefill_worker(a, slots, config):
    """the PD prefill side from the flags, or None (aggregated serving)"""
    if a.pd_prefill_device:
        from opt_prime.accel.pd.pd_batching import PrefillWorker, TorchPrefillEngine
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.pd_prefill_device).eval()
        return PrefillWorker(TorchPrefillEngine(m, max_context=a.max_context, out_device=a.device
                                                if a.backend == "torch" else "cpu"))
    if a.pd_prefill_python:
        import os, shlex
        from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
        c = config.get_text_config()
        env = dict(os.environ, **({"PYTHONPATH": a.pd_prefill_pythonpath} if a.pd_prefill_pythonpath else {}))
        return SubprocessPrefillWorker(a.pd_prefill_python, shlex.split(a.pd_prefill_args) + ["--model", a.model],
                                       slots=slots, layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads,
                                       head_dim=getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
                                       max_context=a.max_context, env=env)
    return None


def build(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    events = None
    if a.kv_events:
        from opt_prime.accel.serve.kv_events import ZmqKVEventPublisher
        events = ZmqKVEventPublisher(a.kv_events, topic=f"kv@{a.pod_ip}:{a.port}@{a.model}", block_size=a.page_size)
    if a.backend == "atom":
        from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
        engine = AtomRblnEngine(devices=a.devices, max_seq_len=a.max_context, batch_size=a.slots)
        engine.start(a.model)
        b, model = engine.batcher(), engine.model
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
        kw = dict(slots=a.slots, max_context=a.max_context, kv=a.kv, page_size=a.page_size,
                  prefix_cache=a.prefix_cache, kv_events=events, host_kv_gib=a.kv_host_memory_gb,
                  reload_bandwidth=a.reload_bandwidth, prefill_rate=a.prefill_rate)
        worker = prefill_worker(a, a.slots, model.config)
        if worker is not None:
            from opt_prime.accel.pd.pd_batching import PDBatcher
            b = PDBatcher(model, worker, **kw)
        else:
            b = ContinuousBatcher(model, prefill_chunk=a.chunk, **kw)
    eos = model.generation_config.eos_token_id
    svc = BatchedService(b, model=a.model, vocab_size=model.config.get_text_config().vocab_size,
                         max_output=a.max_output, render=lambda t: tok.decode(t, skip_special_tokens=True),
                         tokenizer=tok,
                         eos_token_ids=tuple(eos) if isinstance(eos, (list, tuple)) else ((eos,) if eos is not None else ()))
    return BatchedOwner(svc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, default=32); ap.add_argument("--kv", default="paged", choices=["slot", "paged"])
    ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--max-output", type=int, default=1024); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--backend", default="torch", choices=["torch", "atom"])
    ap.add_argument("--devices", type=int, nargs="+", default=[0], help="ATOM devices (tensor parallel)")
    ap.add_argument("--pd-prefill-device", default=None, help="PD: torch prefill on this device (e.g. cuda:0)")
    ap.add_argument("--pd-prefill-python", default=None, help="PD: prefill_server under this interpreter")
    ap.add_argument("--pd-prefill-args", default="--engine torch --device cpu")
    ap.add_argument("--pd-prefill-pythonpath", default=None)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--prefix-cache", action="store_true"); ap.add_argument("--kv-events", default=None)
    ap.add_argument("--pod-ip", default="127.0.0.1"); ap.add_argument("--kv-host-memory-gb", type=float, default=0)
    ap.add_argument("--reload-bandwidth", type=float, default=None, help="bytes/s host->device (reload-or-recompute)")
    ap.add_argument("--prefill-rate", type=float, default=None, help="prefill tokens/s (reload-or-recompute)")
    owner = build(ap.parse_args())
    serve_stage(owner, "decode", ap.parse_args().port)
    owner.run()                                   # owner loop on the main thread
