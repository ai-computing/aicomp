"""Serve a Hugging Face causal LM with continuous batching and chunked prefill over the OpenAI-style HTTP API (#44).

    python examples/pd/serve_batched.py --model unsloth/Llama-3.2-1B-Instruct --port 8000 --slots 32 --kv paged
    # Rebellions ATOM, the optimum-rbln decoder compiled with batch 8 (one lane per request):
    python examples/pd/serve_batched.py --model unsloth/Llama-3.2-1B-Instruct --backend atom --devices 0 --slots 8
    curl localhost:8000/v1/chat/completions -H 'x-request-id: a' -d '{"model": "...", "messages": [...], "stream": true}'
"""
import argparse, threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import serve_stage


def build(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.backend == "atom":
        from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
        engine = AtomRblnEngine(devices=a.devices, max_seq_len=a.max_context, batch_size=a.slots)
        engine.start(a.model)
        b, model = engine.batcher(), engine.model
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
        b = ContinuousBatcher(model, slots=a.slots, max_context=a.max_context, prefill_chunk=a.chunk, kv=a.kv)
    eos = model.generation_config.eos_token_id
    svc = BatchedService(b, model=a.model, vocab_size=model.config.get_text_config().vocab_size,
                         max_output=a.max_output, render=lambda t: tok.decode(t), tokenizer=tok,
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
    owner = build(ap.parse_args())
    serve_stage(owner, "decode", ap.parse_args().port)
    owner.run()                                   # owner loop on the main thread
