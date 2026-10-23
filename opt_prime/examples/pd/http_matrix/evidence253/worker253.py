"""Experimental Qwen3 implementation of SubprocessPrefillWorker's existing wire contract."""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def validate_request(slab, req, width, vocab_size):
    tokens, slot, start = req["tokens"], req["slot"], req["send_from"]
    if type(slot) is not int or not 0 <= slot < slab.shape[0]:
        raise ValueError("invalid shared-memory lane")
    if not 0 < len(tokens) <= min(width, slab.shape[-2]):
        raise ValueError("prompt exceeds compiled capacity or is empty")
    if type(start) is not int or not 0 <= start <= len(tokens):
        raise ValueError("invalid prefix offset")
    if any(type(t) is not int or not 0 <= t < vocab_size for t in tokens):
        raise ValueError("invalid token ID")


def store_result(slab, req, keys, values):
    validate_request(slab, req, slab.shape[-2], 2**63)
    expected = (slab.shape[2], slab.shape[3], len(req["tokens"]), slab.shape[-1])
    if tuple(keys.shape) != expected or tuple(values.shape) != expected:
        raise ValueError("prefill KV geometry mismatch")
    start = req["send_from"]
    arrays = [
        t[:, :, start:]
        .to("cpu", torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
        for t in (keys, values)
    ]
    n = len(req["tokens"]) - start
    dst = slab[req["slot"]]
    dst[0, :, :, :n], dst[1, :, :, :n] = arrays
    return n


def cache_geometry(config):
    head_dim = config.get("head_dim")
    if head_dim is None:
        heads, hidden = config["num_attention_heads"], config["hidden_size"]
        if type(heads) is not int or heads <= 0 or hidden % heads:
            raise ValueError("invalid implicit head dimension")
        head_dim = hidden // heads
    shape = (config["num_hidden_layers"], config["num_key_value_heads"], head_dim)
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid cache geometry")
    return shape


def engine_class(factory):
    """Load an explicit trusted local CLI factory; never select this from request JSON."""
    from importlib import import_module

    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("engine factory must be module:callable")
    cls = getattr(import_module(module), name)
    if not callable(cls):
        raise TypeError("engine factory must be callable")
    return cls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shm", required=True)
    for name in ("slots", "layers", "kv-heads", "head-dim", "max-context"):
        ap.add_argument("--" + name, type=int, required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument(
        "--engine-factory",
        help="trusted local module:callable accepting config, weights, width, chip",
    )
    a = ap.parse_args()
    from multiprocessing import resource_tracker, shared_memory

    from huggingface_hub import snapshot_download
    from opt_prime.accel.pd import SamplingParams
    from opt_prime.accel.pd.sampling import select_token
    from safetensors import safe_open

    out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        path = (
            Path(a.model)
            if Path(a.model).is_dir()
            else Path(snapshot_download(a.model, local_files_only=True))
        )
        config = json.loads((path / "config.json").read_text())
        if a.engine_factory:
            Prefill = engine_class(a.engine_factory)
        else:
            from prefill import Prefill
            from spec import validate

            config = validate(config)
        shape = (a.slots, 2, a.layers, a.kv_heads, a.max_context, a.head_dim)
        if min(shape) <= 0 or (a.layers, a.kv_heads, a.head_dim) != cache_geometry(
            config
        ):
            raise ValueError("shared-memory geometry does not match checkpoint")
        weights = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
                    weights[key] = f.get_tensor(key).float()
        engine = Prefill(config, weights, width=a.width, chip=a.chip)
        shm = shared_memory.SharedMemory(name=a.shm)
        # The parent owns unlinking; this child must not unlink another process's slab.
        resource_tracker.unregister(shm._name, "shared_memory")
        try:
            slab = np.ndarray(shape, dtype=np.uint16, buffer=shm.buf)
            print(json.dumps({"ready": True}), file=out, flush=True)
            for line in sys.stdin:
                req = {}
                try:
                    req = json.loads(line)
                    validate_request(slab, req, a.width, config["vocab_size"])
                    sampling = dict(req.get("sampling") or {})
                    sampling["stop_token_ids"] = tuple(
                        sampling.get("stop_token_ids", ())
                    )
                    sampling = SamplingParams(**sampling)
                    t = time.perf_counter()
                    k, v, logits = engine.prefill(list(req["tokens"]))
                    token = select_token(logits, sampling, len(req["tokens"]))
                    n = store_result(slab, req, k, v)
                    result = {
                        "id": req["id"],
                        "token": int(token),
                        "n": n,
                        "cached": 0,
                        "prefill_s": time.perf_counter() - t,
                    }
                    trace_path = os.environ.get("PD_TRACE_PATH")
                    if trace_path:
                        with open(trace_path, "a") as trace:
                            trace.write(json.dumps(dict(wire_id=req["id"], start_s=t,
                                end_s=t + result["prefill_s"])) + "\n")
                except Exception as error:  # noqa: BLE001 - report errors through the worker protocol
                    result = {"id": req.get("id"), "error": repr(error)}
                print(json.dumps(result), file=out, flush=True)
        finally:
            shm.close()


if __name__ == "__main__":
    main()
