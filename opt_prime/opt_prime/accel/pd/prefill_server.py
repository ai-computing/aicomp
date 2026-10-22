"""A prefill engine in its own process and Python environment (#120): the prefill half of batched PD.

The decode side (`pd_batching.SubprocessPrefillWorker`) starts this module with the interpreter of the environment
that has the prefill device's SDK (e.g. the Furiosa venv for RNGD while ATOM decodes under the RBLN venv), and
creates a shared-memory slab with one KV slot per decode lane. Requests and replies are JSON lines on stdin/stdout:

    -> {"id": 7, "tokens": [...], "send_from": 128, "slot": 3, "sampling": {"temperature": 0.0, ...}}
    <- {"id": 7, "token": 1234, "n": 300, "cached": 0, "prefill_s": 0.031}       (or {"id": 7, "error": "..."})

The KV of positions send_from .. len(tokens) goes into slot `slot` of the slab, laid out
[2 (key, value), layers, kv_heads, max_context, head_dim] in bfloat16, starting at position 0.

Engines (--engine):
  torch      `pd_batching.TorchPrefillEngine` on --device (cpu, cuda:N), any uniform HF causal LM
  rngd-qwen  `platforms.rngd.pd.QwenRngdPrefill` on RNGD chip --chip (Qwen2-family dense models, context 512,
             prompts up to --width tokens); the first token comes from the final norm + LM head on the host
"""
import argparse
import json
import sys
import time


def _sampling(d):
    from opt_prime.accel.pd import SamplingParams
    d = dict(d or {})
    d["stop_token_ids"] = tuple(d.get("stop_token_ids", ()))
    return SamplingParams(**d)


class _RngdQwen:
    def __init__(self, a):
        import torch
        from opt_prime.accel.pd.npu.model import DenseModel, load_weights
        from opt_prime.accel.platforms.rngd.pd import QwenRngdPrefill
        self.spec = DenseModel.load(a.model, a.model_id or a.model)
        w = load_weights(a.model, self.spec)
        self.engine = QwenRngdPrefill(self.spec, w, width=a.width, chip=a.chip, accurate_attention=True)
        self.norm = w["model.norm.weight"].float()
        head = w.get("lm_head.weight")
        self.head = (head if head is not None else w["model.embed_tokens.weight"]).float()
        self.eps = self.spec.epsilon
        self.torch = torch

    def prefill(self, tokens):
        """-> keys, values [layers, kv_heads, n, head_dim] (host, bf16) and last-position logits"""
        torch = self.torch
        k, v, hidden = self.engine.prefill_device(list(tokens), full_context=False)
        n = len(tokens)
        keys = k.cpu()[:, :, :, :n].transpose(-1, -2)
        values = v.cpu()[:, :, :n, :]
        h = hidden.cpu()[n - 1].float()
        h = h * torch.rsqrt(h.pow(2).mean() + self.eps) * self.norm
        return keys, values, h @ self.head.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["torch", "rngd-qwen"])
    ap.add_argument("--model", required=True); ap.add_argument("--model-id", default=None)
    ap.add_argument("--shm", required=True); ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True); ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True); ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--width", type=int, default=512)
    a = ap.parse_args()
    import os
    if not os.path.isdir(a.model):                 # a hub ID: the local snapshot (engines below need a directory)
        from huggingface_hub import snapshot_download
        a.model_id = a.model_id or a.model
        a.model = snapshot_download(a.model, local_files_only=True)
    import numpy as np
    import torch
    from multiprocessing import shared_memory
    from opt_prime.accel.pd.sampling import select_token
    shm = shared_memory.SharedMemory(name=a.shm)
    try:                                       # the decode side owns the segment: do not unlink it on our exit
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:                          # noqa: BLE001 - tracker details differ across Python versions
        pass
    slab = np.ndarray((a.slots, 2, a.layers, a.kv_heads, a.max_context, a.head_dim), dtype=np.uint16, buffer=shm.buf)
    if a.engine == "torch":
        from transformers import AutoModelForCausalLM
        from opt_prime.accel.pd.pd_batching import TorchPrefillEngine
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
        engine = TorchPrefillEngine(model, max_context=a.max_context, chunk=a.chunk)
    else:
        engine = _RngdQwen(a)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            t0 = time.perf_counter()
            from opt_prime.accel.pd import PDRequest
            sampling = _sampling(req.get("sampling"))
            tokens, start = tuple(req["tokens"]), int(req["send_from"])
            if a.engine == "torch":
                out = engine.prefill(PDRequest(f"p{req['id']}", tokens, 1, sampling), start)
                keys, values, token, cached = out.keys, out.values, out.token, out.prefill_cached
            else:
                keys, values, logits = engine.prefill(tokens)
                keys, values = keys[:, :, start:], values[:, :, start:]
                token, cached = select_token(logits, sampling, len(tokens)), 0
            n = keys.shape[2]
            dst = slab[req["slot"]]
            dst[0, :, :, :n] = keys.to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
            dst[1, :, :, :n] = values.to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
            print(json.dumps(dict(id=req["id"], token=int(token), n=n, cached=int(cached),
                                  prefill_s=time.perf_counter() - t0)), flush=True)
        except Exception as error:                                       # noqa: BLE001 - reported per request
            print(json.dumps(dict(id=req.get("id"), error=repr(error))), flush=True)
    shm.close()


if __name__ == "__main__":
    main()
