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
  rngd-qwen  `platforms.rngd.pd.QwenRngdPrefill` on RNGD chip --chip (Qwen2-family dense models, context 512);
             --width takes a comma-separated list of compiled widths, and each prompt runs on the smallest that
             holds it (measured on Qwen2.5-0.5B: width 512 349 ms, 256 217 ms per prompt); the first token comes from
             the final norm + LM head on the host
  rngd-full  rngd-pd's `FullPrefillEngine` on RNGD chip --chip (Llama-3.1-8B geometry: Llama-3.1-8B, KONI-8B; #137):
             two layers per compiled graph, --width buckets (e.g. 256,512), LM head on the card; needs rngd-pd on the
             path (with rngd-pd#16 on furiosa-torch 2026.4). Measured on KONI-8B with the KV on the host: 36-45 ms
             up to 256 tokens, 55-58 ms up to 384, 61-76 ms at 480 (furiosa-llm `serve` on the same card, first
             token over HTTP: 56-61 ms)
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
        widths = sorted(int(x) for x in str(a.width).split(","))
        self.engines = [(width, QwenRngdPrefill(self.spec, w, width=width, chip=a.chip, accurate_attention=True))
                        for width in widths]
        self.norm = w["model.norm.weight"].float()
        head = w.get("lm_head.weight")
        self.head = (head if head is not None else w["model.embed_tokens.weight"]).float()
        self.eps = self.spec.epsilon
        self.torch = torch

    def prefill(self, tokens):
        """-> keys, values [layers, kv_heads, n, head_dim] (host, bf16) and last-position logits"""
        torch = self.torch
        engine = next((e for width, e in self.engines if len(tokens) <= width), None)
        if engine is None:
            raise ValueError(f"prompt of {len(tokens)} tokens exceeds the compiled widths")
        k, v, hidden = engine.prefill_device(list(tokens), full_context=False)
        n = len(tokens)
        keys = k.cpu()[:, :, :, :n].transpose(-1, -2)
        values = v.cpu()[:, :, :n, :]
        h = hidden.cpu()[n - 1].float()
        h = h * torch.rsqrt(h.pow(2).mean() + self.eps) * self.norm
        return keys, values, h @ self.head.T


class _RngdFull:
    """rngd-pd's whole-prompt prefill (FullPrefillEngine): Llama-3.1-8B geometry, one pass per width bucket"""
    GEOMETRY = dict(hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8)

    def __init__(self, a):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from rngd_pd.engines.model import ModelConfig
        from rngd_pd.engines.prefill_full import FullPrefillEngine
        config = AutoConfig.from_pretrained(a.model)
        wrong = {k: getattr(config, k, None) for k, v in self.GEOMETRY.items() if getattr(config, k, None) != v}
        if wrong or config.num_hidden_layers % 2:
            raise ValueError(f"rngd-full needs Llama-3.1-8B geometry {self.GEOMETRY} and an even layer count, got {wrong}")
        hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
        cfg = ModelConfig(hf.config)
        del hf
        widths = sorted(int(x) for x in str(a.width).split(","))
        if widths[-1] > a.max_context:
            raise ValueError(f"width {widths[-1]} exceeds --max-context {a.max_context}")
        self.engine = FullPrefillEngine(cfg, sd, C=widths[0], group=2, chip=a.chip, cache_dir=None)
        for width in widths[1:]:
            self.engine.add_bucket(width)
        self.widths, self.torch = widths, torch

    def prefill(self, tokens):
        """-> keys, values [layers, kv_heads, n, head_dim] (host, bf16) and last-position logits"""
        torch = self.torch
        n = len(tokens)
        if n > self.widths[-1]:
            raise ValueError(f"prompt of {n} tokens exceeds the compiled widths")
        stage = self.engine.begin_device_prefill(list(tokens))
        while not stage.advance(self.engine.ngroups):
            pass
        logits = torch.from_numpy(self.engine._logits(stage.x, n))
        # per layer group: keys [NG, kv_heads, head_dim, C], values [NG, kv_heads, C, head_dim]
        keys = torch.cat([k.cpu()[..., :n] for k in stage.Ks]).transpose(-1, -2)
        values = torch.cat([v.cpu()[:, :, :n] for v in stage.Vs])
        return keys, values, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["torch", "rngd-qwen", "rngd-full"])
    ap.add_argument("--model", required=True); ap.add_argument("--model-id", default=None)
    ap.add_argument("--shm", required=True); ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True); ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True); ap.add_argument("--max-context", type=int, required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--width", default="512", help="compiled prompt widths (rngd-qwen, rngd-full), e.g. 256,512")
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
    elif a.engine == "rngd-full":
        engine = _RngdFull(a)
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
