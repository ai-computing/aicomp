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
        logits = self._last_logits(stage.x, n)
        # per layer group: keys [NG, kv_heads, head_dim, C], values [NG, kv_heads, C, head_dim]
        keys = torch.cat([k.cpu()[..., :n] for k in stage.Ks]).transpose(-1, -2)
        values = torch.cat([v.cpu()[:, :, :n] for v in stage.Vs])
        return keys, values, logits

    def prefill_stream(self, tokens, emit):
        """Export each completed group before scheduling the next group."""
        import torch
        n = len(tokens)
        if not 0 < n <= self.widths[-1]:
            raise ValueError("prompt exceeds compiled capacity or is empty")
        stage = self.engine.begin_device_prefill(list(tokens))
        layer = 0
        while True:
            done = stage.advance(1)
            k, v = stage.Ks[-1], stage.Vs[-1]
            keys = k.cpu()[..., :n].transpose(-1, -2)
            values = v.cpu()[:, :, :n]
            emit(layer, 0, keys, values)
            layer += keys.shape[0]
            if done:
                break
        return self._last_logits(stage.x, n)

    def _last_logits(self, hidden, n):
        import torch
        engine = self.engine
        if callable(getattr(engine, "_logits", None)):
            return torch.from_numpy(engine._logits(hidden, n))
        # Older rngd-pd exposes the same on-card head only through _next_token.
        out = engine.lmh(hidden[n - 1], engine.lm_norm_d, engine.lm_eps_d,
                         engine.lm_nh_d, engine.wlm_d)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits.detach().cpu().float().reshape(-1)[:engine.V]


def stream_result(engine, req, slab, publish):
    """Copy validated source spans into reserved SHM before notifying the receiver.

    Engines call emit(layer_start, token_start, K, V) with dense [L,H,N,D] spans
    and return final logits. Calls are synchronous; source buffers may be reused
    after emit returns. Absolute token spans are clipped to the requested suffix.
    """
    import numpy as np
    import torch
    from opt_prime.accel.pd.layer_progress import LayerProgress
    slot, start, end = req["slot"], req["send_from"], len(req["tokens"])
    if (type(req["id"]) is not int or type(slot) is not int
            or not 0 <= slot < slab.shape[0] or type(start) is not int
            or not 0 <= start <= end <= slab.shape[-2] or end == 0):
        raise ValueError("invalid streamed request reservation")
    coverage = LayerProgress(slab.shape[2], start, end)

    def emit(layer, first, keys, values):
        if (type(layer) is not int or type(first) is not int
                or keys.ndim != 4 or values.shape != keys.shape):
            raise ValueError("invalid source KV span")
        layers, heads, count, dim = keys.shape
        if (layers <= 0 or count <= 0 or not 0 <= layer < layer + layers <= slab.shape[2]
                or heads != slab.shape[3] or dim != slab.shape[-1]
                or not 0 <= first < first + count <= end):
            raise ValueError("invalid source KV geometry")
        lo, hi = max(start, first), first + count
        if lo >= hi:
            return
        # Validate before touching any published memory. Failed copies abort this request.
        coverage.publish(layer, layer + layers, lo, hi)
        arrays = [t[:, :, lo - first:hi - first].to("cpu", torch.bfloat16)
                  .contiguous().view(torch.int16).numpy().view(np.uint16) for t in (keys, values)]
        dst = slab[slot, :, layer:layer + layers, :, lo - start:hi - start]
        dst[0], dst[1] = arrays
        coverage.drain()
        publish(dict(id=req["id"], kv_span=[layer, layer + layers, lo, hi]))

    logits = engine.prefill_stream(list(req["tokens"]), emit)
    coverage.finish()
    return logits


def factory_engine(a):
    """Load a trusted CLI factory using the same FP32 checkpoint inputs as research workers.

    Research-only: `--engine-factory module:callable` is imported and called without any allow-list, and the whole
    checkpoint is materialized on the host as fp32 (4 bytes per parameter) before the factory runs."""
    from importlib import import_module
    from pathlib import Path
    from safetensors import safe_open
    path = Path(a.model)
    config = json.loads((path / "config.json").read_text())
    width = int(a.width)
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    if (not 0 < width <= a.max_context or a.slots <= 0
            or (a.layers, a.kv_heads, a.head_dim) !=
            (config["num_hidden_layers"], config["num_key_value_heads"], head_dim)):
        raise ValueError("factory width or shared-memory geometry does not match checkpoint")
    module, sep, name = (a.engine_factory or "").partition(":")
    if not sep or not module or not name:
        raise ValueError("engine factory must be module:callable")
    factory = getattr(import_module(module), name)
    if not callable(factory):
        raise ValueError("engine factory must be callable")
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key in weights:
                    raise ValueError("duplicate tensor in checkpoint shards")
                weights[key] = f.get_tensor(key).float()
    if not weights:
        raise ValueError("factory requires local safetensors weights")
    return factory(config, weights, width=width, chip=a.chip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["torch", "rngd-qwen", "rngd-full", "factory"])
    ap.add_argument("--engine-factory",
                    help="--engine factory only: module:callable, imported and called as-is (a trusted research "
                         "flag: it runs arbitrary code from PYTHONPATH). Loads every *.safetensors tensor of "
                         "--model into host memory as fp32 first (about 4 bytes per parameter, e.g. ~32 GB for 8B)")
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
    elif a.engine == "factory":
        engine = factory_engine(a)
    else:
        engine = _RngdQwen(a)
    streaming = callable(getattr(engine, "prefill_stream", None))
    print(json.dumps({"ready": True, **({"kv_progress": 1} if streaming else {})}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            t0 = time.perf_counter()
            from opt_prime.accel.pd import PDRequest
            sampling = _sampling(req.get("sampling"))
            tokens, start = tuple(req["tokens"]), int(req["send_from"])
            if req.get("stream_layers"):
                if req["stream_layers"] is not True or not streaming:
                    raise ValueError("engine does not support requested layer streaming")
                logits = stream_result(engine, req, slab,
                                       lambda msg: print(json.dumps(msg), flush=True))
                token = select_token(logits, sampling, len(tokens))
                print(json.dumps(dict(id=req["id"], token=int(token), n=len(tokens) - start,
                                      cached=0, prefill_s=time.perf_counter() - t0)), flush=True)
                continue
            if a.engine == "torch":
                out = engine.prefill(PDRequest(f"p{req['id']}", tokens, 1, sampling), start)
                keys, values, token, cached = out.keys, out.values, out.token, out.prefill_cached
            else:
                keys, values, logits = engine.prefill(list(tokens))
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
