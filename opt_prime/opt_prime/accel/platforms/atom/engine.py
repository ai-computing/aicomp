#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""ATOM whole-model engine: the decoder compiled by optimum-rbln."""
import json
import os
import time
from typing import Optional

from opt_prime.accel.engines.base import ExecutionEngine


class AtomRblnEngine(ExecutionEngine):
    """Rebellions ATOM through optimum-rbln: the whole decoder compiled ahead of time into RBLN prefill and decode
    graphs, so a decode step is one graph call (per-op RBLN calls cost ~60 us each). Compiled once per (model,
    tensor-parallel size, max_seq_len, batch) into `cache_dir` and loaded from there afterwards; a measured
    decode rate saved next to it (`engine_manifest.json`, {"tok_s"}) becomes the selection estimate.

    The engine leases its ATOM chips (role "generate") through opt_prime.accel.platforms while it runs."""
    name = "atom-rbln"
    #: model_type -> optimum-rbln class (0.11)
    CLASSES = {"llama": "RBLNLlamaForCausalLM", "mistral": "RBLNMistralForCausalLM", "qwen2": "RBLNQwen2ForCausalLM",
               "qwen3": "RBLNQwen3ForCausalLM", "gemma3_text": "RBLNGemma3ForCausalLM",
               "qwen3_moe": "RBLNQwen3MoeForCausalLM", "qwen2_moe": "RBLNQwen2MoeForCausalLM",
               "mixtral": "RBLNMixtralForCausalLM", "exaone": "RBLNExaoneForCausalLM"}

    def __init__(self, devices=(0,), max_seq_len: int = 4096, batch_size: int = 1, cache_dir: Optional[str] = None,
                 tok_s: Optional[float] = None):
        self.devices, self.max_seq_len, self.batch_size = list(devices), max_seq_len, batch_size
        self.cache_dir = cache_dir or os.path.join(os.environ.get("OPT_PRIME_CACHE", os.path.expanduser(
            "~/.cache/opt_prime")), "atom-rbln")
        self.tok_s, self.model, self.leases, self.dir = tok_s, None, [], None

    def supports(self, spec):
        reasons = []
        try:
            import optimum.rbln  # noqa: F401
        except ImportError:
            reasons.append("optimum-rbln not installed")
        if spec.model_type not in self.CLASSES:
            reasons.append(f"optimum-rbln class for {spec.model_type} not mapped ({sorted(self.CLASSES)})")
        from opt_prime.accel.platforms import discover
        have = {d.index for d in discover(["atom"])}
        missing = [d for d in self.devices if d not in have]
        if missing:
            reasons.append(f"ATOM devices {missing} not present")
        return reasons

    def _dir(self, model_id):
        slug = str(model_id).rstrip("/").replace("/", "--")
        return os.path.join(self.cache_dir, f"{slug}-tp{len(self.devices)}-s{self.max_seq_len}-b{self.batch_size}")

    def expected_tok_s(self, spec):
        return self.tok_s

    def start(self, model_id):
        if self.model is not None or self.leases:
            raise RuntimeError("engine already started")
        import optimum.rbln as orbln
        from transformers import AutoConfig
        from opt_prime.accel.platforms import acquire
        try:
            for d in self.devices:
                self.leases.append(acquire(f"atom:{d}", "generate", f"atom-rbln engine {model_id}"))
        except BaseException:
            self._release()
            raise
        try:
            cls = getattr(orbln, self.CLASSES[AutoConfig.from_pretrained(model_id).model_type])
            from opt_prime.accel.platforms import discover
            mapping = {d.index: d.extra.get("visible_npu_id") for d in discover(["atom"])}
            if any(mapping.get(d) is None for d in self.devices):
                raise ValueError("physical ATOM devices require a host-group runtime mapping")
            runtime_devices = [int(mapping[d]) for d in self.devices]
            self.dir = self._dir(model_id)
            t0 = time.time()
            if os.path.exists(os.path.join(self.dir, "config.json")):
                self.model = cls.from_pretrained(self.dir, export=False, rbln_config={"device": runtime_devices})
                self.compiled = False
            else:
                self.model = cls.from_pretrained(model_id, export=True, rbln_config={
                    "batch_size": self.batch_size, "max_seq_len": self.max_seq_len,
                    "tensor_parallel_size": len(self.devices), "device": runtime_devices})
                self.model.save_pretrained(self.dir)
                self.compiled = True
            self.load_s = time.time() - t0
            m = os.path.join(self.dir, "engine_manifest.json")
            if self.tok_s is None and os.path.exists(m):
                self.tok_s = json.load(open(m)).get("tok_s")
        except BaseException:
            self.close()
            raise

    def generate(self, ids, max_new=64):
        import torch

        class _Clock:                     # transformers streamer: put(prompt), then put(token) per new token
            def __init__(self):
                self.t = []

            def put(self, value):
                self.t.append(time.time())

            def end(self):
                pass
        clock = _Clock()
        t0 = time.time()
        out = self.model.generate(torch.tensor([list(ids)]), max_new_tokens=max_new, do_sample=False, streamer=clock)
        dt = time.time() - t0
        toks = out[0, len(ids):].tolist()
        t = clock.t[1:]                   # new-token times
        info = dict(seconds=dt, ttft_s=(t[0] - t0) if t else None,
                    decode_tok_s=(len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else None)
        return toks, info

    def batcher(self, **kw):
        """continuous batching over the compiled decode graph (needs batch_size > 1; #44 stage 3)"""
        if self.model is None:
            raise RuntimeError("engine not started")
        from opt_prime.accel.platforms.atom.batching import AtomBatcher
        return AtomBatcher(self.model, **kw)

    def record_tok_s(self, tok_s: float):
        """save a measured decode rate next to the compiled model (the selection estimate from then on)."""
        self.tok_s = tok_s
        with open(os.path.join(self.dir, "engine_manifest.json"), "w") as f:
            json.dump({"tok_s": tok_s, "devices": self.devices, "max_seq_len": self.max_seq_len}, f)

    def _release(self):
        for lease in self.leases:
            lease.release()
        self.leases = []

    def close(self):
        self.model = None
        self._release()
