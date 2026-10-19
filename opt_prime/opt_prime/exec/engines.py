#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Execution engine contract, the built-in engines and selection."""
import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from ..model.blocks import ModelSpec


def engine_geometry(spec: ModelSpec) -> dict:
    """the model properties a compiled engine is specialized for (shapes, block variants, dtype); two models with
    the same geometry run on the same engine build with different weights."""
    attn = {(l.attention.kind, l.attention.heads, l.attention.kv_heads, l.attention.head_dim, l.attention.qk_norm,
             l.attention.qkv_bias, l.attention.sliding_window) for l in spec.layers}
    moe = {(m.router.scoring, m.router.top_k, m.router.n_group, m.router.topk_group, m.experts.num_experts,
            m.experts.ffn, m.experts.gate, m.experts.activation, m.shared_ffn, m.shared_gate)
           for m in (l.moe for l in spec.layers if l.moe is not None)}
    mlp = {(l.mlp.kind, l.mlp.ffn, l.mlp.activation) for l in spec.layers if l.mlp is not None}
    return dict(model_type=spec.model_type, hidden=spec.hidden, layers=len(spec.layers), vocab=spec.vocab,
                dtype=spec.dtype, tied=spec.tied_embeddings, attention=sorted(map(list, attn)),
                moe=sorted(map(list, moe)), mlp=sorted(map(list, mlp)))


class ExecutionEngine:
    """supports(spec) -> reasons it cannot run the model ([] = can); expected_tok_s(spec) -> decode tok/s or None;
    start(model_id); generate(ids, max_new) -> (tokens, info); close()."""
    name = "base"

    def supports(self, spec: ModelSpec) -> List[str]:
        return []

    def expected_tok_s(self, spec: ModelSpec) -> Optional[float]:
        return None

    def start(self, model_id: str):
        raise NotImplementedError

    def generate(self, ids, max_new: int = 64) -> Tuple[List[int], dict]:
        raise NotImplementedError

    def close(self):
        pass


_ENGINES: Dict[str, ExecutionEngine] = {}


def register_engine(engine: ExecutionEngine):
    _ENGINES[engine.name] = engine
    return engine


def registered_engines() -> List[ExecutionEngine]:
    return list(_ENGINES.values())


def select_engine(spec: ModelSpec, candidates=None):
    """-> (engine, table) with table rows (name, reasons, expected tok/s); fastest supporting engine first, engines
    without an estimate after measured ones."""
    rows = []
    for e in candidates if candidates is not None else registered_engines():
        rows.append((e, e.supports(spec), e.expected_tok_s(spec)))
    ok = [r for r in rows if not r[1]]
    if not ok:
        raise ValueError("no engine supports this model: " + "; ".join(f"{e.name}: {', '.join(r)}" for e, r, _ in rows))
    ok.sort(key=lambda r: (r[2] is None, -(r[2] or 0.0)))
    return ok[0][0], [(e.name, reasons, tps) for e, reasons, tps in rows]


class OptimusEngine(ExecutionEngine):
    """Optimus_Inference on the GPU, optionally with MoE expert tiers (moe_offload). Supports any model torch runs;
    its speed depends on the tiers, so it reports no fixed estimate unless given one."""
    name = "optimus"

    def __init__(self, moe_offload=None, tok_s: Optional[float] = None, **inference_kw):
        self.moe_offload, self.tok_s, self.kw = moe_offload, tok_s, inference_kw
        self.engine = None

    def expected_tok_s(self, spec):
        return self.tok_s

    def start(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM
        from ..inference import Optimus_Inference
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        kw = dict(use_gpu=True, dtype=torch.bfloat16, use_kv_cache=True, dynamo_capture=True)
        kw.update(self.kw)
        self.engine = Optimus_Inference(model, moe_offload=self.moe_offload, **kw)
        self.engine.eval()

    def generate(self, ids, max_new=64):
        import torch
        t = torch.tensor([list(ids)]).cuda()
        t0 = time.time()
        out = self.engine.generate(t, max_new_tokens=max_new, do_sample=False, verbose=False)
        dt = time.time() - t0
        return out[0, len(ids):].tolist(), dict(seconds=dt)


class RngdLayerEngine(ExecutionEngine):
    """A native RNGD layer engine binary in serve mode (`<binary> <data_dir> serve`, JSON lines:
    {"ids","max_new"} -> {"tok"}* {"done", ...}). Its data directory holds engine_manifest.json:

        {"geometry": engine_geometry(...) of the model it was built for, "chips": [0, 1],
         "tok_s": measured exact decode tok/s, "env": {...runtime settings...}}

    The engine leases its chips (role "generate") through opt_prime.devices for as long as it runs."""
    name = "rngd-layer"

    def __init__(self, binary: str, data_dir: str, env: Optional[dict] = None, ready_timeout: float = 1800):
        self.binary, self.data_dir, self.env, self.ready_timeout = binary, data_dir, env or {}, ready_timeout
        path = os.path.join(data_dir, "engine_manifest.json")
        self.manifest = json.load(open(path)) if os.path.exists(path) else None
        self.p, self.leases = None, []

    def supports(self, spec):
        if self.manifest is None:
            return [f"no engine_manifest.json in {self.data_dir}"]
        if not os.path.exists(self.binary):
            return [f"engine binary {self.binary} not found"]
        want, have = self.manifest["geometry"], engine_geometry(spec)
        return [f"{k}: engine {want.get(k)!r} != model {have.get(k)!r}" for k in sorted(set(want) | set(have))
                if json.dumps(want.get(k), sort_keys=True) != json.dumps(have.get(k), sort_keys=True)]

    def expected_tok_s(self, spec):
        return (self.manifest or {}).get("tok_s")

    def start(self, model_id=None):
        from ..devices import acquire
        chips = self.manifest.get("chips", [0])
        try:
            for c in chips:
                self.leases.append(acquire(f"rngd:{c}", "generate", f"rngd-layer engine {self.binary}"))
        except BaseException:
            self._release()
            raise
        env = dict(os.environ)
        env.update(self.manifest.get("env", {}))
        env.update(self.env)
        env["FURIOSA_VISIBLE_CHIPS"] = ",".join(map(str, chips))
        self.p = subprocess.Popen([self.binary, self.data_dir, "serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
        t0 = time.time()
        line = self.p.stdout.readline()
        if line.strip() != "READY":
            self.close()
            raise RuntimeError(f"rngd-layer engine did not start (got {line!r})")
        self.load_s = time.time() - t0

    def generate(self, ids, max_new=64):
        self.p.stdin.write(json.dumps({"ids": list(ids), "max_new": int(max_new), "stop": [], "stop_count": 1})
                           + "\n")
        self.p.stdin.flush()
        toks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("rngd-layer engine exited")
            r = json.loads(line)
            if "tok" in r:
                toks.append(r["tok"])
            elif r.get("done"):
                return toks, r
            elif "error" in r:
                raise RuntimeError(r["error"])

    def _release(self):
        for lease in self.leases:
            lease.release()
        self.leases = []

    def close(self):
        if self.p is not None and self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.p = None
        self._release()
