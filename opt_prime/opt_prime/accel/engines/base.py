#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Execution engine contract, the built-in engines and selection."""
import json
from dataclasses import asdict

from typing import Dict, List, Optional, Tuple

from opt_prime.accel.spec.blocks import ModelSpec


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
                dtype=spec.dtype, tied=spec.tied_embeddings, attention=sorted(map(list, attn), key=lambda row: json.dumps(row)),
                moe=sorted(map(list, moe), key=lambda row: json.dumps(row)),
                mlp=sorted(map(list, mlp), key=lambda row: json.dumps(row)),
                geometry_version=2,
                layer_specs=[json.loads(json.dumps(asdict(layer), default=str)) for layer in spec.layers])


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
