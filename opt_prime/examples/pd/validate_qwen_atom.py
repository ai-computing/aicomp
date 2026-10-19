"""Validate external-cache ATOM decode before enabling any P2P export."""

import argparse
import json
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--physical", default="rbln1")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    result = {"passed": False}
    from opt_prime.pd.npu.provenance import record_run

    record_run(a.out, __file__)
    try:
        from opt_prime.pd.npu.atom_device import select_idle_atom, verify_atom_binding

        selection = select_idle_atom(a.physical)
        import torch
        import numpy as np
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from opt_prime.pd.npu.model import DenseModel
        from opt_prime.pd.npu.atom_runtime import QwenExternalDecoder

        torch.set_num_threads(8)
        spec = DenseModel.load(a.model, a.model_id)
        engine = QwenExternalDecoder(a.artifact, a.model, spec)
        with np.load(a.prefix) as data:
            kv = np.stack(
                (data["keys"].transpose(0, 1, 3, 2), data["values"]), axis=1
            ).reshape(spec.caches, spec.kv_heads, 512, spec.head_dim)
        for cache, host in zip(engine.caches, kv):
            cache.copy_(torch.from_numpy(host[None].copy()))
        torch.rbln.synchronize()
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        base = tokenizer.encode(
            "Explain how computer memory works. ", add_special_tokens=False
        )
        prompt = (base * 10)[:32]
        tokens = []
        current = prompt[-1]
        logits = []
        for step in range(32):
            row = engine.step_batch([current], [31 + step])[0]
            if not np.isfinite(row).all():
                raise RuntimeError("nonfinite ATOM logits")
            current = int(row.argmax())
            tokens.append(current)
            logits.append(row.copy())
        torch.rbln.synchronize()
        (a.out / "binding.json").write_text(
            json.dumps(verify_atom_binding(selection), indent=2)
        )
        oracle = AutoModelForCausalLM.from_pretrained(
            a.model,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        with torch.inference_mode():
            target = oracle(
                torch.tensor([prompt + tokens[:-1]]), use_cache=False
            ).logits[0, 31:]
        reference = target.argmax(-1).tolist()
        matches = sum(x == y for x, y in zip(tokens, reference))
        actual = torch.tensor(np.stack(logits))
        result.update(
            passed=matches >= 29,
            matches=matches,
            total=32,
            tokens=tokens,
            prompt=prompt,
            reference=reference,
            logits_relative_l2=float(
                torch.linalg.vector_norm(actual - target)
                / torch.linalg.vector_norm(target)
            ),
            scope="RNGD prefix host-admitted into ATOM; CPU FP32 teacher forced; no P2P",
        )
        print(json.dumps(result), flush=True)
    except BaseException as e:
        result.update(error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        (a.out / "result.json").write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
