"""Probe vendor-native KV APIs without DMA/export or bypassing cache ownership.

Native prefill primes wrapper metadata before each restore. This is a feasibility
probe, not a ready-to-serve native PD adapter or a performance optimization.
"""

import argparse
import json
import importlib.metadata
from pathlib import Path
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--native-artifact", type=Path, required=True)
    p.add_argument("--prefix-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    status = {"passed": False}
    from opt_prime.pd.npu.provenance import record_run

    record_run(a.out, __file__)
    try:
        from opt_prime.pd.npu.atom_device import select_idle_atom, verify_atom_binding

        selection = select_idle_atom(cfg["physical_atom"])
        import torch
        import numpy as np
        from optimum.rbln import RBLNAutoModelForCausalLM

        if importlib.metadata.version("rebel-compiler") != "0.11.0":
            raise RuntimeError("raw KV encoding requires validated rebel0.11.0")
        from rebel.kv_cache import aligned_tensor
        from opt_prime.pd.npu.model import DenseModel
        from rngd_pd.native_v4.dlfloat import bf16_expanded_to_dlfloat

        torch.set_num_threads(8)
        spec = DenseModel.load(cfg["model"], cfg["model_id"])
        provenance = json.loads(
            (a.native_artifact.parent / "build-config.json").read_text()
        )
        if Path(provenance["model"]).resolve() != Path(cfg["model"]).resolve():
            raise ValueError("native checkpoint identity mismatch")
        config = json.loads((a.native_artifact / "rbln_config.json").read_text())
        if config.get("dtype") != "bfloat16":
            raise ValueError("requires the validated BF16 native artifact")
        metas = config["kvcache_metas"]
        if (
            config["batch_size"] != 1
            or config["max_seq_len"] != 512
            or len(metas) != spec.caches
        ):
            raise ValueError("unsupported native cache geometry")
        shape = [1, spec.kv_heads, 512, spec.head_dim]
        if any(
            m["shape"] != shape or m["name"] != f"past_key_values_{i}"
            for i, m in enumerate(metas)
        ):
            raise ValueError("unexpected named native cache order")
        model = RBLNAutoModelForCausalLM.from_pretrained(
            str(a.native_artifact), rbln_config={"create_runtimes": True, "device": 0}
        )
        (a.out / "binding.json").write_text(
            json.dumps(verify_atom_binding(selection), indent=2)
        )
        runtime = model.model[1]
        layer_bytes = spec.kv_heads * 512 * spec.head_dim * 2
        if runtime.get_kv_cache_size() != spec.caches * layer_bytes:
            raise ValueError("unexpected total physical KV bytes")
        saved = []
        for m in metas:
            if runtime.get_kv_cache_size_for_layer(m["name"]) != layer_bytes:
                raise ValueError("unexpected layer physical KV bytes")
            saved.append(
                aligned_tensor(
                    layer_bytes // 2,
                    dtype=np.uint16,
                    alignment=2 * 1024 * 1024,
                    tensor_type="np",
                )
            )
        stage = aligned_tensor(
            layer_bytes // 2,
            dtype=np.uint16,
            alignment=2 * 1024 * 1024,
            tensor_type="np",
        )

        def prime(prompt):
            n = len(prompt) - 1
            result = model.prefill_decoder(
                input_ids=torch.tensor([prompt[:-1]]),
                cache_position=torch.arange(n, dtype=torch.int32)[None, :].contiguous(),
                batch_idx=0,
            )
            if int(result.padded_cache_lengths) != 0:
                raise ValueError(
                    "nonzero prefix position padding needs explicit conversion"
                )

        def decode(prompt):
            token = prompt[-1]
            rows = []
            tokens = []
            for i in range(32):
                out = model.decoders[1](
                    input_ids=torch.tensor([[token]]),
                    cache_position=torch.tensor(
                        [[len(prompt) - 1 + i]], dtype=torch.int32
                    ),
                )
                logits = out.logits[0, -1, :].float().cpu().numpy().copy()
                if not np.isfinite(logits).all():
                    raise ValueError("nonfinite native logits")
                rows.append(logits)
                token = int(logits.argmax())
                tokens.append(token)
            return np.stack(rows), tokens

        rows = []
        for i, prompt in enumerate(cfg["prompt_token_ids"]):
            prime(prompt)
            for meta, buf in zip(metas, saved):
                runtime.get_kv_cache_for_layer(meta["name"], 0, 0, 512, buf)
            reference, tokens = decode(prompt)
            prime(prompt)
            for meta, buf in zip(metas, saved):
                runtime.set_kv_cache_for_layer(meta["name"], 0, 0, 512, buf)
            replay, replayed = decode(prompt)
            if not np.array_equal(reference, replay):
                raise ValueError("native KV roundtrip changed logits")
            prime(prompt)  # Metadata primer only; replaced KV follows.
            with np.load(a.prefix_dir / f"prefix-{i}.npz") as kv:
                values = np.stack(
                    (kv["keys"].transpose(0, 1, 3, 2), kv["values"]), axis=1
                ).reshape(spec.caches, spec.kv_heads, 512, spec.head_dim)
            started = time.perf_counter()
            for meta, value in zip(metas, values):
                stage[:] = bf16_expanded_to_dlfloat(value).reshape(-1)
                runtime.set_kv_cache_for_layer(meta["name"], 0, 0, 512, stage)
            admission = time.perf_counter() - started
            resumed, resumed_tokens = decode(prompt)
            row = dict(
                prompt=prompt,
                tokens=resumed_tokens,
                repeat=0,
                output_length=32,
                input_length=len(prompt),
                native_tokens=tokens,
                native_roundtrip_exact=True,
                admission_seconds=admission,
                agreement_with_native=sum(
                    x == y for x, y in zip(tokens, resumed_tokens)
                ),
            )
            rows.append(row)
            with (a.out / "runs.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print("NATIVE_RESUME", i, row["agreement_with_native"], flush=True)
        status.update(
            passed=True,
            requests=len(rows),
            scope="native KV roundtrip exact; RNGD host-admitted native decode finite; metadata primer retained; not production PD",
        )
    except BaseException as e:
        status.update(error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        (a.out / "result.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
