"""Actual RNGD Qwen prefix against a local FP32 full-attention oracle."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--accurate-attention", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    status = {"passed": False, "accurate_attention": a.accurate_attention}
    from opt_prime.pd.npu.provenance import record_run

    record_run(a.out, __file__)
    try:
        import torch
        import numpy as np
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from opt_prime.pd.npu.model import DenseModel, load_weights
        from hetero_p2p.serving import RngdBinding

        binding = RngdBinding.from_path("/dev/rngd/npu3ch0", 2)
        state = subprocess.check_output(["furiosa-smi", "ps"], text=True)
        (a.out / "device-before.txt").write_text(state)
        if "npu3:" in state:
            raise RuntimeError("RNGD3 is in use")
        torch.set_num_threads(8)
        spec = DenseModel.load(a.model, a.model_id)
        weights = load_weights(a.model, spec)
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        base = tokenizer.encode(
            "Explain how computer memory works. ", add_special_tokens=False
        )
        prompt = (base * 10)[:31]
        oracle = AutoModelForCausalLM.from_pretrained(
            a.model,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        with torch.inference_mode():
            expected = oracle(
                torch.tensor([prompt]), use_cache=True, output_hidden_states=True
            )
        from opt_prime.pd.npu.rngd import QwenRngdPrefill

        start = time.perf_counter()
        engine = QwenRngdPrefill(
            spec, weights, width=32, accurate_attention=a.accurate_attention
        )
        status["build_seconds"] = time.perf_counter() - start
        import os

        channels = []
        for f in Path("/proc/self/fd").iterdir():
            try:
                channels.append(os.readlink(f))
            except OSError:
                pass
        binding.verify_open_channels(channels)
        (a.out / "binding.json").write_text(json.dumps(channels))
        rows = []
        for repeat in range(4):
            start = time.perf_counter()
            keys, values, hidden = engine.prefill_device(prompt)
            seconds = time.perf_counter() - start
            rows.append(dict(repeat=repeat, warmup=repeat == 0, seconds=seconds))
        k = keys.cpu().float()
        v = values.cpu().float()
        h = hidden.cpu().float()
        errors = []
        for i, layer in enumerate(expected.past_key_values.layers):
            for name, actual, target in [
                ("k", k[i, :, :, :31].transpose(-1, -2), layer.keys[0]),
                ("v", v[i, :, :31], layer.values[0]),
            ]:
                relative = float(
                    torch.linalg.vector_norm(actual - target)
                    / torch.linalg.vector_norm(target).clamp_min(1e-9)
                )
                errors.append(
                    dict(
                        layer=i,
                        kind=name,
                        relative_l2=relative,
                        max_abs=float((actual - target).abs().max()),
                    )
                )
        norm = (
            h[30]
            * torch.rsqrt(h[30].square().mean() + spec.epsilon)
            * weights["model.norm.weight"]
        )
        logits = norm @ weights["lm_head.weight"].T
        target = expected.logits[0, -1]
        status.update(
            passed=max(e["relative_l2"] for e in errors) < 0.1
            and int(logits.argmax()) == int(target.argmax()),
            errors=errors,
            rows=rows,
            token=int(logits.argmax()),
            reference_token=int(target.argmax()),
            logits_relative_l2=float(
                torch.linalg.vector_norm(logits - target)
                / torch.linalg.vector_norm(target)
            ),
            scope="RNGD prefix computation; no ATOM or P2P",
        )
        np.savez(
            a.out / "prefix.npz",
            keys=k.numpy(),
            values=v.numpy(),
            prompt=np.array(prompt),
            hidden=h.numpy(),
        )
        (a.out / "device-loaded.txt").write_text(
            subprocess.check_output(["furiosa-smi", "ps"], text=True)
        )
        print(json.dumps(status), flush=True)
    except BaseException as e:
        status.update(error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        (a.out / "result.json").write_text(json.dumps(status, indent=2))
    if not status["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
