"""Finite actual RNGD→ATOM Qwen P2P validation using the common PDRuntime."""

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--production-reference",
        type=Path,
        help="Run without worker oracle; compare tokens with a prior validated runs.jsonl",
    )
    p.add_argument("--test-cancel-drain", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    out = Path(cfg["out"]).resolve()
    expected = None
    if a.production_reference:
        cfg.pop("validation_hook", None)
        expected = {
            (tuple(r["prompt"]), r["output_length"]): r["tokens"]
            for r in (
                json.loads(line)
                for line in a.production_reference.read_text().splitlines()
            )
        }
    else:
        cfg["validation_hook"] = str(
            Path(__file__).with_name("qwen_reference.py").resolve()
        )
    cfg["socket"] = str(out / "bridge.sock")
    if len(os.fsencode(cfg["socket"])) >= 108:
        raise ValueError(
            "output directory is too long for Linux AF_UNIX; choose a shorter path"
        )
    out.mkdir(parents=True, exist_ok=False)
    config_path = out / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    from opt_prime.accel.pd.npu.provenance import record_run

    record_run(out, __file__)
    live = False
    status = {"passed": False}
    sidecar = None
    if cfg.get("sidecar_binary"):
        from sidecar_validation import validate_sidecar

        sidecar = validate_sidecar
    try:
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        from opt_prime.accel.pd import PDRequest
        from opt_prime.accel.pd.npu.model import DenseModel, load_weights
        from opt_prime.accel.pd.npu.factory import create_qwen_p2p
        from hetero_p2p import RngdPdmaBackend
        from hetero_p2p.serving import RngdBinding

        binding = RngdBinding.from_path(
            cfg["rngd_device_path"], cfg["rngd_logical_index"]
        )
        state = subprocess.check_output(["furiosa-smi", "ps"], text=True)
        (out / "rngd-before.txt").write_text(state)
        if f"npu{binding.index}:" in state:
            raise RuntimeError("RNGD in use")
        spec = DenseModel.load(cfg["model"], cfg["model_id"])
        torch.set_num_threads(8)
        from opt_prime.accel.platforms.rngd.pd import QwenRngdPrefill
        from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker

        engine = QwenRngdPrefill(
            spec,
            load_weights(cfg["model"], spec),
            width=64,
            chip=binding.compute_index,
            accurate_attention=cfg.get("accurate_attention", True),
        )
        descriptors = []
        for path in Path("/proc/self/fd").iterdir():
            try:
                descriptors.append(os.readlink(path))
            except OSError:
                pass
        binding.verify_open_channels(descriptors)
        (out / "rngd-binding.json").write_text(json.dumps(descriptors))
        compact_cache = cfg.get("rngd_compact_cache", False)
        if type(compact_cache) is not bool:
            raise ValueError("rngd_compact_cache must be boolean")
        if compact_cache:
            from opt_prime.accel.platforms.rngd.pd_packing import CompactPrefixPacker

            packer_class = CompactPrefixPacker
        else:
            packer_class = BoundedPrefixPacker
        packer = packer_class(
            binding.torch_device, spec.layers, spec.kv_heads, spec.head_dim
        )
        backend = RngdPdmaBackend(cfg["helper"], binding.device_path)
        session = create_qwen_p2p(
            config_path, engine, binding, packer, backend, out / "atom"
        )
        live = True
        runtime = session.runtime
        source, decoder, transport = runtime.prefill, runtime.decode, runtime.transport
        conn = session.worker.conn
        (out / "processes.json").write_text(
            json.dumps(dict(producer=os.getpid(), worker=session.worker.process.pid))
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
        base = tokenizer.encode(
            "Explain how computer memory works. ", add_special_tokens=False
        )
        prompts = cfg.get(
            "prompt_token_ids", [(base * 20)[:length] for length in (32, 64)]
        )
        if not prompts or any(
            not 2 <= len(prompt) <= 65
            or any(type(t) is not int or not 0 <= t < spec.vocabulary for t in prompt)
            for prompt in prompts
        ):
            raise ValueError("prompts must fit the compiled64-token prefix")
        for i, prompt in enumerate(prompts if expected is None else []):
            keys, values, _ = engine.prefill_device(prompt[:-1])
            prefix = out / f"prefix-{i}.npz"
            np.savez(
                prefix,
                keys=keys.cpu().float().numpy(),
                values=values.cpu().float().numpy(),
            )
            conn.send(
                json.dumps(
                    dict(
                        cmd="reference",
                        generation=-i - 1,
                        prompt=prompt,
                        count=32,
                        prefix=str(prefix),
                    )
                ).encode()
            )
            reply = json.loads(conn.recv(65536))
            if (
                reply.get("passed") is not True
                or reply.get("device_completed") is not True
            ):
                raise RuntimeError("reference failed")
            (out / f"reference-{i}.json").write_text(json.dumps(reply, indent=2))
        if a.test_cancel_drain:
            from opt_prime.accel.pd.runtime import CancellationToken, RequestCancelled

            token = CancellationToken()
            original_transfer = transport.transfer
            before_generation = decoder.generation

            def cancel_after_fence(*args):
                receipt = original_transfer(*args)
                token.cancel()
                return receipt

            transport.transfer = cancel_after_fence
            try:
                try:
                    runtime.generate(
                        PDRequest("cancel-fenced", tuple(prompts[0]), 8),
                        cancellation=token,
                    )
                except RequestCancelled:
                    if (
                        runtime.state != "idle"
                        or runtime.retained
                        or decoder.generation != before_generation
                    ):
                        raise RuntimeError("cancelled DMA did not drain without decode")
                    status["cancel_drain"] = dict(
                        passed=True, receipt=runtime.last_receipt, decode_skipped=True
                    )
                else:
                    raise RuntimeError("cancellation was not observed")
            finally:
                transport.transfer = original_transfer
        rows = []
        for repeat in range(2):
            for i, prompt in enumerate(prompts):
                for count in (8, 32):
                    request = PDRequest(
                        f"r{repeat}-p{i}-o{count}", tuple(prompt), count
                    )
                    tokens = runtime.generate(request)
                    if (
                        expected is not None
                        and tokens != expected[(tuple(prompt), count)]
                    ):
                        raise RuntimeError(
                            "production tokens differ from validated reference"
                        )
                    row = dict(
                        request_id=request.request_id,
                        prompt=prompt,
                        tokens=tokens,
                        repeat=repeat,
                        warmup=False,
                        input_length=len(prompt),
                        output_length=count,
                        stage_timings=runtime.last_timings,
                        receipt=runtime.last_receipt,
                        reply=decoder.last_reply,
                    )
                    rows.append(row)
                    with (out / "runs.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print("PASS", request.request_id, flush=True)
        if sidecar is not None:
            cases = [
                dict(prompt=r["prompt"], count=r["output_length"], expected=r["tokens"])
                for r in rows
                if r["repeat"] == 0
            ]
            status["sidecar"] = sidecar(
                source,
                decoder,
                transport,
                cases,
                binary=cfg["sidecar_binary"],
                output=out / "llmd",
                model=spec.model_id,
                vocab_size=spec.vocabulary,
                context=512,
                max_output=32,
                render=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            )
        if cfg.get("http_cancel_test"):
            from validate_http_cancel import validate_http_cancel

            first = rows[0]
            status["http_cancel"] = validate_http_cancel(
                source,
                decoder,
                transport,
                dict(
                    prompt=first["prompt"],
                    count=first["output_length"],
                    expected=first["tokens"],
                ),
                output=out / "http-cancel",
                spec=spec,
                render=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            )
        session.close()
        live = False
        code = session.worker.process.returncode
        status.update(
            passed=True,
            requests=len(rows),
            tokens=sum(len(r["tokens"]) for r in rows),
            commands=sum(r["receipt"]["completed"] for r in rows),
            worker_exit=code,
            scope=(
                "production P2P tokens match prior validated run; worker oracle disabled"
                if expected is not None
                else "actual P2P vs same-decoder host admission full logits; FP32 quality separate"
            ),
        )
        print(json.dumps(status), flush=True)
    except BaseException as error:
        status.update(error=repr(error), traceback=traceback.format_exc())
        (out / "result.json").write_text(json.dumps(status, indent=2))
        if live:
            (out / "quarantine.json").write_text(
                json.dumps(dict(producer=os.getpid(), reason=repr(error)))
            )
            print("QUARANTINED: owners retained pending recovery", flush=True)
            while True:
                time.sleep(30)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
