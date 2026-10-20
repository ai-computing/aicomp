"""Compare fresh native resume without prefill against the primer reference."""

import argparse
import json
import importlib.metadata
from pathlib import Path
import time
import traceback
import subprocess
import selectors
import time as clock


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--native-artifact", type=Path, required=True)
    p.add_argument("--prefix-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--live-rngd-python", type=Path)
    p.add_argument("--compare-shared", action="store_true")
    a = p.parse_args()
    if a.compare_shared and not a.live_rngd_python:
        p.error("--compare-shared requires live RNGD")
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    status = {"passed": False}
    from opt_prime.accel.pd.npu.provenance import record_run

    record_run(a.out, __file__)
    producer = None
    producer_log = None
    shared_fd = None
    slab = None

    def receive(marker):
        deadline = clock.monotonic() + 300
        # Read unbuffered bytes to avoid readline buffering versus select races.
        line = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(producer.stdout, selectors.EVENT_READ)
            while clock.monotonic() < deadline:
                if not selector.select(timeout=1):
                    continue
                chunk = producer.stdout.read(1)
                if not chunk:
                    raise RuntimeError("RNGD worker exited; inspect producer.log")
                line.extend(chunk)
                if chunk == b"\n":
                    text = line.decode(errors="replace").strip()
                    producer_log.write(bytes(line))
                    producer_log.flush()
                    line.clear()
                    if text.startswith(marker):
                        return text[len(marker) :].strip()
        raise TimeoutError("RNGD prefix timed out")

    try:
        if a.live_rngd_python:
            producer_log = (a.out / "producer.log").open("wb")
            extra = []
            if a.compare_shared:
                from opt_prime.accel.pd.npu.model import DenseModel

                geometry = DenseModel.load(cfg["model"], cfg["model_id"])
                from opt_prime.accel.pd.npu.shared_kv import SharedKVSlab

                slab = SharedKVSlab.create(
                    (geometry.caches, geometry.kv_heads, 512, geometry.head_dim)
                )
                shared_fd = slab.fd
                extra = ["--shared-fd", str(shared_fd)]
            producer = subprocess.Popen(
                [
                    str(a.live_rngd_python),
                    str(Path(__file__).with_name("native_rngd_prefix_worker.py")),
                    "--config",
                    str(a.config),
                    "--out",
                    str(a.out / "live-prefixes"),
                ]
                + extra,
                pass_fds=(() if shared_fd is None else (shared_fd,)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=producer_log,
                bufsize=0,
            )
            receive("PREFIX_READY")
        from opt_prime.accel.platforms.atom.device import select_idle_atom, verify_atom_binding

        selection = select_idle_atom(cfg["physical_atom"])
        import torch
        import numpy as np
        from optimum.rbln import RBLNAutoModelForCausalLM

        if importlib.metadata.version("rebel-compiler") != "0.11.0":
            raise RuntimeError("raw KV encoding requires validated rebel0.11.0")
        from rebel.kv_cache import aligned_tensor
        from opt_prime.accel.pd.npu.model import DenseModel
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

        from opt_prime.accel.platforms.atom.pd_native import NativeAtomDecodeLane

        lane = NativeAtomDecodeLane(model, config, spec)
        original_prefill = model.prefill_decoder
        forbidden_calls = []

        def forbidden_prefill(*args, **kwargs):
            forbidden_calls.append(True)
            raise AssertionError("no-primer lane invoked prefill")

        rows = []
        # The first generation runs before ANY native prefill execution.
        for repeat in range(6 if a.compare_shared else 2):
            for i, prompt in enumerate(cfg["prompt_token_ids"]):
                transport = (
                    "shared" if a.compare_shared and repeat in (1, 3, 4) else "npz"
                )
                request_start = time.perf_counter()
                prefix_path = a.prefix_dir / f"prefix-{i}.npz"
                if producer:
                    producer.stdin.write(
                        (
                            json.dumps(
                                dict(index=i, sequence=len(rows), transport=transport)
                            )
                            + "\n"
                        ).encode()
                    )
                    producer.stdin.flush()
                    reply = receive("PREFIX_RESULT ")
                    if transport == "shared":
                        if json.loads(reply) != dict(sequence=len(rows), index=i):
                            raise ValueError("shared KV response/request mismatch")
                    else:
                        prefix_path = Path(reply)
                if transport == "shared":
                    values = slab.view()
                else:
                    with np.load(prefix_path) as kv:
                        values = np.stack(
                            (kv["keys"].transpose(0, 1, 3, 2), kv["values"]), axis=1
                        ).reshape(spec.caches, spec.kv_heads, 512, spec.head_dim)
                prefix_seconds = time.perf_counter() - request_start
                model.prefill_decoder = forbidden_prefill
                try:
                    tokens, candidate, timing = lane.generate(prompt, values, 32)
                finally:
                    model.prefill_decoder = original_prefill
                request_seconds = time.perf_counter() - request_start
                start = time.perf_counter()
                prime(prompt)
                primer_seconds = time.perf_counter() - start
                for meta, value in zip(metas, values):
                    stage[:] = bf16_expanded_to_dlfloat(value).reshape(-1)
                    runtime.set_kv_cache_for_layer(meta["name"], 0, 0, 512, stage)
                reference, reference_tokens = decode(prompt)
                baseline_seconds = time.perf_counter() - start
                exact = bool(np.array_equal(reference, candidate))
                row = dict(
                    warmup=a.compare_shared and repeat < 2,
                    transport=transport,
                    prefix_seconds=prefix_seconds,
                    request_seconds=request_seconds,
                    prompt_index=i,
                    repeat=repeat,
                    input_length=len(prompt),
                    output_length=32,
                    tokens=tokens,
                    full_logits_exact=exact,
                    max_abs_error=float(np.max(np.abs(reference - candidate))),
                    primer_seconds=primer_seconds,
                    baseline_seconds=baseline_seconds,
                    no_primer=timing,
                )
                rows.append(row)
                with (a.out / "runs.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print("NO_PRIMER", repeat, i, exact, flush=True)
                if not exact:
                    raise ValueError(
                        "no-primer full logits differ from primer reference"
                    )
        status.update(
            passed=True,
            requests=len(rows),
            tokens=len(rows) * 32,
            forbidden_prefill_calls=len(forbidden_calls),
            fresh_first_decode=True,
            scope=("live RNGD host KV" if producer else "saved RNGD host KV")
            + "; B1 context512; full-logit parity; no P2P/HTTP",
        )
    except BaseException as e:
        status.update(error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        if producer:
            producer.stdin.close()
            try:
                producer.wait(timeout=15)
            except subprocess.TimeoutExpired:
                producer.terminate()
                try:
                    producer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    producer.kill()
                    producer.wait()
        if producer_log:
            producer_log.close()
        if slab is not None:
            values = None
            slab.close()
        (a.out / "result.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
