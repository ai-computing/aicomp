"""Exclusive external-cache ATOM worker; validation is an optional harness hook."""

import argparse
import array
import ctypes
import json
import socket
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    directory = a.directory
    from opt_prime.accel.pd.npu.provenance import record_run

    record_run(directory, __file__)
    result = {"passed": False}
    exports = None
    try:
        from opt_prime.accel.platforms.atom.device import select_idle_atom, verify_atom_binding

        selection = select_idle_atom(cfg["physical_atom"])
        import torch
        import numpy as np
        from opt_prime.accel.pd.npu.model import DenseModel
        from opt_prime.accel.platforms.atom.pd_runtime import QwenExternalDecoder
        from hetero_p2p.atom_experimental import export_live_caches

        torch.set_num_threads(8)
        spec = DenseModel.load(cfg["model"], cfg["model_id"])
        if json.loads(
            (Path(cfg["artifact"]).parent.parent / "model.json").read_text()
        ) != asdict(spec):
            raise ValueError("compiled checkpoint identity differs")
        engine = QwenExternalDecoder(cfg["artifact"], cfg["model"], spec)
        engine.profile_steps = True
        if type(cfg.get("atom_reuse_inputs", False)) is not bool:
            raise ValueError("atom_reuse_inputs must be boolean")
        engine.step_batch([0], [0])
        torch.rbln.synchronize()
        (directory / "binding.json").write_text(
            json.dumps(verify_atom_binding(selection), indent=2)
        )
        state = json.loads((directory / "live-capture.json").read_text())
        exports = export_live_caches(
            engine,
            engine.cache_device_addresses(),
            spec.kv_heads * 512 * spec.head_dim * 2,
            state,
            experimental=True,
        )
        (directory / "capture-complete").write_text("disable diagnostic breakpoints")
        ctypes.CDLL(None).getppid()
        if not (directory / "breakpoints-disabled").exists():
            raise RuntimeError("capture breakpoints still active")
        validator = None
        if cfg.get("validation_hook"):
            import importlib.util

            module_spec = importlib.util.spec_from_file_location(
                "pd_validation_hook", cfg["validation_hook"]
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            validator = module.ReferenceValidator()

        iteration = 0
        last_generation = 0
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as conn:
            conn.connect(cfg["socket"])
            fds = array.array("i", [r["fd"] for r in exports.records])
            conn.sendmsg(
                [
                    json.dumps(
                        dict(
                            protocol=1,
                            spec=asdict(spec),
                            mappings=exports.mappings,
                            blocks=exports.records,
                        )
                    ).encode()
                ],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)],
            )
            while True:
                raw = conn.recv(65536)
                if not raw:
                    raise RuntimeError("producer disconnected without completion")
                cmd = json.loads(raw)
                if cmd["cmd"] == "stop":
                    break
                if cmd["cmd"] not in ("reference", "decode"):
                    raise ValueError("invalid operation")
                prompt = cmd["prompt"]
                count = cmd["count"]
                if len(prompt) < 2 or count < 1 or len(prompt) + count > 512:
                    raise ValueError("invalid request")
                if cmd["cmd"] == "reference":
                    if validator is None:
                        raise ValueError(
                            "reference commands disabled outside validation harness"
                        )
                    validator.admit(engine, spec, cmd)
                else:
                    generation = cmd.get("generation")
                    if (
                        type(generation) is not int
                        or generation <= last_generation
                        or not cmd.get("request_id")
                    ):
                        raise ValueError("stale generation or missing request identity")
                    last_generation = generation
                    if cmd.get("dma_completed") is not True:
                        raise ValueError("DMA completion required")
                    engine.notify_cache_write(range(spec.caches))
                engine.reuse_inputs = cmd["cmd"] == "decode" and cfg.get(
                    "atom_reuse_inputs", False
                )
                engine.step_profile.clear()
                logits = []
                tokens = []
                current = prompt[-1]
                started = time.perf_counter()
                for step in range(count):
                    row = engine.step_batch([current], [len(prompt) - 1 + step])[0]
                    if not np.isfinite(row).all():
                        raise RuntimeError("nonfinite logits")
                    if validator is not None:
                        logits.append(row.copy())
                    current = int(row.argmax())
                    tokens.append(current)
                torch.rbln.synchronize()
                checked = (
                    validator.check(cmd, np.stack(logits))
                    if validator is not None
                    else {}
                )
                passed = checked.get("passed", True)
                reply = dict(
                    generation=cmd["generation"],
                    request_id=cmd.get("request_id"),
                    reuse_inputs=engine.reuse_inputs,
                    step_timings=list(engine.step_profile),
                    tokens=tokens,
                    passed=passed,
                    reference_checked=validator is not None,
                    max_abs_diff=checked.get("max_abs_diff"),
                    device_completed=True,
                    seconds=time.perf_counter() - started,
                )
                with (directory / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(dict(operation=cmd["cmd"], **reply)) + "\n")
                conn.send(json.dumps(reply).encode())
                iteration += 1
                if not passed:
                    raise RuntimeError("full-logit P2P equality failed")
        exports.close()
        exports = None
        result.update(passed=True, operations=iteration)
    except BaseException as e:
        result.update(error=repr(e), traceback=traceback.format_exc())
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        # Preserve exported owners if the producer may still hold a live DMA lease.
        if exports is not None:
            print("QUARANTINED", flush=True)
            while True:
                time.sleep(30)
        raise
    finally:
        (directory / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
