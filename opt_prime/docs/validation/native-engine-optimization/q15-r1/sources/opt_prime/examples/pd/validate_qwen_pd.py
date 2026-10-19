"""Finite actual RNGD→ATOM Qwen P2P validation using the common PDRuntime."""

import argparse
import array
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    out = Path(cfg["out"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg["socket"] = str(out / "bridge.sock")
    if len(os.fsencode(cfg["socket"])) >= 108:
        raise ValueError("output directory is too long for Linux AF_UNIX; choose a shorter path")
    config_path = out / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    from opt_prime.pd.npu.provenance import record_run

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
        from opt_prime.pd import PDRuntime, PDRequest
        from opt_prime.pd.npu.model import DenseModel, load_weights
        from opt_prime.pd.npu.engines import QwenRngdEngine, QwenAtomWorkerEngine
        from opt_prime.pd.hetero_p2p import HeteroP2PTransport
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
        from opt_prime.pd.npu.rngd import QwenRngdPrefill
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
        packer = BoundedPrefixPacker(
            binding.torch_device, spec.layers, spec.kv_heads, spec.head_dim
        )
        backend = RngdPdmaBackend(cfg["helper"], binding.device_path)
        example = Path(cfg["hetero_repo"]) / "examples/koni"
        sys.path.insert(0, str(example))
        from batch_control import accept_worker

        directory = out / "atom"
        directory.mkdir()
        (directory / "capture.gdb").write_text((example / "capture.gdb").read_text())
        server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        server.bind(cfg["socket"])
        server.listen(1)
        env = dict(os.environ, HETERO_CAPTURE_DIR=str(directory))
        logfile = (directory / "worker.log").open("w")
        worker = subprocess.Popen(
            [
                "gdb",
                "-q",
                "-batch",
                "-x",
                str(directory / "capture.gdb"),
                "--args",
                cfg["atom_python"],
                str(Path(__file__).with_name("qwen_atom_worker.py").resolve()),
                "--config",
                str(config_path),
                "--directory",
                str(directory),
            ],
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
        (out / "processes.json").write_text(
            json.dumps(dict(producer=os.getpid(), worker=worker.pid))
        )
        conn, _ = accept_worker(server, {0: worker})
        raw, anc, flags, _ = conn.recvmsg(65536, socket.CMSG_SPACE(spec.caches * 4))
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated export descriptors")
        fds = array.array("i")
        for level, kind, data in anc:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(data)
        live = True
        metadata = json.loads(raw)
        if (
            metadata["spec"] != asdict(spec)
            or len(metadata["mappings"]) != spec.caches
            or len(fds) != len(metadata["blocks"])
        ):
            raise RuntimeError("export metadata mismatch")
        for fd, block in zip(fds, metadata["blocks"]):
            if os.fstat(fd).st_size != block["size"]:
                raise RuntimeError("export size mismatch")
        owner = dict(conn=conn, fds=fds, metadata=metadata, process=worker)
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
        for i, prompt in enumerate(prompts):
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
        source = QwenRngdEngine(engine, packer, binding)
        decoder = QwenAtomWorkerEngine(owner, source.spec, cfg["physical_atom"])
        transport = HeteroP2PTransport(
            backend,
            decoder.build_plan,
            lambda _: None,
            source=binding.device_path,
            destination=f"atom:{cfg['physical_atom']}",
            spec=source.spec,
        )
        runtime = PDRuntime(source, decoder, transport)
        rows = []
        for repeat in range(2):
            for i, prompt in enumerate(prompts):
                for count in (8, 32):
                    request = PDRequest(
                        f"r{repeat}-p{i}-o{count}", tuple(prompt), count
                    )
                    tokens = runtime.generate(request)
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
        conn.send(json.dumps(dict(cmd="stop")).encode())
        code = worker.wait(timeout=60)
        if (
            code != 0
            or not json.loads((directory / "result.json").read_text())["passed"]
        ):
            raise RuntimeError("owner did not exit normally")
        live = False
        for fd in fds:
            os.close(fd)
        conn.close()
        server.close()
        logfile.close()
        status.update(
            passed=True,
            requests=len(rows),
            tokens=sum(len(r["tokens"]) for r in rows),
            commands=sum(r["receipt"]["completed"] for r in rows),
            worker_exit=code,
            scope="actual P2P vs same-decoder host admission full logits; FP32 quality separate",
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
