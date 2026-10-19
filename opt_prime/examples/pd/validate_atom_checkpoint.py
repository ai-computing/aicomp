"""Compile and measure a local checkpoint on a single ATOM using vendor APIs.

This is a native baseline, not an external-KV or heterogeneous-PD validation.
SDK imports are deliberately local; no changes to the common PD contracts.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", type=int, choices=[2], default=2)
    p.add_argument("--preflight-module-dir", type=Path)
    p.add_argument("--physical", help="Explicit host-group physical ATOM")
    p.add_argument("--prompts-json", type=Path)
    a = p.parse_args()
    if a.repeats < 1 or a.context < 320 or a.chunk <= 0 or a.chunk % 64:
        p.error("requires repeats>=1, context>=320, and a positive chunk multiple of64")
    a.out.mkdir(parents=True, exist_ok=False)
    write(
        a.out / "invocation.json",
        dict(
            argv=sys.argv,
            python=sys.executable,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            git_status=subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ),
        ),
    )
    status = dict(
        passed=False, scope="vendor native ATOM baseline; no external KV or PD"
    )
    try:
        selection = None
        if a.physical:
            from opt_prime.pd.npu.atom_device import (
                select_idle_atom,
                verify_atom_binding,
            )

            selection = select_idle_atom(a.physical)
        import torch
        from transformers import AutoTokenizer
        from optimum.rbln import RBLNAutoModelForCausalLM

        torch.set_num_threads(8)
        write(
            a.out / "versions.json",
            {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "rebel-compiler", "optimum-rbln")
            },
        )
        cfg = dict(
            num_devices=1,
            batch_size=1,
            max_seq_len=a.context,
            decoder_batch_sizes=[1],
            prefill_chunk_size=a.chunk,
            kvcache_num_blocks=1,
            npu="RBLN-CA22",
            create_runtimes=False,
        )
        write(
            a.out / "build-config.json",
            dict(
                model=str(a.model_dir.resolve()), revision=a.model_dir.name, config=cfg
            ),
        )
        artifact = a.artifact
        if artifact is None:
            start = time.perf_counter()
            model = RBLNAutoModelForCausalLM.from_pretrained(
                str(a.model_dir), export=True, rbln_config=cfg, local_files_only=True
            )
            artifact = a.out / "compiled"
            model.save_pretrained(artifact)
            del model
            status["compile_seconds"] = time.perf_counter() - start
            print("COMPILED", artifact, flush=True)
        if a.compile_only:
            status.update(passed=True, compiled_only=True)
            return
        actual = json.loads((artifact / "rbln_config.json").read_text())
        provenance = json.loads((artifact.parent / "build-config.json").read_text())
        if Path(provenance["model"]).resolve() != a.model_dir.resolve():
            raise ValueError("artifact checkpoint differs from the requested model")
        for key, expected in (
            ("batch_size", 1),
            ("max_seq_len", a.context),
            ("prefill_chunk_size", a.chunk),
        ):
            if actual.get(key) != expected:
                raise ValueError(
                    f"artifact {key}={actual.get(key)!r}, expected {expected}"
                )
        manifest = {}
        for path in sorted(artifact.glob("*.rbln")):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            manifest[path.name] = dict(
                bytes=path.stat().st_size, sha256=digest.hexdigest()
            )
        write(
            a.out / "artifact-manifest.json",
            dict(
                path=str(artifact.resolve()),
                files=manifest,
                sdk_declared_dtype=actual.get("dtype"),
                config=actual,
            ),
        )
        if selection is None:
            if a.preflight_module_dir is None:
                raise ValueError("legacy mode requires --preflight-module-dir")
            sys.path.insert(0, str(a.preflight_module_dir.resolve()))
            from batch_control import assert_atom_idle, context_pids

            assert_atom_idle(a.out)
        device = 0 if selection is not None else a.device
        model = RBLNAutoModelForCausalLM.from_pretrained(
            str(artifact), rbln_config=dict(create_runtimes=True, device=device)
        )
        if set(model.rbln_config.device_map.values()) != {device}:
            raise RuntimeError("unexpected logical device map")
        state = subprocess.check_output(["rbln-smi"], text=True)
        (a.out / "device-loaded.txt").write_text(state)
        if selection is not None:
            write(a.out / "binding.json", verify_atom_binding(selection))
        else:
            own = {int(x.name) for x in Path("/proc/self/task").iterdir()}
            if not own.intersection(context_pids(state, (device,))):
                raise RuntimeError("no selected ATOM ownership evidence")
        tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
        base = tokenizer.encode(
            "Explain how computer memory works. ", add_special_tokens=False
        )
        prompts = (
            json.loads(a.prompts_json.read_text())
            if a.prompts_json
            else [
                (base * ((n + len(base) - 1) // len(base)))[:n] for n in (32, 128, 256)
            ]
        )
        if not prompts or any(
            not p
            or len(p) + 32 > a.context
            or any(type(t) is not int or not 0 <= t < len(tokenizer) for t in p)
            for p in prompts
        ):
            raise ValueError("invalid token-ID prompt matrix")
        if len({len(p) for p in prompts}) != len(prompts):
            raise ValueError("summary requires unique input lengths")
        rows = []
        # Fixed token IDs shared by every variant; warm each shape once.
        for prompt in prompts:
            length = len(prompt)
            for count in (8, 32):
                for repeat in range(-1, a.repeats):
                    with torch.inference_mode():
                        start = time.perf_counter()
                        out = model.prefill_decoder(
                            input_ids=torch.tensor([prompt]),
                            cache_position=torch.arange(length, dtype=torch.int32)[
                                None, :
                            ].contiguous(),
                            batch_idx=0,
                        )
                        logits = out.logits[0, -1, :]
                        if not torch.isfinite(logits).all():
                            raise RuntimeError("nonfinite logits")
                        tokens = [int(logits.argmax())]
                        first = time.perf_counter()
                        position = length + int(out.padded_cache_lengths)
                        steps = []
                        for _ in range(count - 1):
                            t = time.perf_counter()
                            out = model.decoders[1](
                                input_ids=torch.tensor([[tokens[-1]]]),
                                cache_position=torch.tensor(
                                    [[position]], dtype=torch.int32
                                ),
                            )
                            logits = out.logits[0, -1, :]
                            if not torch.isfinite(logits).all():
                                raise RuntimeError("nonfinite logits")
                            tokens.append(int(logits.argmax()))
                            position += 1
                            steps.append(time.perf_counter() - t)
                        end = time.perf_counter()
                    row = dict(
                        input_length=length,
                        output_length=count,
                        repeat=repeat,
                        warmup=repeat < 0,
                        prompt=prompt,
                        tokens=tokens,
                        prefill_ms=(first - start) * 1000,
                        decode_ms=sum(steps) * 1000,
                        tpot_ms=statistics.mean(steps) * 1000,
                        total_ms=(end - start) * 1000,
                        output_tok_s=count / (end - start),
                        decode_tok_s=(count - 1) / sum(steps),
                    )
                    rows.append(row)
                    with (a.out / "runs.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(
                        "RUN",
                        length,
                        count,
                        repeat,
                        round(row["output_tok_s"], 2),
                        flush=True,
                    )
        summary = []
        for length in [len(p) for p in prompts]:
            for count in (8, 32):
                group = [
                    r
                    for r in rows
                    if not r["warmup"]
                    and r["input_length"] == length
                    and r["output_length"] == count
                ]
                stable = all(r["tokens"] == group[0]["tokens"] for r in group)
                summary.append(
                    dict(
                        input_length=length,
                        output_length=count,
                        tokens_stable=stable,
                        **{
                            k: statistics.median(r[k] for r in group)
                            for k in (
                                "prefill_ms",
                                "decode_ms",
                                "tpot_ms",
                                "total_ms",
                                "output_tok_s",
                                "decode_tok_s",
                            )
                        },
                    )
                )
                if not stable:
                    raise RuntimeError("nondeterministic native tokens")
        write(a.out / "summary.json", summary)
        status.update(
            passed=True,
            measured_runs=sum(not r["warmup"] for r in rows),
            timing="host synchronous SDK calls including finite checks and argmax; no HTTP or queue",
            quality="finite and repeatable; reference quality must be checked separately",
        )
    except BaseException as exc:
        status.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        write(a.out / "result.json", status)


if __name__ == "__main__":
    main()
