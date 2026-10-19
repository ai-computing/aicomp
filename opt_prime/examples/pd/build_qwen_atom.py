"""Compile Qwen external-KV graph on an explicitly selected idle ATOM."""

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path


class CompileOnlyDone(BaseException):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--physical", default="rbln3")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    result = {"passed": False}
    start = time.perf_counter()
    from opt_prime.pd.npu.provenance import record_run

    record_run(a.out, __file__)
    try:
        from opt_prime.pd.npu.atom_device import select_idle_atom, verify_atom_binding

        selection = select_idle_atom(a.physical)
        (a.out / "preflight.json").write_text(json.dumps(selection, indent=2))
        import torch
        import torch_rbln  # noqa: F401 - registers the ATOM device backend
        import rebel
        from opt_prime.pd.npu.model import DenseModel, load_weights
        from opt_prime.pd.npu.atom import QwenResidentDecode

        torch.set_num_threads(8)
        spec = DenseModel.load(a.model, a.model_id)
        weights = load_weights(a.model, spec)
        (a.out / "model.json").write_text(json.dumps(asdict(spec), indent=2))
        model = QwenResidentDecode(spec, weights).eval().to("rbln:0")
        caches = [
            torch.zeros(1, spec.kv_heads, 512, spec.head_dim).to("rbln:0")
            for _ in range(spec.caches)
        ]
        (a.out / "binding.json").write_text(
            json.dumps(verify_atom_binding(selection), indent=2)
        )
        original = rebel.RBLNCompiledModel.create_runtime

        def stop(self, *args, **kwargs):
            if list((a.out / "compiled").glob("*.rbln")):
                raise CompileOnlyDone()
            return original(self, *args, **kwargs)

        rebel.RBLNCompiledModel.create_runtime = stop
        run = torch.compile(
            model,
            backend="rbln",
            dynamic=False,
            options={"cache_dir": str(a.out / "compiled")},
        )
        host = [
            torch.zeros(1, spec.hidden),
            torch.ones(1, spec.head_dim // 2),
            torch.zeros(1, spec.head_dim // 2),
            torch.ones(1, 512),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int16),
        ]
        with torch.no_grad():
            run(*[x.to("rbln:0") for x in host], *caches)
        raise RuntimeError("compile-only stop was not reached")
    except CompileOnlyDone:
        result.update(passed=True, compiled_only=True)
    except BaseException as e:
        result.update(error=repr(e), traceback=traceback.format_exc())
    finally:
        result.update(
            seconds=time.perf_counter() - start,
            artifacts=[str(p) for p in (a.out / "compiled").glob("*.rbln")],
        )
        (a.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
