"""Actual HTTP validation through serve_batched.build; one model/configuration per process."""

import argparse
import os
import importlib.util
import itertools
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(os.environ["AICOMP_RUNTIME"])/"opt_prime/examples/pd"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--http-module", help="trusted local HTTP module candidate for isolated validation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pd-prefill-device", default=None)
    p.add_argument("--pd-prefill-no-prefix-cache", action="store_true")
    p.add_argument("--pd-prefill-chunk", type=int, default=2048)
    p.add_argument("--prefix-cache", action="store_true")
    p.add_argument("--backend", choices=["torch", "atom"], default="torch")
    p.add_argument("--atom-kv-blocks", type=int, default=3)
    p.add_argument("--host-kv-gib", type=float, default=0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    a = p.parse_args()
    if Path(a.output).exists():
        p.error("output already exists; choose a new result path")
    if a.http_module:
        spec = importlib.util.spec_from_file_location("opt_prime.accel.serve.llmd", a.http_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    # Load consumers after the candidate so StageError identity is shared across the server and owner.
    from opt_prime.accel.serve.llmd import serve_stage
    from serve_batched import build
    if a.backend == "atom" and a.dtype != "bfloat16":
        p.error("ATOM precision is determined by the compiled artifact")
    if a.host_kv_gib < 0 or (a.host_kv_gib and not a.prefix_cache):
        p.error("host KV requires --prefix-cache and a nonnegative capacity")
    devices = []
    if a.backend == "atom":
        nodes = list(Path("/dev").glob("rbln[0-9]*"))
        assert len(nodes) == 1, "requires one isolated allocated ATOM"
        devices = [int(nodes[0].name[4:])]
    torch.set_num_threads(4)
    args = SimpleNamespace(
        model=a.model,
        backend=a.backend,
        device=a.device,
        dtype=a.dtype,
        slots=1,
        devices=devices,
        atom_kv_blocks=a.atom_kv_blocks,
        pd_remote_only=False,
        max_context=9216,
        max_output=256,
        kv="paged",
        page_size=256,
        chunk=256,
        prefix_cache=a.prefix_cache,
        kv_events=None,
        kv_host_memory_gb=a.host_kv_gib,
        reload_bandwidth=None,
        prefill_rate=None,
        remote_kv=None,
        pd_prefill_device=a.pd_prefill_device,
        pd_prefill_no_prefix_cache=a.pd_prefill_no_prefix_cache,
        pd_prefill_chunk=a.pd_prefill_chunk,
        pd_prefill_python=None,
    )
    result = {
        "model_name": Path(a.model).name,
        "prefix_cache": a.prefix_cache,
        "host_kv_gib": a.host_kv_gib,
        "passed": False,
        "scope": "HTTP lifecycle and completion counts; numerical equality reported separately",
        "torch": torch.__version__,
        "dtype": a.dtype if a.backend == "torch" else "compiled artifact",
        "backend": a.backend,
        "decode_device": a.device,
        "pd_prefill_device": a.pd_prefill_device,
        "producer_prefix_cache": None,
        "slots": args.slots,
        "atom_kv_blocks": a.atom_kv_blocks if a.backend == "atom" else None,
        "device": torch.cuda.get_device_name(torch.device(a.device))
        if a.backend == "torch"
        else "allocated ATOM",
        "cases": [],
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        owner = build(args)
        if a.pd_prefill_device:
            producer = owner.service.batcher.worker.engine
            result["producer_prefix_cache"] = producer.kv.prefix_cache
            result["producer_chunk"] = producer.chunk
    except Exception as exc:
        result["startup_error"] = repr(exc)
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    head = [tok.bos_token_id] if tok.bos_token_id is not None else []
    fill = tok.encode(
        "This is background context. The red box contains apples. The blue box contains books.\n",
        add_special_tokens=False,
    )
    tail = tok.encode(
        "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
    )
    prompts = {}
    for key, text in [("A", "Amber archive: the red box contains apples.\n"), ("B", "Violet notebook: the blue box contains books.\n")]:
        fill = tok.encode(text, add_special_tokens=False)
        left = 8192-len(head)-len(tail)
        prompts[key] = head + (fill*((left+len(fill)-1)//len(fill)))[:left] + tail
        assert len(prompts[key]) == 8192
    assert prompts["A"][:256] != prompts["B"][:256]
    result["prompts"] = prompts
    result["sequence"] = ["A", "B", "A", "A", "B", "A"]
    kv = owner.service.batcher.kv
    assert kv.max_pages == 36 and kv.page_size == 256
    if kv.host is not None:
        assert kv.host.blocks >= 64, "host capacity must fit both prompt block sets"
    result["capacity"] = {"device_pages": kv.max_pages, "page_size": kv.page_size,
                          "host_pages": kv.host.blocks if kv.host is not None else 0}
    result["allocations"] = []
    original_alloc = kv.alloc
    def traced_alloc(slot, tokens, prompt=None):
        hit = original_alloc(slot, tokens, prompt=prompt)
        result["allocations"].append({"slot": slot, "reserved_tokens": tokens,
                                      "prompt_tokens": len(prompt) if prompt is not None else None,
                                      "reused_tokens": hit})
        return hit
    kv.alloc = traced_alloc
    server, server_thread = serve_stage(owner, "decode", 0)
    url = f"http://127.0.0.1:{server.server_port}"
    serial = itertools.count()

    def request(path, body=None, rid=None):
        req = urllib.request.Request(
            url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-request-id": rid or f"http211-{next(serial)}",
            },
        )
        return urllib.request.urlopen(req, timeout=180)

    def body(n, count=32, stream=False):
        return {
            "model": a.model,
            "prompt": prompts[n],
            "max_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
            "stream": stream,
        }

    def complete(n):
        start = time.perf_counter()
        with request("/v1/completions", body(n)) as response:
            assert response.status == 200
            data = json.load(response)
        tokens = data["choices"][0]["token_ids"]
        assert len(tokens) == 32 and data["usage"]["completion_tokens"] == 32
        return {
            "input_tokens": n,
            "tokens": tokens,
            "e2e_s": time.perf_counter() - start,
        }

    def drained(rid, timeout=30):
        end = time.monotonic() + timeout
        while rid in owner.service.batcher.by_id:
            if time.monotonic() > end:
                raise TimeoutError(f"undrained request {rid}")
            time.sleep(0.01)

    def client():
        try:
            end = time.monotonic()+30
            while not owner.started:
                if time.monotonic()>end: raise TimeoutError("owner startup")
                time.sleep(.01)
            result["rows"] = []
            refs = {}
            def snapshot():
                return {"hit_tokens": int(kv.hit_tokens), "query_tokens": int(kv.query_tokens),
                        "device_removed": sum(e[0]=="removed" and len(e)==3 and e[2] is None for e in kv.events),
                        "host": dict(kv.host.stats) if kv.host is not None else None,
                        "free_pages": len(kv.free_pages), "cached_pages": len(kv.cached),
                        "live_refs": sum(kv.refs)}
            result["initial"] = snapshot()
            for index, key in enumerate(result["sequence"]):
                before = snapshot(); allocation_start = len(result["allocations"])
                row = complete(key)
                end = time.monotonic()+30
                while not owner.service.batcher.idle:
                    if time.monotonic()>end: raise TimeoutError("request drain")
                    time.sleep(.01)
                refs.setdefault(key,row["tokens"])
                row.update(index=index, prompt_key=key, input_tokens=8192,
                           tokens_equal=row["tokens"]==refs[key], before=before, after=snapshot(),
                           allocations=result["allocations"][allocation_start:])
                result["rows"].append(row)
                assert len(row["allocations"])==1 and row["allocations"][0]["reused_tokens"] is not None
            result["references"] = refs
            result["parity_passed"] = all(r["tokens_equal"] for r in result["rows"])
            result["remaining_requests"] = len(owner.service.batcher.by_id)
            assert result["remaining_requests"]==0
            with request("/ready") as response: assert response.status==200
            with request("/metrics") as response: result["metrics"]=response.read().decode().replace(a.model,Path(a.model).name)
            result["final"] = snapshot()
            if a.prefix_cache:
                assert result["rows"][1]["after"]["device_removed"]>0, "B did not evict A"
                if kv.host is not None:
                    assert result["rows"][2]["after"]["host"]["reloaded"]>result["rows"][2]["before"]["host"]["reloaded"], "A was not reloaded"
                    assert result["rows"][2]["allocations"][0]["reused_tokens"]==7936
                else:
                    assert result["rows"][2]["allocations"][0]["reused_tokens"]==0, "evicted A unexpectedly reused"
            else: assert result["final"]["hit_tokens"]==0
            result["passed"] = True
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            for rid in list(owner.service.batcher.by_id): owner.service.batcher.cancel(rid)
            owner.submit("stop")

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    try:
        owner.run()
    except Exception as exc:  # noqa: BLE001 - preserve failure and stop this process
        result["engine_error"] = repr(exc)
    finally:
        thread.join(190)
        server.shutdown()
        server.server_close()
        server_thread.join(5)
        worker = getattr(owner.service.batcher, "worker", None)
        if worker is not None:
            worker.close()
        output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError(
            result.get("error", result.get("engine_error", "validation failed"))
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cases", "baseline", "metrics")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
