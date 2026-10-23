"""Actual HTTP validation through serve_batched.build; one model/configuration per process."""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opt_prime.accel.serve.llmd import serve_stage
from serve_batched import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix-cache", action="store_true")
    p.add_argument("--backend", choices=["torch", "atom"], default="torch")
    p.add_argument("--atom-kv-blocks", type=int, default=3)
    p.add_argument("--host-kv-gib", type=float, default=0)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    a = p.parse_args()
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
        device="cuda:0",
        dtype=a.dtype,
        slots=4 if a.backend == "torch" else 1,
        devices=devices,
        atom_kv_blocks=a.atom_kv_blocks,
        pd_remote_only=False,
        max_context=9216,
        max_output=256,
        kv="paged",
        page_size=64,
        chunk=256,
        prefix_cache=a.prefix_cache,
        kv_events=None,
        kv_host_memory_gb=a.host_kv_gib,
        reload_bandwidth=None,
        prefill_rate=None,
        remote_kv=None,
        pd_prefill_device=None,
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
        "slots": args.slots,
        "atom_kv_blocks": a.atom_kv_blocks if a.backend == "atom" else None,
        "device": torch.cuda.get_device_name(0)
        if a.backend == "torch"
        else "allocated ATOM",
        "cases": [],
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        owner = build(args)
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
    for n in (128, 2048, 8192):
        left = n - len(head) - len(tail)
        assert left >= 0 and fill
        prompts[n] = head + (fill * ((left + len(fill) - 1) // len(fill)))[:left] + tail
        assert len(prompts[n]) == n
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
            end = time.monotonic() + 30
            while not owner.started:
                if time.monotonic() > end:
                    raise TimeoutError("owner startup")
                time.sleep(0.01)
            with request("/ready") as response:
                assert response.status == 200
            baseline = {n: complete(n) for n in prompts}
            result["baseline"] = baseline
            for concurrency in (4, 8):
                lengths = [(128, 2048, 8192)[i % 3] for i in range(concurrency)]
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(pool.map(complete, lengths))
                for row in rows:
                    row["baseline_equal"] = (
                        row["tokens"] == baseline[row["input_tokens"]]["tokens"]
                    )
                result["cases"].append(
                    {
                        "concurrency": concurrency,
                        "wall_s": time.perf_counter() - start,
                        "rows": rows,
                    }
                )
            # Explicit cancellation and socket disconnect after the first streamed token.
            for mode in ("cancel", "disconnect"):
                before_cancel = complete(128)
                rid = f"http211-{mode}"
                with request("/v1/completions", body(128, 256, True), rid) as response:
                    for raw in response:
                        if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                            data = json.loads(raw[6:])
                            if data.get("choices") and data["choices"][0].get("text"):
                                break
                    else:
                        raise AssertionError("stream had no text")
                    if mode == "cancel":
                        with request("/v1/cancel", {}, rid) as cancelled:
                            assert cancelled.status == 202
                drained(rid)
                after = complete(128)
                result[mode] = {
                    "drained": True,
                    "subsequent_tokens_equal": after["tokens"]
                    == before_cancel["tokens"],
                    "cold_baseline_equal": after["tokens"] == baseline[128]["tokens"],
                    "before_tokens": before_cancel["tokens"],
                    "after_tokens": after["tokens"],
                }
                assert result[mode]["subsequent_tokens_equal"], (
                    "post-cancellation output changed"
                )

            try:
                with request(
                    "/v1/completions",
                    {"model": a.model, "prompt": [], "max_tokens": 32},
                ):
                    raise AssertionError("invalid request accepted")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
                result["invalid_status"] = exc.code
            with request("/ready") as response:
                assert response.status == 200
            result["remaining_requests"] = len(owner.service.batcher.by_id)
            assert result["remaining_requests"] == 0
            with request("/metrics") as response:
                result["metrics"] = (
                    response.read().decode().replace(a.model, result["model_name"])
                )
            hits = [
                float(line.rsplit(" ", 1)[1])
                for line in result["metrics"].splitlines()
                if line.startswith("vllm:prefix_cache_hits_total{")
            ]
            assert hits, "prefix hit metric missing"
            result["prefix_hit_tokens"] = sum(hits)
            if a.prefix_cache:
                assert result["prefix_hit_tokens"] > 0, (
                    "cache enabled without observed hits"
                )
            if a.host_kv_gib:
                host_hits = [
                    float(line.rsplit(" ", 1)[1])
                    for line in result["metrics"].splitlines()
                    if line.startswith("airm_kv_host_hit_tokens_total{")
                ]
                result["host_hit_tokens"] = sum(host_hits)
                assert host_hits and result["host_hit_tokens"] > 0, (
                    "host tier not exercised"
                )
            result["passed"] = True
        except Exception as exc:  # noqa: BLE001 - preserve failed hardware evidence
            result["error"] = repr(exc)
        finally:
            for rid in list(owner.service.batcher.by_id):
                owner.service.batcher.cancel(rid)
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
