"""Warm HTTP native/DP/PD comparison; isolated server processes, exact controls and SSE timings."""
import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def timing_summary(rows, wall_s):
    return {
        "requests": len(rows), "output_tokens": sum(r["output_tokens"] for r in rows),
        "wall_s": wall_s,
        "output_tok_s": sum(r["output_tokens"] for r in rows) / wall_s,
        "median_first_sse_s": statistics.median(r["first_sse_s"] for r in rows),
        "median_tpot_est_s": statistics.median(r["tpot_est_s"] for r in rows),
        "median_e2e_s": statistics.median(r["e2e_s"] for r in rows),
        "all_text_equal": all(r["text_equal"] for r in rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["native", "dp2", "pd"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", help="native benchmark JSON with identical token-ID prompts")
    p.add_argument("--http-module", help="trusted local HTTP candidate")
    p.add_argument("--lengths", nargs="+", type=int, default=[128, 8192])
    p.add_argument("--outputs", nargs="+", type=int, default=[32])
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    a = p.parse_args()
    if min(*a.lengths, *a.outputs, *a.concurrency, a.repeats, a.slots) < 1:
        p.error("sizes must be positive")
    if max(a.lengths) + max(a.outputs) > 9216:
        p.error("input plus output exceeds context9216")
    import torch
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    head = [tok.bos_token_id] if tok.bos_token_id is not None else []
    fill = tok.encode("This is background context. The red box contains apples. The blue box contains books.\n",
                      add_special_tokens=False)
    tail = tok.encode("\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False)
    prompts = {}
    for n in a.lengths:
        left = n - len(head) - len(tail)
        if left < 0 or not fill:
            p.error("prompt length too short")
        prompts[str(n)] = head + (fill * ((left + len(fill) - 1) // len(fill)))[:left] + tail
        assert len(prompts[str(n)]) == n
    reference = json.loads(Path(a.reference).read_text()) if a.reference else None
    if reference:
        assert reference["prompts"] == prompts and reference["dtype"] == a.dtype
    refs = {} if reference is None else reference["references"].copy()
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    procs, logs, ports = [], [], []
    result = {"passed": False, "model": Path(a.model).name, "mode": a.mode, "dtype": a.dtype,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "replicas": 2 if a.mode == "dp2" else 1, "slots_per_decoder": a.slots,
              "physical_gpus": 1 if a.mode == "native" else 2, "cpu_threads_per_server": 4,
              "producer_cache": False, "decoder_cache": False, "producer_chunk": 256,
              "local_chunk": 256, "context": 9216, "prompts": prompts,
              "warmups": [], "correctness_batches": [], "batches": [],
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "timing_scope": "Client first SSE event, full response e2e and estimated TPOT=(e2e-first_event)/(output_tokens-1). SSE can coalesce tokens; this is not per-token kernel timing. Loading/warmup/correctness checks excluded."}
    started = time.perf_counter()
    try:
        for endpoint, device in enumerate(["cuda:0", "cuda:1"] if a.mode == "dp2" else ["cuda:1"]):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log = output.with_suffix(f".server{endpoint}.log").open("w")
            logs.append(log)
            cmd = [sys.executable, str(Path(__file__).with_name("benchmark_server.py")),
                   "--model", a.model, "--device", device, "--dtype", a.dtype,
                   "--slots", str(a.slots), "--max-context", "9216", "--max-output", str(max(a.outputs)),
                   "--kv", "paged", "--page-size", "64", "--chunk", "256", "--port", str(port)]
            if a.http_module:
                cmd += ["--http-module", a.http_module]
            if a.mode == "pd":
                cmd += ["--pd-prefill-device", "cuda:0", "--pd-prefill-no-prefix-cache",
                        "--pd-prefill-chunk", "256"]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
            procs.append(proc)
            ports.append(port)
            end = time.monotonic() + 180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server{endpoint} exited{proc.returncode}; see server log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=1) as response:
                        if response.status == 200 and json.load(response)["ready"]:
                            break
                except (OSError, urllib.error.HTTPError):
                    pass
                if time.monotonic() >= end:
                    raise TimeoutError("server startup")
                time.sleep(.05)
        result["load_s"] = time.perf_counter() - started

        def request(n, count, rid, endpoint, stream=False, gate=None):
            body = {"model": a.model, "prompt": prompts[str(n)], "max_tokens": count,
                    "temperature": 0, "ignore_eos": True, "stream": stream}
            req = urllib.request.Request(f"http://127.0.0.1:{ports[endpoint]}/v1/completions",
                data=json.dumps(body).encode(), headers={"x-request-id": rid})
            if gate is not None:
                gate.wait(timeout=30)
            start = time.perf_counter()
            key = f"{n}:{count}"
            with urllib.request.urlopen(req, timeout=180) as response:
                assert response.status == 200
                if not stream:
                    out = json.load(response)
                    choice = out["choices"][0]
                    assert choice["finish_reason"] == "length" and len(choice["token_ids"]) == count
                    return {"id": rid, "endpoint": endpoint, "tokens": choice["token_ids"],
                            "text": choice["text"], "tokens_equal": key not in refs or choice["token_ids"] == refs[key]["tokens"]}
                first_event = None
                usage = finish = None
                done = False
                text = []
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    if first_event is None:
                        first_event = time.perf_counter() - start
                    if line[6:].strip() == b"[DONE]":
                        done = True
                        break
                    event = json.loads(line[6:])
                    if "usage" in event:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        text.append(choice.get("text", ""))
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
            elapsed = time.perf_counter() - start
            assert done and finish == "length" and usage["completion_tokens"] == count
            text = "".join(text)
            return {"id": rid, "endpoint": endpoint, "input_tokens": n, "output_tokens": count,
                    "text": text, "text_equal": text == refs[key]["text"], "first_sse_s": first_event,
                    "tpot_est_s": (elapsed - first_event) / (count - 1) if count > 1 else 0,
                    "e2e_s": elapsed}

        def batch(n, count, concurrency, rid, repeat, stream):
            gate = threading.Barrier(concurrency + 1)
            with ThreadPoolExecutor(max_workers=concurrency) as clients:
                futures = [clients.submit(request, n, count, f"{rid}-{i}",
                           (repeat + i) % len(ports), stream, gate) for i in range(concurrency)]
                start = time.perf_counter()
                gate.wait(timeout=30)
                rows = [future.result(timeout=240) for future in futures]
                wall = time.perf_counter() - start
            return rows, wall

        warm_start = time.perf_counter()
        for endpoint in range(len(ports)):
            for n in a.lengths:
                for count in a.outputs:
                    row = request(n, count, f"warm-{endpoint}-{n}-{count}", endpoint)
                    refs.setdefault(f"{n}:{count}", {"tokens": row["tokens"], "text": row["text"]})
                    result["warmups"].append(row)
        result["references"] = refs
        for n in a.lengths:
            for count in a.outputs:
                for concurrency in a.concurrency:
                    rows, _ = batch(n, count, concurrency, f"check-{n}-{count}-{concurrency}", 0, False)
                    result["correctness_batches"].append({"input_tokens": n, "output_tokens": count,
                                                         "concurrency": concurrency, "rows": rows})
        result["warmup_and_checks_s"] = time.perf_counter() - warm_start
        for repeat in range(a.repeats):
            for n in (a.lengths if repeat % 2 == 0 else list(reversed(a.lengths))):
                for count in a.outputs:
                    for concurrency in a.concurrency:
                        rows, wall = batch(n, count, concurrency, f"r{repeat}-n{n}-o{count}-c{concurrency}", repeat, True)
                        result["batches"].append({"repeat": repeat, "input_tokens": n,
                            "output_tokens_per_request": count, "concurrency": concurrency,
                            "summary": timing_summary(rows, wall), "rows": rows})
        result["metrics"] = []
        for port in ports:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
                result["metrics"].append(response.read().decode().replace(a.model, result["model"]))
        result["parity_passed"] = all(r["tokens_equal"] for r in result["warmups"]) and all(
            r["tokens_equal"] for b in result["correctness_batches"] for r in b["rows"]) and all(
            b["summary"]["all_text_equal"] for b in result["batches"])
        result["passed"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                proc.wait(timeout=10)
        for log in logs:
            log.close()
        output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["parity_passed"]:
        raise AssertionError("output parity failed; raw data retained, no equivalent-output speedup claim")
    print(json.dumps({"passed": result["passed"], "parity_passed": result["parity_passed"],
                      "batches": len(result["batches"]), "mode": a.mode}), flush=True)


if __name__ == "__main__":
    main()
