"""Actual RNGD subprocess prefill to ATOM HTTP, with native-output and lifecycle checks."""

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--batching", required=True)
p.add_argument("--cache-dir", required=True)
p.add_argument("--fixtures", required=True)
p.add_argument("--cases", nargs="+", required=True)
p.add_argument("--worker-python", required=True)
p.add_argument("--engine-factory", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--kv-blocks", type=int, default=3)
p.add_argument("--model", default="unsloth/gemma-3-1b-it")
p.add_argument("--http-module", required=True)
p.add_argument("--baseline", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
for name, path in [
    ("opt_prime.accel.platforms.atom.batching", a.batching),
    ("opt_prime.accel.serve.llmd", a.http_module),
]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import serve_stage
from transformers import AutoTokenizer

fixtures = json.loads(Path(a.fixtures).read_text())
cases = [r for r in fixtures["cases"] if r["case"] in a.cases]
assert {r["case"] for r in cases} == set(a.cases)
assert cases and all(0 < len(r["ids"]) <= a.width for r in cases)
baseline = {
    r["case"]: r["tokens"]
    for r in map(json.loads, Path(a.baseline).read_text().splitlines())
}


nodes = list(Path("/dev").glob("rbln[0-9]*"))
assert len(nodes) == 1
engine = AtomRblnEngine(
    devices=[int(nodes[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=a.kv_blocks,
    cache_dir=a.cache_dir,
)
worker = None
owner = thread = server = http = None
rows = []
try:
    model = a.model
    engine.start(model)
    c = engine.model.config.get_text_config()
    worker = SubprocessPrefillWorker(
        a.worker_python,
        [
            "--model",
            model,
            "--width",
            str(a.width),
            "--engine-factory",
            a.engine_factory,
        ],
        slots=1,
        layers=c.num_hidden_layers,
        kv_heads=c.num_key_value_heads,
        head_dim=getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
        max_context=a.width,
        env=dict(os.environ),
    )
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    b = engine.batcher(prefix_cache=False, prefill_worker=worker, local_prefill=False)
    service = BatchedService(
        b,
        model=model,
        vocab_size=engine.model.config.vocab_size,
        max_output=64,
        render=lambda ids: tok.decode(ids, skip_special_tokens=True),
        tokenizer=tok,
        eos_token_ids=tuple(fixtures["eos_ids"]),
    )
    owner = BatchedOwner(service)
    thread = threading.Thread(target=owner.run, daemon=True)
    thread.start()
    server, http = serve_stage(owner, "decode", 0)

    def request(path, body, rid):
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps(body).encode(),
            headers={"x-request-id": rid},
        )
        return urllib.request.urlopen(req, timeout=60)

    def complete(r, rid):
        started = time.perf_counter()
        with request(
            "/v1/completions",
            {"model": model, "prompt": r["ids"], "max_tokens": 64},
            rid,
        ) as response:
            out = json.load(response)
        assert out["choices"][0]["finish_reason"] == "stop"
        assert out["choices"][0]["token_ids"] == baseline[r["case"]], out
        rows.append(
            {
                "case": r["case"],
                "request": rid,
                "response": out,
                "latency_s": time.perf_counter() - started,
            }
        )

    # Alternate long and short prefixes while blocks rotate under the same local lane.
    for repeat in range(2):
        for i, r in enumerate(cases):
            complete(r, f"normal-{repeat}-{i}")
    with ThreadPoolExecutor(max_workers=4) as clients:
        list(
            clients.map(
                lambda pair: complete(pair[1], f"concurrent-{pair[0]}"),
                enumerate(cases),
            )
        )
    for mode in ["cancel", "disconnect"]:
        r = cases[-1]
        with request(
            "/v1/completions",
            {
                "model": model,
                "prompt": r["ids"],
                "max_tokens": 64,
                "stream": True,
                "ignore_eos": True,
            },
            mode,
        ) as response:
            for raw in response:
                if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                    event = json.loads(raw[6:])
                    if event.get("choices") and event["choices"][0].get("text"):
                        break
            else:
                raise AssertionError("no streamed text")
            if mode == "cancel":
                with request("/v1/cancel", {}, mode) as cancelled:
                    assert cancelled.status == 202
        end = time.monotonic() + 10
        while mode in b.by_id and time.monotonic() < end:
            time.sleep(0.01)
        assert mode not in b.by_id
        complete(cases[0], "after-" + mode)
    assert b.stats_pd["local"] == 0 and sorted(b.free) == [0]
    assert b.stats_pd["remote"] == len(rows) + 2
    result = {
        "worker": "live_rngd_subprocess_prefill",
        "rows": rows,
        "stats_pd": b.stats_pd,
        "free": b.free,
        "cancellation_reuse": True,
    }
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"requests": len(rows), "stats_pd": b.stats_pd, "cancellation_reuse": True}
        ),
        flush=True,
    )
finally:
    if owner:
        owner.submit("stop")
    if thread:
        thread.join(15)
    if server:
        server.shutdown()
        server.server_close()
        http.join(5)
    if worker:
        worker.close()
    engine.close()
