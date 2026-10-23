"""Real ATOM scheduler/HTTP hybrid PD validation using recorded RNGD KV, not live prefill."""

import argparse
import importlib.util
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--batching", required=True)
p.add_argument("--cache-dir", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--baseline", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
name = "opt_prime.accel.platforms.atom.batching"
spec = importlib.util.spec_from_file_location(name, a.batching)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
from opt_prime.accel.pd.pd_batching import PrefillOut, PrefillWorker
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import serve_stage
from transformers import AutoTokenizer

exports = [torch.load(f, weights_only=True) for f in a.exports]
lookup = {tuple(r["ids"]): r for r in exports}
baseline = {
    r["case"]: r["tokens"]
    for r in map(json.loads, Path(a.baseline).read_text().splitlines())
}


class Replay:
    def prefill(self, request, send_from=0):
        assert send_from == 0
        r = lookup[request.input_ids]
        return PrefillOut(int(r["logits"].argmax()), 0, r["keys"], r["values"])


nodes = list(Path("/dev").glob("rbln[0-9]*"))
assert len(nodes) == 1
engine = AtomRblnEngine(
    devices=[int(nodes[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=3,
    cache_dir=a.cache_dir,
)
worker = PrefillWorker(Replay())
owner = thread = server = http = None
rows = []
try:
    model = "unsloth/gemma-3-1b-it"
    engine.start(model)
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    b = engine.batcher(prefix_cache=False, prefill_worker=worker, local_prefill=False)
    service = BatchedService(
        b,
        model=model,
        vocab_size=engine.model.config.vocab_size,
        max_output=64,
        render=lambda ids: tok.decode(ids, skip_special_tokens=True),
        tokenizer=tok,
        eos_token_ids=tuple(exports[0]["eos_ids"]),
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
        with request(
            "/v1/completions",
            {"model": model, "prompt": r["ids"], "max_tokens": 64},
            rid,
        ) as response:
            out = json.load(response)
        assert out["choices"][0]["finish_reason"] == "stop"
        assert out["choices"][0]["token_ids"] == baseline[r["metadata"]["case"]], out
        rows.append({"case": r["metadata"]["case"], "request": rid, "response": out})

    # Alternate long and short prefixes while blocks rotate under the same local lane.
    for repeat in range(2):
        for i, r in enumerate(exports):
            complete(r, f"normal-{repeat}-{i}")
    for mode in ["cancel", "disconnect"]:
        r = exports[-1]
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
        complete(exports[0], "after-" + mode)
    assert b.stats_pd["local"] == 0 and sorted(b.free) == [0]
    result = {
        "worker": "recorded_rngd_kv_replay_not_live_prefill",
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
    worker.close()
    engine.close()
