"""KONI real-HTTP EOS negative control, override, streaming and reuse check."""

import argparse
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import torch

p = argparse.ArgumentParser()
p.add_argument("--factory", required=True)
p.add_argument("--model", required=True)
p.add_argument("--backend", choices=["torch", "atom"], default="torch")
p.add_argument("--cache-dir")
p.add_argument("--fixtures", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.set_num_threads(4)
spec = importlib.util.spec_from_file_location("factory229", a.factory)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = a.model
if a.backend == "atom":
    if not a.cache_dir:
        p.error("ATOM validation requires an existing artifact cache")
    import opt_prime.accel.platforms.atom.engine as atom_engine

    nodes = list(Path("/dev").glob("rbln[0-9]*"))
    assert len(nodes) == 1, "requires exclusive single-ATOM allocation"

    class ArtifactEngine(atom_engine.AtomRblnEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, cache_dir=a.cache_dir)

    atom_engine.AtomRblnEngine = ArtifactEngine
args = SimpleNamespace(
    model=model,
    kv_events=None,
    backend=a.backend,
    dtype="bfloat16",
    device="cuda:0",
    slots=1,
    max_context=9216 if a.backend == "atom" else 2048,
    kv="slot",
    page_size=64,
    prefix_cache=False,
    kv_host_memory_gb=0,
    reload_bandwidth=None,
    prefill_rate=None,
    chunk=256,
    max_output=64,
    pd_prefill_device=None,
    pd_prefill_python=None,
    remote_kv=None,
    devices=[int(nodes[0].name[4:])] if a.backend == "atom" else [],
    atom_kv_blocks=1,
    pd_remote_only=False,
)
owner = module.build(args)
fixed = owner.service.eos_token_ids
assert fixed == (128001, 128009), fixed
worker = threading.Thread(target=owner.run, daemon=True)
worker.start()
server, http = module.serve_stage(owner, "decode", 0)
ids = json.loads(Path(a.fixtures).read_text())["cases"][0]["ids"]
rows = []
try:
    for name, eos, ignore, stream in [
        ("baseline", (128001,), False, False),
        ("fixed", fixed, False, False),
        ("ignore", fixed, True, False),
        ("stream", fixed, False, True),
        ("reuse", fixed, False, False),
    ]:
        owner.service.eos_token_ids = eos
        body = {
            "model": model,
            "prompt": ids,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": ignore,
            "stream": stream,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"x-request-id": name},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode()
        if stream:
            events = [
                json.loads(line[6:])
                for line in raw.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"
            ]
            assert "[DONE]" in raw
            result = next(
                e
                for e in reversed(events)
                if e.get("choices") and e["choices"][0].get("finish_reason")
            )
        else:
            result = json.loads(raw)
        finish = result["choices"][0]["finish_reason"]
        assert finish == ("length" if name in ["baseline", "ignore"] else "stop"), (
            name,
            result,
        )
        rows.append({"case": name, "eos_ids": eos, "response": result, "raw": raw})
        print(name, finish, flush=True)
    Path(a.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
finally:
    owner.submit("stop")
    worker.join(15)
    server.shutdown()
    server.server_close()
    http.join(5)
