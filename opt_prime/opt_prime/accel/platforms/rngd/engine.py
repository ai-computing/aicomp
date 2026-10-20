#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""RNGD whole-model engine: the native RNGD layer engine binary in serve mode."""
import json
import os
import subprocess
import time
from typing import Optional

from opt_prime.accel.engines.base import ExecutionEngine, engine_geometry  # noqa: F401


class RngdLayerEngine(ExecutionEngine):
    """A native RNGD layer engine binary in serve mode (`<binary> <data_dir> serve`, JSON lines:
    {"ids","max_new"} -> {"tok"}* {"done", ...}). Its data directory holds engine_manifest.json:

        {"geometry": engine_geometry(...) of the model it was built for, "chips": [0, 1],
         "tok_s": measured exact decode tok/s, "env": {...runtime settings...}}

    The engine leases its chips (role "generate") through opt_prime.accel.platforms for as long as it runs."""
    name = "rngd-layer"

    def __init__(self, binary: str, data_dir: str, env: Optional[dict] = None, ready_timeout: float = 1800):
        self.binary, self.data_dir, self.env, self.ready_timeout = binary, data_dir, env or {}, ready_timeout
        path = os.path.join(data_dir, "engine_manifest.json")
        self.manifest = json.load(open(path)) if os.path.exists(path) else None
        self.p, self.leases = None, []

    def supports(self, spec):
        if self.manifest is None:
            return [f"no engine_manifest.json in {self.data_dir}"]
        if not os.path.exists(self.binary):
            return [f"engine binary {self.binary} not found"]
        want, have = self.manifest["geometry"], engine_geometry(spec)
        return [f"{k}: engine {want.get(k)!r} != model {have.get(k)!r}" for k in sorted(set(want) | set(have))
                if json.dumps(want.get(k), sort_keys=True) != json.dumps(have.get(k), sort_keys=True)]

    def expected_tok_s(self, spec):
        return (self.manifest or {}).get("tok_s")

    def start(self, model_id=None):
        from opt_prime.accel.platforms import acquire
        chips = self.manifest.get("chips", [0])
        try:
            for c in chips:
                self.leases.append(acquire(f"rngd:{c}", "generate", f"rngd-layer engine {self.binary}"))
        except BaseException:
            self._release()
            raise
        env = dict(os.environ)
        env.update(self.manifest.get("env", {}))
        env.update(self.env)
        env["FURIOSA_VISIBLE_CHIPS"] = ",".join(map(str, chips))
        self.p = subprocess.Popen([self.binary, self.data_dir, "serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
        t0 = time.time()
        line = self.p.stdout.readline()
        if line.strip() != "READY":
            self.close()
            raise RuntimeError(f"rngd-layer engine did not start (got {line!r})")
        self.load_s = time.time() - t0

    def generate(self, ids, max_new=64):
        self.p.stdin.write(json.dumps({"ids": list(ids), "max_new": int(max_new), "stop": [], "stop_count": 1})
                           + "\n")
        self.p.stdin.flush()
        toks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("rngd-layer engine exited")
            r = json.loads(line)
            if "tok" in r:
                toks.append(r["tok"])
            elif r.get("done"):
                return toks, r
            elif "error" in r:
                raise RuntimeError(r["error"])

    def _release(self):
        for lease in self.leases:
            lease.release()
        self.leases = []

    def close(self):
        if self.p is not None and self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.p = None
        self._release()
