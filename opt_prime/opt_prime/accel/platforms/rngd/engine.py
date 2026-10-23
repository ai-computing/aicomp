#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""RNGD whole-model engine: the native RNGD layer engine binary in serve mode."""
import json
import os
import selectors
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
        if self.p is not None or self.leases:
            raise RuntimeError("engine already started")
        from opt_prime.accel.platforms import acquire
        chips = self.manifest.get("chips", [0])
        try:
            for c in chips:
                self.leases.append(acquire(f"rngd:{c}", "generate", f"rngd-layer engine {self.binary}"))
        except BaseException:
            self._release()
            raise
        try:
            env = dict(os.environ)
            env.update(self.manifest.get("env", {}))
            env.update(self.env)
            env["FURIOSA_VISIBLE_CHIPS"] = ",".join(map(str, chips))
            self.p = subprocess.Popen([self.binary, self.data_dir, "serve"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
            t0 = time.monotonic()
            deadline = t0 + self.ready_timeout
            line = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(self.p.stdout, selectors.EVENT_READ)
                while not line.endswith(b"\n"):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("rngd-layer engine readiness timed out")
                    char = os.read(self.p.stdout.fileno(), 1)
                    if not char or len(line) >= 4096:
                        raise RuntimeError("rngd-layer engine exited or sent an invalid readiness line")
                    line.extend(char)
            if line.strip() != b"READY":
                raise RuntimeError(f"rngd-layer engine did not start (got {bytes(line)!r})")
            self.load_s = time.monotonic() - t0
        except BaseException:
            # Startup never admitted a request. Stop and reap the child before returning its leases.
            if self.p is not None and self.p.poll() is None:
                self.p.kill()
                self.p.wait()
            self.close()
            raise

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
                self.p.wait()
        if self.p is not None:
            self.p.stdin.close()
            self.p.stdout.close()
        self.p = None
        self._release()
