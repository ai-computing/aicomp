"""Test-only native-call entry/completion markers; no execution gate or KV replay."""
import importlib
import json
import os
import time
from pathlib import Path


def make_prefill(config, weights, width, chip=0):
    module, name = os.environ.get(
        "PREFILL_OBSERVER_FACTORY", "long_width_variant:make_prefill"
    ).split(":")
    engine = getattr(importlib.import_module(module), name)(config, weights, width, chip)
    directory = Path(os.environ["PREFILL_OBSERVER_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{os.getpid()}.json"
    original = engine.prefill
    sequence = 0

    def prefill(tokens):
        nonlocal sequence
        sequence += 1

        def record(phase):
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps({"sequence": sequence, "phase": phase,
                                             "tokens": len(tokens), "time": time.monotonic()}))
            temporary.replace(marker)

        record("start")
        try:
            return original(tokens)
        finally:
            record("done")

    engine.prefill = prefill
    return engine
