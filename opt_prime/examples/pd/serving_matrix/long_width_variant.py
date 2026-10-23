"""Issue204: isolated width2048 feasibility test; does not change serving defaults."""

import importlib.util
from pathlib import Path


def load_variant():
    source = Path(importlib.util.find_spec("prefill").origin).read_text()
    guard = "if width <= 0 or width > 512 or width % 32:"
    if source.count(guard) != 1:
        raise ValueError("unexpected prefiller capacity guard; review experiment")
    source = source.replace(guard, "if width <= 0 or width > 2048 or width % 32:")
    source = source.replace("multiple of32 within512", "multiple of32 within2048")
    namespace = {"__name__": "prefill_width2048"}
    exec(compile(source, "prefill_width2048", "exec"), namespace)  # noqa: S102 - trusted local experimental source
    return namespace["Prefill"]
