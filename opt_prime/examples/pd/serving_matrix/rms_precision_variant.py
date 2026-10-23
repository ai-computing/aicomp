"""Issue202 hypothesis only: generate an isolated FP32-gamma variant of PR200."""

import importlib.util
from pathlib import Path


def load_variant():
    source = importlib.util.find_spec("prefill").origin
    text = Path(source).read_text()
    needle = "put(w(k).repeat(n, 1) * math.sqrt(dim))"
    if text.count(needle) != 1 or text.count("rms_dsl(width") != 3:
        raise ValueError(
            "unexpected prefiller source; re-review the precision experiment"
        )
    text = text.replace(
        needle,
        "(w(k).repeat(n, 1) * math.sqrt(dim)).contiguous().to(torch.float32).to(device)",
    )
    text = text.replace("rms_dsl(width", "rms_fp32_dsl(width")
    helper = """
def rms_fp32_dsl(c, n):
    old = f"w: [C={c},N={n}]/bf16"
    dsl = rms_dsl(c, n)
    assert dsl.count(old) == 1
    return dsl.replace(old, f"w: [C={c},N={n}]/f32")
"""
    text = text.replace("class Layer(", helper + "\nclass Layer(", 1)
    namespace = {"__name__": "prefill_fp32_rms"}
    exec(compile(text, "prefill_fp32_rms", "exec"), namespace)  # noqa: S102 - trusted checkout-only experiment
    return namespace["Prefill"]
