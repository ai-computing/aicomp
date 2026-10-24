"""Process-local global text-prefill correction; rebuild into a separate cache."""

import inspect
import linecache
import textwrap

import gemma_causal_compat  # noqa: F401 - preserve the validated sliding correction
import optimum.rbln.transformers.models.decoderonly.decoderonly_architecture as module

for name in ("AttentionOp", "FlashAttentionOp"):
    cls = getattr(module, name)
    source = textwrap.dedent(inspect.getsource(cls.forward))
    needle = 'op_args["is_bidirectional"] = True'
    assert source.count(needle) == 1, "SDK changed; review the global causal patch"
    source = source.replace(
        needle, 'op_args["is_bidirectional"] = self.phase == "image_prefill"'
    )
    filename = f"<gemma-global-causal-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(module.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    cls.forward = namespace["forward"]
