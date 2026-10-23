"""Process-local validation of the one-line upstream causal-prefill fix."""

import inspect
import linecache
import textwrap

import optimum.rbln.transformers.models.decoderonly.decoderonly_architecture as module

old = module.SlidingWindowAttentionOp.forward
source = textwrap.dedent(inspect.getsource(old))
needle = 'op_args["is_bidirectional"] = True'
assert source.count(needle) == 1, "SDK source changed; review patch before using it"
source = source.replace(
    needle, 'op_args["is_bidirectional"] = self.phase == "image_prefill"'
)
filename = "<gemma-causal-prefill-compat>"
linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
namespace = dict(module.__dict__)
exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - trusted installed SDK source
module.SlidingWindowAttentionOp.forward = namespace["forward"]
