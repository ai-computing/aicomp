"""Qwen endpoint adapters; common runtime and transport remain model independent."""

from dataclasses import dataclass
import json
from ..contracts import KVSpec, KVLease, PrefillResult


@dataclass(frozen=True)
class PrefixPayload:
    allocation: object
    padded_tokens: int


class QwenRngdEngine:
    def __init__(self, engine, packer, binding, *, compact_cache=False):
        self.engine, self.packer, self.binding = engine, packer, binding
        if type(compact_cache) is not bool:
            raise ValueError("compact_cache must be boolean")
        self.compact_cache = compact_cache
        self.model = engine.spec
        s = self.model
        self.spec = KVSpec(
            s.model_id,
            s.revision,
            "dlfloat16-token-major-v1",
            "dense-cache-head-token-dim-v1",
            (s.caches, s.kv_heads, 512, s.head_dim),
        )
        self._owned = {}

    def prefill(self, request):
        s = self.model
        if (
            not 2 <= len(request.input_ids) <= self.engine.width + 1
            or len(request.input_ids) + request.max_new_tokens > 512
        ):
            raise ValueError("request exceeds compiled Qwen capacity")
        if any(t >= s.vocabulary for t in request.input_ids):
            raise ValueError("token outside vocabulary")
        import torch
        from rngd_pd.engines.fsdk import sync
        from furiosa.torch._C.tensor import copy_from_tensor
        from furiosa.torch._C.ir import FlatDramShape
        from hetero_p2p import SourceAllocation

        keys, values, _ = self.engine.prefill_device(
            list(request.input_ids[:-1]), full_context=not self.compact_cache
        )
        self._owned = dict(keys=keys, values=values)
        padded = s.padded_prefix(len(request.input_ids) - 1)
        payload, _ = self.packer.encode(keys, values, padded)
        self._owned["payload"] = payload
        sync(self.binding.compute_index)
        source = copy_from_tensor(
            payload,
            FlatDramShape.new_as_broadcast(list(payload.shape), torch.float16),
            self.binding.torch_device,
        )
        self._owned["source"] = source
        sync(self.binding.compute_index)
        allocation = SourceAllocation(
            source.data_ptr(),
            s.caches * s.kv_heads * padded * s.head_dim * 2,
            (source, payload),
            self.spec.representation,
            device=self.binding.device_path,
        )
        lease = KVLease(
            self.spec,
            allocation,
            PrefixPayload(allocation, padded),
            self.binding.device_path,
        )
        return PrefillResult(lease, request.input_ids[-1], len(request.input_ids) - 1)


class QwenAtomWorkerEngine:
    def __init__(self, owner, spec, physical, *, require_reference=True):
        self.owner, self.spec, self.physical = owner, spec, physical
        self.require_reference = require_reference
        self.generation = 0
        self.last_reply = None

    def reserve(self, request):
        if len(request.input_ids) + request.max_new_tokens > 512:
            raise ValueError("context capacity exceeded")
        return KVLease(self.spec, self.owner, self.owner, f"atom:{self.physical}")

    def decode(self, request, cache, continuation):
        continuation.validate(request)
        if cache.owner is not self.owner or continuation.output_prefix:
            raise ValueError("invalid cache continuation")
        self.generation += 1
        self.owner["conn"].send(
            json.dumps(
                dict(
                    cmd="decode",
                    generation=self.generation,
                    request_id=request.request_id,
                    prompt=list(request.input_ids),
                    count=request.max_new_tokens,
                    dma_completed=True,
                )
            ).encode()
        )
        raw = self.owner["conn"].recv(65536)
        if not raw:
            raise RuntimeError("owner disconnected before completion")
        reply = json.loads(raw)
        self.last_reply = reply
        if (
            reply.get("generation") != self.generation
            or reply.get("device_completed") is not True
            or reply.get("passed") is not True
            or reply.get("request_id") != request.request_id
            or (
                self.require_reference
                and (
                    reply.get("reference_checked") is not True
                    or reply.get("max_abs_diff") != 0
                )
            )
            or len(reply.get("tokens", [])) != request.max_new_tokens
        ):
            raise RuntimeError("decode fence or full-logit equality failed")
        return reply["tokens"]

    def build_plan(self, source, destination):
        from hetero_p2p import DestinationAllocation
        from hetero_p2p.layout import TokenMajorLayout

        if destination.owner is not self.owner or source.spec != self.spec:
            raise ValueError("physical specification mismatch")
        targets = tuple(
            DestinationAllocation(fd, self.owner, self.spec.representation)
            for fd in self.owner["fds"]
        )
        caches, heads, context, dim = self.spec.shape
        return TokenMajorLayout(caches, heads, context, dim).prefix_plan(
            source.payload.allocation,
            targets,
            self.owner["metadata"]["mappings"],
            source.payload.padded_tokens,
        )
