"""KONI RNGD prefix and fenced ATOM worker adapters (optional dependencies).

Hardware allocation/export is intentionally kept in the existing owning worker.
No new driver ABI calls or raw-address discovery are introduced by this module.
"""
from dataclasses import dataclass
import json
from opt_prime.accel.pd.runtime import KVLease, KVSpec, PrefillResult


@dataclass(frozen=True)
class RngdPrefix:
    allocation: object
    padded_tokens: int


class KoniRngdEngine:
    def __init__(self, engine, packer, binding, *, revision, context=512, max_prefix=384):
        self.engine, self.packer, self.binding = engine, packer, binding
        self.context, self.max_prefix = context, max_prefix
        self._owned = {}
        self.spec = KVSpec('KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024', revision,
                           'dlfloat16-token-major-v1', 'koni-cache-head-token-dim-v1',
                           (64, 8, context, 128))

    def prefill(self, request):
        if not 2 <= len(request.input_ids) <= self.max_prefix + 1:
            raise ValueError('prompt outside compiled prefill capacity')
        if len(request.input_ids) + request.max_new_tokens > self.context or request.max_new_tokens > 128:
            raise ValueError('request outside compiled decode capacity')
        if any(token >= 128256 for token in request.input_ids):
            raise ValueError('token outside KONI vocabulary')
        import torch
        from rngd_pd.engines.fsdk import sync
        from furiosa.torch._C.tensor import copy_from_tensor
        from furiosa.torch._C.ir import FlatDramShape
        from hetero_p2p import SourceAllocation
        keys, values, _ = self.engine.prefill_device(list(request.input_ids[:-1]))
        self._owned = dict(keys=keys, values=values)
        sync(self.binding.compute_index)
        payload, _ = self.packer.encode(keys, values, len(request.input_ids) - 1)
        self._owned['payload'] = payload
        sync(self.binding.compute_index)
        source = copy_from_tensor(payload, FlatDramShape.new_as_broadcast(list(payload.shape), torch.float16),
                                  self.binding.torch_device)
        self._owned['source'] = source
        sync(self.binding.compute_index)
        padded = ((len(request.input_ids) - 1 + 15) // 16) * 16
        allocation = SourceAllocation(source.data_ptr(), 64 * 8 * padded * 128 * 2,
                                      (source, payload), self.spec.representation, device=self.binding.device_path)
        lease = KVLease(self.spec, allocation, RngdPrefix(allocation, padded), self.binding.device_path)
        return PrefillResult(lease, request.input_ids[-1], len(request.input_ids) - 1)


class KoniAtomWorkerEngine:
    """One exclusive lane in an existing ATOM owner process; no concurrent users."""
    def __init__(self, owner, *, spec, device, batch_size, slot=0):
        if type(slot) is not int or not 0 <= slot < batch_size:
            raise ValueError('invalid decode slot')
        self.owner, self.spec, self.device = owner, spec, device
        self.batch_size, self.slot = batch_size, slot
        self.generation = 0
        self.last_reply = None

    def reserve(self, request):
        if len(request.input_ids) + request.max_new_tokens > self.spec.shape[2] or request.max_new_tokens > 128:
            raise ValueError('request outside compiled ATOM capacity')
        return KVLease(self.spec, self.owner, self.owner, f'atom:{self.device}')

    def decode(self, request, cache, continuation):
        continuation.validate(request)
        if continuation.output_prefix or cache.owner is not self.owner:
            raise ValueError('KONI worker requires prefix-only prefill and its own lease')
        generation = self.generation + 1
        command = dict(cmd='decode_iteration', generation=generation, steps=request.max_new_tokens,
                       requests=[dict(id=request.request_id, slot=self.slot, tokens=list(request.input_ids),
                                      output=request.max_new_tokens, produced=0, admitted=True)])
        conn = self.owner['conn']
        conn.send(json.dumps(command).encode())
        # No in-flight timeout authorizes buffer reuse or owner termination.
        raw = conn.recv(65536)
        if not raw:
            raise RuntimeError('ATOM owner disconnected before completion')
        reply = json.loads(raw)
        self.last_reply = reply
        rows = reply.get('requests', [])
        if (reply.get('generation') != generation or reply.get('device_completed') is not True
                or reply.get('passed') is not True or len(rows) != 1):
            raise RuntimeError('missing matching ATOM completion fence')
        row = rows[0]
        if (row.get('request_id') != request.request_id or row.get('slot') != self.slot
                or row.get('complete') is not True or row.get('passed') is not True
                or row.get('reference_checked') is not True or row.get('max_abs_diff') != 0
                or row.get('produced') != request.max_new_tokens
                or len(row.get('tokens', [])) != request.max_new_tokens):
            raise RuntimeError('ATOM full-logit parity or trajectory check failed')
        self.generation = generation
        return row['tokens']

    def build_plan(self, source, destination):
        from hetero_p2p import DestinationAllocation
        from hetero_p2p.layout import TokenMajorLayout
        if destination.owner is not self.owner or source.spec != self.spec:
            raise ValueError('cache ownership or specification mismatch')
        targets = tuple(DestinationAllocation(fd, self.owner, self.spec.representation)
                        for fd in self.owner['fds'])
        layout = TokenMajorLayout(64, 8, self.spec.shape[2], 128)
        return layout.prefix_plan(source.payload.allocation, targets, self.owner['metadata']['mappings'],
                                  source.payload.padded_tokens, slot=self.slot, batch=self.batch_size)
