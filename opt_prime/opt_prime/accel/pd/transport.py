"""KV transports on hetero-kv sessions (https://github.com/ai-computing/hetero-kv).

hetero-kv moves the bytes and owns the lifecycle reserve -> transfer -> fence -> admit -> finish: decode is admitted
only after every completion is fenced, an uncertain completion poisons the session until an engine quiesce, and
routes are explicit (no fallback). This module is Optimus' glue: it maps Optimus caches and DMA plans onto hetero-kv
buffers. Importing it does not import torch or a vendor SDK.
"""
from dataclasses import dataclass
import os
import re

import hetero_kv as hk


def _settle(session):
    """Bring a session back to idle after the lane failed; True on success."""
    rid = session.request_id
    try:
        if session.state == "admitted":
            try:
                session.finish(rid, passed=False)      # the decoder failed while using the cache
            except hk.PoisonedError:
                pass                                   # expected: a failed consumer poisons
        elif session.state in ("reserved", "fenced"):
            session.cancel(rid)
        elif session.state == "transferring":
            session.cancel(rid)
            session.fence(rid)                         # never abandon in-flight copies
            try:
                session.admit(rid)
            except hk.CancelledError:
                pass
        if session.state == "poisoned":
            session.recover()
    except hk.HeteroKVError:
        return False
    return session.state == "idle"


@dataclass(frozen=True)
class TransferPlan:
    """A plan builder's result: the source buffer, one destination buffer per KV group, and the selections copied
    into each group."""
    source: object
    destinations: tuple
    selections: tuple


class SessionTransport:
    """Explicit route over any hetero-kv engine between two device locations (e.g. rngd:3 -> atom:2).

    plan_builder(source_lease, destination_lease) -> TransferPlan must use live, owned allocations;
    notify(destination_lease) makes the fenced cache visible to the decoder.
    """
    def __init__(self, engine, plan_builder, notify, *, source, destination, spec, src_location, dst_location,
                 representation):
        self.representation = hk.canonical_representation(representation)
        self.session = hk.Session(hk.Route(engine, src_location, dst_location, self.representation))
        self.plan_builder, self.notify = plan_builder, notify
        self.source, self.destination, self.spec = source, destination, spec

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError("unsupported P2P route or physical format")
        plan = self.plan_builder(source, destination)
        if any(b.spec.representation != self.representation for b in (plan.source, *plan.destinations)):
            raise ValueError("DMA plan format differs from negotiated representation")
        if len(plan.selections) != len(plan.destinations):
            raise ValueError("one selection list per destination group required")
        s = self.session
        s.reserve(request_id, list(plan.destinations))
        try:
            for group, selections in enumerate(plan.selections):
                if selections:
                    s.transfer(request_id, plan.source, list(selections), group=group)
            s.fence(request_id)
            return s.admit(request_id, lambda: self.notify(destination))
        except (hk.InvalidError, hk.UnsupportedError, hk.StateError):
            _settle(s)                                 # rejected before the engine saw anything
            raise

    def finish(self, request_id):
        self.session.finish(request_id, passed=True)

    def recover(self):
        return _settle(self.session)


_ROW = 4096   # hetero-npu-p2p DMA spans are 4 KiB aligned


class P2PBackendEngine:
    """hetero-kv engine over a hetero-npu-p2p DMA backend (`backend.transfer(plan) -> receipt`, e.g. RngdPdmaBackend
    for RNGD -> ATOM DMA-BUF). Each job runs as one validated hetero_p2p TransferPlan with the original plan's
    command grouping; the receipt is checked as CopySession did. No quiesce: a poisoned lane stays poisoned."""
    name = "hetero-p2p"

    def __init__(self, backend, src_location, dst_location, representation):
        self.backend = backend
        self.capabilities = [(src_location, dst_location, hk.canonical_representation(representation))]

    def submit(self, job):
        from hetero_p2p.core import CopySpan, TransferPlan as DmaPlan
        original = job.src[0].obj
        plan = DmaPlan(original.source, tuple(r.obj for r in job.dst),
                       tuple(CopySpan(di, so, do, n) for _, so, di, do, n in job.spans),
                       original.descriptors_per_command).validate()
        receipt = self.backend.transfer(plan)
        if (receipt.get("cqe_result") != 0 or receipt.get("completed") != plan.commands
                or receipt.get("requested") != plan.commands or receipt.get("bytes_requested") != plan.bytes):
            raise RuntimeError("incomplete or inconsistent DMA completion receipt")
        return plan.bytes

    def poll(self, ticket):
        return True

    def wait(self, ticket):
        return ticket

    def quiesce(self):
        return False


def p2p_plan(dma, src_location, dst_location, representation):
    """hetero-kv buffers for a validated hetero_p2p TransferPlan: flat 4 KiB rows over the source allocation and
    each destination DMA-BUF, one selection per DMA span, so every job's spans are exactly the plan's."""
    element = hk.Layout(representation, 1, 1, 1, 1, kv="latent").region_bytes

    def flat(location, addr, nbytes, obj):
        layout = hk.Layout(representation, 1, nbytes // _ROW, 1, _ROW // element, kv="latent")
        return hk.KVBuffer(hk.KVSpec("hetero-p2p", "flat-4k-v1", layout),
                           [hk.Region.external(location, addr, layout.region_bytes, obj)])

    source = flat(src_location, dma.source.address, dma.source.size, dma)
    destinations = tuple(flat(dst_location, 0, os.fstat(d.fd).st_size, d) for d in dma.destinations)
    selections = tuple(tuple(hk.Selection.contiguous(s.source_offset // _ROW, s.destination_offset // _ROW,
                                                     s.length // _ROW)
                             for s in dma.spans if s.destination == i)
                       for i in range(len(destinations)))
    return TransferPlan(source, destinations, selections)


def p2p_transport(backend, plan_builder, notify, *, source, destination, spec):
    """SessionTransport over a hetero-npu-p2p backend from an RNGD device path (/dev/rngd/npuNchM) to atom:N.
    plan_builder returns a hetero_p2p TransferPlan in the negotiated physical format (spec.representation, e.g.
    dlfloat16-token-major-v1; its element type is the hetero-kv representation)."""
    m = re.fullmatch(r"/dev/rngd/npu(\d+)ch\d+", str(source))
    if m is None or not re.fullmatch(r"atom:\d+", str(destination)):
        raise ValueError("explicit RNGD device path and ATOM device required")
    locations = (f"rngd:{m.group(1)}", destination)
    representation = hk.canonical_representation(spec.representation.split("-")[0])

    def build(src, dst):
        dma = plan_builder(src, dst)
        if (dma.source.format != spec.representation
                or any(d.format != spec.representation for d in dma.destinations)):
            raise ValueError("DMA plan format differs from negotiated representation")
        return p2p_plan(dma.validate(), *locations, representation)

    return SessionTransport(P2PBackendEngine(backend, *locations, representation), build, notify,
                            source=source, destination=destination, spec=spec, src_location=locations[0],
                            dst_location=locations[1], representation=representation)


class TorchCacheTransport:
    """Explicit tensor-copy transport, never selected as a P2P fallback. Copies the source prefix [0, seq_len) of
    every layer and batch entry with hetero-kv's TorchCopyEngine, fences, then publishes seq_len on the destination
    as the admission step."""
    def __init__(self, *, source, destination, spec):
        self.source, self.destination, self.spec = source, destination, spec
        self._sessions = {}
        self._active = {}

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    def _session(self, src_location, dst_location, representation):
        from hetero_kv.torch_engine import TorchCopyEngine
        key = (src_location, dst_location, representation)
        if key not in self._sessions:
            self._sessions[key] = hk.Session(hk.Route(TorchCopyEngine(), *key))
        return self._sessions[key]

    def transfer(self, request_id, source, destination):
        from hetero_kv.torch_engine import tensor_location
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError("unconfigured tensor-copy route")
        src, dst = source.payload.manager, destination.payload.manager
        position = src.get_seq_len()
        if not 0 < position <= dst.max_seq_len:
            raise ValueError("invalid source sequence length")
        src_groups, src_t = _groups(src, source.spec)
        dst_groups, dst_t = _groups(dst, destination.spec)
        if len(src_groups) != len(dst_groups):
            raise ValueError("source and destination layer groups differ")
        session = self._session(tensor_location(src_t.device), tensor_location(dst_t.device),
                                hk.canonical_representation(str(src_t.dtype)))
        session.reserve(request_id, dst_groups)
        try:
            # One transfer per (group, batch entry): disjoint byte ranges of the same tensors, in flight together.
            for group, buf in enumerate(src_groups):
                for b in range(src_t.shape[0]):
                    session.transfer(request_id, buf, hk.Selection(src_blocks=(b,), dst_blocks=(b,), tokens=position),
                                     group=group)
            receipt = session.fence(request_id)
            session.admit(request_id, lambda: dst.set_seq_len(position))
        except (hk.InvalidError, hk.UnsupportedError, hk.StateError):
            _settle(session)
            raise
        self._active[request_id] = session
        return dict(transport="torch-copy", bytes=receipt["bytes"], cache_position=position, completed=True)

    def finish(self, request_id):
        session = self._active.pop(request_id, None)
        if session is not None:
            session.finish(request_id, passed=True)

    def recover(self):
        """Quiesce (Torch device sync) and settle every session; True when all are idle."""
        ok = all([_settle(s) for s in self._sessions.values()])
        if ok:
            self._active.clear()
        return ok


def _groups(manager, spec):
    """KV buffers over a cache's full-capacity [batch, heads, capacity, dim] tensors: one bhsd group, or one per layer
    when layer shapes differ (e.g. Gemma 4). Blocks are batch entries; block_size is the capacity."""
    from hetero_kv.torch_engine import tensor_region
    pairs = [manager.storage(i) for i in range(manager.layer_start, manager.layer_end)]
    chunks = [pairs] if len({tuple(k.shape) for k, _ in pairs}) == 1 else [[p] for p in pairs]
    groups = []
    for chunk in chunks:
        batch, heads, capacity, dim = chunk[0][0].shape
        layout = hk.Layout.preset("bhsd", str(chunk[0][0].dtype), len(chunk), capacity, heads, dim, blocks=batch)
        groups.append(hk.KVBuffer(hk.KVSpec(spec.model_id, spec.revision, layout),
                                  [tensor_region(t) for pair in chunk for t in pair]))
    return groups, pairs[0][0]
