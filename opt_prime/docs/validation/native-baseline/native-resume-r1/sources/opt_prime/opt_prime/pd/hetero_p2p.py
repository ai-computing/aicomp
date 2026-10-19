"""Optional bridge to the separately installed hetero-npu-p2p library."""


class HeteroP2PTransport:
    """Explicit same-host route; plan_builder must use live owned allocations.

    Backend construction retains existing SDK/driver/helper ABI checks.
    notify(destination) makes the completed cache visible to the decoder.
    No address discovery, reverse import, or implicit host fallback is added.
    """
    def __init__(self, backend, plan_builder, notify, *, source, destination, spec):
        from hetero_p2p.core import CopySession
        self.session = CopySession(backend)
        self.plan_builder, self.notify = plan_builder, notify
        self.source, self.destination, self.spec = source, destination, spec

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError('unsupported P2P route or physical format')
        plan = self.plan_builder(source, destination)
        if (plan.source.format != self.spec.representation or
                any(d.format != self.spec.representation for d in plan.destinations)):
            raise ValueError('DMA plan format differs from negotiated representation')
        receipt = self.session.admit(request_id, plan, lambda: self.notify(destination))
        self.session.start_decode(request_id)
        return receipt

    def finish(self, request_id):
        self.session.finish_decode(request_id, passed=True)
