"""Single-flight PD orchestration; asynchronous engines must fence before return."""
from threading import Lock
# Re-export historical imports for existing adapters.
from .contracts import (KVSpec, KVLease, PrefillResult, PDRequest,
                        PrefillEngine, DecodeEngine, Transport)


class PDRuntime:
    """One cache lane. Keep this object alive after failure until device recovery.

    No automatic host fallback, format conversion, retry, or poisoned-lane reset.
    Endpoints must retain allocations internally if their allocation call raises.
    """
    def __init__(self, prefill: PrefillEngine, decode: DecodeEngine, transport: Transport):
        if prefill.spec != decode.spec:
            raise ValueError('model revision / physical KV specifications differ')
        self.prefill, self.decode, self.transport = prefill, decode, transport
        self.state = 'idle'
        self.retained = []
        self.last_receipt = None
        self._lock = Lock()

    def generate(self, request: PDRequest) -> list[int]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cache lane busy')
        try:
            if self.state != 'idle':
                raise RuntimeError('cache lane unavailable; recovery required')
            self.state = 'prefilling'
            self.last_receipt = None
            try:
                result = self.prefill.prefill(request)
                self.retained.append(result.cache)
                destination = self.decode.reserve(request)
                self.retained.append(destination)
                if result.cache.spec != self.prefill.spec or destination.spec != self.prefill.spec:
                    raise ValueError('allocation does not match negotiated KV specification')
                result.validate(request)
                if not self.transport.supports(result.cache.endpoint, destination.endpoint, result.cache.spec):
                    raise ValueError('unsupported transport route')
                self.state = 'transferring'
                self.last_receipt = self.transport.transfer(request.request_id, result.cache, destination)
                self.state = 'decoding'
                tail = self.decode.decode(request, destination, result)
                if (not isinstance(tail, list) or len(tail) > request.max_new_tokens - len(result.output_prefix)
                        or any(type(x) is not int or x < 0 for x in tail)):
                    raise ValueError('decoder must return only remaining output tokens')
                self.transport.finish(request.request_id)
                self.retained.clear()
                self.state = 'idle'
                return [*result.output_prefix, *tail]
            except BaseException:
                self.state = 'poisoned'
                raise
        finally:
            self._lock.release()
