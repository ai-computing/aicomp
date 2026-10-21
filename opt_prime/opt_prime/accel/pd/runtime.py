"""Single-flight PD orchestration; asynchronous engines must fence before return."""

from threading import Lock, Event
from dataclasses import replace
from uuid import uuid4
from time import perf_counter

# Re-export historical imports for existing adapters.
from opt_prime.accel.pd.contracts import (
    KVSpec as KVSpec,
    KVLease as KVLease,
    PrefillResult as PrefillResult,
    PDRequest,
    PrefillEngine,
    DecodeEngine,
    Transport,
)


class RequestCancelled(Exception):
    """Cancellation completed at a synchronous device boundary."""


class CancellationToken:
    """One token per submission; cooperative cancellation, never device abort."""

    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


class PDRuntime:
    """One cache lane. Keep this object alive after failure until device recovery.

    No automatic host fallback, format conversion, retry, or poisoned-lane reset.
    Endpoints must retain allocations internally if their allocation call raises.
    """

    def __init__(
        self, prefill: PrefillEngine, decode: DecodeEngine, transport: Transport
    ):
        if prefill.spec != decode.spec:
            raise ValueError("model revision / physical KV specifications differ")
        self.prefill, self.decode, self.transport = prefill, decode, transport
        self.state = "idle"
        self.retained = []
        self.last_receipt = None
        # Host wall time of fenced stage calls, not accelerator kernel time.
        self.last_timings = {}
        self._lock = Lock()
        self.lane_id = uuid4().hex
        self.generation = 0

    def generate(self, request: PDRequest, *, cancellation=None) -> list[int]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cache lane busy")
        try:
            if self.state != "idle":
                raise RuntimeError("cache lane unavailable; recovery required")
            self.last_receipt = None
            self.last_timings = {}
            for engine in (self.prefill, self.decode):
                capability = getattr(engine, "capabilities", None)
                if capability is not None:
                    capability.validate(request)
            if cancellation is not None and cancellation.cancelled:
                raise RequestCancelled(request.request_id)
            self.generation += 1
            self.state = "reserving"
            self.last_receipt = None
            self.last_timings = {}
            started = perf_counter()
            try:
                destination = self.decode.reserve(request)
                self.retained.append(destination)
                destination = destination.bind(
                    request.request_id, self.generation, self.lane_id
                )
                self.retained[-1] = destination
                reserved_at = perf_counter()
                self.state = "prefilling"
                result = self.prefill.prefill(request)
                self.retained.append(result.cache)
                source = result.cache.bind(
                    request.request_id, self.generation, self.lane_id
                )
                result = replace(result, cache=source)
                self.retained[-1] = source
                prefilled = perf_counter()
                self.last_timings["prefill_seconds"] = prefilled - reserved_at
                if (
                    result.cache.spec != self.prefill.spec
                    or destination.spec != self.prefill.spec
                ):
                    raise ValueError(
                        "allocation does not match negotiated KV specification"
                    )
                result.validate(request)
                if not self.transport.supports(
                    result.cache.endpoint, destination.endpoint, result.cache.spec
                ):
                    raise ValueError("unsupported transport route")
                self.state = "transferring"
                reserved = perf_counter()
                self.last_timings["reserve_validate_seconds"] = (
                    reserved_at - started + reserved - prefilled
                )
                self.last_receipt = self.transport.transfer(
                    request.request_id, result.cache, destination
                )
                transferred = perf_counter()
                self.last_timings["transfer_seconds"] = transferred - reserved
                self.state = "decoding"
                # transfer() has fenced copies and prepared a decode lease. If
                # cancelled, skip compute but still finish that lease normally.
                cancelled = cancellation is not None and cancellation.cancelled
                tail = (
                    []
                    if cancelled
                    else self.decode.decode(request, destination, result)
                )
                decoded = perf_counter()
                self.last_timings["decode_seconds"] = decoded - transferred
                if (
                    not isinstance(tail, list)
                    or len(tail) > request.max_new_tokens - len(result.output_prefix)
                    or any(type(x) is not int or x < 0 for x in tail)
                ):
                    raise ValueError("decoder must return only remaining output tokens")
                self.transport.finish(request.request_id)
                self.retained.clear()
                self.state = "idle"
                self.last_timings["finish_seconds"] = perf_counter() - decoded
                self.last_timings["total_seconds"] = perf_counter() - started
                if cancellation is not None and cancellation.cancelled:
                    raise RequestCancelled(request.request_id)
                return [*result.output_prefix, *tail]
            except RequestCancelled:
                if self.state != "idle":
                    self.state = "poisoned"
                raise
            except BaseException:
                self.state = "poisoned"
                raise
        finally:
            self._lock.release()
