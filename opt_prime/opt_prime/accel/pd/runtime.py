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

    def generate(self, request: PDRequest, *, cancellation=None, on_token=None) -> list[int]:
        """on_token(token): called on this thread for every output token as soon as it is known (the prefill's
        tokens before decode starts, then each decoded token if the decode engine streams, else after decode)."""
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
                if not request.sampling.greedy and not getattr(engine, "supports_sampling", False):
                    raise ValueError("engine supports greedy decoding only")
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
                if on_token is not None and not (cancellation is not None and cancellation.cancelled):
                    prefix = stop_at(request, list(result.output_prefix))
                    for t in prefix:                       # the prefill's tokens: flushed before the KV copy
                        on_token(t)
                    if prefix and prefix[-1] in request.sampling.stop_token_ids:
                        on_token = None                    # the prefill already ended the output
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
                kw = {}
                if on_token is not None and _accepts(self.decode.decode, "on_token"):
                    kw["on_token"] = on_token
                tail = (
                    []
                    if cancelled
                    else self.decode.decode(request, destination, result, **kw)
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
                out = stop_at(request, [*result.output_prefix, *tail])
                if on_token is not None and not kw:        # the engine did not stream: emit its tokens now
                    for t in out[len(result.output_prefix):]:
                        on_token(t)
                return out
            except RequestCancelled:
                if self.state != "idle":
                    self.state = "poisoned"
                raise
            except BaseException:
                self.state = "poisoned"
                raise
        finally:
            self._lock.release()


def stop_at(request, tokens):
    """Cut after the first stop token (engines that cannot stop early still return a correct output)."""
    stop = request.sampling.stop_token_ids
    for i, t in enumerate(tokens):
        if t in stop:
            return tokens[:i + 1]
    return tokens


def _accepts(fn, name):
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
