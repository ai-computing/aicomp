"""Single-flight PD orchestration; asynchronous engines must fence before return."""
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol


@dataclass(frozen=True)
class KVSpec:
    model_id: str
    revision: str
    representation: str  # Physical encoding, e.g. dlfloat16; not nominal model dtype.
    layout: str          # Versioned physical layout, including shard/head ordering.
    shape: tuple[int, ...]

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in
                   (self.model_id, self.revision, self.representation, self.layout)):
            raise ValueError('explicit model revision and physical KV format required')
        if not self.shape or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError('positive physical KV dimensions required')


@dataclass(frozen=True)
class PDRequest:
    request_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int

    def __post_init__(self):
        if not self.request_id or not self.input_ids:
            raise ValueError('request ID and prompt required')
        if any(type(x) is not int or x < 0 for x in self.input_ids):
            raise ValueError('nonnegative token IDs required')
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError('positive output length required')


@dataclass(frozen=True)
class KVLease:
    spec: KVSpec
    owner: Any  # Strong lifetime reference; adapter owns explicit device cleanup.
    payload: Any
    endpoint: str

    def __post_init__(self):
        if self.owner is None or not self.endpoint:
            raise ValueError('live owner and endpoint required')


@dataclass(frozen=True)
class PrefillResult:
    cache: KVLease
    first_token: int  # Prefill samples token 1; decoder emits only remaining tokens.


class PrefillEngine(Protocol):
    spec: KVSpec
    def prefill(self, request: PDRequest) -> PrefillResult: ...


class DecodeEngine(Protocol):
    spec: KVSpec
    def reserve(self, request: PDRequest) -> KVLease: ...
    def decode(self, request: PDRequest, cache: KVLease, first_token: int) -> list[int]:
        """Return remaining tokens, after all cache accesses have completed."""
        ...


class Transport(Protocol):
    def supports(self, source: str, destination: str, spec: KVSpec) -> bool: ...
    def transfer(self, request_id: str, source: KVLease, destination: KVLease) -> Any:
        """Return only after verified completion and destination visibility."""
        ...
    def finish(self, request_id: str) -> None: ...


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
                if type(result.first_token) is not int or result.first_token < 0:
                    raise ValueError('invalid first token')
                if not self.transport.supports(result.cache.endpoint, destination.endpoint, result.cache.spec):
                    raise ValueError('unsupported transport route')
                self.state = 'transferring'
                self.last_receipt = self.transport.transfer(request.request_id, result.cache, destination)
                self.state = 'decoding'
                tail = self.decode.decode(request, destination, result.first_token)
                if (not isinstance(tail, list) or len(tail) > request.max_new_tokens - 1
                        or any(type(x) is not int or x < 0 for x in tail)):
                    raise ValueError('decoder must return only remaining output tokens')
                self.transport.finish(request.request_id)
                self.retained.clear()
                self.state = 'idle'
                return [result.first_token, *tail]
            except BaseException:
                self.state = 'poisoned'
                raise
        finally:
            self._lock.release()
