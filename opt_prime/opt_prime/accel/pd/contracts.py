"""SDK-free engine, cache lifetime and synchronous transport contracts."""

from dataclasses import dataclass, replace
from typing import Any, Protocol


@dataclass(frozen=True)
class KVSpec:
    model_id: str
    revision: str
    representation: str  # Physical encoding, e.g. dlfloat16; not nominal model dtype.
    layout: str  # Versioned physical layout, including shard/head ordering.
    shape: tuple[int, ...]
    semantic_id: str = ""  # Optional legacy default; versioned model/geometry digest.

    def __post_init__(self):
        if not isinstance(self.semantic_id, str):
            raise ValueError("semantic_id must be a string")
        if not all(
            isinstance(x, str) and x.strip()
            for x in (self.model_id, self.revision, self.representation, self.layout)
        ):
            raise ValueError("explicit model revision and physical KV format required")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(type(x) is not int or x <= 0 for x in self.shape)
        ):
            raise ValueError("positive physical KV dimensions required")


@dataclass(frozen=True)
class SamplingParams:
    """Token selection. The default is greedy; engines without `supports_sampling` accept only that.
    `seed` makes sampling reproducible across processes (each position draws from its own seeded stream), so a
    prefill and a decode stage on different hosts continue one sequence. Generation ends early at any of
    `stop_token_ids` (the stop token is kept in the output)."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.temperature, (int, float)) or not 0 <= self.temperature <= 100:
            raise ValueError("temperature must be in [0, 100]")
        if not isinstance(self.top_p, (int, float)) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.stop_token_ids, tuple) or any(type(t) is not int or t < 0 for t in self.stop_token_ids):
            raise ValueError("stop token IDs must be a tuple of nonnegative integers")

    @property
    def greedy(self):
        return self.temperature == 0 or self.top_k == 1


GREEDY = SamplingParams()


@dataclass(frozen=True)
class PDRequest:
    request_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int                       # upper bound: generation may end earlier at a stop token
    sampling: SamplingParams = GREEDY

    def __post_init__(self):
        if (
            not isinstance(self.request_id, str)
            or not self.request_id.strip()
            or not isinstance(self.input_ids, tuple)
            or not self.input_ids
        ):
            raise ValueError("request ID and prompt required")
        if any(type(x) is not int or x < 0 for x in self.input_ids):
            raise ValueError("nonnegative token IDs required")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("positive output length required")
        if not isinstance(self.sampling, SamplingParams):
            raise ValueError("SamplingParams required")


@dataclass(frozen=True)
class KVLease:
    spec: KVSpec
    owner: Any  # Strong lifetime reference; adapter owns explicit device cleanup.
    payload: Any
    endpoint: str
    request_id: str | None = None
    generation: int | None = None
    lane_id: str | None = None

    def bind(self, request_id, generation, lane_id):
        if (
            not isinstance(request_id, str)
            or not request_id
            or type(generation) is not int
            or generation < 1
            or not isinstance(lane_id, str)
            or not lane_id
        ):
            raise ValueError("valid request, positive generation and lane required")
        identity = (self.request_id, self.generation, self.lane_id)
        expected = (request_id, generation, lane_id)
        if identity != (None, None, None) and identity != expected:
            raise ValueError("stale or foreign KV lease")
        return replace(
            self, request_id=request_id, generation=generation, lane_id=lane_id
        )

    def __post_init__(self):
        if self.owner is None or not self.endpoint:
            raise ValueError("live owner and endpoint required")


@dataclass(frozen=True)
class PrefillResult:
    cache: KVLease
    next_token: int
    cache_position: int
    output_prefix: tuple[int, ...] = ()

    def validate(self, request):
        # Support both full-prompt prefill and KV-only prefix prefill. The latter
        # is used by KONI: ATOM consumes the last prompt token itself.
        if type(self.next_token) is not int or self.next_token < 0:
            raise ValueError("invalid continuation token")
        if type(self.cache_position) is not int:
            raise ValueError("integer cache position required")
        prefix_mode = (
            self.output_prefix == ()
            and self.cache_position == len(request.input_ids) - 1
            and self.next_token == request.input_ids[-1]
        )
        full_mode = self.output_prefix == (
            self.next_token,
        ) and self.cache_position == len(request.input_ids)
        if not (prefix_mode or full_mode):
            raise ValueError("continuation does not match prompt boundary")


class PrefillEngine(Protocol):
    spec: KVSpec

    def prefill(self, request: PDRequest) -> PrefillResult: ...


class DecodeEngine(Protocol):
    spec: KVSpec

    def reserve(self, request: PDRequest) -> KVLease: ...
    def decode(
        self, request: PDRequest, cache: KVLease, continuation: PrefillResult
    ) -> list[int]:
        """Return remaining tokens, after all cache accesses have completed."""
        ...


class Transport(Protocol):
    def supports(self, source: str, destination: str, spec: KVSpec) -> bool: ...
    def transfer(self, request_id: str, source: KVLease, destination: KVLease) -> Any:
        """Return only after verified completion and destination visibility."""
        ...

    def finish(self, request_id: str) -> None: ...
