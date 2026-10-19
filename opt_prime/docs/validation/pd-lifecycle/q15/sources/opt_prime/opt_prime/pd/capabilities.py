"""SDK-free, explicit endpoint limits; no discovery or implicit fallback."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineCapabilities:
    role: str
    max_input: int
    max_context: int
    transports: frozenset[str]
    external_kv: bool = False

    def __post_init__(self):
        if self.role not in ("prefill", "decode"):
            raise ValueError("invalid endpoint role")
        if (
            type(self.max_input) is not int
            or type(self.max_context) is not int
            or not 0 < self.max_input <= self.max_context
        ):
            raise ValueError("invalid endpoint limits")
        if not isinstance(self.transports, frozenset) or not self.transports <= {
            "p2p",
            "host",
        }:
            raise ValueError("explicit transport set required")
        if "p2p" in self.transports and not self.external_kv:
            raise ValueError("P2P requires a validated external KV contract")

    def validate(self, request):
        if (
            len(request.input_ids) > self.max_input
            or len(request.input_ids) + request.max_new_tokens > self.max_context
        ):
            raise ValueError("request exceeds endpoint capability")


def negotiate(source, destination, transport):
    if source.role != "prefill" or destination.role != "decode":
        raise ValueError("endpoint role mismatch")
    if transport not in source.transports & destination.transports:
        raise ValueError("unsupported endpoint/transport combination; no fallback")
