# HTTP SIGPIPE validation

State: review. Per-write MSG_NOSIGNAL preserves BrokenPipeError and avoids the observed native SIGPIPE backtraces. Fourteen tests pass; Qwen0.6 ATOM before/after outputs match. Qwen4B host-cache passes. Full four-model goal remains active; see experiment for scope.
