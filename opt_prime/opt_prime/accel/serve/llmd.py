"""Optional local llm-d shared-storage control plane, separate from the engine.

No KV bytes travel over HTTP. One serialized owner executes all device work;
run OwnerExecutor.run on the main thread for the checked RNGD PDMA backend.
"""

from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time
from opt_prime.accel.pd.runtime import PDRequest, PDRuntime, CancellationToken, RequestCancelled


class StageError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@dataclass
class Prepared:
    request: PDRequest
    result: object
    created: float
    spec: object

    def prefill(self, request):
        if (
            request.request_id != self.request.request_id
            or request.input_ids != self.request.input_ids
        ):
            raise ValueError("prepared continuation belongs to another request")
        return self.result


class PDStageService:
    """Single prepared source slot. Cancel/TTL can release only fenced prefill."""

    def __init__(
        self,
        prefill,
        decode,
        transport,
        *,
        model,
        vocab_size,
        context,
        max_output,
        render,
        ttl=60,
        clock=time.monotonic
    ):
        if prefill.spec != decode.spec:
            raise ValueError("engine specifications differ")
        if min(vocab_size, context, max_output, ttl) <= 0:
            raise ValueError("positive service limits required")
        self.prefill, self.decode, self.transport = prefill, decode, transport
        self.model, self.vocab_size, self.context = model, vocab_size, context
        self.max_output, self.render, self.ttl, self.clock = (
            max_output,
            render,
            ttl,
            clock,
        )
        self.prepared = None
        self.runtime = None
        self.state = "idle"
        self.events = []

    def _event(self, stage, request_id):
        self.events.append(dict(stage=stage, request_id=request_id, time=self.clock()))
        if len(self.events) > 1024:
            del self.events[:-1024]

    def _request(self, rid, body):
        allowed = {
            "model",
            "prompt",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "stream",
            "cache_hit_threshold",
        }
        if not isinstance(rid, str) or not rid.strip() or len(rid) > 128:
            raise StageError(400, "valid x-request-id required")
        if (
            not isinstance(body, dict)
            or set(body) - allowed
            or body.get("model") != self.model
        ):
            raise StageError(400, "invalid model or unsupported request fields")
        prompt = body.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(t) is not int or not 0 <= t < self.vocab_size for t in prompt)
        ):
            raise StageError(400, "one token-ID prompt required")
        count = body.get("max_completion_tokens", body.get("max_tokens", 32))
        if (
            type(count) is not int
            or not 1 <= count <= self.max_output
            or len(prompt) + count > self.context
        ):
            raise StageError(400, "output or context capacity exceeded")
        if (
            "max_tokens" in body
            and "max_completion_tokens" in body
            and body["max_tokens"] != body["max_completion_tokens"]
        ):
            raise StageError(400, "conflicting output limits")
        if body.get("temperature", 0) != 0 or body.get("stream", False) is not False:
            raise StageError(400, "only greedy non-streaming completions supported")
        if "cache_hit_threshold" in body and body["cache_hit_threshold"] != 0:
            raise StageError(400, "unsupported cache threshold")
        return PDRequest(rid, tuple(prompt), count)

    def cancel(self, rid):
        if self.state != "ready" or self.prepared.request.request_id != rid:
            raise StageError(409, "only a matching fenced prefill can be cancelled")
        self.prepared = None
        self.state = "idle"
        self._event("cancelled", rid)

    def execute(self, stage, rid, body, cancellation=None):
        if stage not in ("prefill", "decode"):
            raise StageError(400, "unknown stage")
        request = self._request(rid, body)
        if cancellation is not None and cancellation.cancelled:
            raise StageError(499, "request cancelled before execution")
        if self.state == "poisoned":
            raise StageError(503, "owner quarantined; recovery required")
        if self.state == "ready" and self.clock() - self.prepared.created > self.ttl:
            old = self.prepared.request.request_id
            self.prepared = None
            self.state = "idle"
            self._event("expired", old)
        if self.prepared is not None and self.prepared.request.request_id == rid:
            if self.prepared.request.input_ids != request.input_ids:
                raise StageError(409, "request ID reused with a different prompt")
        if stage == "prefill":
            if self.state == "ready" and self.prepared.request.request_id == rid:
                return self._prefill_reply(rid)
            if self.state != "idle":
                raise StageError(429, "single cache lane occupied")
            self.state = "prefilling"
            prefill_fenced = False
            try:
                result = self.prefill.prefill(request)
                prefill_fenced = True
                self.prepared = Prepared(
                    request, result, self.clock(), self.prefill.spec
                )
                result.validate(request)
                if result.cache.spec != self.prefill.spec:
                    raise ValueError("prefill cache specification mismatch")
                if cancellation is not None and cancellation.cancelled:
                    raise RequestCancelled(rid)
                self.state = "ready"
                self._event("prefill_ready", rid)
                return self._prefill_reply(rid)
            except RequestCancelled:
                if not prefill_fenced:
                    self.state = "poisoned"
                    raise
                self.prepared = None
                self.state = "idle"
                self._event("cancelled", rid)
                raise StageError(499, "prefill cancelled after completion")
            except BaseException:
                self.state = "poisoned"
                raise
        if self.state != "ready" or self.prepared.request.request_id != rid:
            raise StageError(409, "matching completed prefill required")
        self.state = "decoding"
        self._event("decode_start", rid)
        try:
            # PDRuntime reuses Prepared.prefill; the model is not prefetched again.
            self.runtime = PDRuntime(self.prepared, self.decode, self.transport)
            tokens = self.runtime.generate(request, cancellation=cancellation)
            text = self.render(tokens)
            response = dict(
                id=rid,
                object="text_completion",
                model=self.model,
                choices=[
                    dict(index=0, text=text, token_ids=tokens, finish_reason="length")
                ],
                usage=dict(
                    prompt_tokens=len(request.input_ids),
                    completion_tokens=len(tokens),
                    total_tokens=len(request.input_ids) + len(tokens),
                ),
            )
            self.prepared = None
            self.state = "idle"
            self._event("decode_complete", rid)
            return response
        except RequestCancelled:
            if self.runtime.state != "idle":
                self.state = "poisoned"
                raise
            self.prepared = None
            self.state = "idle"
            self._event("cancelled", rid)
            raise StageError(499, "decode cancelled after drain")
        except BaseException:
            self.state = "poisoned"
            raise

    def _prefill_reply(self, rid):
        return dict(
            id=rid,
            object="text_completion",
            model=self.model,
            choices=[dict(index=0, text="", finish_reason="stop")],
        )


class OwnerExecutor:
    """HTTP threads enqueue only. Timeouts never cancel or release device work."""

    def __init__(self, service, capacity=64):
        self.service = service
        self.queue = queue.Queue(capacity)
        self.started = False
        self.accepting = True
        self._submit_lock = threading.Lock()
        self._pending = {}

    def submit(self, operation, *args):
        future = Future()
        token = CancellationToken() if operation == "execute" else None
        rid = args[1] if token is not None else None
        if token is not None and (
            not isinstance(rid, str) or not rid.strip() or len(rid) > 128
        ):
            raise StageError(400, "valid x-request-id required")
        with self._submit_lock:
            if not self.started or not self.accepting:
                raise StageError(503, "owner not accepting work")
            try:
                self.queue.put_nowait((operation, args, future, token))
                if token is not None:
                    self._pending.setdefault(rid, []).append(token)
            except queue.Full:
                raise StageError(429, "owner command queue full")
            if operation == "stop":
                self.accepting = False
        return future.result()

    def cancel_request(self, rid):
        if not isinstance(rid, str) or not rid.strip() or len(rid) > 128:
            raise StageError(400, "valid x-request-id required")
        with self._submit_lock:
            if not self.started or not self.accepting:
                raise StageError(503, "owner not accepting work")
            tokens = self._pending.get(rid, ())
            if tokens:
                for token in tokens:
                    token.cancel()
                return dict(id=rid, cancellation_requested=True, drained=False)
        # No active/queued call: only the owner may release a prepared cache.
        self.submit("cancel", rid)
        return dict(id=rid, cancellation_requested=True, drained=True)

    def run(self):
        self.started = True
        while True:
            operation, args, future, token = self.queue.get()
            error = None
            stopping = operation == "stop"
            try:
                if stopping:
                    if self.service.state not in ("idle", "ready"):
                        raise StageError(409, "owner not quiescent")
                    if self.service.state == "ready":
                        self.service.cancel(self.service.prepared.request.request_id)
                    self.started = False
                    result = None
                elif operation == "execute":
                    result = self.service.execute(*args, cancellation=token)
                elif operation == "cancel":
                    result = self.service.cancel(*args)
                else:
                    raise StageError(400, "unknown owner operation")
            except BaseException as exc:
                error = exc
            # Complete pending-token bookkeeping atomically with cancellation.
            # A cancel accepted after execute() returns must still drain a ready
            # prefill; otherwise the client sees 202 but the slot remains occupied.
            with self._submit_lock:
                if token is not None:
                    try:
                        if error is None and token.cancelled:
                            if (self.service.state == "ready" and
                                    self.service.prepared.request.request_id == args[1]):
                                self.service.cancel(args[1])
                            error = StageError(499, "request cancelled after completion")
                    except BaseException as exc:
                        error = exc
                    finally:
                        pending = self._pending[args[1]]
                        pending.remove(token)
                        if not pending:
                            del self._pending[args[1]]
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)
                if stopping:
                    return


def serve_stage(owner, stage, port):
    if stage not in ("prefill", "decode"):
        raise ValueError("invalid stage")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The completed owner operation is not cancelled by disconnect.

        def do_GET(self):
            if self.path not in ("/ready", "/health"):
                self.reply(404, dict(error="not found"))
                return
            ready = owner.started and owner.service.state != "poisoned"
            self.reply(200 if ready else 503, dict(ready=ready, stage=stage))

        def do_POST(self):
            if self.path not in ("/v1/completions", "/v1/cancel"):
                self.reply(404, dict(error="not found"))
                return
            try:
                if self.path == "/v1/cancel":
                    result = owner.cancel_request(self.headers.get("x-request-id"))
                    self.reply(200 if result["drained"] else 202, result)
                    return
                if not owner.started:
                    raise StageError(503, "owner not running")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1048576:
                    raise StageError(413, "invalid body size")
                body = json.loads(self.rfile.read(length))
                response = owner.submit(
                    "execute", stage, self.headers.get("x-request-id"), body
                )
                self.reply(200, response)
            except StageError as error:
                self.reply(error.status, dict(error=str(error)))
            except (ValueError, TypeError):
                failed = owner.service.state == "poisoned"
                self.reply(
                    500 if failed else 400,
                    dict(error="engine failed" if failed else "invalid request"),
                )
            except Exception:
                self.reply(500, dict(error="engine failed; owner retained"))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
