"""Optional local llm-d shared-storage control plane, separate from the engine.

No KV bytes travel over HTTP. One serialized owner executes all device work;
run OwnerExecutor.run on the main thread for the checked RNGD PDMA backend.
"""

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import select
import socket
import threading
import time
from opt_prime.accel.pd.contracts import RetryableError, SamplingParams
from opt_prime.accel.pd.runtime import PDRequest, PDRuntime, CancellationToken, RequestCancelled
from opt_prime.accel.serve.metrics import StageMetrics


class StageError(Exception):
    """HTTP status + message. retryable: the request may be retried (here later, or on another replica)."""
    def __init__(self, status, message, retryable=False):
        super().__init__(message)
        self.status, self.retryable = status, retryable


@dataclass
class Prepared:
    request: PDRequest
    result: object
    created: float
    spec: object

    def recover(self):
        # host-side record of a fenced prefill; the source engine's own recovery is checked by PDStageService.reset
        return True

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
        clock=time.monotonic,
        eos_token_ids=(),
        tokenizer=None
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
        self.eos_token_ids = tuple(eos_token_ids)
        self.tokenizer = tokenizer          # text prompts and chat messages (token-ID prompts need none)
        self.prepared = None
        self.runtime = None
        self.state = "idle"
        self.events = []
        self.metrics = StageMetrics(model)

    def gauges(self, waiting=0):
        """(running, waiting, KV lane usage) for /metrics: one lane, occupied unless idle"""
        running = 1 if self.state in ("prefilling", "decoding") else 0
        return running, waiting, 0.0 if self.state == "idle" else 1.0

    def _event(self, stage, request_id):
        self.events.append(dict(stage=stage, request_id=request_id, time=self.clock()))
        if len(self.events) > 1024:
            del self.events[:-1024]

    def _request(self, rid, body, chat=False):
        allowed = {
            "messages",
            "stream_options",
            "model",
            "prompt",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "stop",
            "ignore_eos",
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
        prompt = self._prompt(body, chat)
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
        if not isinstance(body.get("stream", False), bool):
            raise StageError(400, "stream must be a boolean")
        stop = body.get("stop", [])
        stop = [stop] if isinstance(stop, str) else stop
        if (not isinstance(stop, list) or len(stop) > 4
                or any(not isinstance(x, str) or not 0 < len(x) <= 64 for x in stop)):
            raise StageError(400, "stop must be up to 4 non-empty strings")
        ignore_eos = body.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise StageError(400, "ignore_eos must be a boolean")
        temperature, seed = body.get("temperature", 0), body.get("seed")
        if isinstance(temperature, bool) or isinstance(body.get("top_p", 1), bool):
            raise StageError(400, "invalid sampling parameters")
        try:
            sampling = SamplingParams(float(temperature) if isinstance(temperature, (int, float)) else temperature,
                                      body.get("top_p", 1.0), body.get("top_k", 0), seed,
                                      () if ignore_eos else self.eos_token_ids)
        except ValueError as error:
            raise StageError(400, str(error))
        if not sampling.greedy and not all(getattr(e, "supports_sampling", False)
                                           for e in (self.prefill, self.decode)):
            raise StageError(400, "engine supports greedy decoding only")
        if "cache_hit_threshold" in body and body["cache_hit_threshold"] != 0:
            raise StageError(400, "unsupported cache threshold")
        return PDRequest(rid, tuple(prompt), count, sampling), tuple(stop)

    def expire(self):
        """Release a fenced prefill whose decode did not arrive within the TTL (owner thread only)."""
        if self.state == "ready" and self.clock() - self.prepared.created > self.ttl:
            old = self.prepared.request.request_id
            self.prepared = None
            self.state = "idle"
            self._event("expired", old)
            return old
        return None

    def _prompt(self, body, chat):
        """token IDs of the request: `messages` through the chat template (chat), a text `prompt` through the
        tokenizer, or a token-ID `prompt` as is"""
        if chat:
            messages = body.get("messages")
            if "prompt" in body or not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or not isinstance(m.get("role"), str)
                    or not isinstance(m.get("content"), str) for m in messages):
                raise StageError(400, "chat requires messages with role and text content")
            if self.tokenizer is None:
                raise StageError(400, "this stage has no tokenizer for chat messages")
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
            return [int(t) for t in ids]
        if "messages" in body:
            raise StageError(400, "messages belong to /v1/chat/completions")
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise StageError(400, "this stage has no tokenizer for text prompts")
            return [int(t) for t in self.tokenizer(prompt, add_special_tokens=True)["input_ids"]]
        return prompt

    def cancel(self, rid):
        if self.state != "ready" or self.prepared.request.request_id != rid:
            raise StageError(409, "only a matching fenced prefill can be cancelled")
        self.prepared = None
        self.state = "idle"
        self._event("cancelled", rid)

    def execute(self, stage, rid, body, cancellation=None, on_token=None, chat=False):
        """on_token(token) streams output tokens on the owner thread (first token after the fenced prefill)."""
        if stage not in ("prefill", "decode"):
            raise StageError(400, "unknown stage")
        request, stops = self._request(rid, body, chat)
        if cancellation is not None and cancellation.cancelled:
            raise StageError(499, "request cancelled before execution")
        if self.state == "poisoned":
            raise StageError(503, "owner quarantined; recovery required", retryable=True)
        self.expire()
        if self.prepared is not None and self.prepared.request.request_id == rid:
            if self.prepared.request.input_ids != request.input_ids:
                raise StageError(409, "request ID reused with a different prompt")
        if stage == "prefill":
            if self.state == "ready" and self.prepared.request.request_id == rid:
                return self._prefill_reply(rid, chat)
            if self.state != "idle":
                raise StageError(429, "single cache lane occupied")
            self.state = "prefilling"
            prefill_fenced = False
            started = self.clock()
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
                elapsed = self.clock() - started
                self.metrics.observe("vllm:time_to_first_token_seconds", elapsed)   # prefill emits the first token
                self.metrics.observe("vllm:e2e_request_latency_seconds", elapsed)
                self.metrics.finished("stop", len(request.input_ids), 1)
                if on_token is not None:
                    for t in result.output_prefix:
                        on_token(t)
                return self._prefill_reply(rid, chat)
            except RequestCancelled:
                if not prefill_fenced:
                    self.state = "poisoned"
                    raise
                self.prepared = None
                self.state = "idle"
                self._event("cancelled", rid)
                raise StageError(499, "prefill cancelled after completion")
            except RetryableError as error:
                self.prepared = None                      # the engine left nothing outstanding
                self.state = "idle"
                self.metrics.failed()
                self._event("retryable_failure", rid)
                raise StageError(503, f"retryable: {error}", retryable=True)
            except BaseException:
                self.state = "poisoned"
                self.metrics.failed()
                raise
        if self.state != "ready" or self.prepared.request.request_id != rid:
            raise StageError(409, "matching completed prefill required")
        self.state = "decoding"
        self._event("decode_start", rid)
        started = self.clock()
        try:
            # PDRuntime reuses Prepared.prefill; the model is not prefetched again.
            self.runtime = PDRuntime(self.prepared, self.decode, self.transport)
            tokens = self.runtime.generate(request, cancellation=cancellation, on_token=on_token)
            tokens, text, finish = self._finish(request, stops, tokens)
            usage = dict(prompt_tokens=len(request.input_ids), completion_tokens=len(tokens),
                         total_tokens=len(request.input_ids) + len(tokens))
            if chat:
                choice = dict(index=0, message=dict(role="assistant", content=text), token_ids=tokens,
                              finish_reason=finish)
            else:
                choice = dict(index=0, text=text, token_ids=tokens, finish_reason=finish)
            response = dict(id=rid, object="chat.completion" if chat else "text_completion", created=int(time.time()),
                            model=self.model, choices=[choice], usage=usage)
            self.prepared = None
            self.state = "idle"
            self._event("decode_complete", rid)
            t = self.runtime.last_timings
            if "transfer_seconds" in t:
                self.metrics.observe("airm_kv_transfer_seconds", t["transfer_seconds"])
            if len(tokens) > 1 and "decode_seconds" in t:
                self.metrics.observe("vllm:time_per_output_token_seconds", t["decode_seconds"] / (len(tokens) - 1))
            self.metrics.observe("vllm:e2e_request_latency_seconds", self.clock() - started)
            self.metrics.finished(finish, len(request.input_ids), len(tokens))
            return response
        except RequestCancelled:
            if self.runtime.state != "idle":
                self.state = "poisoned"
                raise
            self.prepared = None
            self.state = "idle"
            self._event("cancelled", rid)
            raise StageError(499, "decode cancelled after drain")
        except RetryableError as error:
            self.prepared = None                          # runtime released the leases; retry from prefill
            self.state = "idle"
            self.metrics.failed()
            self._event("retryable_failure", rid)
            raise StageError(503, f"retryable: {error}", retryable=True)
        except BaseException:
            self.state = "poisoned"
            self.metrics.failed()
            raise

    def reset(self):
        """Recovery from quarantine: every endpoint must confirm a known-good device state (`recover() -> True`);
        only then are the retained allocations released and the lane reopened. Owner thread only."""
        if self.state != "poisoned":
            raise StageError(409, "owner is not quarantined")
        for part in (self.prefill, self.decode, self.transport):
            fn = getattr(part, "recover", None)
            if fn is None:
                raise StageError(501, f"{type(part).__name__} has no recovery; restart the stage")
            if fn() is not True:
                raise StageError(503, f"{type(part).__name__} not known-good; still quarantined")
        if self.runtime is not None and not self.runtime.recover():
            raise StageError(503, "cache lane not recovered; still quarantined")
        self.prepared = None
        self.state = "idle"
        self._event("recovered", None)
        return dict(state=self.state)

    def _finish(self, request, stop_strings, tokens):
        """(tokens, text, finish_reason): cut at the first stop string (the text excludes it, the tokens end with the
        token that completed it); 'stop' also when the last token is a stop token."""
        text = self.render(tokens)
        stops = [s for s in stop_strings if s in text]
        if stops:
            for i in range(1, len(tokens) + 1):
                partial = self.render(tokens[:i])
                hit = min((partial.find(s) for s in stops if s in partial), default=-1)
                if hit >= 0:
                    return tokens[:i], partial[:hit], "stop"
        stop_ids = request.sampling.stop_token_ids
        return tokens, text, "stop" if tokens and tokens[-1] in stop_ids else "length"

    def _prefill_reply(self, rid, chat=False):
        if chat:
            return dict(id=rid, object="chat.completion", model=self.model,
                        choices=[dict(index=0, message=dict(role="assistant", content=""), finish_reason="stop")])
        return dict(
            id=rid,
            object="text_completion",
            model=self.model,
            choices=[dict(index=0, text="", finish_reason="stop")],
        )


class OwnerExecutor:
    """HTTP threads enqueue only. Timeouts never cancel or release device work.

    While idle, the owner sweeps the TTL every `sweep` seconds, so an abandoned prepared prefill frees the lane
    without waiting for the next request."""

    def __init__(self, service, capacity=64, sweep=1.0):
        self.service, self.sweep = service, sweep
        self.queue = queue.Queue(capacity)
        self.started = False
        self.accepting = True
        self._submit_lock = threading.Lock()
        self._pending = {}

    def submit(self, operation, *args, **kwargs):
        return self.submit_async(operation, *args, **kwargs).result()

    def submit_async(self, operation, *args, **kwargs):
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
                self.queue.put_nowait((operation, args, kwargs, future, token))
                if token is not None:
                    self._pending.setdefault(rid, []).append(token)
            except queue.Full:
                raise StageError(429, "owner command queue full")
            if operation == "stop":
                self.accepting = False
        return future

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
            try:
                operation, args, kwargs, future, token = self.queue.get(timeout=self.sweep)
            except queue.Empty:
                self.service.expire()
                continue
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
                    result = self.service.execute(*args, cancellation=token, **kwargs)
                elif operation == "cancel":
                    result = self.service.cancel(*args)
                elif operation == "reset":
                    result = self.service.reset()
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

        def reply(self, status, payload, retry_after=None):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The completed owner operation is not cancelled by disconnect.

        def client_gone(self):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def do_GET(self):
            if self.path == "/metrics":
                data = owner.service.metrics.render(*owner.service.gauges(owner.queue.qsize())).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path not in ("/ready", "/health"):
                self.reply(404, dict(error="not found"))
                return
            ready = owner.started and owner.service.state != "poisoned"
            self.reply(200 if ready else 503, dict(ready=ready, stage=stage))

        def cancel_if_gone(self, rid, state):
            # A closed client connection is a cancel request through the same fenced path.
            if not state["cancelled"] and self.client_gone():
                state["cancelled"] = True
                try:
                    owner.cancel_request(rid)
                except StageError:
                    pass

        def do_POST(self):
            if self.path not in ("/v1/completions", "/v1/chat/completions", "/v1/cancel", "/v1/admin/reset"):
                self.reply(404, dict(error="not found"))
                return
            try:
                if self.path == "/v1/admin/reset":
                    self.reply(200, owner.submit("reset"))
                    return
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
                rid = self.headers.get("x-request-id")
                chat = self.path == "/v1/chat/completions"
                if isinstance(body, dict) and body.get("stream") is True:
                    self.stream(rid, body, chat)
                    return
                future = owner.submit_async("execute", stage, rid, body, chat=chat)
                state = dict(cancelled=False)
                while True:
                    try:
                        response = future.result(timeout=0.05)
                        break
                    except FutureTimeout:
                        self.cancel_if_gone(rid, state)
                self.reply(200, response)
            except StageError as error:
                self.reply(error.status, dict(error=str(error), retryable=error.retryable),
                           retry_after=1 if error.retryable else None)
            except (ValueError, TypeError):
                failed = owner.service.state == "poisoned"
                self.reply(
                    500 if failed else 400,
                    dict(error="engine failed" if failed else "invalid request", retryable=False),
                )
            except Exception:
                self.reply(500, dict(error="engine failed; owner retained", retryable=False))

        def stream(self, rid, body, chat):
            """OpenAI-compatible SSE: a chunk per token as the owner produces it (the first right after the fenced
            prefill), text held back while it could still be the start of a stop string, then a chunk with
            finish_reason, a usage chunk and [DONE]. Errors before the first chunk are plain JSON replies."""
            tokens = queue.Queue()
            future = owner.submit_async("execute", stage, rid, body, chat=chat, on_token=tokens.put)
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            hold = max((len(x) for x in stops), default=1) - 1
            render, created = owner.service.render, int(time.time())
            kind = "chat.completion.chunk" if chat else "text_completion"
            state, out, sent, started = dict(cancelled=False), [], "", False

            def chunk(delta=None, finish=None, usage=None):
                if chat:
                    choice = dict(index=0, delta={} if delta is None else delta, finish_reason=finish)
                else:
                    choice = dict(index=0, text="" if delta is None else delta, finish_reason=finish)
                payload = dict(id=rid, object=kind, created=created, model=owner.service.model,
                               choices=[] if usage else [choice])
                if usage:
                    payload["usage"] = usage
                return payload

            def send(payload):
                nonlocal started
                if not started:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    started = True
                    if chat:
                        send(chunk(dict(role="assistant", content="")))
                data = payload if isinstance(payload, str) else json.dumps(payload)
                try:
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.cancel_if_gone(rid, state)

            def text_delta(final=None):
                nonlocal sent
                text = render(out) if final is None else final
                if final is None:
                    cut = min((text.find(x) for x in stops if x in text), default=-1)
                    text = text[:cut] if cut >= 0 else text[:max(len(sent), len(text) - hold)]
                if not text.startswith(sent) or len(text) == len(sent):
                    return                                  # nothing new, or a multi-byte character still open
                delta, sent = text[len(sent):], text
                send(chunk(dict(content=delta) if chat else delta))

            response, error = None, None
            while True:
                try:
                    out.append(tokens.get(timeout=0.05))
                    text_delta()
                    continue
                except queue.Empty:
                    pass
                if future.done() and tokens.empty():
                    try:
                        response = future.result()
                    except BaseException as exc:
                        error = exc
                    break
                self.cancel_if_gone(rid, state)
            if error is not None:
                if started:
                    send(dict(error=str(error), status=getattr(error, "status", 500)))
                    send("[DONE]")
                    return
                raise error
            choice = response["choices"][0]
            if "usage" in response:                         # decode stage: the final, stop-cut text
                text_delta(choice["message"]["content"] if chat else choice["text"])
            else:                                           # prefill stage: its token(s), cut at a stop string
                text = render(out)
                cut = min((text.find(x) for x in stops if x in text), default=-1)
                text_delta(text[:cut] if cut >= 0 else text)
            send(chunk(finish=choice["finish_reason"]))
            if "usage" in response:
                send(chunk(usage=response["usage"]))
            send("[DONE]")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
