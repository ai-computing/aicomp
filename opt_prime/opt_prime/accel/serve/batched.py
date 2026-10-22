"""OpenAI-compatible serving with continuous batching and chunked prefill (#44, stage 2).

`BatchedService` parses requests like the PD stage service (`CompletionAPI`: token-ID / text prompts, chat messages,
sampling, stop, finish_reason) and runs them through a `pd.batching.ContinuousBatcher`. `BatchedOwner` drives the
batcher from one owner thread and offers the owner interface of `llmd.OwnerExecutor`, so `llmd.serve_stage` serves
it unchanged: /v1/completions, /v1/chat/completions, SSE streaming, cancellation on client disconnect, /metrics.

Unlike the single-lane PD stage, any number of requests may be in flight; the batcher admits them into free slots
(or pages) and the rest wait. 429 only when the submission queue is full.
"""
from concurrent.futures import Future
import threading
import time

from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.serve.llmd import CompletionAPI, StageError


class BatchedService(CompletionAPI):
    def __init__(self, batcher, **kw):
        kw.setdefault("context", batcher.max_context)
        super().__init__(**kw)
        self.batcher = batcher
        self.state = "idle"                       # "poisoned" after an engine failure (readiness 503)
        if batcher.paged:
            self.metrics.extra = self._cache_metrics
        elif hasattr(batcher, "hit_tokens"):      # NPU engines (ATOM own blocks, furiosa-llm): hit counters
            def extra(lab):
                out = [("vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prefix cache.",
                        [f"vllm:prefix_cache_queries_total{{{lab}}} {batcher.query_tokens}"]),
                       ("vllm:prefix_cache_hits_total", "counter", "Prompt tokens served from the prefix cache.",
                        [f"vllm:prefix_cache_hits_total{{{lab}}} {batcher.hit_tokens}"])]
                size = getattr(batcher, "block_size", None)
                if size:
                    out.append(("vllm:cache_config_info", "gauge", "Information of the LLMEngine CacheConfig.",
                                [f'vllm:cache_config_info{{{lab},block_size="{size}",enable_prefix_caching="True"}} 1']))
                return out
            self.metrics.extra = extra

    def _engines(self):
        return (self.batcher,)

    def gauges(self, waiting=0):
        b = self.batcher
        if hasattr(b, "load"):                    # engines that schedule themselves (furiosa-llm)
            return b.load(waiting)
        running = len(b.prefilling) + len(b.decoding)
        if b.paged:                               # unreferenced cached pages count as free
            used = 1.0 - (len(b.kv.free_pages) + len(b.kv.evictable)) / b.kv.pages
        else:
            used = 1.0 - len(b.free) / len(b.kv.length)
        return running, waiting + len(b.waiting), round(used, 4)

    def _cache_metrics(self, lab):
        """vLLM's cache config (the EPP's prefix scorer reads block_size from it) and prefix-cache counters"""
        kv = self.batcher.kv
        return [
            ("vllm:cache_config_info", "gauge", "Information of the LLMEngine CacheConfig.",
             [f'vllm:cache_config_info{{{lab},block_size="{kv.page_size}",num_gpu_blocks="{kv.pages}",'
              f'enable_prefix_caching="{kv.prefix_cache}"}} 1']),
            ("vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prefix cache.",
             [f"vllm:prefix_cache_queries_total{{{lab}}} {kv.query_tokens}"]),
            ("vllm:prefix_cache_hits_total", "counter", "Prompt tokens served from the prefix cache.",
             [f"vllm:prefix_cache_hits_total{{{lab}}} {kv.hit_tokens}"]),
        ] + (self._host_metrics(lab, kv.host) if kv.host is not None else [])

    @staticmethod
    def _host_metrics(lab, host):
        st = host.stats
        return [
            ("airm_kv_host_blocks", "gauge", "Blocks held by the host-DRAM KV tier.",
             [f"airm_kv_host_blocks{{{lab}}} {len(host.index)}"]),
            ("airm_kv_host_capacity_blocks", "gauge", "Capacity of the host-DRAM KV tier in blocks.",
             [f"airm_kv_host_capacity_blocks{{{lab}}} {host.blocks}"]),
            ("airm_kv_host_hit_tokens_total", "counter", "Prompt tokens reloaded from the host tier.",
             [f"airm_kv_host_hit_tokens_total{{{lab}}} {st['hit_tokens']}"]),
            ("airm_kv_offload_blocks_total", "counter", "Blocks written back from the device to the host tier.",
             [f"airm_kv_offload_blocks_total{{{lab}}} {st['offloaded']}"]),
            ("airm_kv_reload_bytes_total", "counter", "Bytes reloaded from the host tier to the device.",
             [f"airm_kv_reload_bytes_total{{{lab}}} {st['reloaded'] * host.block_bytes}"]),
        ]


class BatchedOwner:
    """owner thread: steps the batcher while work is pending; HTTP threads only submit and cancel"""

    def __init__(self, service, capacity=1024):
        self.service, self.capacity = service, capacity
        self.started, self.accepting = False, True
        self._wake = threading.Condition()
        self._stop = False
        self.queue = _Depth(self)                # /metrics reads queue.qsize()

    # ---- the owner interface of llmd.OwnerExecutor ----
    def submit(self, operation, *args, **kwargs):
        return self.submit_async(operation, *args, **kwargs).result()

    def submit_async(self, operation, *args, chat=False, on_token=None):
        if operation == "stop":
            f = Future()
            with self._wake:
                self.accepting, self._stop = False, True
                self._wake.notify()
            f.set_result(None)
            return f
        if operation == "reset":
            raise StageError(501, "batched service has no reset; restart it")
        if operation != "execute":
            raise StageError(400, "unknown owner operation")
        if not self.started or not self.accepting:
            raise StageError(503, "owner not accepting work")
        svc, b = self.service, self.service.batcher
        if svc.state == "poisoned":
            raise StageError(503, "owner quarantined; restart required", retryable=True)
        _stage, rid, body = args
        request, stops = svc._request(rid, body, chat)
        if len(b.by_id) >= self.capacity:
            raise StageError(429, "owner queue full")
        started = time.monotonic()
        first = []
        def token(t):
            if not first:
                first.append(time.monotonic())
            if on_token is not None:
                on_token(t)
        try:
            inner = b.submit(request, on_token=token)
        except ValueError as error:
            raise StageError(400, str(error))
        outer = Future()

        def done(f):
            try:
                tokens = f.result()
            except RequestCancelled:
                outer.set_exception(StageError(499, "request cancelled"))
                return
            except BaseException as error:
                svc.metrics.failed()
                outer.set_exception(error)
                return
            tokens, text, finish = svc._finish(request, stops, tokens)
            end = time.monotonic()
            if first:
                svc.metrics.observe("vllm:time_to_first_token_seconds", first[0] - started)
                if len(tokens) > 1:
                    svc.metrics.observe("vllm:time_per_output_token_seconds", (end - first[0]) / (len(tokens) - 1))
            svc.metrics.observe("vllm:e2e_request_latency_seconds", end - started)
            svc.metrics.finished(finish, len(request.input_ids), len(tokens))
            outer.set_result(svc._response(rid, request, tokens, text, finish, chat))
        inner.add_done_callback(done)
        with self._wake:
            self._wake.notify()
        return outer

    def cancel_request(self, rid):
        if not isinstance(rid, str) or not rid.strip() or len(rid) > 128:
            raise StageError(400, "valid x-request-id required")
        if not self.service.batcher.cancel(rid):
            raise StageError(409, "no such request in flight")
        with self._wake:
            self._wake.notify()
        return dict(id=rid, cancellation_requested=True, drained=False)

    # ---- the owner thread ----
    def run(self):
        self.started = True
        b = self.service.batcher
        while True:
            with self._wake:
                while b.idle and not self._stop:
                    self._wake.wait(timeout=1.0)
                if self._stop and b.idle:
                    self.started = False
                    return
            try:
                b.step()
            except BaseException:
                self.service.state = "poisoned"           # engine failure: stop serving, keep state for diagnosis
                self.accepting = False
                for r in list(b.by_id.values()):
                    if not r.future.done():
                        r.future.set_exception(StageError(500, "engine failed"))
                self.started = False
                raise


class _Depth:
    def __init__(self, owner):
        self.owner = owner

    def qsize(self):
        return 0                                  # waiting requests are counted by BatchedService.gauges
