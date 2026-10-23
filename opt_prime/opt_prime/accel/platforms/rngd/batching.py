#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Continuous batching on FuriosaAI RNGD through furiosa-llm's native engine (#122).

furiosa-llm 2026.4 already schedules at iteration level on RNGD: paged KV with per-sequence lengths, batched decode,
chunked prefill (extend buckets), prefix caching and an optional host-DRAM KV tier (`KVOffloadConfig`). This adapter
runs its `AsyncLLMEngine` on a private event-loop thread and exposes the `pd.batching.ContinuousBatcher` interface
(`submit`, `cancel`, `step`, `idle`, `by_id`), so `serve.batched.BatchedService` serves RNGD exactly like the GPU
and ATOM batchers: same OpenAI API, SSE, cancellation, sampling, stop handling and vLLM metrics.

Scheduling (`mode`):
  - "rounds" (default): our iteration scheduler drives the engine in rounds. furiosa-llm 2026.4 runs decode first: a
    fresh prompt's prefill waits until every running decode has finished, so with arrivals spread over time the
    native engine serves requests one after another (measured: 16 concurrent requests, KONI-8B, 61 tok/s; the 16th
    request's first token after 15 s). Each round therefore
      1. prefills the waiting requests that fit (one call per request with max_tokens=1: their first tokens), then
      2. decodes every active request for a quantum of Q tokens together (a call with the prompt plus the tokens so
         far; the engine's prefix cache makes the continuation an extend of a few tokens), and the engine batches
         these decodes (B up to its buckets).
    Q is short (`quantum_busy`) while requests wait, so prefills are admitted soon, and long (`quantum_idle`)
    otherwise. Seeded sampling stays deterministic: each quantum's seed is derived from the request seed and position.
  - "native": every request is one engine call (the engine's own scheduling). Used when the engine's prefix cache is
    off: rounds rely on it (without it every quantum recomputes the context; measured 47 vs 247 tok/s).

furiosa-llm publishes no KV events. With `kv_events`, the adapter mirrors the prompt blocks it has sent (chained
64-bit hashes over `block_size` tokens, as `pd.paged.block_hashes`) in an LRU of `mirror_blocks` and publishes
BlockStored / BlockRemoved in vLLM's format, so the llm-d prefix scorer can route to the pod that served a prefix.
The mirror approximates the engine's own cache (same LRU order, capacity from the KV budget); prefix hits measured by
the engine (`num_cached_tokens`) are exported as `vllm:prefix_cache_hits_total`.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import Future
import threading

from opt_prime.accel.pd.paged import block_hashes
from opt_prime.accel.pd.runtime import RequestCancelled


class _Prefixes:
    """the mirror of the engine's prefix cache: LRU of prompt block hashes"""

    def __init__(self, block_size, capacity):
        self.block_size, self.capacity = block_size, capacity
        self.lru = OrderedDict()
        self.events = []

    def served(self, prompt):
        hashes = block_hashes(prompt, self.block_size)
        new = [i for i, h in enumerate(hashes) if h not in self.lru]
        for h in hashes:
            self.lru[h] = None
            self.lru.move_to_end(h)
        if new:
            first = new[0]
            self.events.append(("stored", [hashes[i] for i in new], hashes[first - 1] if first else None,
                                list(prompt[first * self.block_size:(new[-1] + 1) * self.block_size])))
        removed = []
        while len(self.lru) > self.capacity:
            removed.append(self.lru.popitem(last=False)[0])
        if removed:
            self.events.append(("removed", removed))


class FuriosaLLMBatcher:
    supports_sampling = True
    paged = False

    def __init__(self, engine, *, max_context, max_concurrency, kv_events=None, block_size=64, mirror_blocks=4096,
                 mode="rounds", quantum_busy=16, quantum_idle=64):
        """engine: a furiosa_llm AsyncLLMEngine (see `from_artifact`)"""
        if mode not in ("rounds", "native"):
            raise ValueError("mode must be 'rounds' or 'native'")
        self.engine, self.max_context, self.lanes = engine, max_context, max_concurrency
        self.mode, self.quantum_busy, self.quantum_idle = mode, quantum_busy, quantum_idle
        self._pending, self._active = [], []                 # rounds: (request, on_token, future, tokens)
        self.kv_events = kv_events
        self.prefixes = _Prefixes(block_size, mirror_blocks) if kv_events is not None else None
        self.by_id = {}
        self.waiting, self.prefilling, self.decoding = [], [], []        # gauge views (see load())
        self._lock = threading.Lock()
        self._progress = threading.Event()
        self._cancelled = set()
        self.stats = dict(iterations=0, requests=0, prompt_tokens=0, cached_tokens=0, decode_rows=0)
        self.hit_tokens = self.query_tokens = 0
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="furiosa-llm-loop")
        self._thread.start()

    @classmethod
    def from_artifact(cls, model, fxb, *, devices="npu:0", max_context=2048, max_concurrency=8, prefix_cache=True,
                      host_kv_gib=0.0, **kw):
        """build furiosa-llm's engine from a compiled artifact (.fxb) with our scheduler settings"""
        from furiosa_llm import LLM
        from furiosa_llm.llm_engine import AsyncLLMEngine
        from furiosa_llm.metadata.config_types import KVOffloadConfig, PrefixCacheConfig, SchedulerConfig
        offload = KVOffloadConfig(enabled=True, host_memory_bytes=int(host_kv_gib * 2 ** 30), write_back=True) \
            if host_kv_gib else KVOffloadConfig()
        sched = SchedulerConfig(max_concurrency=max_concurrency, prefix_cache_config=PrefixCacheConfig(enabled=prefix_cache),
                                kv_offload_config=offload)
        llm = LLM(model, fxb=fxb, devices=devices, max_model_len=max_context, scheduler_config=sched)
        if not prefix_cache:                     # rounds continue each request through the engine's prefix cache
            kw["mode"] = "native"
        return cls(AsyncLLMEngine.from_llm(llm), max_context=max_context, max_concurrency=max_concurrency, **kw)

    # ---- any thread ----
    def submit(self, request, on_token=None) -> Future:
        if len(request.input_ids) + request.max_new_tokens > self.max_context:
            raise ValueError('request exceeds the batcher context')
        f = Future()
        with self._lock:
            if request.request_id in self.by_id:
                raise ValueError('request ID already active')
            self.by_id[request.request_id] = f
            self.waiting.append(request.request_id)
        if self.mode == "native":
            asyncio.run_coroutine_threadsafe(self._run(request, on_token, f), self._loop)
        else:
            with self._lock:
                self._pending.append([request, on_token, f, []])
        return f

    def cancel(self, request_id):
        with self._lock:
            if request_id not in self.by_id:
                return False
        self._cancelled.add(request_id)
        asyncio.run_coroutine_threadsafe(self._abort(request_id), self._loop)
        return True

    async def _abort(self, request_id):
        for key in [k for k in getattr(self, "_calls", {}) if k.split("#")[0] == request_id] or [request_id]:
            await self.engine.abort(key)

    @property
    def idle(self):
        with self._lock:
            return not self.by_id

    def step(self):
        """rounds: one round (admit + one decode quantum); native: wait for progress (bounded, so it can stop)"""
        self.stats['iterations'] += 1
        if self.mode == "rounds":
            asyncio.run_coroutine_threadsafe(self._round(), self._loop).result()
        else:
            self._progress.wait(0.05)
            self._progress.clear()
        if self.kv_events is not None and self.prefixes.events:
            events, self.prefixes.events = self.prefixes.events, []
            self.kv_events.publish(events)

    def run_until_idle(self, max_iterations=10_000_000):
        for _ in range(max_iterations):
            if self.idle:
                return
            self.step()
        raise RuntimeError('batcher did not drain')

    def load(self, waiting=0):
        """(running, waiting, KV usage) for /metrics: lanes = the engine's max concurrency"""
        with self._lock:
            running, queued = len(self.prefilling) + len(self.decoding), len(self.waiting)
        return running, waiting + queued, round(min(1.0, running / self.lanes), 4)

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)

    # ---- rounds (the engine's loop) ----
    async def _call(self, request, ids, n, key, on_new):
        """one engine call: ids -> up to n tokens; on_new(token) per token; returns (tokens, cached prompt tokens)"""
        self._calls = getattr(self, "_calls", {})
        self._calls[key] = True
        got, cached = [], 0
        try:
            async for out in self.engine.generate({"prompt_token_ids": list(ids)}, self._params(request, n, len(ids)),
                                                  key):
                cached = int(out.num_cached_tokens or 0)
                tokens = list(out.outputs[0].token_ids)
                for t in tokens[len(got):]:
                    got.append(t)
                    on_new(t)
        finally:
            self._calls.pop(key, None)
        return got, cached

    def _done(self, entry):
        request, _, _, tokens = entry
        stops = request.sampling.stop_token_ids
        return len(tokens) >= request.max_new_tokens or (tokens and tokens[-1] in stops)

    def _close(self, entry, error=None):
        request, _, f, tokens = entry
        rid = request.request_id
        if rid in self._cancelled:
            error = RequestCancelled(rid)
        self._cancelled.discard(rid)
        with self._lock:
            self.by_id.pop(rid, None)
            for q in (self.waiting, self.prefilling, self.decoding):
                if rid in q:
                    q.remove(rid)
        self.stats['requests'] += 1
        self.stats['prompt_tokens'] += len(request.input_ids)
        if error is not None:
            f.set_exception(error)
        else:
            f.set_result(list(tokens[:request.max_new_tokens]))

    async def _round(self):
        with self._lock:
            dropped = [e for e in self._pending if e[0].request_id in self._cancelled]
            for e in dropped:
                self._pending.remove(e)
            room = self.lanes - len(self._active)
            new, self._pending = self._pending[:room], self._pending[room:]
        for e in dropped:
            self._close(e)
        for e in [e for e in self._active if e[0].request_id in self._cancelled]:
            self._active.remove(e)
            self._close(e)
        # 1. prefill the admitted requests (first token each)
        for e in new:
            self._move(e[0].request_id, self.waiting, self.prefilling)

        async def prefill(e):
            request, on_token = e[0], e[1]
            emit = (lambda t: (e[3].append(t), on_token and on_token(t)))
            try:
                _, cached = await self._call(request, request.input_ids, 1, f"{request.request_id}#p", emit)
            except BaseException as error:                                  # noqa: BLE001
                return e, error
            self.hit_tokens += cached
            self.query_tokens += len(request.input_ids)
            self.stats['cached_tokens'] += cached
            if self.prefixes is not None:
                self.prefixes.served(request.input_ids)
            return e, None

        for e, error in await asyncio.gather(*[prefill(e) for e in new]):
            if error is not None or e[0].request_id in self._cancelled or self._done(e):
                self._close(e, error)
            else:
                self._move(e[0].request_id, self.prefilling, self.decoding)
                self._active.append(e)
        if not self._active:
            if not new:
                await asyncio.sleep(0.005)
            return
        # 2. one decode quantum for every active request, batched by the engine
        q = self.quantum_busy if self._pending else self.quantum_idle

        async def decode(e):
            request, on_token, _, tokens = e
            n = min(q, request.max_new_tokens - len(tokens))
            emit = (lambda t: (tokens.append(t), on_token and on_token(t)))
            try:
                await self._call(request, tuple(request.input_ids) + tuple(tokens), n,
                                 f"{request.request_id}#{len(tokens)}", emit)
                return e, None
            except BaseException as error:                                  # noqa: BLE001
                return e, error

        rows = 0
        for e, error in await asyncio.gather(*[decode(e) for e in self._active]):
            rows += 1
            if error is not None or e[0].request_id in self._cancelled or self._done(e):
                self._active.remove(e)
                self._close(e, error)
        self.stats['decode_rows'] += rows
        if self.kv_events is not None and self.prefixes.events:
            events, self.prefixes.events = self.prefixes.events, []
            self.kv_events.publish(events)

    # ---- the engine's loop ----
    def _params(self, request, max_tokens=None, position=0):
        import inspect
        from furiosa_llm import SamplingParams
        s = request.sampling
        seed = None if s.seed is None else (s.seed * 1000003 + position) % (2 ** 31)  # per call, deterministic
        kw = dict(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k if s.top_k > 0 else -1, seed=seed,
                  max_tokens=max_tokens or request.max_new_tokens, ignore_eos=True,
                  stop_token_ids=list(s.stop_token_ids) or None)
        try:
            params = inspect.signature(SamplingParams).parameters
            open_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            accepted = kw if open_kw else params
        except (TypeError, ValueError):
            accepted = kw
        if s.seed is not None and "seed" not in accepted:
            raise ValueError("this furiosa-llm has no seeded sampling")
        return SamplingParams(**{k: v for k, v in kw.items() if k in accepted})

    def _move(self, rid, src, dst):
        with self._lock:
            if rid in src:
                src.remove(rid)
                if dst is not None:
                    dst.append(rid)

    async def _run(self, request, on_token, f):
        rid, tokens = request.request_id, []
        self._move(rid, self.waiting, self.prefilling)
        try:
            async for out in self.engine.generate({"prompt_token_ids": list(request.input_ids)},
                                                  self._params(request), rid):
                ids = list(out.outputs[0].token_ids)
                if not tokens and ids:                          # first token: prefill done
                    self._move(rid, self.prefilling, self.decoding)
                    cached = int(out.num_cached_tokens or 0)
                    self.hit_tokens += cached
                    self.query_tokens += len(request.input_ids)
                    self.stats['cached_tokens'] += cached
                    if self.prefixes is not None:
                        self.prefixes.served(request.input_ids)
                new, tokens = ids[len(tokens):], ids
                self.stats['decode_rows'] += len(new)
                if on_token is not None:
                    for t in new:
                        on_token(t)
                self._progress.set()
            if rid in self._cancelled:
                raise RequestCancelled(rid)
            f.set_result(tokens[:request.max_new_tokens])
        except BaseException as error:                          # noqa: BLE001 - delivered to the waiting client
            f.set_exception(RequestCancelled(rid) if rid in self._cancelled else error)
        finally:
            self._cancelled.discard(rid)
            with self._lock:
                self.by_id.pop(rid, None)
                for q in (self.waiting, self.prefilling, self.decoding):
                    if rid in q:
                        q.remove(rid)
            self.stats['requests'] += 1
            self.stats['prompt_tokens'] += len(request.input_ids)
            self._progress.set()
