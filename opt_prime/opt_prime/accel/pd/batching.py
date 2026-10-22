"""Continuous batching with chunked prefill for torch causal LMs (#44).

`SlotKV` keeps the KV of up to `slots` sequences in one store [layers, slots, kv_heads, max_ctx, head_dim] with a
length per slot. A decode step runs every active slot in one forward: the slots' prefixes are padded to the longest,
masked per slot, given their own positions, and each slot's new K/V column is written at its own length.

`ContinuousBatcher` schedules at iteration granularity. Every iteration it
  1. admits waiting requests into free slots,
  2. runs one prefill chunk (at most `prefill_chunk` prompt tokens of one request), and
  3. runs one batched decode step over every slot that is decoding,
so requests join and leave the batch every step and a long prompt delays the running batch by at most one chunk per
step (chunked prefill). A finished or cancelled request frees its slot at once.

Scope: uniform full-attention families (dense and MoE, `pd.torch_models`); per-layer or sliding geometries are
rejected. One owner thread drives `step()`; `submit` / `cancel` are thread-safe.
"""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
from typing import Callable, Optional

import torch
from transformers.cache_utils import DynamicCache

from opt_prime.accel.pd.contracts import PDRequest
from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.pd.sampling import select_token
from opt_prime.accel.pd.torch_models import describe_model


class SlotKV:
    def __init__(self, layers, slots, kv_heads, max_ctx, head_dim, dtype, device):
        shape = (layers, slots, kv_heads, max_ctx, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = [0] * slots
        self.max_ctx = max_ctx

    def prefix(self, slots, n):
        """DynamicCache over the first n positions of `slots` (a copy: the model's cache appends to it)"""
        idx = torch.tensor(slots, device=self.k.device)
        return DynamicCache(ddp_cache_data=[(self.k[layer].index_select(0, idx)[:, :, :n],
                                             self.v[layer].index_select(0, idx)[:, :, :n])
                                            for layer in range(self.k.shape[0])])

    def write(self, slots, positions, cache, src):
        """copy cache positions `src` (a slice) of every row into slot rows at `positions` (one start per slot)"""
        for layer, entry in enumerate(cache.layers):
            k, v = entry.keys[:, :, src], entry.values[:, :, src]
            n = k.shape[2]
            for row, (slot, pos) in enumerate(zip(slots, positions)):
                self.k[layer, slot, :, pos:pos + n] = k[row]
                self.v[layer, slot, :, pos:pos + n] = v[row]


@dataclass
class _Request:
    request: PDRequest
    future: Future
    on_token: Optional[Callable[[int], None]]
    slot: int = -1
    prefilled: int = 0                 # prompt tokens already in the slot
    tokens: list = field(default_factory=list)
    cancelled: bool = False


class ContinuousBatcher:
    supports_sampling = True          # SamplingParams (torch engine: select_token), stop tokens

    def __init__(self, model, *, slots=8, max_context=2048, prefill_chunk=256, kv="slot", pages=None, page_size=64):
        """kv="slot": one max_context store per slot (copies the slots' prefixes every step); kv="paged": a page pool
        of `pages` pages (default: slots * max_context worth) read in place by flex attention (#108)."""
        geometry = describe_model(model.config)
        if not geometry.uniform or geometry.window is not None:
            raise ValueError('continuous batching supports uniform full-attention families only')
        params = next(model.parameters())
        self.model, self.device, self.dtype = model.eval(), params.device, params.dtype
        if slots < 1 or prefill_chunk < 1 or max_context < 2:
            raise ValueError('positive slots, chunk and context required')
        if kv == "paged":
            from opt_prime.accel.pd.paged import PagedKV
            self.kv = PagedKV(geometry.layers, slots, geometry.kv_heads, geometry.head_dim, self.dtype, self.device,
                              pages=pages or slots * max_context // page_size, page_size=page_size,
                              max_context=max_context)
        elif kv == "slot":
            self.kv = SlotKV(geometry.layers, slots, geometry.kv_heads, max_context, geometry.head_dim,
                             self.dtype, self.device)
        else:
            raise ValueError("kv must be 'slot' or 'paged'")
        self.paged = kv == "paged"
        self.chunk, self.max_context = prefill_chunk, max_context
        self.free = list(range(slots))
        self.waiting, self.prefilling, self.decoding = deque(), deque(), []
        self.by_id = {}
        self._lock = threading.Lock()
        self.stats = dict(iterations=0, prefill_chunks=0, decode_steps=0, decode_rows=0)

    # ---- any thread ----
    def submit(self, request: PDRequest, on_token=None) -> Future:
        if len(request.input_ids) + request.max_new_tokens > self.max_context:
            raise ValueError('request exceeds the batcher context')
        vocab = self.model.config.get_text_config().vocab_size
        if any(t >= vocab for t in request.input_ids):
            raise ValueError('token outside model vocabulary')
        r = _Request(request, Future(), on_token)
        with self._lock:
            if request.request_id in self.by_id:
                raise ValueError('request ID already active')
            self.by_id[request.request_id] = r
            self.waiting.append(r)
        return r.future

    def cancel(self, request_id):
        """the request leaves at the next iteration boundary; its future raises RequestCancelled"""
        with self._lock:
            r = self.by_id.get(request_id)
            if r is not None:
                r.cancelled = True
            return r is not None

    @property
    def idle(self):
        with self._lock:
            return not (self.waiting or self.prefilling or self.decoding)

    # ---- owner thread ----
    @torch.inference_mode()
    def step(self):
        """one iteration: admit, one prefill chunk, one batched decode step"""
        self.stats['iterations'] += 1
        with self._lock:
            for queue in (self.waiting, self.prefilling):
                for r in [r for r in queue if r.cancelled]:
                    queue.remove(r)
                    self._finish(r, RequestCancelled(r.request.request_id))
            for r in [r for r in self.decoding if r.cancelled]:
                self.decoding.remove(r)
                self._finish(r, RequestCancelled(r.request.request_id))
            while self.free and self.waiting:
                r = self.waiting[0]
                slot = self.free[0]
                need = len(r.request.input_ids) + r.request.max_new_tokens
                if self.paged and not self.kv.alloc(slot, need):
                    break                                 # wait for pages (FIFO: no overtaking)
                self.waiting.popleft()
                r.slot = self.free.pop(0)
                self.kv.length[r.slot] = 0
                self.prefilling.append(r)
        if self.prefilling:
            self._prefill_chunk(self.prefilling[0])
        if self.decoding:
            self._decode_step()

    def run_until_idle(self, max_iterations=1_000_000):
        for _ in range(max_iterations):
            if self.idle:
                return
            self.step()
        raise RuntimeError('batcher did not drain')

    def _prefill_chunk(self, r):
        ids = r.request.input_ids
        start, end = r.prefilled, min(r.prefilled + self.chunk, len(ids))
        cache = self.kv.prefix([r.slot], start) if start else None
        out = self.model(torch.tensor([ids[start:end]], device=self.device), past_key_values=cache or DynamicCache(),
                         use_cache=True,
                         position_ids=torch.arange(start, end, device=self.device)[None],
                         attention_mask=torch.ones(1, end, dtype=torch.long, device=self.device), logits_to_keep=1)
        self.kv.write([r.slot], [start], out.past_key_values, slice(start, end))
        self.kv.length[r.slot] = r.prefilled = end
        self.stats['prefill_chunks'] += 1
        if end < len(ids):
            return
        self.prefilling.popleft()
        token = select_token(out.logits[0, -1], r.request.sampling, len(ids))
        if self._emit(r, token):
            self.decoding.append(r)

    def _decode_step(self):
        rows = list(self.decoding)
        slots = [r.slot for r in rows]
        lengths = [self.kv.length[s] for s in slots]
        if self.paged:
            logits = self.kv.decode(self.model, slots, lengths, [r.tokens[-1] for r in rows])
            return self._decoded(rows, lengths, logits)
        L = max(lengths)
        mask = torch.zeros(len(rows), L + 1, dtype=torch.long, device=self.device)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        mask[:, L] = 1
        out = self.model(torch.tensor([[r.tokens[-1]] for r in rows], device=self.device),
                         past_key_values=self.kv.prefix(slots, L), use_cache=True, attention_mask=mask,
                         position_ids=torch.tensor([[n] for n in lengths], device=self.device), logits_to_keep=1)
        self.kv.write(slots, lengths, out.past_key_values, slice(L, L + 1))
        self._decoded(rows, lengths, out.logits[:, -1])

    def _decoded(self, rows, lengths, logits):
        self.stats['decode_steps'] += 1
        self.stats['decode_rows'] += len(rows)
        done = []
        for i, r in enumerate(rows):
            self.kv.length[r.slot] = lengths[i] + 1
            token = select_token(logits[i], r.request.sampling, lengths[i] + 1)
            if not self._emit(r, token):
                done.append(r)
        for r in done:
            self.decoding.remove(r)

    def _emit(self, r, token):
        """record a generated token; False (and the request finished) when it was the last one"""
        r.tokens.append(token)
        if r.on_token is not None:
            r.on_token(token)
        if len(r.tokens) >= r.request.max_new_tokens or token in r.request.sampling.stop_token_ids:
            with self._lock:
                self._finish(r, None)
            return False
        return True

    def _finish(self, r, error):
        """lock held; frees the slot and completes the future"""
        if r.slot >= 0:
            if self.paged:
                self.kv.release(r.slot)
            self.free.append(r.slot)
            self.free.sort()
            r.slot = -1
        self.by_id.pop(r.request.request_id, None)
        if error is not None:
            r.future.set_exception(error)
        else:
            r.future.set_result(list(r.tokens))
