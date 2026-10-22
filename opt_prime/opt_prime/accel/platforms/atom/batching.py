#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Continuous batching on Rebellions ATOM (#44 stage 3): the optimum-rbln decoder compiled with batch_size B.

optimum-rbln compiles one prefill graph (a single sequence, written into the KV pages of lane `batch_idx`) and one
decode graph over all B lanes. `AtomBatcher` schedules requests onto the lanes with the scheduling, cancellation,
sampling and stop handling of `pd.batching.ContinuousBatcher`, so `serve.batched.BatchedService` serves it unchanged.
Every iteration it
  1. admits at most one waiting request into a free lane and prefills its whole prompt (the SDK prefill call chunks
     internally and blocks: one prefill per iteration bounds how long the running batch waits), then
  2. runs one decode step of the B-lane graph for every lane that is decoding.
A finished request frees its lane at once (refill). The KV stays in the SDK's own pages; nothing is imported.

Lanes without a request still run in the decode graph. At start every lane is prefilled with one dummy token, so it
owns its own page, and idle lanes decode the dummy token at position 0 of their own page; they never alias live KV
(validated on hardware in hetero-npu-p2p: B8 refills token-exact against the vendor reference).
"""
from collections import deque
import threading

import torch

from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.sampling import select_token


class _Lanes:
    """the per-lane decode position (what `ContinuousBatcher` calls kv.length)"""

    def __init__(self, lanes):
        self.length = [0] * lanes


class AtomBatcher(ContinuousBatcher):
    supports_sampling = True          # the decode graph returns logits; sampling runs on the host

    def __init__(self, model, *, pad_token_id=None):
        rc = model.rbln_config
        self.lanes = rc.batch_size
        decoders = getattr(model, "decoders", None) or {}
        if self.lanes not in decoders:
            raise ValueError(f"no compiled decode graph for batch {self.lanes} (have {sorted(decoders)})")
        self.model, self.decoder = model, decoders[self.lanes]
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.vocab = cfg.vocab_size
        self.pad = pad_token_id if pad_token_id is not None else (cfg.bos_token_id or 0)
        if isinstance(self.pad, (list, tuple)):
            self.pad = self.pad[0]
        self.max_context = rc.max_seq_len
        self.paged = False
        self.kv = _Lanes(self.lanes)
        self.free = list(range(self.lanes))
        self.waiting, self.prefilling, self.decoding = deque(), deque(), []
        self.by_id = {}
        self._lock = threading.Lock()
        self.stats = dict(iterations=0, prefill_chunks=0, decode_steps=0, decode_rows=0)
        self.poisoned = False
        with torch.inference_mode():
            for lane in range(self.lanes):             # every lane owns a page before the B-lane graph runs
                self._prefill(lane, [self.pad])

    def kv_usage(self):
        return 1.0 - len(self.free) / self.lanes

    def _prefill(self, lane, prompt):
        return self.model.prefill_decoder(input_ids=torch.tensor([list(prompt)], dtype=torch.int64),
                                          cache_position=torch.arange(len(prompt), dtype=torch.int32)[None],
                                          batch_idx=lane)

    @torch.inference_mode()
    def step(self):
        """one iteration: admit and prefill at most one request, one B-lane decode step"""
        if self.poisoned:
            raise RuntimeError("ATOM batcher quarantined after an SDK failure")
        self.stats['iterations'] += 1
        self._drop_cancelled()
        with self._lock:
            r = None
            if self.free and self.waiting:
                r = self.waiting.popleft()
                r.slot = self.free.pop(0)
                self.prefilling.append(r)
        try:
            if r is not None:
                self._prefill_request(r)
            if self.decoding:
                self._decode_step()
        except BaseException:
            self.poisoned = True
            raise

    def _prefill_request(self, r):
        ids = r.request.input_ids
        out = self._prefill(r.slot, ids)
        self.stats['prefill_chunks'] += 1
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite ATOM logits")
        padded = int(getattr(out, "padded_cache_lengths", 0) or 0)
        self.kv.length[r.slot] = r.prefilled = len(ids) + padded
        self.prefilling.popleft()
        if self._emit(r, select_token(logits, r.request.sampling, len(ids))):
            self.decoding.append(r)

    def _decode_step(self):
        rows = list(self.decoding)
        tokens, positions = [self.pad] * self.lanes, [0] * self.lanes     # idle lanes: dummy token at position 0
        for r in rows:
            tokens[r.slot], positions[r.slot] = r.tokens[-1], self.kv.length[r.slot]
        out = self.decoder(input_ids=torch.tensor(tokens, dtype=torch.int64)[:, None].contiguous(),
                           cache_position=torch.tensor(positions, dtype=torch.int32)[:, None].contiguous())
        logits = out.logits[[r.slot for r in rows], -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite ATOM logits")
        self._decoded(rows, [self.kv.length[r.slot] for r in rows], logits)
