"""Batched PD disaggregation (#120): prefill on its own device or process, decode lanes batched.

`PDBatcher` is the decode side. It schedules like `ContinuousBatcher` (paged KV, flex-attention decode over all
lanes), but a request's prompt is not prefilled in the decode loop. On admission the request gets its decode lane
and pages, and its prefill goes to a `PrefillWorker` that runs concurrently on another device (thread) or process.
When the worker's result arrives, the KV span lands in the lane's pages in one indexed copy, the first token is
emitted, and the request joins the next decode step. Decode steps never wait for a prefill.

Prefix reuse works on both sides:
  - prefill side: `TorchPrefillEngine` keeps its own paged prefix cache, so a cached prompt prefix is not recomputed;
  - decode side: blocks already cached in the decode pool (or its host-DRAM tier) are not transferred at all: the
    worker is asked only for positions from `send_from` on.
KV events and the host tier are the decode pool's (`PagedKV`), as in aggregated serving.

Workers:
  - `PrefillWorker(engine)`: a thread running `engine.prefill` (another GPU in the same process, or CPU).
  - `ProcessPrefillWorker(factory)`: the engine in a child process. CUDA KV comes back by IPC handle; a receiver
    thread in this process copies it to the decode device (peer copy) and only then lets the child free it.
  - `SubprocessPrefillWorker(python, engine_args)`: `pd.prefill_server` under another interpreter / environment (an
    NPU SDK's venv), KV handed over in a shared-memory slab with one slot per decode lane.
Any engine with `prefill(request, send_from) -> PrefillOut` can be a worker's engine;
`platforms.atom.batching.AtomBatcher(prefill_worker=...)` decodes PD requests on ATOM (#123).
"""
from collections import deque
from concurrent.futures import Future, wait, FIRST_COMPLETED
from dataclasses import dataclass
import queue
import threading
import time

from contextlib import nullcontext as _nullctx
import os

import torch

from opt_prime.accel.pd.batching import ContinuousBatcher, prefill_chunk
from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.pd.sampling import select_token
from opt_prime.accel.pd.torch_models import describe_model


PEER_COPY = os.environ.get("OPT_PRIME_PEER_COPY", "0") == "1"


def move_kv(t, device):
    """copy a KV span to `device`. Between GPUs the copy is staged through pinned host memory unless
    OPT_PRIME_PEER_COPY=1: on a PCIe host whose GPUs sit on different sockets, direct peer copies were found to deliver
    wrong data and to overwrite unrelated memory on the destination GPU (see docs/validation/batching/PD.md). Enable
    peer copies only where they are verified (NVLink, same PCIe switch)."""
    device = torch.device(device)
    if t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda" and not PEER_COPY:
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        host.copy_(t)                                          # device -> host completes before returning
        return host.to(device, non_blocking=True)
    return t.to(device, non_blocking=True)


@dataclass
class PrefillOut:
    token: int                 # the first output token (sampled on the prefill side)
    start: int                 # the span's first position (= send_from)
    keys: torch.Tensor         # [layers, kv_heads, n, head_dim]
    values: torch.Tensor
    prefill_cached: int = 0    # prompt tokens the prefill side did not recompute (its prefix cache)
    prefill_s: float = 0.0
    transfer_s: float = 0.0


class TorchPrefillEngine:
    """prefill only, on its own device: chunked over a paged KV with a prefix cache; returns the KV span to send"""

    def __init__(self, model, *, max_context=2048, chunk=2048, page_size=64, pages=None, prefix_cache=True,
                 out_device=None):
        from opt_prime.accel.pd.paged import PagedKV
        g = describe_model(model.config)
        if not g.uniform or g.window is not None:
            raise ValueError("PD batching supports uniform full-attention families only")
        p = next(model.parameters())
        self.model, self.device, self.dtype = model.eval(), p.device, p.dtype
        self.kv = PagedKV(g.layers, 1, g.kv_heads, g.head_dim, self.dtype, self.device,
                          pages=pages or 4 * max_context // page_size, page_size=page_size, max_context=max_context,
                          prefix_cache=prefix_cache)
        self.chunk, self.max_context = chunk, max_context
        self.out_device = torch.device(out_device) if out_device is not None else None
        self._stream = torch.cuda.Stream(self.out_device) if self.out_device is not None and \
            self.out_device.type == "cuda" and self.out_device != self.device else None

    @torch.inference_mode()
    def prefill(self, request, send_from=0):
        ids = request.input_ids
        t0 = time.perf_counter()
        cached = self.kv.alloc(0, len(ids), prompt=ids)
        if cached is None:
            raise RuntimeError("prefill KV pool too small for the prompt")
        try:
            start, logits = cached, None
            while start < len(ids):
                end = min(start + self.chunk, len(ids))
                logits = prefill_chunk(self.model, self.kv, 0, ids, start, end, self.device)
                start = end
            self.kv.commit_prompt(0)
            token = select_token(logits, request.sampling, len(ids))
            keys, values = self.kv.read_span(0, send_from, len(ids))       # gathered copies, own memory
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)                          # prefill time = compute, not launch
            t1 = time.perf_counter()
            if self.out_device is not None and self.out_device != self.device:
                with torch.cuda.stream(self._stream) if self._stream is not None else _nullctx():
                    keys, values = move_kv(keys, self.out_device), move_kv(values, self.out_device)
                if self._stream is not None:
                    self._stream.synchronize()
            t2 = time.perf_counter()
            return PrefillOut(token, send_from, keys, values, cached, t1 - t0, t2 - t1)
        finally:
            self.kv.release(0)


class PrefillWorker:
    """runs `engine.prefill` on one thread, in submission order"""

    def __init__(self, engine):
        self.engine = engine
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="prefill-worker")
        self._thread.start()

    def submit(self, request, send_from=0, slot=None) -> Future:
        f = Future()
        self._q.put((request, send_from, f))
        return f

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            request, send_from, f = item
            if not f.set_running_or_notify_cancel():
                continue
            try:
                f.set_result(self.engine.prefill(request, send_from))
            except BaseException as error:
                f.set_exception(error)

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=10)


def _child(factory, requests, replies, acks):
    """the prefill process: build the engine, serve prefills; keep each reply's tensors until the parent acks"""
    engine = factory()
    held = {}
    while True:
        while not acks.empty():
            held.pop(acks.get(), None)
        item = requests.get()
        if item is None:
            return
        key, request, send_from = item
        try:
            out = engine.prefill(request, send_from)
            held[key] = (out.keys, out.values)
            replies.put((key, out, None))
        except BaseException as error:                                  # noqa: BLE001 - report to the parent
            replies.put((key, None, repr(error)))


class ProcessPrefillWorker:
    """`factory()` builds a prefill engine in a child process (spawned: CUDA-safe). Results come back over a
    torch.multiprocessing queue (CUDA tensors by IPC handle); a receiver thread copies them to `out_device` and acks."""

    def __init__(self, factory, out_device):
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._req, self._rep, self._ack = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self._proc = ctx.Process(target=_child, args=(factory, self._req, self._rep, self._ack), daemon=True)
        self._proc.start()
        self.out_device = torch.device(out_device)
        self._futures, self._next = {}, 0
        self._lock = threading.Lock()
        self._recv = threading.Thread(target=self._receive, daemon=True, name="prefill-receiver")
        self._recv.start()

    def submit(self, request, send_from=0, slot=None) -> Future:
        f = Future()
        with self._lock:
            key, self._next = self._next, self._next + 1
            self._futures[key] = f
        self._req.put((key, request, send_from))
        return f

    def _receive(self):
        stream = torch.cuda.Stream(self.out_device) if self.out_device.type == "cuda" else None
        while True:
            item = self._rep.get()
            if item is None:
                return
            key, out, error = item
            with self._lock:
                f = self._futures.pop(key)
            if error is not None:
                f.set_exception(RuntimeError(f"prefill process: {error}"))
                continue
            t0 = time.perf_counter()
            if stream is not None:
                with torch.cuda.stream(stream):
                    k, v = move_kv(out.keys, self.out_device), move_kv(out.values, self.out_device)
                stream.synchronize()
            else:
                k, v = out.keys.to(self.out_device).clone(), out.values.to(self.out_device).clone()
            self._ack.put(key)                                           # the child may free its tensors now
            f.set_result(PrefillOut(out.token, out.start, k, v, out.prefill_cached, out.prefill_s,
                                    out.transfer_s + time.perf_counter() - t0))

    def close(self):
        self._req.put(None)
        self._rep.put(None)
        self._proc.join(timeout=30)


class SubprocessPrefillWorker:
    """`opt_prime.accel.pd.prefill_server` run by `python` (another environment); one KV slot per decode lane in a
    shared-memory slab. A result's keys/values are bfloat16 views of the lane's slot: valid until that lane is
    submitted again, which the decode side does only after importing them."""

    def __init__(self, python, engine_args, *, slots, layers, kv_heads, head_dim, max_context, env=None,
                 startup_timeout=1800):
        import json, subprocess
        import numpy as np
        from multiprocessing import shared_memory
        shape = (slots, 2, layers, kv_heads, max_context, head_dim)
        self._shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 2)
        self.slab = np.ndarray(shape, dtype=np.uint16, buffer=self._shm.buf)
        cmd = [python, "-m", "opt_prime.accel.pd.prefill_server", "--shm", self._shm.name, "--slots", str(slots),
               "--layers", str(layers), "--kv-heads", str(kv_heads), "--head-dim", str(head_dim),
               "--max-context", str(max_context), *engine_args]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                      env=env)
        self._json = json
        self._futures, self._next, self._lock = {}, 0, threading.Lock()
        ready = Future()
        self._recv = threading.Thread(target=self._receive, args=(ready,), daemon=True, name="prefill-subprocess")
        self._recv.start()
        ready.result(timeout=startup_timeout)

    def submit(self, request, send_from=0, slot=None) -> Future:
        if slot is None:
            raise ValueError("SubprocessPrefillWorker needs the decode lane (slot)")
        f = Future()
        with self._lock:
            key, self._next = self._next, self._next + 1
            self._futures[key] = (f, slot, send_from)
        s = request.sampling
        line = self._json.dumps(dict(id=key, tokens=list(request.input_ids), send_from=send_from, slot=slot,
                                     sampling=dict(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k, seed=s.seed,
                                                   stop_token_ids=list(s.stop_token_ids))))
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        return f

    def _receive(self, ready):
        for line in self._proc.stdout:
            try:
                msg = self._json.loads(line)
            except ValueError:
                continue                                       # SDK log lines on stdout
            if msg.get("ready"):
                ready.set_result(True)
                continue
            with self._lock:
                f, slot, start = self._futures.pop(msg["id"])
            if "error" in msg:
                f.set_exception(RuntimeError(f"prefill subprocess: {msg['error']}"))
                continue
            n = msg["n"]
            kv = torch.from_numpy(self.slab[slot, :, :, :, :n].view("int16")).view(torch.bfloat16)
            f.set_result(PrefillOut(msg["token"], start, kv[0], kv[1], msg.get("cached", 0), msg.get("prefill_s", 0.0)))
        if not ready.done():
            ready.set_exception(RuntimeError(f"prefill subprocess exited ({self._proc.wait()})"))
        with self._lock:
            for f, _, _ in self._futures.values():
                f.set_exception(RuntimeError("prefill subprocess exited"))
            self._futures.clear()

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=60)
        finally:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class PDBatcher(ContinuousBatcher):
    """decode side of batched PD: paged decode lanes; prompts are prefilled by `worker` concurrently"""

    def __init__(self, model, worker, **kw):
        kw.setdefault("kv", "paged")
        if kw["kv"] != "paged":
            raise ValueError("PD batching needs kv='paged'")
        super().__init__(model, **kw)
        self.worker = worker
        self.stats.update(prefills=0, transferred_tokens=0, decode_cached_tokens=0, prefill_cached_tokens=0,
                          prefill_s=0.0, transfer_s=0.0)

    def step(self):
        self.stats['iterations'] += 1
        self._drop_cancelled()
        with self._lock:
            while self.free and self.waiting:
                r = self.waiting[0]
                slot = self.free[0]
                ids = r.request.input_ids
                cached = self.kv.alloc(slot, len(ids) + r.request.max_new_tokens, prompt=ids)
                if cached is None:
                    break                                     # wait for pages (FIFO)
                self.waiting.popleft()
                r.slot = self.free.pop(0)
                self.kv.length[r.slot] = r.prefilled = cached
                r.pending = self.worker.submit(r.request, send_from=cached, slot=r.slot)   # cached blocks not sent
                self.prefilling.append(r)
        self._admit_prefilled(block=not self.decoding)
        if self.decoding:
            self._decode()
        if self.kv_events is not None:
            events = self.kv.drain_events()
            if events:
                self.kv_events.publish(events)

    def _admit_prefilled(self, block):
        pending = [r for r in self.prefilling if r.pending is not None]
        if not pending:
            return
        if block:                                             # nothing to decode: wait for the first prefill
            wait([r.pending for r in pending], timeout=0.05, return_when=FIRST_COMPLETED)
        for r in pending:
            if not r.pending.done():
                continue
            with self._lock:
                if r not in self.prefilling:                  # cancelled meanwhile: lane already released
                    continue
                self.prefilling.remove(r)
            try:
                out = r.pending.result()
            except BaseException as error:
                with self._lock:
                    self._finish(r, error)
                continue
            ids = r.request.input_ids
            with torch.inference_mode():
                self.kv.write_span(r.slot, out.start, out.keys, out.values)
            self.kv.length[r.slot] = r.prefilled = len(ids)
            self.kv.commit_prompt(r.slot)
            s = self.stats
            s['prefills'] += 1
            s['transferred_tokens'] += len(ids) - out.start
            s['decode_cached_tokens'] += out.start
            s['prefill_cached_tokens'] += out.prefill_cached
            s['prefill_s'] += out.prefill_s
            s['transfer_s'] += out.transfer_s
            r.pending = None
            if self._emit(r, out.token):
                self.decoding.append(r)

    @torch.inference_mode()
    def _decode(self):
        self._decode_step()
