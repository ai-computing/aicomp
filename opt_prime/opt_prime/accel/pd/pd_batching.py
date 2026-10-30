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
    if isinstance(t, (list, tuple)):                           # hybrid models: one span per KV group
        return [move_kv(x, device) for x in t]
    return _move_one(t, device)


def _move_one(t, device):
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
    state: dict = None         # linear-attention layers: {layer: (conv, recurrent)} after the prompt (#127)


class TorchPrefillEngine:
    """prefill only, on its own device: chunked over a paged KV with a prefix cache; returns the KV span to send"""

    def __init__(self, model, *, max_context=2048, chunk=2048, page_size=64, pages=None, prefix_cache=True,
                 out_device=None):
        from opt_prime.accel.pd.paged import PagedKV
        g = describe_model(model.config)
        p = next(model.parameters())
        self.model, self.device, self.dtype = model.eval(), p.device, p.dtype
        self.kv = PagedKV(g.layers, 1, g.kv_heads, g.head_dim, self.dtype, self.device,
                          pages=pages or 4 * max_context // page_size, page_size=page_size, max_context=max_context,
                          prefix_cache=prefix_cache, groups=g.groups)
        if g.recurrent:
            self.kv.attach_recurrent(model, g.recurrent)
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
            state = self.kv.lane_state.pop(0, None) if self.kv.rstate else None
            token = select_token(logits, request.sampling, len(ids))
            keys, values = self.kv.read_span(0, send_from, len(ids))       # gathered copies, own memory
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)                          # prefill time = compute, not launch
            t1 = time.perf_counter()
            if self.out_device is not None and self.out_device != self.device:
                with torch.cuda.stream(self._stream) if self._stream is not None else _nullctx():
                    keys, values = move_kv(keys, self.out_device), move_kv(values, self.out_device)
                    if state:
                        state = {k: tuple(move_kv(list(v), self.out_device)) for k, v in state.items()}
                if self._stream is not None:
                    self._stream.synchronize()
            t2 = time.perf_counter()
            return PrefillOut(token, send_from, keys, values, cached, t1 - t0, t2 - t1, state)
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

    @property
    def ready(self):
        """False once the child has died or been closed (the serving readiness gate reads this)."""
        return self._futures is not None and self._proc.is_alive()

    def submit(self, request, send_from=0, slot=None) -> Future:
        f = Future()
        with self._lock:
            if self._futures is None:
                raise RuntimeError("prefill process exited")
            key, self._next = self._next, self._next + 1
            self._futures[key] = f
        self._req.put((key, request, send_from))
        return f

    def _receive(self):
        stream = torch.cuda.Stream(self.out_device) if self.out_device.type == "cuda" else None
        while True:
            try:
                item = self._rep.get(timeout=0.5)
            except queue.Empty:
                if self._proc.is_alive():
                    continue
                with self._lock:                                     # the child died: fail what it will never answer
                    for f in self._futures.values():
                        f.set_exception(RuntimeError(f"prefill process exited ({self._proc.exitcode})"))
                    self._futures = None
                return
            if item is None:
                return
            key, out, error = item
            with self._lock:
                f = self._futures.pop(key)
            if error is not None:
                f.set_exception(RuntimeError(f"prefill process: {error}"))
                continue
            t0 = time.perf_counter()

            def copy(x):                                             # own memory on this side (the child frees its)
                if isinstance(x, (list, tuple)):
                    return [copy(y) for y in x]
                return move_kv(x, self.out_device) if stream is not None else x.to(self.out_device).clone()

            with torch.cuda.stream(stream) if stream is not None else _nullctx():
                k, v = copy(out.keys), copy(out.values)
                state = {layer: tuple(copy(list(t))) for layer, t in out.state.items()} if out.state else None
            if stream is not None:
                stream.synchronize()
            self._ack.put(key)                                           # the child may free its tensors now
            f.set_result(PrefillOut(out.token, out.start, k, v, out.prefill_cached, out.prefill_s,
                                    out.transfer_s + time.perf_counter() - t0, state))

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
        self._json = json
        self._futures, self._next, self._lock = {}, 0, threading.Lock()
        self._write_lock = threading.Lock()
        self._failed = None
        self._proc = self._recv = None
        self._closed = False
        ready = Future()
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                          bufsize=1, env=env)
            self._recv = threading.Thread(target=self._receive, args=(ready,), daemon=True,
                                          name="prefill-subprocess")
            self._recv.start()
            ready.result(timeout=startup_timeout)
        except BaseException:
            self.close()
            raise

    @property
    def ready(self):
        """Terminal producer failure/closure is distinct from a single rejected request."""
        with self._lock:
            return (not self._closed and self._failed is None and self._proc.poll() is None
                    and self._recv.is_alive())

    def submit(self, request, send_from=0, slot=None) -> Future:
        if slot is None or not 0 <= slot < self.slab.shape[0]:
            raise ValueError("SubprocessPrefillWorker needs a valid decode lane (slot)")
        f = Future()
        s = request.sampling
        write_error = None
        # Keep pipe writes ordered without holding the response-map lock across blocking I/O.
        with self._write_lock:
            with self._lock:
                error = self._failed
                key, self._next = self._next, self._next + 1
                if error is None:
                    self._futures[key] = (f, slot, send_from)
            if error is not None:
                f.set_exception(error)
                return f
            try:
                line = self._json.dumps(dict(id=key, tokens=list(request.input_ids), send_from=send_from, slot=slot,
                                             sampling=dict(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k,
                                                           seed=s.seed, stop_token_ids=list(s.stop_token_ids))))
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError, TypeError) as error:
                write_error = RuntimeError(f"prefill subprocess write failed: {error}")
        if write_error is not None:
            self._fail(write_error)
        return f

    def _fail(self, error):
        with self._lock:
            if self._failed is None:
                self._failed = error
            pending, self._futures = self._futures, {}
            error = self._failed
        # Future callbacks may call back into the worker; never invoke them under its lock.
        for f, _, _ in pending.values():
            if f.set_running_or_notify_cancel():
                f.set_exception(error)

    def _receive(self, ready):
        try:
            for line in self._proc.stdout:
                try:
                    msg = self._json.loads(line)
                except ValueError:
                    continue                                  # SDK log lines on stdout
                if not isinstance(msg, dict):
                    continue
                if msg.get("ready"):
                    if not ready.done():
                        ready.set_result(True)
                    continue
                if "id" not in msg:
                    continue                                  # structured SDK logs
                with self._lock:
                    entry = self._futures.get(msg["id"])
                    failed = self._failed
                if failed is not None:
                    return
                if entry is None:
                    raise ValueError("unknown response ID")
                f, slot, start = entry
                error = None
                if "error" in msg:
                    error = RuntimeError(f"prefill subprocess: {msg['error']}")
                else:
                    n = msg["n"]
                    if type(n) is not int or not 0 <= n <= self.slab.shape[-2]:
                        raise ValueError("invalid KV span length")
                    kv = torch.from_numpy(self.slab[slot, :, :, :, :n].view("int16")).view(torch.bfloat16)
                    out = PrefillOut(msg["token"], start, kv[0], kv[1], msg.get("cached", 0),
                                     msg.get("prefill_s", 0.0))
                # Keep the request registered until validation succeeds so failure settles it too.
                with self._lock:
                    entry = self._futures.pop(msg["id"], None)
                if entry is not None and f.set_running_or_notify_cancel():
                    if error is not None:
                        f.set_exception(error)
                    else:
                        f.set_result(out)
            raise RuntimeError("prefill subprocess exited")
        except Exception as error:
            failure = RuntimeError(f"prefill subprocess response failed: {error}")
            self._fail(failure)
            if not ready.done():
                ready.set_exception(failure)

    def close(self, timeout=5):
        import subprocess
        if self._closed:
            return
        self._closed = True
        self._fail(RuntimeError("prefill subprocess closed"))
        if self._proc is not None:
            # Stop the writer before unmapping its slab, including when startup never completed. Closing stdin first
            # lets prefill_server leave its request loop and exit normally (the SDK releases the device on exit);
            # SIGTERM, then SIGKILL, only for a child that does not exit within `timeout`.
            try:
                self._proc.stdin.close()
            except OSError:
                pass                                         # a broken pipe must not leak the slab
            for stop in (None, self._proc.terminate, self._proc.kill):
                if self._proc.poll() is not None:
                    break
                if stop is not None:
                    stop()
                try:
                    self._proc.wait(timeout=None if stop == self._proc.kill else timeout)
                except subprocess.TimeoutExpired:
                    pass
            if self._recv is not None and self._recv is not threading.current_thread():
                self._recv.join(timeout=timeout)
            self._proc.stdout.close()
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


class PrefillWorkerPool:
    """several prefill workers (e.g. one `SubprocessPrefillWorker` per NPU chip): each request goes to the worker with
    the fewest prefills in flight, so prefill capacity scales with the workers while the decode lanes stay batched"""

    def __init__(self, workers):
        if not workers:
            raise ValueError("a prefill pool needs at least one worker")
        self.workers = list(workers)
        self._load = [0] * len(self.workers)
        self._lock = threading.Lock()

    @property
    def ready(self):
        # The pool still routes to every member; no implicit removal/retry of a failed producer.
        return all(getattr(worker, "ready", True) for worker in self.workers)

    def submit(self, request, send_from=0, slot=None) -> Future:
        with self._lock:
            i = min(range(len(self.workers)), key=self._load.__getitem__)
            self._load[i] += 1
        f = self.workers[i].submit(request, send_from=send_from, slot=slot)

        def done(_):
            with self._lock:
                self._load[i] -= 1
        f.add_done_callback(done)
        return f

    def close(self):
        for w in self.workers:
            w.close()


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
            if self.kv.rstate:
                self.kv.commit_state(r.slot, out.state)
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
