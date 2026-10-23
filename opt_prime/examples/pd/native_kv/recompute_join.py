"""Bounded two-producer ownership, with GPU work advanced by the decode owner."""
from collections import deque
import threading
import time


class RecomputeJoin:
    def __init__(self, capacity, fraction, chunk_tokens, prefiller, synchronize):
        self.capacity, self.fraction, self.chunk_tokens = capacity, fraction, chunk_tokens
        self.prefiller, self.synchronize = prefiller, synchronize
        self.lock = threading.Lock()
        self.entries = {}
        self.waiting = deque()
        self.current = None

    def admit(self, job, tokens, count):
        with self.lock:
            if id(job) in self.entries or len(self.entries) >= self.capacity:
                raise RuntimeError('recompute admission exceeds lifetime bound')
            n = min(int(len(tokens)*self.fraction), len(tokens)-2)
            self.entries[id(job)] = dict(job=job, tokens=tokens, count=count,
                length=n, native_done=False, gpu_done=n==0, suffix=None, prefix=None)
            if n: self.waiting.append(id(job))
            return n

    def native_complete(self, job, suffix):
        with self.lock:
            entry = self.entries[id(job)]
            if entry['native_done']: raise RuntimeError('duplicate native completion')
            entry.update(native_done=True, suffix=suffix)

    def advance(self):
        """GPU owner only: at most one chunk, between ordinary decode steps."""
        if self.current is None:
            with self.lock:
                if not self.waiting: return
                entry = self.entries[self.waiting.popleft()]
            if entry['job'].cancel.is_set():
                with self.lock: entry['gpu_done'] = True
                return
            session = self.prefiller.start([entry['tokens'][:entry['length']]], self.chunk_tokens)
            self.current = entry, session, time.perf_counter()
        entry, session, started = self.current
        if entry['job'].cancel.is_set():
            # A previous chunk may still be queued on CUDA even with no decode lanes.
            self.synchronize()
            result = None
        else:
            result = session.step()
            if result is None: return
        with self.lock:
            entry['prefix'] = None if result is None else result[1]
            entry['gpu_done'] = True
            entry['job'].stage_metrics['gpu_prefix_ms'] = (time.perf_counter()-started)*1000
        self.current = None

    def pop_ready(self):
        with self.lock:
            for key, entry in self.entries.items():
                if entry['native_done'] and entry['gpu_done']:
                    del self.entries[key]
                    return entry
        return None

    def close(self):
        """GPU owner, after native producer joins: fence before dropping ownership."""
        self.synchronize()
        self.current = None
        with self.lock:
            jobs = [entry['job'] for entry in self.entries.values()]
            self.entries.clear()
            self.waiting.clear()
        return jobs
