"""Opt-in research host-wall timings; never interpret sums as a device overlap trace.

Remote completion includes queueing, producer work and response delivery. Import and
decode measure synchronous Python-call boundaries, not independently fenced DMA.
Installing this probe changes overhead; compare both instrumented arms and retain
un-instrumented measurements. It does not change KV ownership or scheduling.
"""
import threading
import time


class StageProbe:
    def __init__(self, *, clock=time.perf_counter):
        self.clock = clock
        self._lock = threading.Lock()
        self._stats = {}

    def _record(self, stage, start, error=False):
        elapsed = self.clock() - start
        with self._lock:
            row = self._stats.setdefault(stage, dict(calls=0, errors=0, wall_s=0.0))
            row['calls'] += 1
            row['errors'] += int(error)
            row['wall_s'] += elapsed

    def snapshot(self):
        with self._lock:
            return {stage: dict(row) for stage, row in self._stats.items()}

    def _wrap(self, obj, name, stage):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            key = stage()
            start = self.clock()
            failed = True
            try:
                result = original(*args, **kwargs)
                failed = False
                return result
            finally:
                self._record(key, start, failed)
        setattr(obj, name, measured)

    def install(self, batcher):
        self._wrap(batcher, '_decode_step', lambda: f'decode/{len(batcher.decoding)}')
        if batcher.blocks is not None:
            self._wrap(batcher.blocks, 'import_layer_span', lambda: 'import')
        if batcher.worker is not None:
            original = batcher.worker.submit
            def submit(*args, **kwargs):
                start = self.clock()
                try:
                    future = original(*args, **kwargs)
                except BaseException:
                    self._record('remote_completion', start, True)
                    raise
                def completed(done):
                    failed = done.cancelled() or done.exception() is not None
                    self._record('remote_completion', start, failed)
                future.add_done_callback(completed)
                return future
            batcher.worker.submit = submit
