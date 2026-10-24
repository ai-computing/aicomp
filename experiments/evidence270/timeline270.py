"""Research-only host timeline; not a device trace. Includes all harness phases.

Install on the owner's thread before serving. Queue snapshots are taken by that
thread; completion records are written by the producer reader thread. Decode
intervals measure Python call duration and do not prove DMA/compute overlap.
"""
import copy
import threading
import time

class TimelineProbe:
    def __init__(self, *, clock=time.perf_counter):
        self.clock=clock
        self._lock=threading.Lock()
        self._data={'remote':[], 'decode':[]}
        self._next=0

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._data)

    def _append(self, kind, row):
        with self._lock:
            self._data[kind].append(row)

    def install(self, batcher):
        original_decode=batcher._decode_step
        def decode(*args, **kwargs):
            row=dict(start=self.clock(),lanes=len(batcher.decoding),
                     prefilling=len(batcher.prefilling),
                     ready=sum(r.pending is not None and r.pending.done() for r in batcher.prefilling),
                     waiting=len(batcher.waiting),error=True)
            try:
                out=original_decode(*args, **kwargs)
                row['error']=False
                return out
            finally:
                row['end']=self.clock()
                self._append('decode',row)
        batcher._decode_step=decode
        if batcher.worker is None:
            return
        original_submit=batcher.worker.submit
        def submit(request, *args, **kwargs):
            with self._lock:
                key=self._next
                self._next+=1
            row=dict(id=key,slot=kwargs.get('slot'),input_tokens=len(request.input_ids),start=self.clock())
            try:
                future=original_submit(request,*args,**kwargs)
            except BaseException:
                row.update(end=self.clock(),error=True)
                self._append('remote',row)
                raise
            def completed(done):
                row.update(end=self.clock(),error=done.cancelled() or done.exception() is not None)
                self._append('remote',row)
            future.add_done_callback(completed)
            return future
        batcher.worker.submit=submit
