"""Bounded producer/consumer queue; backend owns copies and completion fences.

Factories return stage(first,last), finish(), close() transfers. close() must
fence outstanding work before releasing the retained source packet. The queue
is single-consumer; publish is thread-safe. Callers validate chunk geometry.
"""
class TransferQueue:
    """Receiver publishes ranges; only the GPU owner allocates or submits CUDA work."""
    def __init__(self, capacity, device, *, slots, upload_factory):
        import queue
        if type(capacity) is not int or capacity < 1 or type(slots) is not int or slots < 1:
            raise ValueError('positive capacity and slots required')
        self.queue=queue.Queue(capacity*slots)
        self.upload_factory=upload_factory
        self.slots=slots
        self.device=device
        self.uploads={}

    def publish(self, job, first, last, packet):
        self.queue.put_nowait((job,first,last,packet))

    def advance(self):
        import queue
        import time
        # At most one publication per slot per lifetime credit.
        for _ in range(self.queue.maxsize):
            try: job,first,last,packet=self.queue.get_nowait()
            except queue.Empty: break
            if job.cancel.is_set(): continue
            started=time.perf_counter()
            if id(job) not in self.uploads:
                self.uploads[id(job)]=self.upload_factory(packet,self.device)
                job.stage_metrics['upload_alloc_ms']=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            self.uploads[id(job)].stage(first,last)
            elapsed=(time.perf_counter()-started)*1000
            metrics=job.stage_metrics
            metrics['upload_submit_ms']=metrics.get('upload_submit_ms',0)+elapsed
            metrics['upload_submit_max_ms']=max(metrics.get('upload_submit_max_ms',0),elapsed)

    def take(self, job):
        # Completion may have arrived after this iteration's earlier advance().
        self.advance()
        return self.uploads.pop(id(job))

    def discard(self, job):
        self.advance()
        upload=self.uploads.pop(id(job),None)
        if upload is not None: upload.close()

    def close(self):
        import queue
        # Producer must be joined first; no further ranges can arrive.
        while True:
            try:self.queue.get_nowait()
            except queue.Empty:break
        for upload in self.uploads.values():upload.close()
        self.uploads.clear()
