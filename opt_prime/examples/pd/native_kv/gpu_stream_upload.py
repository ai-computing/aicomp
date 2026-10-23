"""Receive-owner H2D copies into isolated storage, never an active decode cache."""
import torch


class StreamingUpload:
    def __init__(self, packet, device, stream=None):
        host=packet.host
        if (host.device.type!='cpu' or host.dtype!=torch.bfloat16 or host.ndim!=4
                or host.shape[0]!=72 or host.shape[2:]!=(8,128)
                or not 1<=host.shape[1]<2048 or not host.is_contiguous()
                or not host.is_pinned('cuda')):
            raise ValueError('expected owned pinned native KV')
        self.packet=packet
        self.stream=stream if stream is not None else torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.device_data=torch.empty(host.shape,dtype=host.dtype,device=device)
        self.event=torch.cuda.Event()
        self.seen=set()
        self.closed=False

    @torch.inference_mode()
    def stage(self, first, last):
        if self.closed: raise RuntimeError('upload closed')
        if (type(first) is not int or type(last) is not int or not 0<=first<last<=72
                or self.seen.intersection(range(first,last))):
            raise ValueError('invalid or repeated slots')
        with torch.cuda.stream(self.stream):
            self.device_data[first:last].copy_(self.packet.host[first:last],non_blocking=True)
            self.event.record(self.stream)
        self.seen.update(range(first,last))

    def finish(self):
        if self.closed: raise RuntimeError('upload closed')
        if len(self.seen)!=72: raise ValueError('incomplete upload')
        consumer=torch.cuda.current_stream(self.device_data.device)
        consumer.wait_event(self.event)
        self.device_data.record_stream(consumer)
        return self.device_data

    def close(self):
        if not self.closed:
            if self.seen: self.event.synchronize()
            self.closed=True
            self.packet=None


class UploadQueue:
    """Receiver publishes ranges; only the GPU owner allocates or submits CUDA work."""
    def __init__(self, capacity, device):
        import queue
        self.queue=queue.Queue(capacity*72)
        self.device=device
        self.uploads={}

    def publish(self, job, first, last, packet):
        self.queue.put_nowait((job,first,last,packet))

    def advance(self):
        import queue
        import time
        # At most 72 validated, non-repeated slots per lifetime credit.
        for _ in range(self.queue.maxsize):
            try: job,first,last,packet=self.queue.get_nowait()
            except queue.Empty: break
            if job.cancel.is_set(): continue
            started=time.perf_counter()
            if id(job) not in self.uploads:
                self.uploads[id(job)]=StreamingUpload(packet,self.device)
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
