"""Catch fixed Qwen geometry and backend-specific queue ownership."""
import threading
import unittest
from types import SimpleNamespace
import torch
from gpu_stream_upload import StreamingUpload, UploadQueue

class TransferBackendsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_non_qwen_geometry(self):
        host=torch.arange(4*7*2*16,dtype=torch.float32).reshape(4,7,2,16).to(torch.bfloat16).pin_memory('cuda')
        upload=StreamingUpload(SimpleNamespace(host=host),'cuda:0')
        try:
            upload.stage(2,4)
            with self.assertRaises(ValueError): upload.finish()
            upload.stage(0,2)
            self.assertTrue(torch.equal(upload.finish().cpu(),host))
        finally: upload.close()

    def test_queue_injects_backend_and_retires_cancelled_transfer(self):
        class HostCopy:
            def __init__(self, packet, device):
                self.packet=packet; self.data=torch.empty_like(packet.host); self.closed=False
            def stage(self,first,last): self.data[first:last].copy_(self.packet.host[first:last])
            def finish(self): return self.data
            def close(self): self.closed=True; self.packet=None
        q=UploadQueue(1,'cpu',slots=4,upload_factory=HostCopy)
        job=SimpleNamespace(cancel=threading.Event(),stage_metrics={})
        p=SimpleNamespace(host=torch.arange(8).reshape(4,2))
        q.publish(job,0,2,p); q.advance()
        u=q.uploads[id(job)]
        self.assertTrue(torch.equal(u.data[:2],p.host[:2]))
        job.cancel.set();q.discard(job)
        self.assertTrue(u.closed);self.assertFalse(q.uploads)
        q.close()
