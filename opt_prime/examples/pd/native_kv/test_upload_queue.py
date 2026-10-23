import threading
import unittest
from types import SimpleNamespace
import torch
from pinned_receiver import PinnedPacket


@unittest.skipUnless(torch.cuda.is_available(),'CUDA upload owner')
class UploadQueueTest(unittest.TestCase):
    def test_receive_thread_publishes_without_cuda_and_gpu_owner_uploads(self):
        from gpu_stream_upload import UploadQueue
        q=UploadQueue(1,'cuda:0');job=SimpleNamespace(cancel=threading.Event(),stage_metrics={})
        packet=PinnedPacket(17);packet.host.fill_(7)
        failure=[]
        def receive():
            try:
                for first in range(0,72,4):q.publish(job,first,first+4,packet)
            except BaseException as exc:failure.append(exc)
        thread=threading.Thread(target=receive);thread.start();thread.join()
        self.assertFalse(failure);self.assertFalse(q.uploads)
        q.advance()
        upload=q.take(job)
        try:self.assertTrue(torch.all(upload.finish().cpu()==7))
        finally:upload.close();q.close()

    def test_cancelled_pending_chunks_do_not_create_upload_and_retire_partial(self):
        from gpu_stream_upload import UploadQueue
        q=UploadQueue(1,'cuda:0');job=SimpleNamespace(cancel=threading.Event(),stage_metrics={})
        packet=PinnedPacket(17);packet.host.fill_(9)
        q.publish(job,0,4,packet);q.advance()
        upload=q.uploads[id(job)]
        job.cancel.set();q.publish(job,4,8,packet);q.discard(job)
        self.assertFalse(q.uploads);self.assertTrue(upload.closed)
        self.assertEqual(upload.seen,set(range(4)))
        packet.host.fill_(3)
        self.assertTrue(torch.all(upload.device_data[:4].cpu()==9));q.close()

    def test_queue_has_fixed_lifetime_capacity(self):
        import queue
        from gpu_stream_upload import UploadQueue
        q=UploadQueue(1,'cuda:0');job=SimpleNamespace(cancel=threading.Event(),stage_metrics={})
        for i in range(72):q.publish(job,i,i+1,None)
        with self.assertRaises(queue.Full):q.publish(job,0,1,None)
        q.close()
