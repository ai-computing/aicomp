import unittest
import torch
import zmq
from pinned_receiver import PinnedAssembler
from test_memory_receiver import batch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA copy stream')
class PinnedStreamTest(unittest.TestCase):
    def test_validated_chunk_available_before_final_payload(self):
        context=zmq.Context();tx=context.socket(zmq.PAIR);rx=context.socket(zmq.PAIR)
        tx.bind('inproc://early-pinned');rx.connect('inproc://early-pinned')
        rx.setsockopt(zmq.RCVTIMEO,2000)
        assembler=PinnedAssembler('memory-test-139',17)
        received=[]
        assembler.on_batch=lambda first,last,packet:received.append((first,last,packet.host[first:last].clone()))
        try:
            header,payload=batch(range(4),17,False)
            tx.send_multipart([header,payload]);assembler.receive(rx.recv(),rx)
            self.assertEqual(len(received),1,'validated first chunk must publish before remaining KV arrives')
            self.assertEqual(received[0][:2],(0,4))
            self.assertTrue(torch.equal(received[0][2].view(torch.uint8),assembler.data.host[:4].view(torch.uint8)))
            self.assertFalse(assembler.final)
            # Repeated slots are invalid and must never trigger a second copy.
            tx.send_multipart([header,payload])
            with self.assertRaises(ValueError):assembler.receive(rx.recv(),rx)
            self.assertEqual(len(received),1)
        finally:tx.close(0);rx.close(0);context.term()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA copy stream')
class StreamingUploadTest(unittest.TestCase):
    def test_early_gpu_copy_and_complete_byte_equality(self):
        from gpu_stream_upload import StreamingUpload
        a=PinnedAssembler('memory-test-139',17)
        a.data.host.view(torch.int16).copy_(torch.arange(a.data.host.numel(),dtype=torch.int32).reshape(a.data.host.shape).to(torch.int16))
        upload=StreamingUpload(a.data,'cuda:0')
        try:
            upload.stage(0,4)
            upload.event.synchronize()
            self.assertTrue(torch.equal(upload.device_data[:4].cpu().view(torch.uint8),a.data.host[:4].view(torch.uint8)))
            with self.assertRaises(ValueError):upload.finish()
            with self.assertRaises(ValueError):upload.stage(0,4)
            for first in range(4,72,4):upload.stage(first,first+4)
            actual=upload.finish().cpu()
            self.assertTrue(torch.equal(actual.view(torch.uint8),a.data.host.view(torch.uint8)))
        finally:upload.close()

    def test_partial_close_fences_source_before_reuse(self):
        from gpu_stream_upload import StreamingUpload
        a=PinnedAssembler('memory-test-139',511);a.data.host.fill_(7)
        upload=StreamingUpload(a.data,'cuda:0');upload.stage(0,4)
        upload.close();a.data.host.fill_(3)
        self.assertTrue(torch.all(upload.device_data[:4].cpu()==7))
        with self.assertRaises(RuntimeError):upload.stage(4,8)

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA pinned allocation')
class MultiplexStreamingTest(unittest.TestCase):
    def test_callback_routes_context_before_completion(self):
        import concurrent.futures
        from collections import deque
        from unittest.mock import Mock,patch
        from test_multiplex import MultiplexTest,batch as packet_batch
        from protocol import control
        c=MultiplexTest().client();messages=deque();payload=[None];events=[];pool=Mock()
        future=concurrent.futures.Future();response=Mock();response.json.return_value={};future.set_result(response)
        def submit(fn,url,json,timeout):
            rid=json['kv_transfer_params']['id'];messages.append([control(rid,1)])
            for first in range(0,72,4):messages.append(packet_batch(rid,range(first,first+4),1,5,first==68))
            return future
        def recv(copy):
            parts=messages.popleft();payload[0]=parts[1] if len(parts)==2 else None
            return parts[0]
        def recv_into(target):
            target[:]=payload[0];n=len(payload[0]);payload[0]=None;return n
        pool.submit.side_effect=submit;c.rx.poll.side_effect=lambda _:bool(messages)
        c.rx.recv.side_effect=recv;c.rx.recv_into.side_effect=recv_into
        c.rx.getsockopt.side_effect=lambda _:payload[0] is not None
        context=object();remaining=[True]
        def acquire():remaining.clear();return [1,2],context
        def chunk(ctx,first,last,packet):
            self.assertIs(ctx,context)
            self.assertTrue(torch.all(packet.host[first:last].view(torch.int16)==5))
            events.append((first,last))
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool):
            c.prefill_stream('model',acquire,lambda *args:events.append('complete'),lambda:not remaining,
                             direct=True,on_batch=chunk)
        self.assertEqual(events,[(i,i+4) for i in range(0,72,4)]+['complete'])

if __name__=='__main__':unittest.main()
