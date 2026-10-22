import unittest
from unittest.mock import Mock
import numpy as np
import torch
import zmq
from memory_receiver import CanonicalAssembler
from pinned_receiver import PinnedAssembler
from test_memory_receiver import batch


@unittest.skipUnless(torch.cuda.is_available(),'CUDA pinned allocation')
class PinnedReceiveTest(unittest.TestCase):
    def test_real_multipart_exact_and_owned(self):
        context=zmq.Context();tx=context.socket(zmq.PAIR);rx=context.socket(zmq.PAIR)
        tx.bind('inproc://pinned-test');rx.connect('inproc://pinned-test');rx.setsockopt(zmq.RCVTIMEO,2000)
        try:
            for n in (1,17,511):
                direct=PinnedAssembler('memory-test-139',n);reference=CanonicalAssembler('memory-test-139',n)
                for first in range(0,72,4):
                    header,payload=batch(range(first,first+4),n,first==68)
                    reference.accept(header,payload);tx.send_multipart([header,payload])
                    actual_header=rx.recv();self.assertTrue(rx.getsockopt(zmq.RCVMORE))
                    done=direct.receive(actual_header,rx)
                self.assertTrue(done)
                self.assertTrue(direct.data.host.is_pinned('cuda'))
                np.testing.assert_array_equal(direct.array,reference.data.transpose(0,2,1,3))
        finally:tx.close(0);rx.close(0);context.term()

    def test_bad_header_not_received_or_acknowledged(self):
        a=PinnedAssembler('memory-test-139',3);socket=Mock()
        header,payload=batch(range(4),3)
        with self.assertRaises(ValueError):a.receive(header[:-1],socket)
        socket.recv_into.assert_not_called()
        with self.assertRaises(RuntimeError):a.receive(header,socket)

    def test_size_extra_frame_and_nonconsecutive_rejected(self):
        for extra,delta in ((False,-1),(False,1),(True,0)):
            a=PinnedAssembler('memory-test-139',3);socket=Mock()
            header,payload=batch(range(4),3)
            socket.recv_into.return_value=len(payload)+delta;socket.getsockopt.return_value=extra
            with self.assertRaises(ValueError):a.receive(header,socket)
            self.assertFalse(a.seen.any());self.assertFalse(a.final)
        a=PinnedAssembler('memory-test-139',3);socket=Mock();header,_=batch((0,2,4,6),3)
        with self.assertRaises(ValueError):a.receive(header,socket)
        socket.recv_into.assert_not_called()

    def test_real_truncation_is_poisoned(self):
        context=zmq.Context();tx=context.socket(zmq.PAIR);rx=context.socket(zmq.PAIR)
        tx.bind('inproc://truncate-test');rx.connect('inproc://truncate-test');rx.setsockopt(zmq.RCVTIMEO,2000)
        try:
            a=PinnedAssembler('memory-test-139',3);header,payload=batch(range(4),3)
            tx.send_multipart([header,payload+b'x']);rx.recv()
            with self.assertRaises(ValueError):a.receive(header,rx)
            self.assertTrue(a.failed);self.assertFalse(a.seen.any())
        finally:tx.close(0);rx.close(0);context.term()

    def test_eight_slot_tcp_frames_exact_at_capacity(self):
        context=zmq.Context();tx=context.socket(zmq.PUSH);rx=context.socket(zmq.PULL)
        rx.setsockopt(zmq.MAXMSGSIZE,36*1024*1024);rx.setsockopt(zmq.RCVTIMEO,5000)
        port=rx.bind_to_random_port('tcp://127.0.0.1');tx.connect(f'tcp://127.0.0.1:{port}')
        try:
            for n in (1,511,2047):
                direct=PinnedAssembler('memory-test-139',n,max_batch_slots=8)
                reference=CanonicalAssembler('memory-test-139',n)
                # Reference consumes default four-slot frames built from the same bytes.
                for first in range(0,72,8):
                    header,payload=batch(range(first,first+8),n,first==64)
                    tx.send_multipart([header,payload]);done=direct.receive(rx.recv(),rx)
                    size=4*n*2048
                    for offset in (0,4):
                        h,_=batch(range(first+offset,first+offset+4),n,first+offset==68)
                        reference.accept(h,payload[(offset//4)*size:(offset//4+1)*size])
                self.assertTrue(done)
                np.testing.assert_array_equal(direct.array,reference.data.transpose(0,2,1,3))
                self.assertTrue(direct.data.host.is_pinned('cuda'))
        finally:tx.close(0);rx.close(0);context.term()

    def test_group_bound_rejects_before_receive(self):
        socket=Mock()
        for limit,count in ((4,8),(8,9)):
            a=PinnedAssembler('memory-test-139',3,max_batch_slots=limit)
            header,_=batch(range(count),3)
            with self.assertRaises(ValueError):a.receive(header,socket)
            self.assertTrue(a.failed);self.assertFalse(a.seen.any())
        socket.recv_into.assert_not_called()
        with self.assertRaises(ValueError):PinnedAssembler('memory-test-139',3,max_batch_slots=16)

    def test_extended_group_requires_direct_client_path(self):
        from multiplex_receiver import MultiplexKVClient
        a=MultiplexKVClient.__new__(MultiplexKVClient);a.max_batch_slots=8;a.poisoned=False
        with self.assertRaises(ValueError):a.prefill_many('test',[[1,2]])
        with self.assertRaises(ValueError):a.prefill('test',[1,2])

if __name__=='__main__':unittest.main()
