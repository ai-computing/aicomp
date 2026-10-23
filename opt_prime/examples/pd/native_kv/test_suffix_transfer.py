import concurrent.futures
from collections import deque
import unittest
from unittest.mock import Mock,patch
import numpy as np
from memory_receiver import NativeKVClient
from protocol import control,u32
from test_multiplex import batch


class SuffixTransferTest(unittest.TestCase):
    def test_offset_wire_field_and_bounds(self):
        header=control('suffix-test-123',2,[0,1,2],start_pos=17)
        self.assertEqual(u32(header,len(header)-20),17)
        for offset in (-1,2047,True,1.5):
            with self.subTest(offset=offset),self.assertRaises(ValueError):
                control('suffix-test-123',2,[0],start_pos=offset)
        for kind,ids,offset in [(3,[],1),(1,[],1),(2,[],1),(2,[0,1],2046)]:
            with self.assertRaises(ValueError):control('suffix-test-123',kind,ids,start_pos=offset)
        control('suffix-test-123',2,[0],start_pos=2046)

    def test_invalid_offset_fails_before_network(self):
        c=NativeKVClient.__new__(NativeKVClient);c.poisoned=False
        for offset in (-1,3,True,1.5):
            with self.subTest(offset=offset),self.assertRaises(ValueError):
                c.prefill('model',[1,2,3,4],'views',start_pos=offset)

    def test_shorter_packet_ack_and_exact_data(self):
        for mode in ('views','fast','legacy'):
            with self.subTest(mode=mode):
                c=NativeKVClient.__new__(NativeKVClient)
                c.poisoned=False;c.timeout=2;c.receiver_port=19202;c.http_url='http://127.0.0.1:19101'
                c.http=Mock();c.executor=Mock();c.rx=Mock();c.tx=Mock()
                rid='perf-kv-suffix';messages=deque([[control(rid,1)]])
                for slot in range(0,72,4):messages.append(batch(rid,range(slot,slot+4),1,77,slot==68))
                c.rx.poll.return_value=True;c.rx.recv_multipart.side_effect=lambda **_:messages.popleft()
                response=Mock();response.json.return_value={};future=concurrent.futures.Future();future.set_result(response)
                c.executor.submit.return_value=future
                with patch('memory_receiver.uuid.uuid4',return_value=Mock(hex='suffix')):
                    data,stats=c.prefill('model',[1,2,3,4],mode,start_pos=2)
                self.assertEqual(stats['exported_positions'],1)
                self.assertEqual(stats['bytes'],72*2048)
                self.assertTrue(np.all(np.frombuffer(data.tobytes(),dtype=np.uint16)==77))
                sent=c.tx.send.call_args_list
                self.assertEqual(len(sent),2)
                self.assertEqual(sent[0].args[0],control(rid,2,[0],start_pos=2))
                self.assertEqual(sent[1].args[0],control(rid,3))
                self.assertFalse(c.poisoned)

if __name__=='__main__':unittest.main()
