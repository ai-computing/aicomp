import concurrent.futures
import struct
import unittest
from collections import deque
from unittest.mock import Mock, patch
import numpy as np
import requests
from multiplex_receiver import MultiplexKVClient
from protocol import control, pad, string_ref, read_string, u32


def batch(rid, slots, n, marker, final):
    header=bytearray(rid.encode());pad(header);start=len(header)
    for slot in slots:
        for pos in range(n):header.extend(struct.pack('<4I',0,slot,pos,2048))
    root=len(header);header.extend(b'\0'*36)
    struct.pack_into('<I',header,root,1);string_ref(header,root+4,0,len(rid))
    struct.pack_into('<iII',header,root+12,start-root-12,len(slots)*n,int(final))
    return [bytes(header), np.full(len(slots)*n*1024,marker,dtype=np.uint16).tobytes()]


class MultiplexTest(unittest.TestCase):
    def client(self):
        c=MultiplexKVClient.__new__(MultiplexKVClient)
        c.poisoned=False;c.timeout=2;c.receiver_port=19202;c.http_url='http://127.0.0.1:19101'
        c.rx=Mock();c.tx=Mock()
        return c

    def test_interleaved_ids_and_out_of_order_completion(self):
        c=self.client();messages=deque();submitted=[];futures={};completed=[]
        pool=Mock()
        def submit(fn,url,json,timeout):
            rid=json['kv_transfer_params']['id'];future=concurrent.futures.Future()
            submitted.append((rid,len(json['prompt'])-1));futures[rid]=future
            if len(submitted)==2:
                for rid,n in reversed(submitted):messages.append([control(rid,1)])
                for first in range(0,72,4):
                    for index in [1,0]:
                        rid,n=submitted[index]
                        messages.append(batch(rid,range(first,first+4),n,index+1,first==68))
            return future
        def send(header):
            root=len(header)-36
            if u32(header,root+4)==3:
                rid=read_string(header,root+8)
                response=Mock();response.json.return_value={'ok':True}
                futures[rid].set_result(response)
        pool.submit.side_effect=submit;c.tx.send.side_effect=send
        c.rx.poll.side_effect=lambda timeout: bool(messages)
        c.rx.recv_multipart.side_effect=lambda copy: messages.popleft()
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool):
            results=c.prefill_many('model',[[1,2],[3,4,5]],lambda i,*_:completed.append(i))
        self.assertEqual(completed,[1,0])
        for index,(kv,stats) in enumerate(results):
            self.assertTrue(np.all(np.frombuffer(kv.tobytes(),dtype=np.uint16)==index+1))
            self.assertEqual(stats['batches'],18)
        self.assertFalse(c.poisoned)

    def test_unknown_id_poisoned(self):
        c=self.client();pool=Mock();pool.submit.return_value=concurrent.futures.Future()
        c.rx.poll.return_value=True;c.rx.recv_multipart.return_value=[control('unregistered-id',1)]
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool):
            with self.assertRaises(ValueError):c.prefill_many('model',[[1,2]])
        self.assertTrue(c.poisoned)
        with self.assertRaises(RuntimeError):c.prefill_many('model',[[1,2]])

    def test_http_failure_preempts_receive(self):
        c=self.client();pool=Mock();future=concurrent.futures.Future()
        response=requests.Response();response.status_code=503;future.set_result(response)
        pool.submit.return_value=future
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool):
            with self.assertRaises(requests.HTTPError):c.prefill_many('model',[[1,2]])
        self.assertTrue(c.poisoned);c.rx.poll.assert_not_called()


if __name__=='__main__':unittest.main()
