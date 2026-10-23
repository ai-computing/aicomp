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

    def test_stream_replenishes_before_old_tail_and_bounds_pending(self):
        c=self.client(); messages=deque(); submitted=[]; futures={}; completed=[]
        pool=Mock(); remaining=deque(range(3)); live=set(); peak=[]
        def acquire():
            if not remaining:return None
            index=remaining.popleft()
            return [1,2],index
        def enqueue(rid,index):
            messages.append([control(rid,1)])
            for first in range(0,72,4):
                messages.append(batch(rid,range(first,first+4),1,index+1,first==68))
        def submit(fn,url,json,timeout):
            rid=json['kv_transfer_params']['id'];index=len(submitted)
            submitted.append(rid);live.add(rid);peak.append(len(live))
            future=concurrent.futures.Future();futures[rid]=future
            if index==0:enqueue(rid,index)
            if index==2:
                # Third request must enter while second still waits for KV.
                self.assertEqual(completed,[0])
                self.assertIn(submitted[1],live)
                enqueue(rid,index);enqueue(submitted[1],1)
            return future
        def send(header):
            root=len(header)-36
            if u32(header,root+4)==3:
                rid=read_string(header,root+8)
                response=Mock();response.json.return_value={}
                futures[rid].set_result(response)
        def complete(index,data,stats):
            live.remove(submitted[index]);completed.append(index)
            self.assertTrue(np.all(np.frombuffer(data.tobytes(),dtype=np.uint16)==index+1))
        pool.submit.side_effect=submit;c.tx.send.side_effect=send
        c.rx.poll.side_effect=lambda timeout:bool(messages)
        c.rx.recv_multipart.side_effect=lambda copy:messages.popleft()
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool):
            c.prefill_stream('model',acquire,complete,lambda:not remaining,max_pending=2)
        self.assertEqual(completed,[0,2,1]);self.assertEqual(max(peak),2)
        self.assertFalse(live);self.assertFalse(c.poisoned)

    def test_stream_deadline_is_per_request_not_stream_lifetime(self):
        c=self.client();c.timeout=1;clock=[0.0];messages=deque();remaining=deque(range(3))
        futures={};completed=[];pool=Mock()
        def acquire():
            if not remaining:return None
            return [1,2],remaining.popleft()
        def submit(fn,url,json,timeout):
            rid=json['kv_transfer_params']['id'];future=concurrent.futures.Future();futures[rid]=future
            messages.append([control(rid,1)])
            for first in range(0,72,4):messages.append(batch(rid,range(first,first+4),1,1,first==68))
            return future
        def send(header):
            root=len(header)-36
            if u32(header,root+4)==3:
                clock[0]+=0.8
                rid=read_string(header,root+8);response=Mock();response.json.return_value={}
                futures[rid].set_result(response)
        pool.submit.side_effect=submit;c.tx.send.side_effect=send
        c.rx.poll.side_effect=lambda timeout:bool(messages)
        c.rx.recv_multipart.side_effect=lambda copy:messages.popleft()
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool),patch('multiplex_receiver.time.perf_counter',side_effect=lambda:clock[0]):
            c.prefill_stream('model',acquire,lambda index,*_:completed.append(index),lambda:not remaining,max_pending=1)
        self.assertEqual(completed,[0,1,2]);self.assertGreater(clock[0],c.timeout)
        self.assertFalse(c.poisoned)

    def test_suffix_offset_and_short_packet(self):
        c=self.client();pool=Mock();messages=deque();future=concurrent.futures.Future()
        response=Mock();response.json.return_value={};future.set_result(response)
        def submit(fn,url,json,timeout):
            rid=json['kv_transfer_params']['id'];messages.append([control(rid,1)])
            for first in range(0,72,4):messages.append(batch(rid,range(first,first+4),1,77,first==68))
            return future
        pool.submit.side_effect=submit;c.rx.poll.return_value=True
        c.rx.recv_multipart.side_effect=lambda copy:messages.popleft()
        with patch('multiplex_receiver.concurrent.futures.ThreadPoolExecutor',return_value=pool):
            result=c.prefill_many('model',[[1,2,3,4]],start_pos=2)
        self.assertEqual(result[0][1]['exported_positions'],1)
        self.assertEqual(result[0][0].nbytes,72*2048)
        self.assertTrue(np.all(np.frombuffer(result[0][0].tobytes(),dtype=np.uint16)==77))
        header=c.tx.send.call_args_list[0].args[0]
        self.assertEqual(u32(header,len(header)-20),2)
        for offset in (-1,True,3):
            with self.assertRaises(ValueError):c.prefill_many('model',[[1,2,3,4]],start_pos=offset)

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
