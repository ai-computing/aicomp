"""Single-flight in-memory native KV receiver; experimental fixed-build protocol.

The fast path validates a complete contiguous run per slot and writes directly to
[72, heads, positions, head_dim] BF16 storage. No per-token Python objects or disk IO.
"""
import concurrent.futures
import time
import uuid

import numpy as np
import requests
import zmq

from protocol import MAX_HEADER, MAX_POSITIONS, control, handshake, parse_message, read_string, rel, u32


class CanonicalAssembler:
    def __init__(self, request_id, positions):
        if not 1 <= positions <= MAX_POSITIONS:
            raise ValueError('unsupported position count')
        self.request_id, self.positions = request_id, positions
        self.data = np.empty((72, 8, positions, 128), dtype=np.uint16)
        self.seen = np.zeros(72, dtype=bool)
        self.expected_positions = np.arange(positions, dtype=np.uint32)
        self.final = False

    def validate_header(self, header, payload_size=None):
        if self.final:
            raise ValueError('batch after final')
        if not 36 <= len(header) <= MAX_HEADER or len(header) % 4:
            raise ValueError('invalid header size')
        root = len(header) - 36
        if u32(header, root) != 1 or read_string(header, root+4) != self.request_id:
            raise ValueError('unexpected message or request ID')
        start, count = rel(header, root+12), u32(header, root+16)
        final = u32(header, root+20)
        n = self.positions
        if (start < 0 or start + count*16 != root or not 0 < count <= getattr(self, "max_batch_slots", 4)*n or
            count % n or final not in (0,1) or header[root+24:] != b'\0'*12 or
            (payload_size is not None and payload_size != count*2048)):
            raise ValueError('invalid metadata or payload size')
        meta = np.frombuffer(header, dtype='<u4', count=count*4, offset=start).reshape(-1,n,4)
        slots = meta[:,0,1]
        if (np.any(meta[:,:,0] != 0) or np.any(meta[:,:,3] != 2048) or
            np.any(slots >= 72) or np.any(meta[:,:,1] != slots[:,None]) or
            np.any(meta[:,:,2] != self.expected_positions) or
            np.unique(slots).size != len(slots) or np.any(self.seen[slots])):
            raise ValueError('noncontiguous, duplicate or unsupported blocks')
        if final and np.count_nonzero(self.seen) + len(slots) != 72:
            raise ValueError('incomplete final batch')
        return slots, bool(final)

    def accept(self, header, payload):
        slots, final = self.validate_header(header, len(payload))
        n = self.positions
        values = np.frombuffer(payload, dtype='<u2').reshape(len(slots),n,8,128).transpose(0,2,1,3)
        self._store(slots, values)
        self.seen[slots] = True
        self.final = bool(final)
        return self.final

    def _store(self, slots, values):
        # Basic slices avoid advanced-index temporary copies when slots are contiguous.
        if np.array_equal(slots, np.arange(int(slots[0]),int(slots[0])+len(slots))):
            self.data[int(slots[0]):int(slots[-1])+1] = values
        else:
            for i,slot in enumerate(slots):self.data[slot] = values[i]


class KVViews:
    """Owns NumPy views that retain ZMQ frames; does not imply GPU visibility."""
    def __init__(self, positions):
        self.slots = [None] * 72
        self.nbytes = 72 * positions * 2048

    def tobytes(self):
        return b''.join(slot.tobytes() for slot in self.slots)

    def torch_layers(self):
        import torch
        if any(slot is None for slot in self.slots):raise ValueError('incomplete cache')
        return [(torch.from_numpy(self.slots[2*i]).view(torch.bfloat16).unsqueeze(0),
                 torch.from_numpy(self.slots[2*i+1]).view(torch.bfloat16).unsqueeze(0)) for i in range(36)]


class ViewAssembler(CanonicalAssembler):
    def __init__(self, request_id, positions, on_batch=None):
        if not 1 <= positions <= MAX_POSITIONS:raise ValueError('unsupported position count')
        self.request_id, self.positions = request_id, positions
        self.data = KVViews(positions)
        self.seen = np.zeros(72, dtype=bool)
        self.expected_positions = np.arange(positions, dtype=np.uint32)
        self.final = False
        self.on_batch = on_batch

    def _store(self, slots, values):
        if self.on_batch is not None:self.on_batch(slots,values)
        for i,slot in enumerate(slots):self.data.slots[slot] = values[i]


class LegacyAssembler:
    """Disk-free baseline retaining the original parser and per-block copying."""
    def __init__(self, request_id, positions):
        self.request_id, self.positions = request_id, positions
        self.blocks = {}
        self.data = None
        self.final = False

    def accept(self, header, payload):
        if self.final:raise ValueError('batch after final')
        message = parse_message([header, payload])
        if message['id'] != self.request_id:raise ValueError('unexpected ID')
        offset = 0
        for _,slot,block,size in message['blocks']:
            key = slot,block
            if key in self.blocks or not 0 <= block < self.positions:raise ValueError('invalid block')
            self.blocks[key] = payload[offset:offset+size]
            offset += size
        if message['final']:
            if len(self.blocks) != 72*self.positions:raise ValueError('incomplete transfer')
            self.data = np.empty((72,8,self.positions,128),dtype=np.uint16)
            for slot in range(72):
                packed = b''.join(self.blocks[slot,block] for block in range(self.positions))
                self.data[slot] = np.frombuffer(packed,dtype='<u2').reshape(self.positions,8,128).transpose(1,0,2)
            self.final = True
        return self.final


class NativeKVClient:
    def __init__(self, http_url, prefill_port, receiver_port, layout, timeout=90, *, max_batch_slots=4):
        if max_batch_slots not in (4,8): raise ValueError("max_batch_slots must be 4 or 8")
        self.max_batch_slots = max_batch_slots
        from urllib.parse import urlparse
        u = urlparse(http_url)
        if u.scheme!='http' or u.hostname!='127.0.0.1' or u.username or u.password:
            raise ValueError('loopback HTTP only')
        if not 1 <= timeout <= 600 or not all(1024 <= p <= 65535 for p in (prefill_port,receiver_port)) or prefill_port==receiver_port:
            raise ValueError('invalid ports/timeout')
        self.http_url, self.receiver_port, self.timeout = http_url.rstrip('/'),receiver_port,timeout
        self.http = requests.Session()
        self.executor = concurrent.futures.ThreadPoolExecutor(1)
        self.ctx = zmq.Context()
        self.rx, self.tx = self.ctx.socket(zmq.PULL),self.ctx.socket(zmq.PUSH)
        self.rx.setsockopt(zmq.MAXMSGSIZE,32*1024*1024)   # per frame; 8 slots x 2047 x 2048 B still fits
        self.rx.setsockopt(zmq.RCVHWM,32)
        self.tx.setsockopt(zmq.SNDTIMEO,int(timeout*1000))
        self.poisoned = False
        try:
            self.rx.bind(f'tcp://127.0.0.1:{receiver_port}')
            self.tx.connect(f'tcp://127.0.0.1:{prefill_port}')
            self.tx.send(handshake(f'127.0.0.1:{receiver_port}',layout))
        except BaseException:
            self.close()
            raise

    def prefill(self, model, tokens, mode='fast', on_batch=None):
        if getattr(self, 'max_batch_slots', 4) != 4:raise ValueError('larger groups require direct prefill_many')
        if self.poisoned:raise RuntimeError('failed client cannot be reused; restart dedicated server')
        if not 2 <= len(tokens) <= 2048:raise ValueError('prompt length must be 2..2048')
        if mode not in ('fast','legacy','views'):raise ValueError('unknown assembly mode')
        if on_batch is not None and mode!='views':raise ValueError('batch callback requires views')
        start = time.perf_counter()
        request_id = 'perf-kv-'+uuid.uuid4().hex
        assembler = (ViewAssembler(request_id,len(tokens)-1,on_batch) if mode=='views' else
                     {'fast':CanonicalAssembler,'legacy':LegacyAssembler}[mode](request_id,len(tokens)-1))
        body={'model':model,'prompt':tokens,'max_tokens':1,'min_tokens':0,'temperature':0,'ignore_eos':True,'stream':False,
              'kv_transfer_params':{'remote_decoder':f'127.0.0.1:{self.receiver_port}','id':request_id}}
        future = self.executor.submit(self.http.post,self.http_url+'/v1/completions',json=body,timeout=self.timeout)
        deadline=start+self.timeout
        ready=False;parse_s=0;batch_count=0;ready_ms=None;batch_ms=[]
        try:
            while True:
                remaining=deadline-time.perf_counter()
                if future.done():
                    # Successful prefill waits for Ack; early completion exposes HTTP failures.
                    response=future.result()
                    response.raise_for_status()
                if remaining<=0:raise TimeoutError('KV receive')
                if not self.rx.poll(max(1,min(500,int(remaining*1000)))):continue
                parts=self.rx.recv_multipart(copy=False)
                header=bytes(parts[0])
                if len(parts)==1:
                    msg=parse_message([header])
                    if ready or msg['id']!=request_id:raise ValueError('unexpected ready')
                    ready=True
                    ready_ms=(time.perf_counter()-start)*1000
                    self.tx.send(control(request_id,2,range(len(tokens)-1)))
                    continue
                if not ready or len(parts)!=2:raise ValueError('unexpected batch')
                batch_ms.append((time.perf_counter()-start)*1000)
                t=time.perf_counter()
                payload=memoryview(parts[1]) if mode!='legacy' else bytes(parts[1])
                done=assembler.accept(header,payload)
                parse_s+=time.perf_counter()-t
                batch_count+=1
                if done:break
            kv_ready=time.perf_counter()
            self.tx.send(control(request_id,3))
            response=future.result(timeout=max(0.001,deadline-time.perf_counter()))
            response.raise_for_status()
            result={'mode':mode,'input_tokens':len(tokens),'kv_ready_ms':(kv_ready-start)*1000,
                    'complete_ms':(time.perf_counter()-start)*1000,'parse_pack_ms':parse_s*1000,
                    'bytes':assembler.data.nbytes,'batches':batch_count,'ready_ms':ready_ms,
                    'batch_arrival_ms':batch_ms,'response':response.json()}
            return assembler.data,result
        except BaseException:
            self.poisoned=True
            raise

    def close(self):
        self.tx.close(linger=0);self.rx.close(linger=0);self.ctx.term()
        self.executor.shutdown(wait=True,cancel_futures=True);self.http.close()
