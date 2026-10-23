"""Bounded concurrent native requests on one owned ZMQ receiver, keyed by request ID."""
import concurrent.futures
import time
import uuid
import requests
import zmq
from memory_receiver import NativeKVClient, ViewAssembler
from protocol import control, parse_message, read_string, u32, MAX_HEADER


class MultiplexKVClient(NativeKVClient):
    def prefill_many(self, model, prompts, on_complete=None, *, direct=False, start_pos=0):
        if self.poisoned: raise RuntimeError('failed client cannot be reused')
        if not 1 <= len(prompts) <= 16: raise ValueError('expected 1..16 prompts')
        if any(not 2 <= len(tokens) <= 2048 for tokens in prompts):
            raise ValueError('prompt length must be 2..2048')
        if type(start_pos) is not int or any(not 0 <= start_pos < len(t)-1 for t in prompts):
            raise ValueError('invalid shared source prefix offset')
        remaining = iter(enumerate(prompts))
        exhausted = False
        results = [None] * len(prompts)
        def acquire():
            nonlocal exhausted
            item = next(remaining, None)
            if item is None: exhausted = True
            return None if item is None else (item[1], item[0])
        def complete(index, data, stats):
            results[index] = (data, stats)
            if on_complete is not None: on_complete(index, data, stats)
        self.prefill_stream(model, acquire, complete, lambda: exhausted,
                            max_pending=len(prompts), direct=direct, start_pos=start_pos)
        return results

    def prefill_stream(self, model, acquire, on_complete, stopped, *, max_pending=8, direct=False, start_pos=0,
                       get_start_pos=None):
        """Continuously replenish bounded requests from nonblocking acquire().

        acquire() returns (tokens, context) or None. Stop admission when stopped()
        is true, then drain pending transfers. Deadlines are per request; the
        callback owns completed data and the caller owns all acquired contexts.
        get_start_pos(tokens, context), when supplied, selects and snapshots each
        request's source offset before HTTP submission; do not also set start_pos.
        """
        if self.poisoned: raise RuntimeError('failed client cannot be reused')
        if type(start_pos) is not int or not 0 <= start_pos < 2047:
            raise ValueError('invalid shared source prefix offset')
        if get_start_pos is not None and (not callable(get_start_pos) or start_pos != 0):
            raise ValueError('per-request offset requires a callable and shared start_pos=0')
        if not 1 <= max_pending <= 16: raise ValueError('expected 1..16 pending requests')
        if direct:
            from pinned_receiver import PinnedAssembler
            if not hasattr(self.rx, 'recv_into'): raise RuntimeError('direct receive requires pyzmq recv_into')
        pending = {}
        pool = concurrent.futures.ThreadPoolExecutor(max_pending)
        try:
            while pending or not stopped():
                while len(pending) < max_pending and not stopped():
                    item = acquire()
                    if item is None: break
                    tokens, context = item
                    if not 2 <= len(tokens) <= 2048: raise ValueError('prompt length must be 2..2048')
                    offset = start_pos if get_start_pos is None else get_start_pos(tokens, context)
                    if type(offset) is not int or not 0 <= offset < len(tokens)-1:
                        raise ValueError('prefix offset must be an integer leaving a suffix')
                    positions = len(tokens)-1-offset
                    start = time.perf_counter()
                    rid = 'multi-kv-'+uuid.uuid4().hex
                    body = {'model':model, 'prompt':tokens, 'max_tokens':1, 'min_tokens':0,
                        'temperature':0, 'ignore_eos':True, 'stream':False,
                        'kv_transfer_params':{'remote_decoder':f'127.0.0.1:{self.receiver_port}', 'id':rid}}
                    future = pool.submit(requests.post, self.http_url+'/v1/completions', json=body, timeout=self.timeout)
                    pending[rid] = {'index':context, 'start':start, 'n':positions, 'start_pos':offset, 'future':future,
                        'assembler':(PinnedAssembler(rid,positions) if direct else ViewAssembler(rid,positions)),
                        'ready':False, 'done':False, 'ready_ms':None, 'kv_ready_ms':None, 'batches':0}
                for rid, state in list(pending.items()):
                    if time.perf_counter()-state['start'] >= self.timeout:
                        raise TimeoutError('multiplex KV receive')
                    if not state['future'].done(): continue
                    response = state['future'].result()
                    response.raise_for_status()
                    if not state['done']: continue
                    stats = {k:state[k] for k in ('ready_ms','kv_ready_ms','batches')}
                    stats.update(start_pos=state['start_pos'],exported_positions=state['n'],complete_ms=(time.perf_counter()-state['start'])*1000, response=response.json())
                    result = (state['assembler'].data, stats)
                    if on_complete is not None: on_complete(state['index'], *result)
                    del pending[rid]
                if not pending:
                    if not stopped(): time.sleep(0.001)
                    continue
                if not self.rx.poll(1): continue
                if direct:
                    header = bytes(self.rx.recv(copy=False))
                    part_count = 2 if self.rx.getsockopt(zmq.RCVMORE) else 1
                else:
                    parts = self.rx.recv_multipart(copy=False)
                    header = bytes(parts[0])
                    part_count = len(parts)
                if part_count == 1:
                    msg = parse_message([header])
                    state = pending.get(msg['id'])
                    if state is None or state['ready'] or state['done']: raise ValueError('unexpected ready ID')
                    state['ready'] = True
                    state['ready_ms'] = (time.perf_counter()-state['start'])*1000
                    self.tx.send(control(msg['id'],2,range(state['n']),start_pos=state['start_pos']))
                elif part_count == 2:
                    # Only route here; assembler validates the complete header and payload.
                    if not 36 <= len(header) <= MAX_HEADER or len(header)%4 or u32(header,len(header)-36)!=1:
                        raise ValueError('invalid batch envelope')
                    rid = read_string(header,len(header)-32)
                    state = pending.get(rid)
                    if state is None or not state['ready'] or state['done']: raise ValueError('unexpected batch ID')
                    state['batches'] += 1
                    done = (state['assembler'].receive(header,self.rx) if direct
                            else state['assembler'].accept(header,memoryview(parts[1])))
                    if done:
                        state['done'] = True
                        state['kv_ready_ms'] = (time.perf_counter()-state['start'])*1000
                        self.tx.send(control(rid,3))
                else: raise ValueError('invalid multipart count')
        except BaseException:
            self.poisoned = True
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
