"""Bounded concurrent native requests on one owned ZMQ receiver, keyed by request ID."""
import concurrent.futures
import time
import uuid
import requests
from memory_receiver import NativeKVClient, ViewAssembler
from protocol import control, parse_message, read_string, u32, MAX_HEADER


class MultiplexKVClient(NativeKVClient):
    def prefill_many(self, model, prompts, on_complete=None):
        if self.poisoned: raise RuntimeError('failed client cannot be reused')
        if not 1 <= len(prompts) <= 16: raise ValueError('expected 1..16 prompts')
        if any(not 2 <= len(tokens) <= 2048 for tokens in prompts):
            raise ValueError('prompt length must be 2..2048')
        start = time.perf_counter()
        pending = {}
        results = [None] * len(prompts)
        pool = concurrent.futures.ThreadPoolExecutor(len(prompts))
        try:
            for index, tokens in enumerate(prompts):
                rid = 'multi-kv-'+uuid.uuid4().hex
                body = {'model':model, 'prompt':tokens, 'max_tokens':1, 'min_tokens':0,
                    'temperature':0, 'ignore_eos':True, 'stream':False,
                    'kv_transfer_params':{'remote_decoder':f'127.0.0.1:{self.receiver_port}', 'id':rid}}
                # Each concurrent HTTP call owns its session; ZMQ stays on this thread.
                future = pool.submit(requests.post, self.http_url+'/v1/completions', json=body, timeout=self.timeout)
                pending[rid] = {'index':index, 'n':len(tokens)-1, 'future':future,
                    'assembler':ViewAssembler(rid,len(tokens)-1), 'ready':False, 'done':False,
                    'ready_ms':None, 'kv_ready_ms':None, 'batches':0}
            while pending:
                if time.perf_counter()-start >= self.timeout: raise TimeoutError('multiplex KV receive')
                for rid, state in list(pending.items()):
                    if not state['future'].done(): continue
                    response = state['future'].result()
                    response.raise_for_status()
                    if not state['done']: continue
                    stats = {k:state[k] for k in ('ready_ms','kv_ready_ms','batches')}
                    stats.update(complete_ms=(time.perf_counter()-start)*1000, response=response.json())
                    result = (state['assembler'].data, stats)
                    results[state['index']] = result
                    if on_complete is not None: on_complete(state['index'], *result)
                    del pending[rid]
                if not pending: break
                if not self.rx.poll(1): continue
                parts = self.rx.recv_multipart(copy=False)
                header = bytes(parts[0])
                if len(parts) == 1:
                    msg = parse_message([header])
                    state = pending.get(msg['id'])
                    if state is None or state['ready'] or state['done']: raise ValueError('unexpected ready ID')
                    state['ready'] = True
                    state['ready_ms'] = (time.perf_counter()-start)*1000
                    self.tx.send(control(msg['id'],2,range(state['n'])))
                elif len(parts) == 2:
                    # Only route here; assembler validates the complete header and payload.
                    if not 36 <= len(header) <= MAX_HEADER or len(header)%4 or u32(header,len(header)-36)!=1:
                        raise ValueError('invalid batch envelope')
                    rid = read_string(header,len(header)-32)
                    state = pending.get(rid)
                    if state is None or not state['ready'] or state['done']: raise ValueError('unexpected batch ID')
                    state['batches'] += 1
                    if state['assembler'].accept(header,memoryview(parts[1])):
                        state['done'] = True
                        state['kv_ready_ms'] = (time.perf_counter()-start)*1000
                        self.tx.send(control(rid,3))
                else: raise ValueError('invalid multipart count')
            return results
        except BaseException:
            self.poisoned = True
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
