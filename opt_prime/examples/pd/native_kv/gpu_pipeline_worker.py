"""Native prefill socket owner feeding a continuous GPU lane scheduler."""
import json
import queue
import threading
import concurrent.futures
import time
from collections import deque
from pathlib import Path
from serve_gpu import Worker, TextDecoder, validate


class Registry:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.cancel = self
    def add(self, job):
        with self.lock: self.jobs[id(job)] = job
    def remove(self, job):
        with self.lock: self.jobs.pop(id(job), None)
    def count(self):
        with self.lock: return len(self.jobs)
    def set(self):
        with self.lock:
            for job in self.jobs.values(): job.cancel.set()


class PipelineWorker(Worker):
    def run(self):
        producer = None
        registry = Registry()
        self.active = registry
        a = self.args
        prepared = queue.Queue(maxsize=a.batch_size)
        credits = threading.BoundedSemaphore(a.batch_size)
        producer_ready = threading.Event()
        states = {}
        def retire(job):
            registry.remove(job)
            credits.release()
        try:
            import torch
            from transformers import AutoTokenizer
            from gpu_decoder import load_model
            from gpu_batch_decoder import BatchGraphDecoder, pack_prefix
            from gpu_norms import install_fused_norms
            from gpu_native_pack import pack_native_views
            from multiplex_receiver import MultiplexKVClient
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            model = load_model(a.model_dir, a.device, 'marlin', fuse_mlp=a.fuse_mlp, fuse_qkv=a.fuse_qkv)
            if a.norm_mode != 'none': install_fused_norms(model, a.norm_mode)
            if getattr(a, 'fused_elementwise', False):
                from gpu_activation import install_fused_swiglu, install_fused_post_attention
                install_fused_swiglu(model); install_fused_post_attention(model)
            graph = BatchGraphDecoder(model, a.batch_size, adaptive_single=getattr(a, 'adaptive_single', False))
            tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
            self.tokenizer = tokenizer
            eos = json.loads((Path(a.model_dir)/'generation_config.json').read_text())['eos_token_id']
            eos = {eos} if isinstance(eos, int) else set(eos)
            def prefill():
                client = None
                pack_pool = concurrent.futures.ThreadPoolExecutor(1) if getattr(a, 'async_pack', False) else None
                try:
                    client = MultiplexKVClient(a.native, 19201, 19202,
                        json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
                    producer_ready.set()
                    if getattr(a, 'rolling_native', False):
                        from rolling_prefill import run_rolling
                        run_rolling(self, client, tokenizer, model.config.vocab_size,
                                    credits, registry, prepared, retire)
                        return
                    credit_wait_ms = 0.0
                    wave_id = 0
                    while not self.stopping.is_set():
                        credit_start = time.perf_counter()
                        acquired = credits.acquire(timeout=0.1)
                        credit_wait_ms += (time.perf_counter()-credit_start)*1000
                        if not acquired: continue
                        try: first = self.jobs.get(timeout=0.1)
                        except queue.Empty:
                            credits.release()
                            credit_wait_ms = 0.0
                            continue
                        first.admitted_at = time.perf_counter()
                        wave_credit_wait_ms = credit_wait_ms
                        credit_wait_ms = 0.0
                        batch = [first]
                        registry.add(first)
                        deadline = time.monotonic()+getattr(a, 'native_batch_wait_ms', 2)/1000
                        limit = min(a.batch_size, getattr(a, 'native_batch_size', 1))
                        while len(batch) < limit and credits.acquire(blocking=False):
                            try: job = self.jobs.get(timeout=max(0,deadline-time.monotonic()))
                            except queue.Empty:
                                credits.release()
                                break
                            job.admitted_at = time.perf_counter()
                            batch.append(job)
                            registry.add(job)
                        valid = []
                        for job in batch:
                            if self.stopping.is_set(): job.cancel.set()
                            if job.cancel.is_set():
                                retire(job)
                                continue
                            try:
                                count = validate(job.body, a.model)
                                tokens = job.body['prompt']
                                if isinstance(tokens, str): tokens = tokenizer.encode(tokens, add_special_tokens=False)
                                if not 2 <= len(tokens) <= 2048 or len(tokens)-1+count > 2048:
                                    raise ValueError('prompt/output exceeds capacity')
                                if any(t < 0 or t >= model.config.vocab_size for t in tokens):
                                    raise ValueError('token out of range')
                            except ValueError as exc:
                                job.send('error', (400,str(exc)))
                                retire(job)
                                continue
                            job.send('start')
                            valid.append((job,tokens,count))
                        if not valid: continue
                        wave_id += 1
                        received_offsets = []
                        admitted_count = registry.count()
                        pending_count = self.jobs.qsize()
                        handed_off = set()
                        packing = []
                        def pack_one(index, views, stats, queued_at):
                            job,tokens,count = valid[index]
                            delivered = False
                            try:
                                if self.stopping.is_set(): job.cancel.set()
                                if job.cancel.is_set(): return
                                started = time.perf_counter()
                                mode = getattr(a, 'native_pack', 'torch')
                                packed = (views.host if mode == 'direct' else pack_native_views(views)
                                          if mode == 'numpy' else pack_prefix(views.torch_layers()))
                                job.stage_metrics = {'native_wave_id':wave_id, 'native_batch':len(valid), 'input_tokens':len(tokens),
                                    'admission_wait_ms':(job.admitted_at-job.submitted_at)*1000,
                                    'gather_validate_ms':(job.native_submitted_at-job.admitted_at)*1000,
                                    'wave_credit_wait_ms':wave_credit_wait_ms,
                                    'admitted_at_native_start':admitted_count,
                                    'pending_at_native_start':pending_count,
                                    'native_ready_ms':stats.get('ready_ms'), 'native_kv_ms':stats.get('kv_ready_ms'),
                                    'native_complete_ms':stats.get('complete_ms'),
                                    'cpu_pack_queue_ms':(started-queued_at)*1000,
                                    'cpu_pack_ms':(time.perf_counter()-started)*1000}
                                job.prepared_at = time.perf_counter()
                                prepared.put_nowait((job,tokens,count,packed))
                                delivered = True
                            finally:
                                if not delivered: retire(job)
                        def received(index, views, stats):
                            # Claim lifetime/credit before handing the packet to a pack worker.
                            handed_off.add(index)
                            queued_at = time.perf_counter()
                            received_offsets.append((queued_at-native_wave_start)*1000)
                            if pack_pool is None: pack_one(index,views,stats,queued_at)
                            else: packing.append(pack_pool.submit(pack_one,index,views,stats,queued_at))
                        native_wave_start = time.perf_counter()
                        for job,_,_ in valid: job.native_submitted_at = native_wave_start
                        try:
                            if len(valid) == 1 and getattr(a, 'native_pack', 'torch') != 'direct':
                                views,stats = client.prefill(a.model,valid[0][1],'views')
                                received(0,views,stats)
                                del views
                            else:
                                client.prefill_many(a.model,[tokens for _,tokens,_ in valid],received,
                                    direct=getattr(a, 'native_pack', 'torch') == 'direct')
                            for future in packing: future.result()
                            if getattr(a, 'profile_stages', None):
                                wave = {'native_wave_id':wave_id,'requests':len(valid),
                                    'wave_ms':(time.perf_counter()-native_wave_start)*1000,
                                    'credit_wait_ms':wave_credit_wait_ms,
                                    'pending_before':pending_count,'pending_after':self.jobs.qsize(),
                                    'admitted_before':admitted_count,'admitted_after':registry.count(),
                                    'callback_ms':received_offsets}
                                with open(str(a.profile_stages)+'.waves.jsonl','a') as stream:
                                    stream.write(json.dumps(wave)+'\n')
                        finally:
                            for index,(job,_,_) in enumerate(valid):
                                if index not in handed_off: retire(job)
                except BaseException as exc:
                    self.error = exc
                    self.stopping.set()
                    producer_ready.set()
                finally:
                    if pack_pool is not None: pack_pool.shutdown(wait=True)
                    if client is not None: client.close()
            producer = threading.Thread(target=prefill, name='native-prefill-owner', daemon=False)
            producer.start()
            producer_ready.wait()
            if self.error: raise self.error
            self.ready.set()
            while not self.stopping.is_set():
                free = [lane for lane in range(a.batch_size) if lane not in states]
                for lane in free:
                    try: job, tokens, count, packed = prepared.get(timeout=0.02 if not states else 0)
                    except queue.Empty: break
                    if job.cancel.is_set():
                        del packed
                        retire(job)
                        continue
                    import_start = time.perf_counter()
                    job.stage_metrics['prepared_wait_ms'] = (import_start-job.prepared_at)*1000
                    job.stage_metrics['decode_active_at_import'] = sum(n>0 for n in graph.remaining)
                    graph.load_packed_lane(lane, packed, tokens[-1], count)
                    job.stage_metrics['gpu_import_ms'] = (time.perf_counter()-import_start)*1000
                    job.stage_metrics['decode_start_ms'] = (time.perf_counter()-job.submitted_at)*1000
                    states[lane] = [job, TextDecoder(tokenizer), len(tokens), 0, deque()]
                    del packed
                for lane, (job, _, _, _, pending) in list(states.items()):
                    if job.cancel.is_set():
                        graph.cancel_lane(lane)
                        del states[lane]
                        retire(job)
                        continue
                    while pending:
                        try: job.events.put_nowait(pending[0])
                        except queue.Full: break
                        pending.popleft()
                    if not graph.remaining[lane] and not pending:
                        if getattr(a, 'profile_stages', None):
                            with open(a.profile_stages, 'a') as stream:
                                stream.write(json.dumps(job.stage_metrics)+'\n')
                        del states[lane]
                        retire(job)
                paused = [lane for lane, state in states.items() if state[4]]
                output = graph.step(ignore_eos=True, paused=paused)
                if not output and states: time.sleep(0.001)
                for lane, token in output.items():
                    state = states[lane]
                    job, decoder, prompt_count, generated, pending = state
                    state[3] = generated = generated+1
                    if generated==1:job.stage_metrics['first_gpu_token_ms']=(time.perf_counter()-job.submitted_at)*1000
                    if getattr(a, 'async_text', False): pending.append(('token', token))
                    else:
                        delta = decoder.append(token)
                        if delta:
                            if 'first_text_ready_ms' not in job.stage_metrics:job.stage_metrics['first_text_ready_ms']=(time.perf_counter()-job.submitted_at)*1000
                            pending.append(('text', delta))
                    stopped = token in eos and not job.body.get('ignore_eos', False)
                    if stopped: graph.cancel_lane(lane)
                    if not graph.remaining[lane]:
                        if not getattr(a, 'async_text', False):
                            delta = decoder.append(final=True)
                            if delta: pending.append(('text', delta))
                        pending.append(('done', {'finish_reason': 'stop' if stopped else 'length',
                            'usage': {'prompt_tokens': prompt_count, 'completion_tokens': generated,
                                      'total_tokens': prompt_count+generated}}))
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            self.error = exc
            self.stopping.set()
            self.ready.set()
        finally:
            self.stopping.set()
            registry.set()
            if producer is not None: producer.join(timeout=110)
            if producer is not None and producer.is_alive():
                self.error = RuntimeError('native prefill owner did not stop')
            while True:
                try: self.jobs.get_nowait().cancel.set()
                except queue.Empty: break
