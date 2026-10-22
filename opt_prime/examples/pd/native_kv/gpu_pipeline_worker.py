"""Native prefill socket owner feeding a continuous GPU lane scheduler."""
import json
import queue
import threading
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
            from memory_receiver import NativeKVClient
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            model = load_model(a.model_dir, a.device, 'marlin', fuse_mlp=a.fuse_mlp, fuse_qkv=a.fuse_qkv)
            if a.norm_mode != 'none': install_fused_norms(model, a.norm_mode)
            graph = BatchGraphDecoder(model, a.batch_size, adaptive_single=getattr(a, 'adaptive_single', False))
            tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
            eos = json.loads((Path(a.model_dir)/'generation_config.json').read_text())['eos_token_id']
            eos = {eos} if isinstance(eos, int) else set(eos)
            def prefill():
                client = None
                try:
                    client = NativeKVClient(a.native, 19201, 19202,
                        json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
                    producer_ready.set()
                    while not self.stopping.is_set():
                        if not credits.acquire(timeout=0.1): continue
                        try: job = self.jobs.get(timeout=0.1)
                        except queue.Empty:
                            credits.release()
                            continue
                        registry.add(job)
                        handed_off = False
                        started = False
                        try:
                            if self.stopping.is_set(): job.cancel.set()
                            if job.cancel.is_set(): continue
                            count = validate(job.body, a.model)
                            tokens = job.body['prompt']
                            if isinstance(tokens, str): tokens = tokenizer.encode(tokens, add_special_tokens=False)
                            if not 2 <= len(tokens) <= 2048 or len(tokens)-1+count > 2048:
                                raise ValueError('prompt/output exceeds capacity')
                            if any(t < 0 or t >= model.config.vocab_size for t in tokens):
                                raise ValueError('token out of range')
                            started = True
                            job.send('start')
                            views, _ = client.prefill(a.model, tokens, 'views')
                            if self.stopping.is_set(): job.cancel.set()
                            if job.cancel.is_set(): continue
                            # CPU packing overlaps GPU decode; the queued packet owns its bytes.
                            packed = pack_prefix(views.torch_layers())
                            del views
                            prepared.put_nowait((job, tokens, count, packed))
                            del packed
                            handed_off = True
                        except ValueError as exc:
                            job.send('error', (500, 'prefill failed') if started else (400, str(exc)))
                            if client.poisoned: raise
                        finally:
                            if not handed_off: retire(job)
                except BaseException as exc:
                    self.error = exc
                    self.stopping.set()
                    producer_ready.set()
                finally:
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
                    graph.load_packed_lane(lane, packed, tokens[-1], count)
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
                        del states[lane]
                        retire(job)
                paused = [lane for lane, state in states.items() if state[4]]
                output = graph.step(ignore_eos=True, paused=paused)
                if not output and states: time.sleep(0.001)
                for lane, token in output.items():
                    state = states[lane]
                    job, decoder, prompt_count, generated, pending = state
                    state[3] = generated = generated+1
                    delta = decoder.append(token)
                    if delta: pending.append(('text', delta))
                    stopped = token in eos and not job.body.get('ignore_eos', False)
                    if stopped: graph.cancel_lane(lane)
                    if not graph.remaining[lane]:
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
