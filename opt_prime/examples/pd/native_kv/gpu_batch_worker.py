"""Experimental wave batching; native prefill remains serialized on its socket owner."""
import json
import queue
import time
from collections import deque
from pathlib import Path
from serve_gpu import Worker, TextDecoder, validate

# The single GPU thread owns every lane, so it never blocks on one HTTP consumer:
# a lane whose event queue is full is paused, and cancelled if it stays full this long.
STALL_SECONDS = 1.0


class ActiveWave:
    """Worker.close can cancel the entire owned wave through the existing interface."""
    def __init__(self, jobs):
        self.jobs = jobs
        self.cancel = self
    def set(self):
        for job in self.jobs: job.cancel.set()


class BatchWorker(Worker):
    def run(self):
        client = None
        try:
            import torch
            from transformers import AutoTokenizer
            from gpu_decoder import load_model
            from gpu_batch_decoder import BatchGraphDecoder
            from gpu_norms import install_fused_norms
            from memory_receiver import NativeKVClient
            a = self.args
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            model = load_model(a.model_dir, a.device, 'marlin', fuse_mlp=a.fuse_mlp, fuse_qkv=a.fuse_qkv)
            if a.norm_mode != 'none': install_fused_norms(model, a.norm_mode)
            if getattr(a, 'fused_elementwise', False):
                from gpu_activation import install_fused_swiglu, install_fused_post_attention
                install_fused_swiglu(model); install_fused_post_attention(model)
            graph = BatchGraphDecoder(model, a.batch_size)
            tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
            eos = json.loads((Path(a.model_dir)/'generation_config.json').read_text())['eos_token_id']
            eos = {eos} if isinstance(eos, int) else set(eos)
            client = NativeKVClient(a.native, 19201, 19202,
                json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
            self.ready.set()
            while not self.stopping.is_set():
                try: first = self.jobs.get(timeout=0.2)
                except queue.Empty: continue
                wave = [first]
                self.active = ActiveWave(wave)
                deadline = time.monotonic()+0.002
                while len(wave) < a.batch_size:
                    try: wave.append(self.jobs.get(timeout=max(0, deadline-time.monotonic())))
                    except queue.Empty: break
                states = {}
                try:
                    for lane, job in enumerate(wave):
                        if self.stopping.is_set(): job.cancel.set()
                        if job.cancel.is_set(): continue
                        started = False
                        try:
                            body = job.body
                            count = validate(body, a.model)
                            tokens = body['prompt']
                            if isinstance(tokens, str): tokens = tokenizer.encode(tokens, add_special_tokens=False)
                            if not 2 <= len(tokens) <= 2048 or len(tokens)-1+count > 2048:
                                raise ValueError('prompt/output exceeds capacity')
                            if any(t < 0 or t >= model.config.vocab_size for t in tokens):
                                raise ValueError('token out of range')
                            started = True
                            job.send('start')
                            views, _ = client.prefill(a.model, tokens, 'views')
                            if job.cancel.is_set(): continue
                            graph.load_lane(lane, views.torch_layers(), tokens[-1], count)
                            states[lane] = [job, TextDecoder(tokenizer), len(tokens), 0, deque(), None]
                            # HTTP handler arrivals can exceed the short admission window.
                            # Fill free lanes from arrivals during the first native prefill.
                            while len(wave) < a.batch_size:
                                try: wave.append(self.jobs.get_nowait())
                                except queue.Empty: break
                        except ValueError as exc:
                            job.fail(*((500, 'generation failed') if started else (400, str(exc))))
                            if client.poisoned: raise
                    while states and not self.stopping.is_set():
                        now = time.monotonic()
                        for lane, state in list(states.items()):
                            job, pending = state[0], state[4]
                            while pending and job.try_send(*pending[0]): pending.popleft()
                            if not pending: state[5] = None
                            elif state[5] is None: state[5] = now
                            elif now-state[5] > STALL_SECONDS: job.cancel.set()
                            if job.cancel.is_set():
                                graph.cancel_lane(lane)
                                del states[lane]
                            elif not pending and not graph.remaining[lane]:
                                del states[lane]
                        paused = [lane for lane, state in states.items() if state[4]]
                        # EOS is per request; the graph handles count retirement.
                        emitted = graph.step(ignore_eos=True, paused=paused)
                        for lane, token in emitted.items():
                            state = states[lane]
                            job, decoder, prompt_count, generated, pending, _ = state
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
                        if not emitted: self.stopping.wait(0.001)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    # Never block here: a stalled consumer must not keep the worker alive.
                    for job in wave: job.fail(500, 'batch generation failed')
                    # Unknown CUDA/native failures require a fresh worker, not reuse.
                    self.stopping.set()
                finally:
                    for lane in range(a.batch_size): graph.cancel_lane(lane)
                    if self.stopping.is_set(): self.active.cancel.set()
                    self.active = None
        except BaseException as exc:
            self.error = exc
            self.stopping.set()
            self.ready.set()
        finally:
            if client is not None: client.close()
            while True:
                try: self.jobs.get_nowait().fail(503, 'worker stopped')
                except queue.Empty: break
