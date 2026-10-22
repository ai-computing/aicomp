"""Experimental local BF16 prefill / Marlin decode wave worker."""
import json
import queue
import time
from collections import deque
from pathlib import Path
from serve_gpu import Worker, TextDecoder, validate
from gpu_batch_worker import ActiveWave


class LocalWorker(Worker):
    def run(self):
        try:
            import torch
            from transformers import AutoTokenizer
            from gpu_decoder import load_model
            from gpu_batch_decoder import BatchGraphDecoder
            from gpu_local_prefill import LocalPrefiller
            from gpu_norms import install_fused_norms
            a = self.args
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            model = load_model(a.model_dir, a.device, 'marlin', fuse_mlp=a.fuse_mlp, fuse_qkv=a.fuse_qkv)
            if a.norm_mode != 'none': install_fused_norms(model, a.norm_mode)
            if a.fused_elementwise:
                from gpu_activation import install_fused_swiglu, install_fused_post_attention
                install_fused_swiglu(model); install_fused_post_attention(model)
            batch_graph = BatchGraphDecoder(model, a.batch_size)
            single_graph = (BatchGraphDecoder(model, 1) if getattr(a, 'local_single_width', False)
                            and a.batch_size > 1 else batch_graph)
            graph = batch_graph
            prefill = LocalPrefiller(a.model_dir, a.device)
            tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
            self.tokenizer = tokenizer
            eos = json.loads((Path(a.model_dir)/'generation_config.json').read_text())['eos_token_id']
            eos = {eos} if isinstance(eos, int) else set(eos)
            self.ready.set()
            while not self.stopping.is_set():
                try: first = self.jobs.get(timeout=0.2)
                except queue.Empty: continue
                wave = [first]; self.active = ActiveWave(wave)
                deadline = time.monotonic()+a.native_batch_wait_ms/1000
                while len(wave) < a.batch_size:
                    try: wave.append(self.jobs.get(timeout=max(0, deadline-time.monotonic())))
                    except queue.Empty: break
                valid = []
                try:
                    for job in wave:
                        if job.cancel.is_set(): continue
                        try:
                            count = validate(job.body, a.model)
                            tokens = job.body['prompt']
                            if isinstance(tokens, str): tokens = tokenizer.encode(tokens, add_special_tokens=False)
                            if not 1 <= len(tokens) <= 2048 or len(tokens)-1+count > 2048:
                                raise ValueError('prompt/output exceeds capacity')
                            if any(t < 0 or t >= model.config.vocab_size for t in tokens):
                                raise ValueError('token out of range')
                            valid.append((job, tokens, count))
                        except ValueError as exc: job.send('error', (400, str(exc)))
                    if not valid: continue
                    graph = single_graph if len(valid) == 1 else batch_graph
                    if getattr(a, 'profile_stages', None):
                        with open(a.profile_stages, 'a') as log:
                            log.write(json.dumps({'stage':'local_wave','requests':len(valid),
                                'width':graph.batch_size,'input_lengths':[len(x[1]) for x in valid]})+'\n')
                    for job, _, _ in valid: job.send('start')
                    states = {}
                    def emit(state, token, finished):
                        job, decoder, prompt_count, generated, pending = state
                        state[3] = generated = generated+1
                        delta = decoder.append(token)
                        if delta: pending.append(('text', delta))
                        stopped = token in eos and not job.body.get('ignore_eos', False)
                        if finished or stopped:
                            delta = decoder.append(final=True)
                            if delta: pending.append(('text', delta))
                            pending.append(('done', {'finish_reason':'stop' if stopped else 'length',
                                'usage':{'prompt_tokens':prompt_count,'completion_tokens':generated,
                                         'total_tokens':prompt_count+generated}}))
                        return stopped
                    chunk_size = getattr(a, 'local_prefill_batch_size', 16)
                    for begin in range(0, len(valid), chunk_size):
                        if self.stopping.is_set(): break
                        chunk = valid[begin:begin+chunk_size]
                        first_tokens, layers, offsets, _ = prefill.prefill([x[1] for x in chunk])
                        for index, (job, tokens, count) in enumerate(chunk):
                            lane = begin+index
                            if job.cancel.is_set(): continue
                            state = [job, TextDecoder(tokenizer), len(tokens), 0, deque()]
                            states[lane] = state
                            stopped = emit(state, first_tokens[index], count == 1)
                            # Deliver this chunk's first tokens before computing the next.
                            # Never block the GPU owner on a stalled HTTP consumer.
                            while state[4]:
                                try: job.events.put_nowait(state[4][0])
                                except queue.Full: break
                                state[4].popleft()
                            if count > 1 and not stopped:
                                prefix = [(k[offsets[index]:offsets[index+1]],v[offsets[index]:offsets[index+1]]) for k,v in layers]
                                graph.load_device_lane(lane, prefix, first_tokens[index], count-1)
                    while states and not self.stopping.is_set():
                        for lane, state in list(states.items()):
                            job, _, _, _, pending = state
                            if job.cancel.is_set():
                                graph.cancel_lane(lane); del states[lane]; continue
                            while pending:
                                try: job.events.put_nowait(pending[0])
                                except queue.Full: break
                                pending.popleft()
                            if not pending and not graph.remaining[lane]: del states[lane]
                        paused = [lane for lane,state in states.items() if state[4]]
                        emitted = graph.step(ignore_eos=True, paused=paused)
                        for lane, token in emitted.items():
                            if emit(states[lane], token, not graph.remaining[lane]): graph.cancel_lane(lane)
                        if not emitted: self.stopping.wait(0.001)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    for job in wave: job.send('error', (500, 'local generation failed'))
                    self.stopping.set()
                finally:
                    for lane in range(graph.batch_size): graph.cancel_lane(lane)
                    if self.stopping.is_set(): self.active.cancel.set()
                    self.active = None
        except BaseException as exc:
            self.error = exc; self.stopping.set(); self.ready.set()
        finally:
            while True:
                try: self.jobs.get_nowait().send('error', (503, 'worker stopped'))
                except queue.Empty: break
