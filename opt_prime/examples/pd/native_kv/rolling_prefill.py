"""Bounded native submission without a whole-wave completion barrier."""
import queue
import time
from serve_gpu import validate


def run_rolling(worker, client, tokenizer, vocab_size, credits, registry, prepared, retire, recompute=None):
    a = worker.args
    owned = {}

    def acquire():
        if worker.stopping.is_set() or not credits.acquire(blocking=False): return None
        try:
            job = worker.jobs.get_nowait()
        except queue.Empty:
            credits.release()
            return None
        registry.add(job)
        owned[id(job)] = job
        job.admitted_at = time.perf_counter()
        try:
            if job.cancel.is_set() or worker.stopping.is_set():
                job.cancel.set()
                owned.pop(id(job))
                retire(job)
                return None
            count = validate(job.body, a.model)
            tokens = job.body['prompt']
            if isinstance(tokens, str): tokens = tokenizer.encode(tokens, add_special_tokens=False)
            if not 2 <= len(tokens) <= 2048 or len(tokens)-1+count > 2048:
                raise ValueError('prompt/output exceeds capacity')
            if any(t < 0 or t >= vocab_size for t in tokens): raise ValueError('token out of range')
        except ValueError as exc:
            job.send('error', (400, str(exc)))
            owned.pop(id(job))
            retire(job)
            return None
        job.send('start')
        job.native_submitted_at = time.perf_counter()
        job.stage_metrics = {
            'native_mode':'rolling', 'input_tokens':len(tokens),
            'admission_wait_ms':(job.admitted_at-job.submitted_at)*1000,
            'gather_validate_ms':(job.native_submitted_at-job.admitted_at)*1000,
            'admitted_at_native_start':registry.count(),
            'pending_at_native_start':worker.jobs.qsize(),
        }
        if recompute is not None:
            job.prefix_length = recompute.admit(job, tokens, count)
            job.stage_metrics['prefix_length'] = job.prefix_length
        return tokens, (job, tokens, count)

    def complete(context, packet, stats):
        job, tokens, count = context
        owned.pop(id(job))
        delivered = False
        try:
            if worker.stopping.is_set(): job.cancel.set()
            if job.cancel.is_set() and recompute is None: return
            job.stage_metrics.update(native_ready_ms=stats.get('ready_ms'),
                native_kv_ms=stats.get('kv_ready_ms'), native_complete_ms=stats.get('complete_ms'),
                cpu_pack_queue_ms=0.0, cpu_pack_ms=0.0)
            job.prepared_at = time.perf_counter()
            if recompute is not None:
                recompute.native_complete(job, None if packet is None else packet.host)
            else:
                prepared.put_nowait((job, tokens, count, packet.host))
            delivered = True
        finally:
            if not delivered: retire(job)

    try:
        client.prefill_stream(a.model, acquire, complete, worker.stopping.is_set,
                             max_pending=min(a.batch_size, a.native_batch_size), direct=True,
                             **({'get_start_pos':lambda tokens, ctx:ctx[0].prefix_length}
                                if recompute is not None else {}))
    finally:
        for job in owned.values():
            if recompute is None: retire(job)
            else:
                job.cancel.set()
                recompute.native_complete(job, None)
