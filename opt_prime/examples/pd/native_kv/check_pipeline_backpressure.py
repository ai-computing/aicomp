"""Real worker test: leave one event consumer stalled while its peer completes."""
import argparse
import json
import queue
import time
from types import SimpleNamespace
from serve_gpu import Job
from gpu_pipeline_worker import PipelineWorker

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir', required=True)
p.add_argument('--out', required=True)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--adaptive-single', action='store_true')
a = p.parse_args()
args = SimpleNamespace(model_dir=a.model_dir, device='cuda:0', model='Qwen/Qwen3-8B-FP8',
    native='http://127.0.0.1:19101', norm_mode='fused', fuse_mlp=True, fuse_qkv=True, batch_size=a.batch_size, adaptive_single=a.adaptive_single)
worker = PipelineWorker(args)
body = {'model':args.model, 'prompt':[1000]*128, 'max_tokens':128,
        'temperature':0, 'ignore_eos':True}
slow, fast = Job(body), Job(body)
def consume(job):
    text = ''
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        kind, value = job.events.get(timeout=10)
        if kind == 'error': raise RuntimeError(value)
        if kind == 'text': text += value
        if kind == 'done': return text, value
    raise TimeoutError('consumer')
try:
    worker.submit(slow)
    worker.submit(fast)
    fast_text, fast_done = consume(fast)
    stalled_events = slow.events.qsize()
    assert stalled_events == slow.events.maxsize, stalled_events
    slow_text, slow_done = consume(slow)
    if slow_text != fast_text:
        open(a.out, 'w').write(json.dumps({'texts_equal':False,'slow':slow_text,'fast':fast_text},indent=2)+'\n')
        raise AssertionError('stalled/resumed text differs from peer')
    assert slow_done['usage']['completion_tokens'] == fast_done['usage']['completion_tokens'] == 128
    # A cancelled non-consuming request must not hold its credit or prevent reuse.
    cancelled = Job(body)
    worker.submit(cancelled)
    cancelled.events.get(timeout=10)
    cancelled.cancel.set()
    followup = Job(body)
    worker.submit(followup)
    follow_text, follow_done = consume(followup)
    assert follow_text == fast_text
    assert follow_done['usage']['completion_tokens'] == 128
    result = {'stalled_events':stalled_events, 'peer_tokens':128, 'resumed_tokens':128,
              'texts_equal':True, 'cancel_followup_tokens':128}
    open(a.out, 'w').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
finally:
    worker.close()
