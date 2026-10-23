"""Retain exact short-prompt live RNGD KV for numerical diagnosis; no GPU work."""
import json
import time
from pathlib import Path
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
root = Path(__file__).resolve().parent
output = root/'koni261-short-kv.pt'
summary = root/'koni261-short-export.json'
if output.exists() or summary.exists():
    raise FileExistsError('Refusing to overwrite KONI short export')
data = json.loads((root/'local201-koni-matrix-a.json').read_text())
ids = data['prompts']['128']
assert len(ids) == 128
torch.set_num_threads(4)
started = time.monotonic()
worker = SubprocessPrefillWorker('/tmp/rngd-gpu201-src/python-worker.sh',
    ['--model', '/tmp/serving-matrix-models/KONI-8B', '--width', '8192', '--engine-factory', 'koni_prefill:Prefill'],
    slots=1, layers=32, kv_heads=8, head_dim=128, max_context=9216)
loaded = time.monotonic() - started
try:
    retained = None
    times = []
    for i in range(2):
        begin = time.monotonic()
        out = worker.submit(PDRequest(f'export-{i}', tuple(ids), 128), slot=0).result(timeout=300)
        times.append(time.monotonic()-begin)
        assert out.start == 0 and out.keys.shape == out.values.shape == (32, 8, 128, 128)
        current = dict(ids=ids, token=out.token, keys=out.keys.clone(), values=out.values.clone())
        if retained is None:
            retained = current
        else:
            assert current['token'] == retained['token']
            assert torch.equal(current['keys'], retained['keys']) and torch.equal(current['values'], retained['values'])
    torch.save(retained, output)
    summary.write_text(json.dumps(dict(complete=True, repeat_exact=True, token=retained['token'],
        shape=list(retained['keys'].shape), dtype=str(retained['keys'].dtype), load_s=loaded,
        observed_prefill_export_s=times, width=8192, logical_prompt=128,
        runtime='32ce5d5', revision='0172bf7e73977ae213613529f3d2d5400a33f678',
        comparison='Two owned copies from the same live producer; GPU replay not yet checked'), indent=2)+'\n')
finally:
    worker.close()
