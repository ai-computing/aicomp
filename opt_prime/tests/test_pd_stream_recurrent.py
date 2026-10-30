"""Layer streaming cannot carry recurrent (linear-attention) state: reject it at setup, not at the first import."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pd_hybrid import _qwen35


def test_pd_batcher_rejects_streaming_worker_for_recurrent_model():
    from opt_prime.accel.pd.pd_batching import PDBatcher
    worker = SimpleNamespace(stream_layers=True, submit=lambda *a, **k: pytest.fail('must not submit'))
    with pytest.raises(ValueError, match='recurrent'):
        PDBatcher(_qwen35(), worker, slots=1, max_context=64, page_size=8)


def test_subprocess_worker_rejects_streaming_with_recurrent_state_before_spawn(monkeypatch):
    import subprocess
    from multiprocessing import shared_memory
    from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
    monkeypatch.setattr(shared_memory, 'SharedMemory', lambda *a, **k: pytest.fail('must not allocate'))
    with pytest.raises(ValueError, match='recurrent'):
        SubprocessPrefillWorker('python', [], slots=1, layers=4, kv_heads=2, head_dim=16, max_context=64,
                                stream_layers=True, recurrent_state=True)


def test_cli_rejects_streaming_for_recurrent_model_before_worker(monkeypatch):
    import opt_prime.accel.pd.pd_batching as pd
    path = Path(__file__).parents[1] / 'examples/pd/serve_batched.py'
    spec = importlib.util.spec_from_file_location('stream_recurrent_factory', path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    monkeypatch.setattr(pd, 'SubprocessPrefillWorker', lambda *a, **k: pytest.fail('must not start a producer'))
    args = SimpleNamespace(pd_prefill_stream_layers=True, backend='torch', pd_prefill_python='python',
                           pd_prefill_device=None, pd_prefill_pythonpath=None, pd_prefill_args='--engine torch',
                           model='local', max_context=64)
    with pytest.raises(ValueError, match='recurrent'):
        factory.prefill_worker(args, 1, _qwen35().config)
