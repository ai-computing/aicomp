"""The public factory must make remote producer cache policy controllable."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from opt_prime.accel.pd import PDRequest
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.mark.parametrize("disabled,expected_hit", [(None, 64), (False, 64), (True, 0)])
def test_producer_cache_policy_controls_real_repeated_prefill(tmp_path, disabled, expected_hit):
    torch.set_num_threads(2)
    torch.manual_seed(241)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
    )).eval()
    model.save_pretrained(tmp_path)
    path = Path(__file__).parents[1] / "examples/pd/serve_batched.py"
    spec = importlib.util.spec_from_file_location("serve_cache_test", path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    args = SimpleNamespace(model=str(tmp_path), pd_prefill_device="cpu", dtype="float32",
                           backend="torch", device="cpu", max_context=256)
    if disabled is not None:
        args.pd_prefill_no_prefix_cache = disabled
    worker = factory.prefill_worker(args, 1, model.config)
    try:
        prompt = tuple((i % 90) + 1 for i in range(128))
        first = worker.submit(PDRequest("first", prompt, 1)).result(timeout=10)
        repeated = worker.submit(PDRequest("repeated", prompt, 1)).result(timeout=10)
        assert first.prefill_cached == 0
        assert repeated.prefill_cached == expected_hit
        assert first.token == repeated.token
        assert first.keys.shape == repeated.keys.shape == (1, 2, 128, 8)
    finally:
        worker.close()


def test_producer_chunk_controls_real_prefill_boundaries(tmp_path, monkeypatch):
    import opt_prime.accel.pd.pd_batching as pd

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
    )).eval()
    model.save_pretrained(tmp_path)
    path = Path(__file__).parents[1] / "examples/pd/serve_batched.py"
    spec = importlib.util.spec_from_file_location("serve_chunk_test", path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    spans = []
    original = pd.prefill_chunk

    def observe(model, kv, slot, ids, start, end, device):
        spans.append((start, end))
        return original(model, kv, slot, ids, start, end, device)

    monkeypatch.setattr(pd, "prefill_chunk", observe)
    args = SimpleNamespace(model=str(tmp_path), pd_prefill_device="cpu", dtype="float32",
                           backend="torch", device="cpu", max_context=256, pd_prefill_chunk=32)
    worker = factory.prefill_worker(args, 1, model.config)
    try:
        out = worker.submit(PDRequest("chunk", tuple(range(1, 65)), 1)).result(timeout=10)
        assert spans == [(0, 32), (32, 64)]
        assert out.keys.shape == (1, 2, 64, 8)
    finally:
        worker.close()
    for chunk in (0, -1):
        args.pd_prefill_chunk = chunk
        with pytest.raises(ValueError, match="positive"):
            factory.prefill_worker(args, 1, model.config)


@pytest.mark.parametrize('backend,source,target', [('torch', 'cpu', 'cpu'), ('atom', 'cuda:0', 'cpu'),
                                                   ('torch', None, 'cuda:1')])
def test_copy_overlap_rejects_unsupported_route_before_model_load(backend, source, target, monkeypatch):
    path = Path(__file__).parents[1] / 'examples/pd/serve_batched.py'
    spec = importlib.util.spec_from_file_location('serve_overlap_test', path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    args = SimpleNamespace(model='must-not-load', pd_prefill_device=source, pd_prefill_python=None,
                           backend=backend, device=target, dtype='float32', pd_prefill_copy_overlap=True)
    def unexpected_load(*args, **kwargs):
        pytest.fail('unsupported copy route reached model loading')
    monkeypatch.setattr(factory.AutoModelForCausalLM, 'from_pretrained', unexpected_load)
    with pytest.raises(ValueError, match='CUDA'):
        factory.prefill_worker(args, 1, None)


@pytest.mark.parametrize('backend,python,source', [('atom','python',None),('rngd','python',None),('torch',None,None),('torch','python','cpu')])
def test_layer_stream_rejects_unsupported_owner(backend,python,source):
    spec=importlib.util.spec_from_file_location('stream_factory',Path(__file__).parents[1]/'examples/pd/serve_batched.py')
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    args=SimpleNamespace(pd_prefill_stream_layers=True,backend=backend,pd_prefill_python=python,pd_prefill_device=source)
    with pytest.raises(ValueError,match='layer streaming'):
        factory.prefill_worker(args,1,None)


def test_layer_stream_reaches_subprocess_receiver(monkeypatch):
    import opt_prime.accel.pd.pd_batching as pd
    spec=importlib.util.spec_from_file_location('stream_factory',Path(__file__).parents[1]/'examples/pd/serve_batched.py')
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    received={}
    def worker(*args,**kw):received.update(kw);return object()
    monkeypatch.setattr(pd,'SubprocessPrefillWorker',worker)
    args=SimpleNamespace(pd_prefill_stream_layers=True,backend='torch',pd_prefill_python='python',pd_prefill_device=None,
                         pd_prefill_pythonpath=None,pd_prefill_args='--engine rngd-full',model='local',max_context=16)
    config=LlamaConfig(hidden_size=32,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=2)
    factory.prefill_worker(args,1,config)
    assert received['stream_layers'] is True
