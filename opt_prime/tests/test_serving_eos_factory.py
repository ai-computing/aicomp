"""The public factory must honor model EOS and tokenizer chat turn end."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "configured,tokenizer_eos,expected",
    [
        (8, 9, (8, 9)),
        ([8, 9], 9, (8, 9)),
        ((8, 10), 9, (8, 10, 9)),
        (None, 9, (9,)),
        (8, None, (8,)),
        (None, None, ()),
    ],
)
def test_factory_preserves_all_eos_and_ignore_override(
    monkeypatch, configured, tokenizer_eos, expected
):
    path = Path(__file__).parents[1] / "examples/pd/serve_batched.py"
    spec = importlib.util.spec_from_file_location("serve_eos_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = SimpleNamespace(
        eos_token_id=tokenizer_eos, decode=lambda ids, **kw: str(ids)
    )
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=configured),
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=32)),
    )
    model.to = lambda device: model
    model.eval = lambda: model
    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    monkeypatch.setattr(
        module.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model
    )
    monkeypatch.setattr(
        module,
        "ContinuousBatcher",
        lambda *a, **kw: SimpleNamespace(max_context=64, paged=False),
    )
    monkeypatch.setattr(module, "prefill_worker", lambda *a: None)
    monkeypatch.setattr(module, "remote_tier", lambda *a: None)
    args = SimpleNamespace(
        model="test",
        kv_events=None,
        backend="torch",
        dtype="float32",
        device="cpu",
        slots=1,
        max_context=64,
        kv="slot",
        page_size=16,
        prefix_cache=False,
        kv_host_memory_gb=0,
        reload_bandwidth=1,
        prefill_rate=1,
        chunk=16,
        max_output=16,
    )
    service = module.build(args).service
    request, _ = service._request(
        "r", {"model": "test", "prompt": [1], "max_tokens": 4}
    )
    assert request.sampling.stop_token_ids == expected
    request, _ = service._request(
        "r2", {"model": "test", "prompt": [1], "max_tokens": 4, "ignore_eos": True}
    )
    assert request.sampling.stop_token_ids == ()
