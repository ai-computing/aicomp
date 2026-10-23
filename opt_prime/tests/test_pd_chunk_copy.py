"""Two-device oracle for producer chunk-copy overlap; run only on allocated idle GPUs."""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

torch = pytest.importorskip('torch')
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import TorchPrefillEngine
from test_pd_batching import _llama
from test_pd_hybrid import _gemma3

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two allocated CUDA devices')


@pytest.mark.parametrize('factory', [_llama, _gemma3], ids=['dense', 'hybrid'])
@pytest.mark.parametrize('send_from', [0, 7, 29, 35])
def test_chunk_copy_matches_whole_span_with_cached_prefix(factory, send_from):
    model = factory()
    options = dict(max_context=128, chunk=8, page_size=8, prefix_cache=True, out_device='cuda:1')
    whole = TorchPrefillEngine(copy.deepcopy(model).to('cuda:0'), **options)
    chunks = TorchPrefillEngine(copy.deepcopy(model).to('cuda:0'), copy_overlap=True, **options)
    prompt = tuple(range(1, 36))
    for repeat in range(2):
        request = PDRequest(f'copy-{repeat}', prompt, 4)
        expected = whole.prefill(request, send_from)
        actual = chunks.prefill(request, send_from)
        assert actual.token == expected.token
        assert actual.start == expected.start == send_from
        assert actual.prefill_cached == expected.prefill_cached
        if repeat:
            assert actual.prefill_cached > 0
        for got, want in [(actual.keys, expected.keys), (actual.values, expected.values)]:
            if isinstance(want, list):
                assert isinstance(got, list) and len(got) == len(want)
            pairs = zip(got, want) if isinstance(want, list) else [(got, want)]
            for left, right in pairs:
                assert left.shape[-2] == len(prompt) - send_from
                torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)


def test_compute_advances_while_copy_is_pending_and_copy_failure_releases_lane(monkeypatch):
    from opt_prime.accel.pd import pd_batching as module
    engine = TorchPrefillEngine(_llama().to('cuda:0'), max_context=128, chunk=8, page_size=8,
                                out_device='cuda:1', copy_overlap=True)
    copying, computed, release = threading.Event(), threading.Event(), threading.Event()
    original_copy, original_compute = module._ChunkCopy._copy, module.prefill_chunk

    def delayed_copy(self, spans, ready, offset):
        copying.set()
        if not release.wait(10):
            raise TimeoutError('test did not release transfer')
        original_copy(self, spans, ready, offset)
        raise RuntimeError('injected transfer failure')

    def observed_compute(*args):
        result = original_compute(*args)
        if args[4] >= 8:
            computed.set()
        return result

    monkeypatch.setattr(module._ChunkCopy, '_copy', delayed_copy)
    monkeypatch.setattr(module, 'prefill_chunk', observed_compute)
    request = PDRequest('delayed', tuple(range(1, 36)), 4)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.prefill, request)
        try:
            assert copying.wait(10)
            assert computed.wait(10), 'producer blocked on the first pending copy'
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='injected transfer failure'):
            future.result(timeout=10)
    monkeypatch.setattr(module._ChunkCopy, '_copy', original_copy)
    result = engine.prefill(request)
    assert result.keys.shape[-2] == 35
    assert result.keys.device == torch.device('cuda:1')


def test_decoder_import_retains_cross_stream_source_until_consumed():
    from opt_prime.accel.pd.paged import PagedKV
    device = torch.device('cuda:1')
    producer, consumer = torch.cuda.Stream(device), torch.cuda.Stream(device)
    kv = PagedKV(1, 1, 2, 16, torch.float32, device, pages=16, page_size=8, max_context=128)
    kv.alloc(0, 32)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(producer):
        keys = torch.full((1, 2, 32, 16), 3.0, device=device)
        values = torch.full_like(keys, 7.0)
    producer.synchronize()
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(100_000_000)
        kv.write_span(0, 0, keys, values)
    del keys, values
    with torch.cuda.stream(producer):
        overwrite = [torch.zeros((1, 2, 32, 16), device=device) for _ in range(4)]
    torch.cuda.synchronize(device)
    actual_keys, actual_values = kv.read_span(0, 0, 32)
    assert bool((actual_keys == 3).all())
    assert bool((actual_values == 7).all())
    assert len(overwrite) == 4
