"""Completed handoff durations reach public metrics without scrape-side accumulation."""
import copy

import pytest

pytest.importorskip('torch')
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import PDBatcher, PrefillWorker, TorchPrefillEngine
from opt_prime.accel.serve.batched import BatchedService
from test_pd_batching import _llama, _prompts


def test_completed_handoff_timings_are_exported_once():
    model = _llama()
    engine = TorchPrefillEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=16,
                                prefix_cache=False)
    original = engine.prefill

    def measured(request, send_from=0):
        result = original(request, send_from)
        assert result.prefill_s > 0
        # Deterministic worker-reported durations; real prefill/KV and admission remain exercised.
        result.prefill_s, result.transfer_s = .25, .125
        return result

    engine.prefill = measured
    worker = PrefillWorker(engine)
    batcher = PDBatcher(model, worker, slots=1, max_context=128, page_size=16, prefix_cache=False)
    service = BatchedService(batcher, model='m', vocab_size=97, max_output=8, render=str)
    try:
        for i, prompt in enumerate(_prompts(2, 17, 18)):
            future = batcher.submit(PDRequest(str(i), prompt, 2))
            batcher.run_until_idle()
            assert len(future.result()) == 2
        for _ in range(2):
            text = service.metrics.render(*service.gauges())
            assert 'airm_pd_producer_seconds_total{model_name="m",stage="prefill"} 0.5' in text
            assert 'airm_pd_producer_seconds_total{model_name="m",stage="transfer"} 0.25' in text
        del batcher.stats['transfer_s']
        text = service.metrics.render(*service.gauges())
        assert 'airm_pd_producer_seconds_total{model_name="m",stage="prefill"} 0.5' in text
        assert 'stage="transfer"' not in text
        del batcher.stats['prefill_s']
        assert 'airm_pd_producer_seconds_total' not in service.metrics.render(*service.gauges())
    finally:
        worker.close()
