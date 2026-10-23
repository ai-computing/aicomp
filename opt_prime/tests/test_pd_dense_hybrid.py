"""Dense subprocess KV must preserve model-layer order in grouped GPU pools (#258)."""
import pytest
import torch
from opt_prime.accel.pd.paged import PagedKV
from opt_prime.accel.pd.torch_models import KVGroup


def pool():
    groups = (KVGroup('sliding', 4, 1, 2, (0, 2)), KVGroup('full', None, 1, 2, (1,)))
    cache = PagedKV(3, 1, 1, 2, torch.float32, 'cpu', pages=4, page_size=4,
                    max_context=16, groups=groups)
    assert cache.alloc(0, 12) == 0
    return cache


def test_dense_span_maps_model_layers_at_nonzero_start():
    cache = pool()
    keys = torch.arange(30).reshape(3, 1, 5, 2).float()
    cache.write_span(0, 3, keys, keys + 100)
    idx = cache.span_index(0, 3, 8)
    for layer in range(3):
        k, v = cache.layer_kv(layer)
        torch.testing.assert_close(k[:, idx], keys[layer])
        torch.testing.assert_close(v[:, idx], keys[layer] + 100)
    assert all(not x[:, :, :3].any() for x in cache.gk)


@pytest.mark.parametrize('bad', ['missing_group', 'bad_later_shape'])
def test_invalid_grouped_span_rejected_before_any_write(bad):
    cache = pool()
    keys = [torch.ones(2, 1, 5, 2), torch.ones(1, 1, 5, 2)]
    values = [x + 1 for x in keys]
    if bad == 'missing_group':
        keys, values = keys[:1], values[:1]
    else:
        values[1] = torch.ones(1, 1, 4, 2)
    with pytest.raises(ValueError):
        cache.write_span(0, 0, keys, values)
    assert all(not x.any() for x in cache.gk + cache.gv)


def test_dense_gemma_pd_matches_independent_hf_beyond_window():
    from dataclasses import replace
    from test_pd_hybrid import _gemma3, _greedy, _prompts
    from opt_prime.accel.pd import PDRequest
    from opt_prime.accel.pd.pd_batching import TorchPrefillEngine, PrefillWorker, PDBatcher
    import copy

    class DenseEngine(TorchPrefillEngine):
        def prefill(self, request, send_from=0):
            out = super().prefill(request, send_from)
            def dense(groups):
                by_layer = {layer: span[i] for g, span in zip(self.kv.groups, groups)
                            for i, layer in enumerate(g.layers)}
                return torch.stack([by_layer[i] for i in range(len(by_layer))])
            return replace(out, keys=dense(out.keys), values=dense(out.values))

    model = _gemma3()
    engine = DenseEngine(copy.deepcopy(model), max_context=128, chunk=16, page_size=8)
    batcher = PDBatcher(model, PrefillWorker(engine), slots=2, max_context=128, page_size=8,
                        prefix_cache=False)
    prompts = _prompts(3, 25, 40, seed=258)
    futures = [batcher.submit(PDRequest(str(i), p, 20)) for i, p in enumerate(prompts)]
    batcher.run_until_idle()
    assert [f.result() for f in futures] == [_greedy(model, p, 20) for p in prompts]
