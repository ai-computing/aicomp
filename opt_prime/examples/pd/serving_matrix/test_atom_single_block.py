from types import SimpleNamespace

import pytest
from atom_single_block import SingleBlockTarget


def model(**overrides):
    cfg = {
        "batch_size": 1,
        "kvcache_num_blocks": 1,
        "cache_impl": "static",
        "max_seq_len": 32,
        "kvcache_metas": [
            {
                "name": f"past_key_values_{i}",
                "shape": [1, 2, 32, 4],
                "dtype": "bfloat16",
                "layer_type": "full_attention",
            }
            for i in range(4)
        ],
    }
    cfg.update(overrides)
    return SimpleNamespace(rbln_config=SimpleNamespace(**cfg), model=[None, object()])


def test_fixed_cache_owns_only_block_zero_without_scheduler():
    m = model()
    t = SingleBlockTarget(m)
    assert t.lane_block == [0]
    assert t.io_rt is m.model[1]
    assert t.metas == m.rbln_config.kvcache_metas


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 2},
        {"kvcache_num_blocks": 3},
        {"cache_impl": "hybrid"},
        {"max_seq_len": 64},
        {"kvcache_metas": []},
    ],
)
def test_rejects_non_single_full_cache(overrides):
    with pytest.raises(ValueError):
        SingleBlockTarget(model(**overrides))
