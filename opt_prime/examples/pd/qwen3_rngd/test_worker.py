"""Shared-memory boundary: wrong offsets/lanes must not overwrite another request."""

import unittest

import numpy as np
import pytest
import torch
import worker


class WorkerContract(unittest.TestCase):
    def test_suffix_is_compacted_into_owned_slot(self):
        slab = np.full((2, 2, 1, 1, 4, 2), 65535, dtype=np.uint16)
        k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=torch.bfloat16)
        req = {"slot": 1, "tokens": [10, 20, 30], "send_from": 1}
        n = worker.store_result(slab, req, k, -k)
        self.assertEqual(n, 2)
        got = torch.from_numpy(slab[1, 0, :, :, :2].copy().view(np.int16)).view(
            torch.bfloat16
        )
        self.assertTrue(
            torch.equal(
                got, torch.tensor([[[[3.0, 4.0], [5.0, 6.0]]]], dtype=torch.bfloat16)
            )
        )
        self.assertTrue((slab[0] == 65535).all())
        self.assertTrue((slab[1, :, :, :, 2:] == 65535).all())

    def test_invalid_requests_leave_slab_untouched(self):
        slab = np.full((1, 2, 1, 1, 4, 2), 65535, dtype=np.uint16)
        k = torch.zeros((1, 1, 3, 2), dtype=torch.bfloat16)
        base = {"slot": 0, "tokens": [10, 20, 30], "send_from": 0}
        for patch in [
            {"slot": -1},
            {"slot": 1},
            {"send_from": -1},
            {"send_from": 4},
            {"tokens": []},
        ]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                worker.store_result(slab, {**base, **patch}, k, k)
            self.assertTrue((slab == 65535).all())
        with self.assertRaises(ValueError):
            worker.store_result(slab, base, k, torch.zeros(1))
        self.assertTrue((slab == 65535).all())


if __name__ == "__main__":
    unittest.main()


def test_explicit_engine_factory_is_local_callable():
    import sys
    import types
    from unittest.mock import patch

    from worker import engine_class

    module = types.ModuleType("test_local_prefill")

    class Engine:
        def __init__(self, config, weights, width, chip):
            self.args = config, weights, width, chip

    module.Prefill = Engine
    with patch.dict(sys.modules, {"test_local_prefill": module}):
        cls = engine_class("test_local_prefill:Prefill")
        assert cls({}, {}, width=128, chip=0).args == ({}, {}, 128, 0)
        for invalid in ["test_local_prefill", ":Prefill", "test_local_prefill:"]:
            with pytest.raises(ValueError):
                engine_class(invalid)


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            {
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "num_attention_heads": 32,
            },
            (32, 8, 128),
        ),
        (
            {
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "hidden_size": 2560,
                "num_attention_heads": 32,
                "head_dim": 128,
            },
            (36, 8, 128),
        ),
        (
            {
                "num_hidden_layers": 26,
                "num_key_value_heads": 1,
                "hidden_size": 1152,
                "num_attention_heads": 4,
                "head_dim": 256,
            },
            (26, 1, 256),
        ),
    ],
)
def test_cache_geometry_preserves_explicit_head_dim(config, expected):
    assert worker.cache_geometry(config) == expected
