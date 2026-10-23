import unittest
from types import SimpleNamespace

import torch
from hybrid_tables import install


class HybridTablesTest(unittest.TestCase):
    def batcher(self):
        recorder = lambda **kw: kw
        recorder.dec_attn_mask = torch.ones(2, 16)
        return SimpleNamespace(
            lanes=2,
            worker=None,
            model=SimpleNamespace(
                rbln_config=SimpleNamespace(cache_impl="hybrid"),
                prefill_decoder=recorder,
            ),
            decoder=recorder,
            blocks=SimpleNamespace(prefix_cache=False, host_bytes=0, lane_block=[2, 0]),
        )

    def test_lane_local_tables_are_distinct_from_global_blocks(self):
        b = self.batcher()
        install(b)
        mask = b.model.prefill_decoder.dec_attn_mask
        pre = b._prefill(1, [11, 12])
        self.assertEqual(mask[1].tolist(), [1, 1] + [0] * 14)
        self.assertEqual(mask[0].tolist(), [1] * 16)
        self.assertEqual(pre["block_tables"].tolist(), [0])
        self.assertEqual(pre["local_block_tables"].tolist(), [1])
        out = b.decoder(
            cache_position=torch.tensor([[2], [0]]),
            block_tables=torch.tensor([[2], [3]]),
        )
        self.assertEqual(out["local_block_tables"].tolist(), [[0], [1]])
        with self.assertRaises(ValueError):
            b._prefill(0, [11, 12], 1)

    def test_rejects_unrestored_sliding_cache_modes(self):
        for key, value in [("prefix_cache", True), ("host_bytes", 1024)]:
            b = self.batcher()
            setattr(b.blocks, key, value)
            with self.assertRaises(ValueError):
                install(b)
        b = self.batcher()
        b.worker = object()
        with self.assertRaises(ValueError):
            install(b)
