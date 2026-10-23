"""Ownership and mask checks for the bounded native decode adapter."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import pytest  # noqa: E402

pytest.importorskip("rngd_pd")   # optional: the RNGD/ATOM PD runtime package

from opt_prime.accel.platforms.atom.pd_native import NativeAtomDecodeLane


class NativeLaneTests(unittest.TestCase):
    def lane(self):
        config = dict(
            batch_size=1,
            max_seq_len=512,
            kvcache_num_blocks=1,
            kvcache_block_size=512,
            cache_impl="static",
            dtype="bfloat16",
            kvcache_metas=[
                dict(name="past_key_values_0", shape=[1, 1, 512, 64], dtype="bfloat16")
            ],
        )
        runtime = SimpleNamespace(
            get_kv_cache_size=lambda: 65536,
            get_kv_cache_size_for_layer=lambda name: 65536,
            set_kv_cache_for_layer=lambda *args: None,
        )

        class Decoder:
            dec_attn_mask = torch.zeros(1, 1, 1, 512)

            def __call__(self, **kwargs):
                self.last = kwargs
                return SimpleNamespace(logits=torch.zeros(1, 1, 4))

        decoder = Decoder()
        model = SimpleNamespace(model={1: runtime}, decoders={1: decoder})
        fake = SimpleNamespace(
            aligned_tensor=lambda n, **kw: np.zeros(n, dtype=np.uint16)
        )
        with patch.dict(sys.modules, {"rebel.kv_cache": fake}), patch(
            "importlib.metadata.version",
            side_effect=lambda p: "0.11.0" if p == "rebel-compiler" else "0.11.0.post1",
        ):
            lane = NativeAtomDecodeLane(
                model,
                config,
                SimpleNamespace(caches=1, kv_heads=1, head_dim=64, vocabulary=4),
            )
        return lane

    def test_mask_reset_after_longer_request(self):
        lane = self.lane()
        kv = np.zeros((1, 1, 512, 64), dtype=np.float32)
        lane.generate([1] * 20, kv, 2)
        lane.generate([1] * 2, kv, 1)
        self.assertEqual(int(lane.mask.sum()), 2)
        self.assertEqual(lane.state, "idle")
        self.assertIsNone(lane._owned)
        self.assertEqual(lane.blocks.tolist(), [[0]])

    def test_write_failure_poisoned_and_buffers_retained(self):
        lane = self.lane()

        def fail(*args):
            raise RuntimeError("write failed")

        lane.runtime.set_kv_cache_for_layer = fail
        kv = np.zeros((1, 1, 512, 64), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            lane.generate([1, 2], kv, 1)
        self.assertEqual(lane.state, "poisoned")
        self.assertIsNotNone(lane._owned)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            lane.generate([1, 2], kv, 1)

    def test_invalid_kv_never_admitted(self):
        lane = self.lane()
        kv = np.full((1, 1, 512, 64), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            lane.generate([1, 2], kv, 1)
        self.assertEqual(lane.state, "idle")
