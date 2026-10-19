"""Persistent inputs preserve scalar values and cache identity across steps."""

import unittest
from types import SimpleNamespace
import numpy as np
import torch
from opt_prime.pd.npu.atom_runtime import QwenExternalDecoder


class InputReuseTests(unittest.TestCase):
    def engine(self):
        e = QwenExternalDecoder.__new__(QwenExternalDecoder)
        e.reuse_inputs = True
        e._step_inputs = None
        e.profile_steps = False
        e.context = 8
        e.spec = SimpleNamespace(hidden=4, head_dim=4)
        e.embed = np.arange(40, dtype=np.float32).reshape(10, 4)
        e.inv = np.array([1, 0.01], dtype=np.float32)
        e.torch, e.device = torch, "cpu"
        e.caches = [torch.zeros(1) for _ in range(4)]

        def runner(*args, out=None):
            self.assertIs(args[4], e.caches[0])
            self.assertIs(args[5], e.caches[1])
            self.assertIs(args[8], e.caches[2])
            return [args[0].clone()]

        e.runner = runner
        return e

    def test_buffers_reused_with_changed_values(self):
        e = self.engine()
        np.testing.assert_array_equal(e.step_batch([2], [3]), e.embed[[2]])
        pointers = [x.data_ptr() for x in e._step_inputs]
        np.testing.assert_array_equal(e.step_batch([4], [5]), e.embed[[4]])
        self.assertEqual(pointers, [x.data_ptr() for x in e._step_inputs])
        np.testing.assert_array_equal(
            e._step_inputs[1].numpy(), np.cos(np.array([[5]], np.float32) * e.inv)
        )
        np.testing.assert_array_equal(
            e._step_inputs[3].numpy(), [[1, 1, 1, 1, 1, 1, 0, 0]]
        )
        self.assertEqual(e._step_inputs[4].item(), 5)
        self.assertEqual(e._step_inputs[5].item(), 0)

    def test_invalid_inputs_fail_before_allocating(self):
        e = self.engine()
        for tokens, pos in [
            ([-1], [0]),
            ([10], [0]),
            ([0], [8]),
            ([0, 1], [0, 1]),
            ([1.5], [0]),
        ]:
            with self.assertRaises(ValueError):
                e.step_batch(tokens, pos)
            self.assertIsNone(e._step_inputs)
