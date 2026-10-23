import unittest

import torch
from semantic_export import valid_prefix


class ExportTests(unittest.TestCase):
    def test_padding_is_excluded_and_last_real_hidden_is_selected(self):
        k = torch.arange(2 * 1 * 8 * 3).reshape(2, 1, 8, 3)
        v = k + 100
        h = torch.arange(8 * 4).reshape(8, 4)
        keys, values, last = valid_prefix(k, v, h, 5)
        self.assertEqual(keys.shape, (2, 1, 5, 3))
        self.assertTrue(torch.equal(keys, k[:, :, :5]))
        self.assertTrue(torch.equal(values, v[:, :, :5]))
        self.assertTrue(torch.equal(last, h[4]))
        for n in [0, 9]:
            with self.assertRaises(ValueError):
                valid_prefix(k, v, h, n)
