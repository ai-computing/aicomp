import unittest
import torch
from opt_prime.pd.npu.packing import CompactPrefixPacker
from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker, PackKVPayload


class CompactTests(unittest.TestCase):
    def packer(self, cls, dim):
        p = cls.__new__(cls)
        p.layers, p.heads, p.head_dim, p.device = 2, 2, dim, "cpu"
        p.module = PackKVPayload()
        return p

    def test_identical_payload_with_short_storage(self):
        for dim in (64, 128):
            torch.manual_seed(7)
            k = torch.randn(2, 2, dim, 64).bfloat16()
            v = torch.randn(2, 2, 64, dim).bfloat16()
            fullk = torch.cat((k, torch.zeros(2, 2, dim, 448).bfloat16()), -1)
            fullv = torch.cat((v, torch.zeros(2, 2, 448, dim).bfloat16()), -2)
            for tokens in (32, 64):
                a, ma = self.packer(CompactPrefixPacker, dim).encode(k, v, tokens)
                b, mb = self.packer(BoundedPrefixPacker, dim).encode(
                    fullk, fullv, tokens
                )
                self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
                self.assertEqual(ma, mb)

    def test_nonfinite_and_short_capacity_rejected(self):
        p = self.packer(CompactPrefixPacker, 64)
        k = torch.zeros(2, 2, 64, 32).bfloat16()
        v = k.transpose(-1, -2).contiguous()
        with self.assertRaises(ValueError):
            p.encode(k, v, 33)
        k[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "range"):
            p.encode(k, v, 32)
