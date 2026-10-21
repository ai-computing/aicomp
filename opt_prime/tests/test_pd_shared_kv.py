import os
import unittest
import numpy as np
from opt_prime.accel.pd.npu.shared_kv import SharedKVSlab


class SharedKVTests(unittest.TestCase):
    def test_visibility_readonly_and_sealed_size(self):
        reader = SharedKVSlab.create((2, 1, 4, 2))
        writer = SharedKVSlab(reader.fd, reader.shape, writable=True)
        try:
            writer.view()[:] = 7
            self.assertTrue(np.all(reader.view() == 7))
            with self.assertRaises(ValueError):
                reader.view()[:] = 2
            with self.assertRaises(OSError):
                os.ftruncate(reader.fd, 1)
            with self.assertRaises(ValueError):
                SharedKVSlab(reader.fd, (1, 1, 1, 1))
        finally:
            writer.close()
            reader.close()
        reader.close()
        with self.assertRaises(RuntimeError):
            reader.view()
