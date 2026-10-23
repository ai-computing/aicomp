import unittest

import numpy as np
from atom_stream_upload import bf16_to_dlfloat


def reference(value):
    sign = value & 0x8000
    exponent = (value >> 7) & 255
    if exponent < 96:
        return sign
    if exponent > 159:
        return sign | 0x7FFF
    return sign | ((exponent - 96) << 9) | ((value & 127) << 2)


class EncodingTests(unittest.TestCase):
    def test_all_bit_patterns_match_existing_dlfloat_contract(self):
        bits = np.arange(65536, dtype=np.uint16)
        expected = np.array([reference(int(x)) for x in bits], dtype=np.uint16)
        np.testing.assert_array_equal(bf16_to_dlfloat(bits), expected)

    def test_strided_input_preserves_shape_and_storage(self):
        bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)[:, ::-2]
        before = bits.copy()
        result = bf16_to_dlfloat(bits)
        expected = np.array(
            [reference(int(x)) for x in bits.flat], dtype=np.uint16
        ).reshape(bits.shape)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(bits, before)
        self.assertFalse(np.shares_memory(bits, result))


if __name__ == "__main__":
    unittest.main()
