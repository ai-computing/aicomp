import json
from pathlib import Path
import struct
import unittest

import protocol as p

# Header observed for a synthetic prompt, with no model payload distributed.
CAPTURED_PULL = bytes.fromhex(
    '636f6465782d6b762d30303034000000c0270400bf270400be27040000000000'
    '020000008d000000dcffffff00000000e0ffffff03000000e4ffffff00000000')
CAPTURED_BATCH = bytes.fromhex(
    '636f6465782d6b762d303030320000000000000000000000c027040000080000'
    '0000000001000000c0270400000800000000000002000000c027040000080000'
    '0000000003000000c027040000080000010000008d000000acffffffb4ffffff'
    '0400000000000000000000000000000000000000')


class ProtocolTest(unittest.TestCase):
    def test_pull_matches_observed_header(self):
        self.assertEqual(p.control('codex-kv-0004', 2, [272320,272319,272318]), CAPTURED_PULL)

    def test_observed_batch(self):
        value = p.parse_message([CAPTURED_BATCH, b'\0' * 8192])
        self.assertEqual(value['id'], 'codex-kv-0002')
        self.assertEqual(value['blocks'], [(0, slot, 272320, 2048) for slot in range(4)])
        self.assertFalse(value['final'])

    def test_incomplete_payload(self):
        with self.assertRaises(ValueError):
            p.parse_message([CAPTURED_BATCH, b'\0' * 8191])

    def test_truncated_headers(self):
        for n in range(len(CAPTURED_BATCH)):
            with self.subTest(n=n), self.assertRaises((ValueError, UnicodeError)):
                p.parse_message([CAPTURED_BATCH[:n], b'\0' * 8192])

    def test_bad_vector_reference(self):
        data = bytearray(CAPTURED_BATCH)
        struct.pack_into('<i', data, len(data)-24, -100000)
        with self.assertRaises(ValueError):
            p.parse_message([data, b'\0' * 8192])

    def test_unknown_variant_and_slot(self):
        for offset, value in [(len(CAPTURED_BATCH)-36, 99), (20,72)]:
            data = bytearray(CAPTURED_BATCH)
            struct.pack_into('<I', data, offset, value)
            with self.assertRaises(ValueError):
                p.parse_message([data, b'\0' * 8192])

    def test_bounds(self):
        for request in ['tiny', 'x'*128]:
            with self.assertRaises(ValueError):p.control(request, 3)
        for ids in [[1,1],[-1],[272321],list(range(2048))]:
            with self.assertRaises(ValueError):p.control('request-123', 2, ids)

    def test_variable_ids(self):
        for n in (8,13,32,64,127):
            request = 'x' * n
            self.assertEqual(p.parse_message([p.control(request,1)])['id'],request)

    def test_bad_layout(self):
        layout = json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
        p.validate_layout(layout)
        for field, value in [('element_type','Float16'),('num_chips',2),('block_size',4096)]:
            with self.assertRaises(ValueError):p.handshake('127.0.0.1:19202', {**layout,field:value})

    def test_no_auxiliary(self):
        data=bytearray(CAPTURED_BATCH);data[-1]=1
        with self.assertRaises(ValueError):p.parse_message([data,b'\0'*8192])


if __name__ == '__main__':
    unittest.main()
