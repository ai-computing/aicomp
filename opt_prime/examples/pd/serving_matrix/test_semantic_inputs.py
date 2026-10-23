import unittest

from semantic_inputs import eos_ids, generate_until_stop


class StopTests(unittest.TestCase):
    def test_tokenizer_turn_end_is_not_lost_to_stale_generation_config(self):
        self.assertEqual(eos_ids(128001, 128009), [128001, 128009])
        self.assertEqual(eos_ids([1, 2], 2), [1, 2])
        self.assertEqual(eos_ids(None, 2), [2])

    def test_first_token_eos_does_not_decode(self):
        def fail(_):
            self.fail("decoded after EOS")

        self.assertEqual(generate_until_stop(9, fail, {9, 10}, 8), ([9], "eos"))

    def test_later_eos_stops_without_extra_call(self):
        calls = []

        def step(token):
            calls.append(token)
            return token + 1

        self.assertEqual(generate_until_stop(3, step, {5}, 9), ([3, 4, 5], "eos"))
        self.assertEqual(calls, [3, 4])

    def test_length_limit_and_invalid_capacity(self):
        self.assertEqual(
            generate_until_stop(1, lambda x: x + 1, {9}, 2), ([1, 2], "length")
        )
        with self.assertRaises(ValueError):
            generate_until_stop(1, lambda x: x, set(), 0)
