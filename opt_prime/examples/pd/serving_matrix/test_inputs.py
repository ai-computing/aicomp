import unittest

from inputs import exact_prompt, through_eos


class PromptTests(unittest.TestCase):
    def test_length_and_question_tail_are_preserved(self):
        self.assertEqual(exact_prompt([1], [2, 3], [8, 9], 8), [1, 2, 3, 2, 3, 2, 8, 9])

    def test_eos_ends_semantic_comparison(self):
        self.assertEqual(through_eos([10, 20, 99, 30], [99, 100]), [10, 20, 99])
        self.assertEqual(through_eos([10, 20], [99]), [10, 20])

    def test_invalid_capacity_and_empty_filler_rejected(self):
        for args in [([1], [2], [8, 9], 2), ([1], [], [8], 8)]:
            with self.assertRaises(ValueError):
                exact_prompt(*args)


if __name__ == "__main__":
    unittest.main()
