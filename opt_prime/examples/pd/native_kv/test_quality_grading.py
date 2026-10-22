import unittest
from check_live_quality import grade


class QualityGradingTest(unittest.TestCase):
    def test_structured_answer_types_and_order(self):
        self.assertTrue(grade(' [1, 2, 3] \n',[1,2,3]))
        for answer in ('[true,2,3]','[1.0,2,3]','[3,2,1]','[1,2]','{"a":1}','```json\n[1,2,3]\n```'):
            self.assertFalse(grade(answer,[1,2,3]),answer)

    def test_strict_format_is_not_semantic_grading(self):
        self.assertTrue(grade(' paris\n','Paris'))
        self.assertTrue(grade('323','323'))
        for answer in ('The answer is 323.','**323**','3230'):
            self.assertFalse(grade(answer,'323'))
        self.assertFalse(grade('No.','no'))
        self.assertFalse(grade('서울특별시','서울'))


if __name__=='__main__':unittest.main()
