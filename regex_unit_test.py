import unittest
from regex_utils import *


class TestAddFishToAquarium(unittest.TestCase):

    def test1(self):
        string_test = "987 MinuTes 78 927 9876"
        actual = pattern_finder(string_test)
        expected = [('987 MinuTes 78 927 ', 'MinuTe', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test2(self):
        string_test = "987 hourS 78 927 9876"
        actual = pattern_finder(string_test)
        expected = [('987 hourS 78 927 ', 'hour', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test3(self):
        string_test = "987 days 78 927 9876"
        actual = pattern_finder(string_test)
        expected = [('987 days 78 927 ', 'day', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test4(self):
        string_test = "987 days 654 months 78 927 9876"
        actual = pattern_finder(string_test)
        expected = [('', '', '987 days 654 ', 'day', '', ''), (' 654 months 78 927 ', 'month', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test5(self):
        string_test = "987  days  654  decades 43  months   234     765"
        actual = pattern_finder(string_test)
        expected = [('', '', ' 654  decades 43 ', 'decade', '', ''), (' 43  months   234     765', 'month', '', '', '', ''), ('', '', '987  days  654 ', 'day', '', '')]
        expected = expected
        self.assertCountEqual(actual, expected)
        print([co_finder(x) for x in actual])

    def test6(self):
        string_test = "  2 months 4.5  months 3.4.5"
        actual = pattern_finder(string_test)
        expected = []
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test7(self):
        string_test = "  2.5 months 1 3"
        actual = pattern_finder(string_test)
        expected = []
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test8(self):
        string_test = "2. 5 months 1 3"
        actual = pattern_finder(string_test)
        expected = [(' 5 months 1 3', 'month', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test9(self):
        string_test = "2. 05 moNThs 1 3"
        actual = pattern_finder(string_test)
        expected = [(' 05 moNThs 1 3', 'moNTh', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])


    def test10(self):
        string_test = "1 months 2 3\n1 months 2 3\n1 months 2 3\n1 months 2 3"
        actual = pattern_finder(string_test)
        print(actual)
        expected = [('\n1 months 2 3\n', 'month', '', '', '', ''), ('\n1 months 2 3', 'month', '', '', '', ''), ('\n1 months 2 3\n', 'month', '', '', '', '') ,('1 months 2 3\n', 'month', '', '', '', '')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])

    def test11(self):
        string_test = "1 months !!! 3 "
        actual = pattern_finder(string_test)
        expected = [('', '', '', '', '1 months !!! 3 ', 'month')]
        self.assertCountEqual(actual, expected)
        # print([co_finder(x) for x in actual])



if __name__ == "__main__":
    unittest.main()
    print("okay")