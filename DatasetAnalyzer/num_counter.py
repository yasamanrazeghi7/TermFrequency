from itertools import combinations
from typing import List
import re
from DatasetAnalyzer.utils import surrounding_search


class NumCounter:
    # TODO: fix keyword_list typing
    def __init__(self, window_size: int = 5, digit_limit: int = 6, keyword_list: List = None):
        self.window_size = window_size
        self.num_query = re.compile(r'^\d{{1,{0}}}$'.format(digit_limit))
        if keyword_list is None:
            keyword_list = [("minute", True)]
        keyword_patterns = []
        for keyword, is_plural in keyword_list:
            if is_plural:
                keyword_patterns.append(r"(?:^|\s|$)({0})s?(?:^|\s|$)".format(keyword))
            else:
                keyword_patterns.append(r"(?:^|\s|$)({0})(?:^|\s|$)".format(keyword))
        self.keyword_query = re.compile("|".join(keyword_patterns), re.IGNORECASE)


class OriginalNumCounter(NumCounter):
    """
        This class is used for computing numerical related information in the original dataset.
    """
    def surrounding_keyword_search(self, text: str):
        return list(surrounding_search(text, self.window_size, self.keyword_query))

    def word_num_occurrence(self, keyword: str, window: List[str], num_count: int = 2):
        """
        Given a window where its middle value is keyword, the co-occurrences of num_count numbers
        will be counted
        :param window:
        :param num_re_query:
        :param num_count:
        :return:
        """
        list_of_tuples = []
        number_with_positions = [(word, pos) for (pos, word) in enumerate(window) if word and self.num_query.match(word)]
        middle_pos = len(window) // 2
        for i, (num1, pos1) in enumerate(number_with_positions):
            if num_count == 1:
                list_of_tuples += [(num1, keyword)]
                continue
            other_numbers = []
            for (num2, pos2) in number_with_positions[i+1:]:
                if pos2 - pos1 < self.window_size and abs(pos2 - middle_pos) < self.window_size:
                    other_numbers.append(num2)
            for combs in combinations(other_numbers, num_count-1):
                list_of_tuples += [(*sorted([num1] + list(combs)), keyword)]
        return list_of_tuples


class ProcessedWindowNumCounter(NumCounter):
    """
        This class is used for calculating numerical related information in preprocessed datasets.
        Each line in this dataset is a window of window_size words where the middle word is a number with
        at most digit_limit digits. This dataset may have duplicate data, e.g., the original line "1 2 3"
        will be converted into three windows where in each of them the middle number is 1, 2, or 3.
    """
    def word_num_occurrence(self, window: List[str], num_count: int = 2):
        """
        TODO: write the description
        """
        list_of_tuples = []
        middle_pointer = len(window) // 2
        middle_number = window[middle_pointer]
        keyword_with_positions = []
        for i, word in enumerate(window):
            if abs(middle_pointer-i) >= self.window_size:
                continue
            keywords = self.keyword_query.findall(word)
            if len(keywords) != 1:
                assert len(keywords) == 0, f"Oh no! This assertion failed!\n\tKeyword: '{keywords}'\n\tWord: '{word}'\n\tWindow: '{window}'"  # window should contain the word
                continue
            keyword = "".join(keywords[0]).lower()
            keyword_with_positions += [(keyword, i)]
        number_with_positions = [(word, pos+middle_pointer+1) for (pos, word) in enumerate(window[middle_pointer+1:]) if word and self.num_query.match(word)]
        for keyword, keyword_pos in keyword_with_positions:
            if num_count == 1:
                list_of_tuples += [(middle_number, keyword)]
                continue
            other_numbers = []
            for (num, pos) in number_with_positions:
                if abs(pos - middle_pointer) < self.window_size and abs(pos - keyword_pos) < self.window_size:
                    other_numbers.append(num)
            for combs in combinations(other_numbers, num_count-1):
                list_of_tuples += [(*sorted([middle_number] + list(combs)), keyword)]
        return list_of_tuples

    def num_co_occurrence(self, window: List[str], num_count: int):
        middle_pointer = len(window) // 2
        middle_number = window[middle_pointer]
        if num_count == 1:
            return [middle_number]
        number_with_positions = [(word, pos+middle_pointer+1) for (pos, word) in enumerate(window[middle_pointer+1:]) if word and self.num_query.match(word)]
        list_of_tuples = []
        other_numbers = []
        for (num, pos) in number_with_positions:
            if abs(pos - middle_pointer) < self.window_size:
                other_numbers.append(num)
        for combs in combinations(other_numbers, num_count - 1):
            list_of_tuples.append(tuple([*sorted([middle_number] + list(combs))]))
        return list_of_tuples


if __name__ == "__main__":
    text = "1 second *3 4 - 5 6 * 7 8 + 9 10  10:12 3.2     !!!!! 11  1234567  +minUte 1 MinutEs$Seconds 768 plus minutes 0909 653 725 78678675.9898 alaki @     776 seconds"
    # text = "1243 minutes and 23 MInutes and 3 minutes and     3   minutes and 3minutes"
    keyword_regex = re.compile(r"(?:^|\s|$)(minute)s?(?:^|\s|$)|(?:^|\s|$)(second)s?(?:^|\s|$)", re.IGNORECASE)
    # num_counter = OriginalNumCounter(5, 6, [('minute', True), ('second', True), ('plus', False)])
    # keyword_windows = num_counter.surrounding_keyword_search(text)
    # # print("\n".join(str(x) for x in keyword_windows))
    # for keyword, window in keyword_windows:
    #     print(keyword, " ### ", window)
    #     print("\n".join(str(x) for x in num_counter.word_num_occurrence(keyword, window, 2)))
    #     print("---------")

    # num_counter = ProcessedWindowNumCounter(4, 6, [('minute', True), ('second', True), ('plus', False), ('\+', False), ('\*', False), ('-', False)])
    # print(num_counter.keyword_query.findall("minutes seconds"))

    # processed_window = ["NONEWORD", "NONEWORD", "NONEWORD", "NONEWORD", "NONEWORD", "1", "2", "+", "3", "4", "5"]
    # my_tuples = num_counter.word_num_occurrence(processed_window, 2)
    # all_tuples = []
    # for keyword, processed_window in surrounding_search(text, num_counter.window_size, num_counter.num_query):
    #     # processed_window = [None, "1", "2", "seconds", "3", "4", "5", "6", "minute", "7", "8"]
    #     print(processed_window)
    #     print("-------Start-----")
    #     # my_tuples = num_counter.num_co_occurrence(processed_window, 2)
    #     my_tuples = num_counter.word_num_occurrence(processed_window, 2)
    #     print("\n".join(str(x) for x in my_tuples))
    #     all_tuples.extend(my_tuples)
    #     print("-------End-------")
    #
    # print("\n".join(str(x) for x in all_tuples))


