from itertools import combinations
from typing import List
import re


class SimpleNumCounter:
    # TODO: fix keyword_list typing
    def __init__(self, window_size: int, digit_limit: int, keyword_list: List):
        self.window_size = window_size
        self.num_query = re.compile(r'^\d{{1,{0}}}$'.format(digit_limit))
        keyword_patterns = []
        for keyword, is_plural in keyword_list:
            if is_plural:
                keyword_patterns.append(r"(?:^|\s|$)({0})s?(?:^|\s|$)".format(keyword))
            else:
                keyword_patterns.append(r"(?:^|\s|$)({0})(?:^|\s|$)".format(keyword))
        self.keyword_query = re.compile("|".join(keyword_patterns), re.IGNORECASE)

    def text_window_slider(self, text: str):
        ret_list = []
        tokens = text.split()
        window = [None] * (self.window_size * 2 + 1)
        none_tokens = [None] * (self.window_size * 2 + 1)
        for token in (tokens + none_tokens):
            window.pop(0)
            window.append(token)
            ret_list.append(list(window))
        return ret_list

    def surrounding_search(self, text: str) -> (List[str], str):
        ret_list = []
        for window in self.text_window_slider(text):
            middle = window[len(window) // 2]
            if middle is None:
                continue
            keywords = self.keyword_query.findall(middle)
            if len(keywords) != 1:
                assert len(keywords) == 0,  f"Oh no! This assertion failed!\n\tKeyword: '{keywords}'\n\tMiddle: '{middle}'\n\tWindow: '{window}'" # window should contain the word
                continue
            keyword = "".join(keywords[0]).lower()
            ret_list.append((keyword, window))
        return ret_list

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
        for i, (num1, pos1) in enumerate(number_with_positions):
            if num_count == 1:
                list_of_tuples += [(num1, keyword)]
                continue
            other_numbers = []
            for (num2, pos2) in number_with_positions[i+1:]:
                if pos2 - pos1 < self.window_size:
                    other_numbers.append(num2)
            for combs in combinations(other_numbers, num_count-1):
                list_of_tuples += [(*sorted([num1] + list(combs)), keyword)]
        return list_of_tuples

    def word_num_occurrence_in_processed_window(self, window: List[str], num_count: int = 2):
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

    def num_co_occurrence(self, text: str, num_re_query: str, num_count: int):
        pass


if __name__ == "__main__":
    text = "1 second 3 4 5 6 7 8 9 10  10:12 3.2     !!!!! 11  1234567  minUte 1 MinutEs$Seconds 768 plus minutes 0909 653 725 78678675.9898 alaki @     776 seconds"
    # keyword_regex = re.compile(r"(?:^|\s|$)(minute)s?(?:^|\s|$)|(?:^|\s|$)(second)s?(?:^|\s|$)", re.IGNORECASE)
    num_counter = SimpleNumCounter(5, 6, [('minute', True), ('second', True), ('plus', False)])
    keyword_windows = num_counter.surrounding_search(text)
    # print("\n".join(str(x) for x in keyword_windows))
    # for keyword, window in keyword_windows:
    #     print(keyword, " ### ", window)
    #     print("\n".join(str(x) for x in num_counter.word_num_occurrence(keyword, window, 3)))
    #     print("---------")

    # processed_window = [None, None, None, None, None, "1", "2", "minute", "3", "4", "5"]
    processed_window = [None, "1", "2", "seconds", "3", "4", "5", "6", "minute", "7", "8"]
    print("\n".join(str(x) for x in num_counter.word_num_occurrence_in_processed_window(processed_window, 2)))

