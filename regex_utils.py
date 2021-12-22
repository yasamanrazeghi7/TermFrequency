from typing import List, AnyStr, Any
import re


window_size = 5
time_stamp = "second|minute|hour|day|month|week|year|decade"
from_navid = r'((?:[^\s]+\s*){{0,{0}}}(?:\s+|^)\bminutes\b(?:\s+|$)(?:[^\s]+\s*){{0,{0}}})'


##########################################################################################
#########################################################################################
# these can be eventually be loaded from a file
re_patterns = [
    r"(?=((?:^|\s)\d+\s+({0})s?\s+\d+\s+\d+(?:\s|$)))",  # finds number time_stamp number number
    r"(?=((?:^|\s)\d+\s+({0})s?\s+\d+(?:\s|$)))",  # number time_stamp number
    r"(?=((?:^|\s)\d+\s+({0})s?\s+[^\s]+\s+\d+(?:\s|$)))"  # number time_stamp non_number number
]
pattern_finder = lambda x: re.compile("|".join(re_pattern.format(time_stamp) for re_pattern in re_patterns),
                                      re.IGNORECASE).findall(x)  # this must match all the patterns including timestamp

def co_finder(x):  # this should prepare the keys as tuples and triplets with time_stamps from the pattern_finder
    this_time_stamp = ""
    tuple_list = []
    for z in x:
        if z != "":
            numbers = re.findall(r'\b(\d+)\b', z)
            if len(numbers) == 0:
                this_time_stamp = z
            else:
                if len(numbers) == 3:
                    tuple_list.append((numbers[0], numbers[1]))
                    tuple_list.append((numbers[0], numbers[2]))
                    tuple_list.append((numbers[0], min(numbers[1], numbers[2]), max(numbers[1], numbers[2])))
                elif len(numbers) == 2:
                    tuple_list.append((numbers[0], numbers[1]))
    result = []
    for z in tuple_list:
        result.append((z, this_time_stamp.lower()))
    return result

################################################################################
#todo test this with re.compile(pattern).findall(text)

def find_word_in_ws(ws, word, text):
    UNUSED_TOKENS = " _____ " * (ws+1)
    pattern = r'(?=(\b(?:\S+\s+){{{0}}}\b\b(({1})s?)\b\s*\b(?:\S+\s+){{{0}}}\b))'.format(ws,word)
    return re.compile(pattern, re.IGNORECASE).findall(UNUSED_TOKENS+text+UNUSED_TOKENS)

text = "1 second 3 4 5 6 7 8 9 10 11    minute 1 768 minutes 0909 653 725 alaki @     776 seconds"


print("\n".join(str(x) for x in find_word_in_ws(5, "minute|second", text)))

def text_window_slider(text: str, window_size: int):
    tokens = text.split()
    window = [None]*(window_size * 2 + 1)
    for token in (tokens + [None] * (window_size*2+1)):
        window.pop(0)
        window.append(token)
        yield list(window)


def surrounding_search(text: str, window_size: int, regex_search_query: str):
    for window in text_window_slider(text, window_size):
        middle = window[len(window)//2]
        if middle is None or not re.compile(regex_search_query).findall(middle):
            continue
        yield window

# print("\n".join(str(x) for x in text_window_slider(text, 5)))
print("\n".join(str(x) for x in surrounding_search(text, 5, r"(minute|second)s?")))

# print("\n".join(str(x) for x in find_word_in_ws(5, "minute", text)))

##implemenet a function that gets a time_stamp and text and produces tuples and triples

##




if __name__ == "__main__":
    print("okay")
