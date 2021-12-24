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

text = "1 second 3 4 5 6 7 8 9 10  10:12 3.2     !!!!! 11  1234567  minUte 1 768 minutes 0909 653 725 78678675.9898 alaki @     776 seconds"


print("\n".join(str(x) for x in find_word_in_ws(5, "minute|second", text)))

def text_window_slider(text: str, window_size: int):
    ret_list = []
    tokens = text.split()
    window = [None]*(window_size * 2 + 1)
    for token in (tokens + [None] * (window_size*2+1)):
        window.pop(0)
        window.append(token)
        ret_list.append(list(window))
    return ret_list


def surrounding_search(text: str, window_size: int, regex_search_query: str):
    ret_list = []
    for window in text_window_slider(text, window_size):
        middle = window[len(window)//2]
        if middle is None or not re.compile(regex_search_query, re.IGNORECASE).findall(middle):
            continue
        ret_list.append(window)
    return ret_list

def create_tuples(window, key_words_regex, window_size=5, digit_limit=6):
    #check the middle word should be one of the keywords
    middle = window[len(window) // 2]
    keyword = re.compile(key_words_regex, re.IGNORECASE).findall(middle)
    keyword = ["".join(x) for x in keyword]
    print(keyword)
    assert len(keyword)==1 # window should contain the word
    list_of_tuples=[]
    # reg = re.compile(r'\b\d{{1,{0}}}\b'.format(digit_limit))
    reg = re.compile(r'^\d{{1,{0}}}$'.format(digit_limit))
    print(reg)
    for i, word in enumerate(window):
        if word == None:
            continue
        for j, word2 in enumerate(window[i+1:]):
            if j-i>window_size or word2==None:
                break
            if reg.match(word)!=None and reg.match(word2)!=None:
                list_of_tuples.append((min(word, word2), max(word, word2), keyword[0].lower()))
                # yield (min(word, word2), max(word, word2), keyword[0].lower())
    return list_of_tuples





key_word_regex = r"(?:^|\s|$)(minute)s?(?:^|\s|$)|(?:^|\s|$)(second)s?(?:^|\s|$)"
# print("\n".join(str(x) for x in text_window_slider(text, 5)))
print("\n".join(str(x) for x in surrounding_search(text, 5, key_word_regex)))
print("***************************************************************************")
all_windows = surrounding_search(text, 5, key_word_regex)
for x in all_windows:
    print("******************************")
    print(x)
    print("\n".join(str(y) for y in create_tuples(x, key_word_regex)))
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(["\n".join(str(y) for y in create_tuples(z, key_word_regex)) for z in surrounding_search(text, 5, key_word_regex))
# print("\n".join(str(x) for x in find_word_in_ws(5, "minute", text)))

##implemenet a function that gets a time_stamp and text and produces tuples and triples

##
##text should have middle
#list($hour,$minute,$second)


if __name__ == "__main__":
    print("okay")

