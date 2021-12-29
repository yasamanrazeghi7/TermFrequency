#this file should contain all the clean and fully tested code that I use :)
import re
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
