#this file should contain all the clean and fully tested code that I use :)
from typing import List
import re


def text_window_slider(text: str, window_size: int):
    tokens = text.split()
    # TODO: make it a constant
    window = ["NONEWORD"] * ((window_size - 1) * 2 + 1)
    none_tokens = ["NONEWORD"] * ((window_size - 1) * 2 + 1)
    for token in (tokens + none_tokens):
        window.pop(0)
        window.append(token)
        yield list(window)


def surrounding_search(text: str, window_size: int, re_search_query: re.Pattern) -> (List[str], str):
    for window in text_window_slider(text, window_size):
        middle = window[len(window) // 2]
        if middle is None:
            continue
        keywords = re_search_query.findall(middle)
        if len(keywords) != 1:
            assert len(
                keywords) == 0, f"Oh no! This assertion failed!\n\tKeyword: '{keywords}'\n\tMiddle: '{middle}'\n\tWindow: '{window}'"  # window should contain the word
            continue
        keyword = "".join(keywords[0]).lower()
        yield keyword, window
