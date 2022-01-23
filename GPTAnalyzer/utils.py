"""General utilities."""
import codecs
import boto3
import random
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)

factory_type_dict = {
    #each factory type should have a math operator and keyword for the templete
    "Num1Mult": ('*', 'times'),
    "Num2Mult": ('*', 'times'),
    "Num1#*": ('*', '#'),
    "Num2#*": ('*', '#'),
    "Num1Plus": ('+', 'plus'),
    "Num2Plus": ('+', 'plus'),
    "Num1#+": ('+', '#'),
    "Num2#+": ('+', '#'),
    "Num1concat#": ('concat', '#'), #this should have a separate factorty
    "Num1mode10#": ('%', '#'),
    "Num1Less": ('comp', 'lower'),
    "Num1More": ('comp', 'higher')
}

# MATH_OPERATORS_WORD_Template = {
#     '+': 'plus',
#     '*': 'times',
#     '#*': '#',
#     '#+': '#',
#     'concat#': 'concat by',
#     'mode10#': '#'
# }

# MATH_OPERATORS_From_WORD_Template = {
#     'plus': '+',
#     'times': '*',
#     '#*': '#',
#     '#+': '#',
#     'concat#': 'concat by',
#     'mode10#': '#'
# }
TIME_UNIT_CONVERTORS = {
    'minute': 60,
    'hour': 60,
    'day': 24,
    'week': 7,
    'month': 4,
    'year': 12,
    'decade': 10
}
TIME_UNIT_CONVERTORS_DES = {
    'minute': 'seconds',
    'hour': 'minutes',
    'day': 'hours',
    'week': 'days',
    'month': 'weeks',
    'year': 'months',
    'decade': 'years'
}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_frequency_file(path: str) -> str:
    if path.startswith("s3://"):
        return read_frequency_file_from_s3(s3_path=path)
    return read_frequency_file_local(path)


def read_frequency_file_local(path: str) -> str:
    result = ""
    with open(path) as f:
        for line in f.readlines():
            result += line
    return result


def read_frequency_file_from_s3(s3_path: str) -> str:
    if not s3_path.startswith("s3://"):
        raise Exception(f"The s3 path is invalid: '{s3_path}'")
    bucket_name = s3_path[len("s3://"):].split('/')[0]
    input_file = s3_path[len("s3://")+len(bucket_name)+1:]
    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket_name, input_file)
    line_stream = codecs.getreader("utf-8")
    result = ""
    for i, line in enumerate(line_stream(s3_object.get()['Body'])):
        result += line
    return result


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
