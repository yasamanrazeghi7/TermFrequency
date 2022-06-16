from operator import add
from pyspark import SparkConf, SparkContext
import json

# -------------- START HACK ---------------
# TODO: Must use setup.py
import os
import sys
import inspect
import pickle
import boto3

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from num_counter import ProcessedWindowNumCounter
# -------------- END HACK ---------------

if __name__ == "__main__":

    input_file_pattern = "s3://pilebucketyasaman/pile/new_processed/00_0.jsonl"
    output_folder = "s3://pilebucketyasaman/results/arithmetic_pattern_counting"
    window_size = 5
    digit_limit = 6
    conf = (SparkConf()
            .setAppName("ArithmeticPattern"))
            # .set("spark.executor.memory", "100g"))
    sc = SparkContext(conf=conf)
    print(
        f"Counting arithmetic pattern occurrences in file {input_file_pattern} with window_size {window_size} and digit_limit {digit_limit}. The result will be written in {output_folder}")
    num_counter = ProcessedWindowNumCounter(window_size=window_size, digit_limit=digit_limit)
    lines = sc.textFile(input_file_pattern)
    lines = lines\
        .map(json.loads)\
        .map(lambda x: x["text"])\
        .flatMap(lambda x: num_counter.arithmetic_pattern_counter(x.split()))\
        .map(lambda x: (x, 1))\
        .reduceByKey(add)
    # lines.saveAsTextFile(output_folder)  # This one writes the output in multiple files in the specified folder
    collected_lines = lines.collect()
    result_dict = {}
    for ter_freq in collected_lines:
        result_dict[ter_freq[0]] = ter_freq[1]

    pickle_object = pickle.dumps(result_dict)
    s3 = boto3.resource('s3')
    s3.Object('pilebucketyasaman', 'results/arithmetic_patterns_whitespace_tokenizer.pkl').put(Body=pickle_object)
    print(f"success")
