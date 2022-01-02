from operator import add
from pyspark import SparkConf, SparkContext
import json

# -------------- START HACK ---------------
# TODO: Must use setup.py
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from DatasetAnalyzer.num_counter import ProcessedWindowNumCounter
# -------------- END HACK ---------------

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Missing parameters")
        exit()

    input_file_pattern = sys.argv[1]
    output_folder = sys.argv[2]
    window_size = int(sys.argv[3])
    num_count = int(sys.argv[4])
    digit_limit = 6
    keyword_list = [('second', True),
                    ('minute', True),
                    ('hour', True),
                    ('day', True),
                    ('week', True),
                    ('month', True),
                    ('year', True),
                    ('decade', True),
                    ('plus', False),
                    ('\+', False),
                    ('minus', False),
                    ('-', False),
                    ('times', False),
                    ('\*', False),
                    ]
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("WordCounter"))
            # .set("spark.executor.memory", "100g"))
    sc = SparkContext(conf=conf)
    print(
        f"Counting word and {num_count} numbers occurrences in file {input_file_pattern} with window_size {window_size} and digit_limit {digit_limit}. The result will be written in {output_folder}")
    num_counter = ProcessedWindowNumCounter(window_size=window_size, digit_limit=digit_limit, keyword_list=keyword_list)
    lines = sc.textFile(input_file_pattern)
    lines = lines\
        .map(json.loads)\
        .map(lambda x: x["text"])\
        .flatMap(lambda x: num_counter.word_num_occurrence(x.split(), num_count=num_count))\
        .map(lambda x: (x, 1))\
        .reduceByKey(add)
    lines.saveAsTextFile(output_folder)  # This one writes the output in multiple files in the specified folder
    print(f"success")
