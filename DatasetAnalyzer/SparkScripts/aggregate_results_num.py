import sys
from operator import add
from pyspark import SparkConf, SparkContext


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Missing parameters")
        exit()
    input_folder_pattern = sys.argv[1]
    aggregated_output_folder = sys.argv[2]

    input_files = []
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("WordCounter")
            .set("spark.executor.memory", "100g"))
    sc = SparkContext(conf=conf)
    all_lines = sc.textFile(input_folder_pattern)
    aggregated_result = all_lines.map(lambda x: eval(x)).reduceByKey(add).collect()
    aggregated_result.sort(key=lambda x: -x[1])
    with open(f"{aggregated_output_folder}/aggregated_results.txt", "w") as f:
        for line in aggregated_result:
            f.write(str(line) + "\n")