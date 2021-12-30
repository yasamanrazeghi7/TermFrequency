import codecs
import boto3
import re
import json
import time
from num_counter import surrounding_search, SimpleNumCounter


def preprocess_dataset(bucket_name: str,
                       input_file: str,
                       output_file: str,
                       window_size: int = 5,
                       digit_limit: int = 6,
                       max_file_lines: int = 10_000_000,
                       verbose: int = 0):
    def write_to_bucket(content: str):
        object_writer = s3.Object(bucket_name, f'{output_file}_{file_counter}.jsonl')
        object_writer.put(Body=bytes(content.encode('UTF-8')))

    if output_file.endswith(".jsonl"):
        output_file = output_file[:-len(".jsonl")]
    num_query = SimpleNumCounter(digit_limit=digit_limit).num_query
    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket_name, input_file)
    line_stream = codecs.getreader("utf-8")
    data_lines = []
    file_counter = 0
    start_time = time.time()
    for i, line in enumerate(line_stream(s3_object.get()['Body'])):
        data_line = json.loads(line)
        for middle, window in surrounding_search(data_line['text'], window_size, num_query):
            new_data = {"meta": data_line["meta"], "text": " ".join(window), "middle": middle}
            data_lines.append(json.dumps(new_data))
        if len(data_lines) >= max_file_lines:
            if verbose > 0:
                print(f"Elapsed Time: {(time.time() - start_time)}s, Dumping file {file_counter}...")
            write_to_bucket("\n".join(data_lines))
            data_lines = []
            file_counter += 1

    if len(data_lines) >= 0:
        write_to_bucket("\n".join(data_lines))
        file_counter += 1
    if verbose > 0:
        print(f"--- Finished in {time.time() - start_time} seconds, Created {file_counter} files ---")