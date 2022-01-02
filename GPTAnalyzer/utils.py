import codecs
import boto3


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_from_file(path: str) -> str:
    result = ""
    with open(path) as f:
        for line in f.readlines():
            result += line
    return result


def read_from_s3(s3_path: str = "", bucket_name: str = "", input_file: str = "") -> str:
    if len(s3_path) > 0:
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
