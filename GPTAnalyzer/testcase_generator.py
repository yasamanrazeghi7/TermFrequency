from typing import Callable, Any, List, Tuple
import json
import math
import random
import argparse
import logging
from pandas.io.json import json_normalize
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTJForCausalLM, AutoTokenizer, GPTNeoForCausalLM
from utils import read_from_s3, read_from_file, chunks

logger = logging.getLogger(__name__)


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class RawFrequencyData:
    def __init__(self, key: Tuple[str], frequency: int):
        self.key = key
        self.frequency = frequency


MATH_OPERATORS = {
    '+': 'plus',
    '*': 'times',
    '-': 'minus',
    '/': 'divides',
}


class CompleteFrequencyData:
    def __init__(self, x: str, y: str, z: str, word: str, math_operator: str,
                 raw_frequency_data: RawFrequencyData):
        self.x = x
        self.y = y
        self.z = z
        self.word = word
        self.math_operator = math_operator
        self.raw_frequency_data = raw_frequency_data

    def __str__(self):
        if self.raw_frequency_data:
            return f"CFD({self.x}, {self.y}, {self.z}, {self.word}, {self.math_operator}, F: {self.raw_frequency_data.frequency}, K: {self.raw_frequency_data.key})"
        return f"CFD({self.x}, {self.y}, {self.z}, {self.word}, {self.math_operator})"


class CompleteFrequencyFactory:
    def __init__(self):
        pass

    def build(self, raw_frequency_data: RawFrequencyData) -> List[CompleteFrequencyData]:
        pass


class Num1MultiplyFactory(CompleteFrequencyFactory):
    def __init__(self, multiplicands: List[int]):
        super().__init__()
        self.multiplicands = multiplicands

    def build(self, raw_frequency_data: RawFrequencyData) -> List[CompleteFrequencyData]:
        result = []
        x = int(raw_frequency_data.key[0])
        for multiplicand in self.multiplicands:
            complete_data = CompleteFrequencyData(str(x), str(multiplicand), str(x * multiplicand), 'times', '*',
                                                  raw_frequency_data)
            result.append(complete_data)
        return result


class DataPoint:
    def __init__(self, question: str, answer: str, frequency_data: CompleteFrequencyData):
        self.question = question
        self.answer = answer
        self.frequency_data = frequency_data

    def __str__(self):
        return f"({self.question}, {self.answer}, {self.frequency_data})"


# ------- Start Filters -----------

class FrequencyDataFilter:
    def is_valid(self, complete_frequency_data: CompleteFrequencyData) -> bool:
        pass


class DigitLimitFilter(FrequencyDataFilter):
    def __init__(self, digit_limit: int = 2):
        super().__init__()
        self.digit_limit = digit_limit

    def is_valid(self, complete_frequency_data: CompleteFrequencyData) -> bool:
        return len(complete_frequency_data.x) <= self.digit_limit and len(complete_frequency_data.y) <= self.digit_limit


# ------- End Filters -----------

# ------- Start Templates -----------


class FrequencyDataTemplate:
    def generate(self, complete_frequency_data: CompleteFrequencyData) -> DataPoint:
        pass


class MultiplyTemplate(FrequencyDataTemplate):
    def generate(self, complete_frequency_data: CompleteFrequencyData) -> DataPoint:
        question = "What is {0} times {1}?".format(complete_frequency_data.x, complete_frequency_data.y)
        answer = complete_frequency_data.z
        return DataPoint(question, answer, complete_frequency_data)


# ------- End Templates -----------


class DataPointGenerator:
    def __init__(self):
        self.filters = []
        self.templates = []

    def add_filter(self, data_filter: FrequencyDataFilter):
        self.filters.append(data_filter)
        return self

    def add_template(self, datapoint_template: FrequencyDataTemplate):
        self.templates.append(datapoint_template)
        return self

    def generate(self, frequency_dataset: List[CompleteFrequencyData]) -> List[DataPoint]:
        result = []
        for frequency_data in frequency_dataset:
            is_valid = True
            for f_filter in self.filters:
                if not f_filter.is_valid(frequency_data):
                    is_valid = False
                    break
            if not is_valid:
                continue
            for template in self.templates:
                result.append(template.generate(frequency_data))
        return result


class TestCase:
    def __init__(self, body: str, datapoint: DataPoint):
        self.body = body
        self.data_point = datapoint

    def __str__(self):
        return f"----- TestCase {self.data_point} ------\n{self.body}\n-------"


class TestCaseResult:
    def __init__(self, testcase: TestCase, generated_answer: str, is_correct: bool):
        self.testcase = testcase
        self.generated_answer = generated_answer
        self.is_correct = is_correct

    def __str__(self):
        return f"===== TestCaseResult GeneratedAnswer:{self.generated_answer}, IsCorrect? {self.is_correct} =======\n{self.testcase}\n======"


class TestCaseTemplate:
    def __init__(self, question_prefix: str, answer_prefix: str, prompt_title: str):
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.prompt_title = prompt_title

    def generate_testcase(self, train_datapoints: List[DataPoint], test_datapoint: DataPoint) -> TestCase:
        result = self.prompt_title
        result += "\n\n"
        for dp in train_datapoints:
            result += f"{self.question_prefix} {dp.question}\n"
            result += f"{self.answer_prefix} {dp.answer}\n"
            result += "\n"
        result += f"{self.question_prefix} {test_datapoint.question}\n"
        result += f"{self.answer_prefix}"
        return TestCase(result, test_datapoint)


def testcase_splitter(datapoints: List[DataPoint], shots_number: int) -> (List[DataPoint], List[DataPoint]):
    train_set, test_set = torch.utils.data.random_split(datapoints, [shots_number, len(datapoints) - shots_number])
    return (list(train_set), list(test_set))


def testcase_generator(train_datapoints: List[DataPoint], test_datapoints: List[DataPoint],
                       template: TestCaseTemplate) -> List[TestCase]:
    return [template.generate_testcase(train_datapoints, test_datapoint) for test_datapoint in test_datapoints]


def GPTJ_Analysis(model, tokenizer, device, batch_size: int, testcases: List[TestCase]) -> List[TestCaseResult]:
    results = []
    for testcase_chunk in chunks(testcases, batch_size):
        testcases_bodies = [x.body for x in testcase_chunk]
        input_ids = tokenizer.batch_encode_plus(testcases_bodies, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                           max_length=5 + len(input_ids['input_ids'][0]), do_sample=False, min_length=2)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_answers = [x[len(testcases_bodies[i]) + 1:].split()[0] for i, x in enumerate(generated_texts)]
        for (tc, ans) in zip(testcase_chunk, generated_answers):
            results.extend([TestCaseResult(tc, ans, ans.strip() == tc.data_point.answer.strip())])
    return results


def setup_model(device, model_name: str) -> (Any, Any):
    if model_name == 'GPTNEO-small':
        logger.debug("in first model")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
    elif model_name == 'GPTNEO-large':
        logger.debug("in second model")
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
        model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B').to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
        model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "left"  # check this please
    tokenizer.pad_token = tokenizer.eos_token
    # model = GPTJForCausalLM.from_pretrained(model).to(device)
    model.eval()
    return (model, tokenizer)


def main(args):
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
    logger.info(f"Device is detected as {device}!")
    model, tokenizer = setup_model(device, args.model)
    logger.info("Model is loaded!")
    # -------------------
    input_path = args.input_path
    random.seed(args.seed)  # TODO: set all
    torch.manual_seed(args.seed)
    # Read from aggregated results
    logger.debug(f"The input path is '{input_path}'")
    # my_file = read_from_s3(s3_path=input_path)
    my_file = read_from_file(input_path)
    # key_parser = lambda key: key
    complete_frequency_factory = Num1MultiplyFactory(list(range(1, 21)))
    frequency_data = []
    for line in my_file.split("\n")[:args.top]:
        key, frequency = eval(line)
        frequency_data.extend(complete_frequency_factory.build(RawFrequencyData(key, frequency)))
    logger.info(f"Reading is done! Total Complete Frequency Data: {len(frequency_data)}")
    logger.debug("A sample of complete frequency data:")
    logger.debug("\n".join(str(x) for x in frequency_data[:40]))
    datapoint_generator = DataPointGenerator()
    datapoint_generator.add_filter(DigitLimitFilter(2))
    datapoint_generator.add_template(MultiplyTemplate())
    datapoints = datapoint_generator.generate(frequency_data)
    logger.info(f"Generated {len(datapoints)} data points!")
    logger.debug("A sample of data points:")
    logger.debug("\n".join(str(x) for x in datapoints[:40]))
    testcase_template = TestCaseTemplate("Q:", "A:", "Calculate the math question below:")
    # test_case = testcase_template.generate_testcase(datapoints[:2], datapoints[3])
    # print(test_case.body)
    train_datapoints, test_datapoints = testcase_splitter(datapoints, args.shots)
    logger.info(f"Train Datapoints: {len(train_datapoints)}, Test Datapoints: {len(test_datapoints)}")
    logger.debug("A sample of train data points:")
    logger.debug("\n".join(str(x) for x in train_datapoints[:3]))
    logger.debug("A sample of test data points:")
    logger.debug("\n".join(str(x) for x in test_datapoints[:3]))
    testcases = testcase_generator(train_datapoints, test_datapoints, testcase_template)
    logger.info(f"Generated {len(testcases)} Test Cases!")
    logger.debug("A sample of train data points:")
    logger.debug("\n------\n".join(x.body for x in testcases[:3]))
    results = GPTJ_Analysis(model, tokenizer, device, args.bs, testcases)
    logger.info(f"Finished: {len(results)} test case is analyzed!")
    json_results = MyEncoder().encode(results)
    flatten_json_result = json_normalize(json.loads(json_results))
    flatten_json_result.to_csv(args.output_path)
    logger.info(f"The final result is written in '{args.output_path}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help="The s3 or local path to the aggregated result")
    parser.add_argument('--output-path', type=str, required=True, help="The local path to write the output")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--bs', type=int, default=20, help="Batch size")
    parser.add_argument('--shots', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--top', type=int, default=200, help="Filter the top frequent data")
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--model', type=str, default='GPT-J', help="model-name")

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
