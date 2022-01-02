from typing import Callable, Any, List, Tuple
import argparse
from utils import read_from_s3


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
            complete_data = CompleteFrequencyData(str(x), str(multiplicand), str(x*multiplicand), 'times', '*', raw_frequency_data)
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


class TestCaseResult:
    def __init__(self, testcase: TestCase, result: str, is_correct: bool):
        self.testcase = testcase
        self.result = result
        self.is_correct = is_correct


class TestCaseTemplate:
    def __init__(self, question_prefix: str, answer_prefix: str, prompt: str):
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.prompt = prompt

    def generate_testcase(self, train_datapoints: List[DataPoint], test_datapoint: DataPoint) -> TestCase:
        pass


def testcase_splitter(datapoints: List[DataPoint]) -> (List[DataPoint], List[DataPoint]):
    return ([], [])


def testcase_generator(train_datapoints: List[DataPoint], test_datapoints: List[DataPoint], template: TestCaseTemplate) -> List[TestCase]:
    pass


def GPTJ_Analysis(testcases: List[TestCase]) -> List[TestCaseResult]:
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="The s3 path to the aggregated result")
    parser.add_argument('--top', type=int, default=200, help="Filter the top frequent data")
    args = parser.parse_args()
    input_path = args.input_path
    # Read from aggregated results
    print(f"The input path is '{input_path}'")
    my_file = read_from_s3(s3_path=input_path)
    # key_parser = lambda key: key
    complete_frequency_factory = Num1MultiplyFactory(list(range(1,21)))
    frequency_data = []
    for line in my_file.split("\n")[:args.top]:
        key, frequency = eval(line)
        frequency_data.extend(complete_frequency_factory.build(RawFrequencyData(key, frequency)))
    print(f"Reading is done! Total Complete Frequency Data: {len(frequency_data)}")
    print("A sample of complete frequency data:")
    print("\n".join(str(x) for x in frequency_data[:40]))
    datapoint_generator = DataPointGenerator()
    datapoint_generator.add_filter(DigitLimitFilter(2))
    datapoint_generator.add_template(MultiplyTemplate())
    datapoints = datapoint_generator.generate(frequency_data)
    print(f"Generated {len(datapoints)} data points!")
    print("A sample of data points:")
    print("\n".join(str(x) for x in datapoints[:40]))
    testcase_template = TestCaseTemplate("Q:", "A:", "hahaha")
    train_datapoints, test_datapoints = testcase_splitter(datapoints)
    testcases = testcase_generator(train_datapoints, test_datapoints, testcase_template)
    results = GPTJ_Analysis(testcases)
    print(f"Finished: {len(results)} test case is analyzed!")



