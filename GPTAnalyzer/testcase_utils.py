from typing import List
import torch
from datapoint_utils import DataPoint


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


def testcase_generator(datapoints: List[DataPoint], template: TestCaseTemplate, shots_number: int) -> List[TestCase]:
    train_set, test_set = torch.utils.data.random_split(datapoints, [shots_number, len(datapoints) - shots_number])
    return [template.generate_testcase(train_set, test_datapoint) for test_datapoint in test_set]
