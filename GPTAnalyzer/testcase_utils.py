from typing import List
import random
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
        result = ""
        if self.prompt_title:
            result = self.prompt_title
            result += "\n\n"
        for dp in train_datapoints:
            result += f"{self.question_prefix} {dp.question}\n"  # It does have a space between Q: and question
            result += f"{self.answer_prefix} {dp.answer}\n"
            result += "\n"
        result += f"{self.question_prefix} {test_datapoint.question}\n"
        result += f"{self.answer_prefix}"
        return TestCase(result, test_datapoint)


def testcase_generator(datapoints: List[DataPoint], template: TestCaseTemplate, shots_number: int) -> List[TestCase]:
    train_set, test_set = torch.utils.data.random_split(datapoints, [shots_number, len(datapoints) - shots_number])
    return [template.generate_testcase(train_set, test_datapoint) for test_datapoint in test_set]


def special_testcase_generator(list_of_datapoints: List[List[DataPoint]], template: TestCaseTemplate, shots_number: int) -> List[TestCase]:
    shot_number_per_group = shots_number // len(list_of_datapoints)
    remaining = shots_number % len(list_of_datapoints)
    all_train_set = []
    all_test_set = []
    for datapoints in list_of_datapoints:
        my_shot_number = shot_number_per_group
        if remaining > 0:
            my_shot_number += 1
            remaining -= 1
        train_set, test_set = torch.utils.data.random_split(datapoints, [my_shot_number, len(datapoints) - my_shot_number])
        for dp in train_set:
            all_train_set.append(dp)
        for dp in test_set:
            all_test_set.append(dp)
    random.shuffle(all_train_set)
    random.shuffle(all_test_set)
    return [template.generate_testcase(all_train_set, test_datapoint) for test_datapoint in all_test_set]
