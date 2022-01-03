from typing import List
from collections import namedtuple
from frequency_data_utils import FrequencyData

# DataPoint = namedtuple('DataPoint', ['question', 'answer', 'frequency_data'])
class DataPoint:
    def __init__(self, question: str, answer: str, frequency_data: FrequencyData):
        self.question = question
        self.answer = answer
        self.frequency_data = frequency_data

    def __str__(self):
        return f"({self.question}, {self.answer}, {self.frequency_data})"


# ------- Start Filters -----------

class FrequencyDataFilter:
    def is_valid(self, frequency_data: FrequencyData) -> bool:
        pass


class DigitLimitFilter(FrequencyDataFilter):
    def __init__(self, digit_limit: int = 2):
        super().__init__()
        self.digit_limit = digit_limit

    def is_valid(self, frequency_data: FrequencyData) -> bool:
        return len(frequency_data.x) <= self.digit_limit and len(frequency_data.y) <= self.digit_limit


# ------- End Filters -----------

# ------- Start Templates -----------


class FrequencyDataTemplate:
    def generate(self, complete_frequency_data: FrequencyData) -> DataPoint:
        pass


class MultiplyTemplate(FrequencyDataTemplate):
    def generate(self, frequency_data: FrequencyData) -> DataPoint:
        question = "What is {0} times {1}?".format(frequency_data.x, frequency_data.y)
        answer = frequency_data.z
        return DataPoint(question, answer, frequency_data)


class Day1Template(FrequencyDataTemplate):
    def generate(self, frequency_data: FrequencyData) -> DataPoint:
        question = "What is {0} days in hours?".format(frequency_data.x)
        answer = frequency_data.z
        return DataPoint(question, answer, frequency_data)


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

    def generate(self, frequency_dataset: List[FrequencyData]) -> List[DataPoint]:
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
