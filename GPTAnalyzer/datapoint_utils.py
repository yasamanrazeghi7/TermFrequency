from typing import List
from collections import namedtuple
from frequency_data_utils import FrequencyData
import utils

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


class ArithmeticsTemplate(FrequencyDataTemplate):
    def __init__(self, factory_type):
        if factory_type in utils.factory_type_dict:
            self.keyword = utils.factory_type_dict[factory_type][1]
        else:
            assert True, 'factory type does not have a template'

    def generate(self, frequency_data: FrequencyData) -> DataPoint:
        question = "What is {0} {2} {1}?".format(frequency_data.x, frequency_data.y, self.keyword)
        answer = frequency_data.z
        return DataPoint(question, answer, frequency_data)


class ComparisonTemplate(FrequencyDataTemplate):
    def __init__(self, factory_type):
        if factory_type in utils.factory_type_dict.keys():
            self.keyword = utils.factory_type_dict[factory_type][1]
        else:
            assert True, 'factory type does not have a template'

    def generate(self, frequency_data: FrequencyData) -> DataPoint:
        question = "Which one is {2}? {0} or {1}".format(frequency_data.x, frequency_data.y, self.keyword)
        answer = frequency_data.z
        return DataPoint(question, answer, frequency_data)


class TimeUnitConversionTemplate(FrequencyDataTemplate):
    def __init__(self, factory_type):
        if 'minute' in factory_type.lower():
            self.time_unit_source = 'minutes'
            self.time_unit_des = 'seconds'
        elif 'hour' in factory_type.lower():
            self.time_unit_source = 'hours'
            self.time_unit_des = 'minutes'
        elif 'day' in factory_type.lower():
            self.time_unit_source = 'days'
            self.time_unit_des = 'hours'
        elif 'week' in factory_type.lower():
            self.time_unit_source = 'weeks'
            self.time_unit_des = 'days'
        elif 'month' in factory_type.lower():
            self.time_unit_source = 'months'
            self.time_unit_des = 'weeks'
        elif 'year' in factory_type.lower():
            self.time_unit_source = 'years'
            self.time_unit_des = 'months'
        elif 'decade' in factory_type.lower():
            self.time_unit_source = 'decades'
            self.time_unit_des = 'years'
        else:
            assert True, 'factory type does not have a template'

    def generate(self, frequency_data: FrequencyData) -> DataPoint:
        question = "What is {0} {1} in {2}?".format(frequency_data.x, self.time_unit_source, self.time_unit_des)
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

    def add_template(self, factory_type):
        if factory_type.startswith('Num') and ('less' in factory_type.lower() or 'more' in factory_type.lower()):
            self.templates.append(ComparisonTemplate(factory_type))
        elif factory_type.startswith('Num'):
            self.templates.append(ArithmeticsTemplate(factory_type))
        else:
            self.templates.append(TimeUnitConversionTemplate(factory_type))
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
