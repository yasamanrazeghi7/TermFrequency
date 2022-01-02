from typing import Callable, Any, List


class DataPoint:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer


class DataPointGenerator:
    def __init__(self):
        self.filters = []
        self.templates = []

    def add_filter(self, data_filter: Callable[[Any], bool]):
        self.filters.append(data_filter)
        return self

    def add_template(self, datapoint_template: Callable[[Any], DataPoint]):
        self.templates.append(datapoint_template)
        return self

    def generate(self, dataset) -> List[DataPoint]:
        pass


class TestCase:
    def __init__(self, body: str):
        self.body = body


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
    datapoint_generator = DataPointGenerator()\
            .add_filter(lambda x: True)\
            .add_template(lambda x: DataPoint(f"Hey {x}?", "q"))
    datapoints = datapoint_generator.generate("Asdas")
    testcase_template = TestCaseTemplate("Q:", "A:", "hahaha")
    train_datapoints, test_datapoints = testcase_splitter(datapoints)
    testcases = testcase_generator(train_datapoints, test_datapoints, testcase_template)
    results = GPTJ_Analysis(testcases)
    print(f"Finished: {len(results)} test case is analyzed!")



