from typing import List, Tuple

MATH_OPERATORS = {
    '+': 'plus',
    '*': 'times',
    '-': 'minus',
    '/': 'divides',
}


class FrequencyData:
    def __init__(self, key: str, frequency: int, x: str, y: str, z: str, word: str, math_operator: str):
        self.x = x
        self.y = y
        self.z = z
        self.word = word
        self.math_operator = math_operator
        self.key = key
        self.frequency = frequency

    def __str__(self):
        return f"FD(K: {self.key}, F: {self.frequency}, {self.x}, {self.y}, {self.z}, {self.word}, {self.math_operator})"


class CompleteFrequencyFactory:
    def __init__(self):
        pass

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        pass


class Num1MultiplyFactory(CompleteFrequencyFactory):
    def __init__(self, multiplicands: List[int]):
        super().__init__()
        self.multiplicands = multiplicands

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key)
        for multiplicand in self.multiplicands:
            complete_data = FrequencyData(key, frequency, str(x), str(multiplicand), str(x * multiplicand), 'times', '*')
            result.append(complete_data)
        return result


class Num2MultiplyFactory(CompleteFrequencyFactory):
    def build(self, key: Tuple[str, str], frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key[0])
        y = int(key[1])
        complete_data = FrequencyData(str(key), frequency, key[0], key[1], str(x * y), 'times', '*')
        result.append(complete_data)
        return result


class DayNum1ConvertFactory(CompleteFrequencyFactory):
    def build(self, key: Tuple[str, str], frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key[0])
        word = key[1]
        assert word == 'day', f"The word in the dataset must be day, but it's {word}"
        complete_data = FrequencyData(str(key), frequency, key[0], str(24), str(x*24), word, '*')
        result.append(complete_data)
        return result
