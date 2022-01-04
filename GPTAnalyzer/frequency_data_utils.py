from typing import List, Tuple
import utils



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


class Num1PlusFactory(CompleteFrequencyFactory):
    def __init__(self, adding_numbers: List[int]):
        super().__init__()
        self.adding_numbers = adding_numbers

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key)
        for adding_number in self.adding_numbers:
            complete_data = FrequencyData(key, frequency, str(x), str(adding_number), str(x + adding_number), 'plus', '+')
            result.append(complete_data)
        return result


class Num2PlusFactory(CompleteFrequencyFactory):
    def build(self, key: Tuple[str, str], frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key[0])
        y = int(key[1])
        complete_data = FrequencyData(str(key), frequency, key[0], key[1], str(x + y), 'plus', '+')
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


class TimeUnitNum1ConvertFactory(CompleteFrequencyFactory):
    def __init__(self, time_unit_word):
        super().__init__()
        self.time_unit_word = time_unit_word
        if self.time_unit_word not in utils.TIME_UNIT_CONVERTORS.keys():
            assert True, 'time_unit word is wrong'
        self.multiplicand = utils.TIME_UNIT_CONVERTORS[self.time_unit_word]

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key[0])
        word = key[1]
        assert word == self.time_unit_word, f"The word in the dataset must be {self.time_unit_word}, but it's {word}"
        complete_data = FrequencyData(str(key), frequency, key[0], str(self.multiplicand),
                                      str(x * self.multiplicand), word, '*')
        result.append(complete_data)
        return result


def create_complete_frequency_factory(factory_type: str):
    if factory_type == "Num1Mult":
        complete_frequency_factory = Num1MultiplyFactory(list(range(1, 51)))
    elif factory_type == "Num2Mult":
        complete_frequency_factory = Num2MultiplyFactory()
    elif factory_type == "Num1Plus":
        complete_frequency_factory = Num1PlusFactory(list(range(1, 51)))
    elif factory_type == "Num2Plus":
        complete_frequency_factory = Num2PlusFactory()
    elif factory_type == "Minute1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('minute')
    elif factory_type == "Hour1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('hour')
    elif factory_type == "Day1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('day')
    elif factory_type == "Week1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('week')
    elif factory_type == "Month1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('month')
    elif factory_type == "Year1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('year')
    elif factory_type == "Decade1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('decade')
    else:
        assert False, 'args.factory-type is not correct'

    return complete_frequency_factory

