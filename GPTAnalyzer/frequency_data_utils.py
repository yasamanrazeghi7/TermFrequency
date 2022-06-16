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
    def __init__(self, factory_type: str):
        self.factory_type = factory_type
        pass

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        pass


class Num1MultiplyFactory(CompleteFrequencyFactory):
    def __init__(self, multiplicands: List[int], factory_type: str) -> object:
        super().__init__(factory_type)
        self.multiplicands = multiplicands

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key)
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        for multiplicand in self.multiplicands:
            complete_data = FrequencyData(key, frequency, str(x), str(multiplicand), str(x * multiplicand), word_in_template, math_operator)
            result.append(complete_data)
        return result


class Num1PlusFactory(CompleteFrequencyFactory):
    def __init__(self, adding_numbers: List[int], factory_type: str):
        super().__init__(factory_type)
        self.adding_numbers = adding_numbers

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key)
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        for adding_number in self.adding_numbers:
            complete_data = FrequencyData(key, frequency, str(x), str(adding_number), str(x + adding_number), word_in_template, math_operator)
            result.append(complete_data)
        return result


class Num1ComparisonFactory(CompleteFrequencyFactory):
    def __init__(self, comparing_numbers: List[int], factory_type: str):
        super().__init__(factory_type)
        self.comparing_numbers = comparing_numbers

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key)
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        for comparing_number in self.comparing_numbers:
            if comparing_number == x:
                continue
            else:
                if word_in_template == 'lower':
                    z = min(comparing_number, x)
                elif word_in_template == 'higher':
                    z = max(comparing_number, x)
                complete_data = FrequencyData(key, frequency, str(x), str(comparing_number), str(z), word_in_template, math_operator)
            result.append(complete_data)
        return result



class Num2PlusFactory(CompleteFrequencyFactory):
    def build(self, key: Tuple[str, str], frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key[0])
        y = int(key[1])
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        complete_data = FrequencyData(str(key), frequency, key[0], key[1], str(x + y), word_in_template , math_operator)
        result.append(complete_data)
        return result


class Num2MultiplyFactory(CompleteFrequencyFactory):
    def build(self, key: Tuple[str, str], frequency: int) -> List[FrequencyData]:
        result = []
        x = int(key[0])
        y = int(key[1])
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        complete_data = FrequencyData(str(key), frequency, key[0], key[1], str(x * y), word_in_template , math_operator)
        result.append(complete_data)
        return result


class Num1concatFactory(CompleteFrequencyFactory):
    def __init__(self, concatinations: List[int], factory_type: str) -> object:
        super().__init__(factory_type)
        self.concatinations = concatinations

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = key
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        for concat in self.concatinations:
            complete_data = FrequencyData(key, frequency, x, str(concat), x + str(concat),
                                          word_in_template, math_operator)
            result.append(complete_data)
        return result



class  Num1Mode10Factory(CompleteFrequencyFactory):
    def __init__(self, factory_type: str) -> object:
        super().__init__(factory_type)

    def build(self, key: str, frequency: int) -> List[FrequencyData]:
        result = []
        x = key
        (math_operator, word_in_template) = utils.factory_type_dict[self.factory_type]
        complete_data = FrequencyData(key, frequency, x, '10', x[-1], word_in_template, math_operator)
        result.append(complete_data)
        return result


class TimeUnitNum1ConvertFactory(CompleteFrequencyFactory):
    def __init__(self, time_unit_word, factory_type: str):
        super().__init__(factory_type)
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
    if factory_type == "Num1Mult" or factory_type == "Num1#*":
        complete_frequency_factory = Num1MultiplyFactory(list(range(0, 100)), factory_type=factory_type)
    elif factory_type == "Num2Mult":
        complete_frequency_factory = Num2MultiplyFactory(factory_type=factory_type)
    #check
    elif factory_type == "Num1concat#":
        complete_frequency_factory = Num1concatFactory(list(range(1, 51)), factory_type=factory_type)
    #check
    elif factory_type == "Num1mode10#":
        complete_frequency_factory = Num1Mode10Factory(factory_type=factory_type)
    elif factory_type == "Num1Plus" or factory_type == "Num1#+":
        complete_frequency_factory = Num1PlusFactory(list(range(0, 100)), factory_type=factory_type)
    elif factory_type == "Num2Plus":
        complete_frequency_factory = Num2PlusFactory(factory_type=factory_type)
    elif factory_type == "Num1Less" or factory_type == "Num1More":
        complete_frequency_factory = Num1ComparisonFactory(list(range(1, 101)), factory_type=factory_type)
    elif factory_type == "Minute1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('minute', factory_type=factory_type)
    elif factory_type == "Hour1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('hour', factory_type=factory_type)
    elif factory_type == "Day1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('day', factory_type=factory_type)
    elif factory_type == "Week1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('week', factory_type=factory_type)
    elif factory_type == "Month1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('month', factory_type=factory_type)
    elif factory_type == "Year1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('year', factory_type=factory_type)
    elif factory_type == "Decade1":
        complete_frequency_factory = TimeUnitNum1ConvertFactory('decade', factory_type=factory_type)
    else:
        assert False, 'args.factory-type is not correct'

    return complete_frequency_factory

