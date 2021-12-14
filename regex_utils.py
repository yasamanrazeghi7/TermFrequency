import re

time_stamp = "second|minute|hour|day|month|week|year|decade"

# these can be eventually be loaded from a file
re_patterns = [
    r"(?=((?:^|\s)\d+\s+({0})s?\s+\d+\s+\d+(?:\s|$)))",  # finds number time_stamp number number
    r"(?=((?:^|\s)\d+\s+({0})s?\s+\d+(?:\s|$)))",  # number time_stamp number
    r"(?=((?:^|\s)\d+\s+({0})s?\s+[^\s]+\s+\d+(?:\s|$)))"  # number time_stamp non_number number
]

pattern_finder = lambda x: re.compile("|".join(re_pattern.format(time_stamp) for re_pattern in re_patterns),
                                      re.IGNORECASE).findall(x)  # this must match all the patterns including timestamp


def co_finder(x):  # this should prepare the keys as tuples and triplets with time_stamps
    this_time_stamp = ""
    tuple_list = []
    for z in x:
        if z != "":
            numbers = re.findall(r'\b(\d+)\b', z)
            if len(numbers) == 0:
                this_time_stamp = z
            else:
                if len(numbers) == 3:
                    tuple_list.append((numbers[0], numbers[1]))
                    tuple_list.append((numbers[0], numbers[2]))
                    tuple_list.append((numbers[0], min(numbers[1], numbers[2]), max(numbers[1], numbers[2])))
                elif len(numbers) == 2:
                    tuple_list.append((numbers[0], numbers[1]))
    result = []
    for z in tuple_list:
        result.append((z, this_time_stamp.lower()))
    return result


if __name__ == "__main__":
    print("okay")
