import random


def get_random_number_range_int(start: int, end: int, step: int) -> int:
    random_number = random.randrange(start, end, step)
    return random_number


def get_random_element_from_tuple(data: tuple):
    random_number = random.choice(data)
    return random_number
