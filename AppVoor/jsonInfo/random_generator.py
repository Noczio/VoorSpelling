import random
from typing import Any


class Randomizer:

    @staticmethod
    def get_random_number_range_int(start: int, end: int, step: int) -> int:
        random_number = random.randrange(start, end, step)
        return random_number

    @staticmethod
    def get_random_element_from_tuple(data: tuple) -> Any:
        random_choice = random.choice(data)
        return random_choice
