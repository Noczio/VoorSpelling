import random
from typing import Any


class Randomizer:

    @staticmethod
    def get_random_number_range_int(start: int, end: int, step: int) -> int:
        random_number = random.randrange(start, end, step)
        return random_number
