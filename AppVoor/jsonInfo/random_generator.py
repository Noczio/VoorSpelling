import random


class Randomizer:

    @staticmethod
    def get_random_number_range_int(start: int, end: int, step: int) -> int:
        if step < 0:
            raise ValueError("Step must be a positive integer greater than zero")
        elif start == end:
            return start
        else:
            random_number = random.randrange(start, end + 1, step)
            return random_number
