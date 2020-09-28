import pathlib

from jsonInfo.json_to_data import ABCJson
from jsonInfo.random_generator import get_random_number_range_int


class WelcomeMessage(ABCJson):

    def __init__(self, file_name="welcomeMessage.json"):
        super().__init__()
        self.load_file(file_name)

    def get_data(self, index: int) -> tuple:
        if (index < len(self._data)) and (index >= 0):
            author = self._data[index]["Author"]
            quote = self._data[index]["Quote"]
            return author, quote
        raise IndexError


class WelcomeMessenger:

    def __init__(self):
        self._json_message = WelcomeMessage()
        random_start = 0
        random_end = len(self._json_message.get_file_data())
        self._random_index = get_random_number_range_int(random_start, random_end, 1)

    def __str__(self):
        return self.get_welcome_message()

    def get_welcome_message(self) -> str:
        author, quote = self._json_message.get_data(self._random_index)
        message = quote + "." + " " + author
        return message
