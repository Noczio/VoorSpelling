from typing import Any

from is_data import DataEnsurer
from jsonInfo.json_to_data import JSONMessage
from jsonInfo.random_generator import Randomizer


class WelcomeMessage(JSONMessage):

    # abstract class method implementation
    def __getitem__(self, key: int) -> tuple:
        if DataEnsurer.validate_py_data(key, int):
            # make sure index is not out of boundaries
            if (key < len(self.data)) and (key >= -len(self.data)):
                # initialize local var author and quote
                author = self.data[key]["Author"]
                quote = self.data[key]["Quote"]
                return author, quote
            # index is out of boundaries. Raise IndexError
            raise IndexError
        raise TypeError


class WelcomeMessenger:

    def __init__(self, file_path: str = ".\\welcomeMessage.json", data_type: Any = list) -> None:
        # initialize a WelcomeMessage
        json_path = file_path
        json_data_type = data_type
        self._json_message: JSONMessage = WelcomeMessage(file_path=json_path, data_type=json_data_type)
        # start, end and step for random choice
        random_start = 0
        random_end = len(self._json_message)
        random_step = 1
        # initialize random num using get_random_number_range_int
        self._random_index = Randomizer.get_random_number_range_int(random_start, random_end, random_step)

    def __str__(self) -> str:
        # get author and quote from the json_message using get_data_by_index
        author, quote = self._json_message[self._random_index]
        # message is supposed to be "bla bla bla. author name"
        message = quote + "." + " " + author
        return message

    def __len__(self) -> int:
        return len(self.__str__())
