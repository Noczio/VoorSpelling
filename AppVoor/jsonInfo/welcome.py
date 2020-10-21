from jsonInfo.json_to_data import ABCJson
from jsonInfo.random_generator import Randomizer
from load_data import JSONDataTypeLoader, DataReturner


class WelcomeMessage(ABCJson):

    def __init__(self, file_path=".\\welcomeMessage.json") -> None:
        # call super class init
        super().__init__(file_path)
        # by default load file when object is created. _data is now set
        self._load_file()

    # abstract class method implementation
    def get_data_by_index(self, index: int) -> tuple:
        # make sure index is not out of boundaries
        if (index < len(self._data)) and (index >= -len(self._data)):
            # initialize local var author and quote
            author = self._data[index]["Author"]
            quote = self._data[index]["Quote"]
            return author, quote
        # index is out of boundaries. Raise IndexError
        raise IndexError

    # abstract class method implementation
    def _load_file(self) -> None:
        # data setter using JSONDataTypeLoader
        json_type = JSONDataTypeLoader(self._file_path)
        data_returner = DataReturner(json_type)
        self._data = data_returner.get_data()


class WelcomeMessenger:

    def __init__(self, json_message: ABCJson) -> None:
        # initialize a WelcomeMessage
        self._json_message = json_message
        # start, end and step for random choice
        random_start = 0
        random_end = len(self._json_message)
        random_step = 1
        # initialize random num using get_random_number_range_int
        self._random_index = Randomizer.get_random_number_range_int(random_start, random_end, random_step)

    # method to treat class as str. Returns get_welcome_message
    def __str__(self) -> str:
        return str(self.get_welcome_message())

    def __len__(self) -> int:
        return len(self.__str__())

    def get_welcome_message(self) -> str:
        # get author and quote from the json_message using get_data_by_index
        author, quote = self._json_message.get_data_by_index(self._random_index)
        # message is supposed to be "bla bla bla. author name"
        message = quote + "." + " " + author
        return message
