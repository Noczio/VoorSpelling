from jsonInfo.json_to_data import ABCJson
from jsonInfo.random_generator import get_random_number_range_int
from load_data import JSONDataTypeLoader


class WelcomeMessage(ABCJson):

    def __init__(self, file_name="welcomeMessage.json"):
        # using JSONDataTypeLoader
        self._json_type = JSONDataTypeLoader(file_name)
        # data by default is None
        self._data = None
        # call super class init
        super().__init__()
        # by default load file when object is created
        self._load_file()

    # abstract class method implementation
    def get_data_by_index(self, index: int) -> tuple:
        # make sure index is not out of boundaries
        if (index < len(self._data)) and (index >= 0):
            # initialize local var author and quote
            author = self._data[index]["Author"]
            quote = self._data[index]["Quote"]
            return author, quote
        # index is out of boundaries. Raise IndexError
        raise IndexError

    # abstract class method implementation
    def _load_file(self):
        # data setter using JSONDataTypeLoader
        self._data = self._json_type.get_file_transformed()


class WelcomeMessenger:

    def __init__(self):
        # initialize a WelcomeMessage
        self._json_message = WelcomeMessage()
        # start, end and step for random choice
        random_start = 0
        random_end = len(self._json_message.get_file_data())
        random_step = 1
        # initialize random num using get_random_number_range_int
        self._random_index = get_random_number_range_int(random_start, random_end, random_step)

    # method to treat class as str. Returns get_welcome_message
    def __str__(self):
        return self.get_welcome_message()

    def get_welcome_message(self) -> str:
        # get author and quote from the json_message using get_data_by_index
        author, quote = self._json_message.get_data_by_index(self._random_index)
        # message is supposed to be "bla bla bla. author name"
        message = quote + "." + " " + author
        return message
