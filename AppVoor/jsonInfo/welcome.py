import json
import pathlib
from abc import ABC, abstractmethod


class ABCJson(ABC):

    def __init__(self, file_name: str):
        self._file_name = file_name
        self._data = None

    def _load_file(self, file_name: str):
        file = pathlib.Path(file_name)
        if file.exists():
            with open(file_name, 'r', encoding="utf-8") as f:
                self._data = json.load(f)
        raise FileNotFoundError

    @abstractmethod
    def get_data(self, index: int) -> tuple:
        pass


class WelcomeMessenger(ABCJson):

    def __init__(self, file_name="welcomeMessage.json"):
        super().__init__(file_name)
        self._load_file(self._file_name)

    def get_data(self, index: int) -> tuple:
        if (index < len(self._data)) and (index >= 0):
            author = self._data[index]["Author"]
            quote = self._data[index]["Quote"]
            return author, quote
        raise IndexError
