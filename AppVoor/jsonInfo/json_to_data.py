import json
import pathlib
from abc import ABC, abstractmethod


class ABCJson(ABC):

    def __init__(self):
        self._data = None

    def load_file(self, file_name: str):
        try:
            with open(file_name, 'r', encoding="utf-8") as f:
                self._data = json.load(f)
        except():
            raise FileNotFoundError

    def get_file_data(self) -> tuple:
        return tuple(self._data)

    @abstractmethod
    def get_data(self, index: int) -> tuple:
        pass
