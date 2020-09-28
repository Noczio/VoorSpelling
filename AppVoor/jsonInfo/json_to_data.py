import json

from abc import ABC, abstractmethod


class ABCJson(ABC):

    def __init__(self):
        self._data = None

    def get_file_data(self) -> tuple:
        return tuple(self._data)

    @abstractmethod
    def _load_file(self):
        pass

    @abstractmethod
    def get_data_by_index(self, index: int) -> tuple:
        pass
