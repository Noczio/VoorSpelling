from abc import ABC, abstractmethod
from typing import Union


class ABCJson(ABC):

    def __init__(self, file_path: str) -> None:
        # initialize data and file_path
        self._data: Union[list, dict] = []
        self._file_path: str = file_path

    @abstractmethod
    def _load_file(self) -> None:
        pass

    @abstractmethod
    def get_data_by_index(self, index: int) -> tuple:
        pass

    @property
    def data(self) -> Union[list, dict]:
        return self._data

    @data.setter
    def data(self, value: Union[list, dict]) -> None:
        self._data = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value
