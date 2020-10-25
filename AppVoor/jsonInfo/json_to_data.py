from abc import ABC, abstractmethod
from typing import Union, Any, TypeVar

DesJson = Union[list, dict]
T = TypeVar('T')


class ABCJson(ABC):

    def __init__(self, file_path: str, data_type: Any) -> None:
        # initialize data and file_path
        if data_type is list:
            self._data: list = []
        elif data_type is dict:
            self._data: dict = {}
        else:
            raise TypeError

        self._file_path: str = file_path
        # load file into data variable when an object of this class is created
        self._load_file()

    @abstractmethod
    def _load_file(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self, key: Union[int, str]) -> tuple:
        pass

    @property
    def data(self) -> DesJson:
        return self._data

    @data.setter
    def data(self, value: DesJson) -> None:
        self._data: DesJson = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path: str = value

    def __len__(self) -> int:
        return len(self.data)
