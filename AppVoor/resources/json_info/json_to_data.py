from abc import ABC, abstractmethod
from typing import Any

from resources.backend_scripts.load_data import LoaderCreator


class JSONMessage(ABC):

    def __init__(self, file_path: str, data_type: Any) -> None:
        # initialize data and file_path
        if data_type is list:
            self._data: list = []
        elif data_type is dict:
            self._data: dict = {}
        else:
            raise TypeError("Parameter data_type is not a list or a dict")

        self._file_path: str = file_path
        # load file into data variable when an object of this class is created
        self._load_file()

    def _load_file(self) -> None:
        # data setter using JSONDataTypeLoader
        json_type = LoaderCreator.create_loader(self.file_path, "JSON")
        self.data = json_type.get_file_transformed()

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        pass

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

    def __len__(self) -> int:
        return len(self.data)
