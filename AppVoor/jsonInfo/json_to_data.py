from abc import ABC, abstractmethod
from typing import Union, Any

from load_data import LoaderCreator, DataLoaderReturner


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
        loader = LoaderCreator.get_instance()
        json_type = loader.create_loader(self.file_path, "JSON")
        # use json_type as parameter for DataLoaderReturner and then get the data
        data_returner = DataLoaderReturner(json_type)
        self.data = data_returner.get_data()

    @abstractmethod
    def __getitem__(self, key: Union[int, str]) -> Any:
        pass

    @property
    def data(self) -> Union[list, dict]:
        return self._data

    @data.setter
    def data(self, value: Union[list, dict]) -> None:
        self._data: Union[list, dict] = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path: str = value

    def __len__(self) -> int:
        return len(self.data)
