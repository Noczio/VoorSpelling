import json
from abc import abstractmethod, ABC
from typing import Union, TypeVar, Generic, Any

import pandas as pd

from is_data import DataEnsurer

T = TypeVar('T')
DataFrame = pd.DataFrame


class DataLoader(ABC, Generic[T]):

    def __init__(self, file_path: str) -> None:
        # initialization when obj is created. By default _data is None
        self._file_path: str = file_path
        self._data: T = None

    @property
    def data(self) -> T:
        return self._data

    @data.setter
    def data(self, value: T) -> None:
        self._data = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

    @abstractmethod
    def get_file_transformed(self) -> T:
        pass


class JSONDataLoader(DataLoader[Union[list, dict]]):

    # abstract class method implementation
    def get_file_transformed(self) -> Union[list, dict]:
        # try to load file and set data, if error raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = json.load(f)
                if DataEnsurer.validate_py_data(temp, list) or DataEnsurer.validate_py_data(temp, dict):
                    self.data = temp
                    return self.data
                raise TypeError
        except():
            raise FileNotFoundError


class CSVDataLoader(DataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as ","
        separator = ","
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:

            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if DataEnsurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except():
            raise FileNotFoundError


class TSVDataLoader(DataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as "\t"
        separator = "\t"
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if DataEnsurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except():
            raise FileNotFoundError


class DataLoaderReturner:

    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    def get_data(self) -> Any:
        data = self._data_loader.get_file_transformed()
        return data
