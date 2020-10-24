import json
from abc import abstractmethod, ABC
from typing import Union, TypeVar, Generic, Any

import pandas as pd

from is_data import DataEnsurer

T = TypeVar('T')
DataFrame = pd.DataFrame


class ABCDataLoader(ABC, Generic[T]):

    def __init__(self, full_path: str) -> None:
        # initialization when obj is created. By default _data is None
        self._full_path: str = full_path
        self._data: T = None

    @property
    def data(self) -> T:
        return self._data

    @data.setter
    def data(self, value: T) -> None:
        self._data = value

    @property
    def full_path(self) -> str:
        return self._full_path

    @full_path.setter
    def full_path(self, value: str) -> None:
        self._full_path = value

    @abstractmethod
    def get_file_transformed(self) -> T:
        pass


class JSONDataTypeLoader(ABCDataLoader[Union[list, dict]]):

    # abstract class method implementation
    def get_file_transformed(self) -> Union[list, dict]:
        data_ensurer = DataEnsurer()
        # try to load file and set data, if error raise FileNotFoundError
        try:
            with open(self.full_path, 'r', encoding="utf-8") as f:
                temp = json.load(f)
                if data_ensurer.validate_py_data(temp, list) or data_ensurer.validate_py_data(temp, dict):
                    self.data = temp
                    return self.data
                raise TypeError
        except():
            raise FileNotFoundError


class CSVDataTypeLoader(ABCDataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as ","
        separator = ","
        data_ensurer = DataEnsurer()
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:

            with open(self.full_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if data_ensurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except():
            raise FileNotFoundError


class TSVDataTypeLoader(ABCDataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as "\t"
        separator = "\t"
        data_ensurer = DataEnsurer()
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self.full_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if data_ensurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError
        except():
            raise FileNotFoundError


class DataReturner:

    def __init__(self, data_loader: ABCDataLoader) -> None:
        self._data_loader = data_loader

    def get_data(self) -> Any:
        data = self._data_loader.get_file_transformed()
        return data
