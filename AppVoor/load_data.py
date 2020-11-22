import json
from abc import abstractmethod, ABC
from typing import Union, TypeVar, Generic

import pandas as pd

from is_data import DataEnsurer

T = TypeVar('T')
DataFrame = pd.DataFrame


class DataLoader(ABC, Generic[T]):
    _data: T = None

    def __init__(self, file_path: str) -> None:
        # initialization when obj is created.
        self._file_path: str = file_path

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
                raise TypeError("Deserialized JSON file is neither a list nor a dict")
        except():
            raise FileNotFoundError("Path to JSON file was not found")


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
                raise TypeError("Data does not meet sample or column requirements to train a model")
        except():
            raise FileNotFoundError("Path to CSV file was not found")


class SCSVDataLoader(DataLoader[DataFrame]):

    # abstract class method implementation
    def get_file_transformed(self) -> DataFrame:
        # initialize separator  as ","
        separator = ";"
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if DataEnsurer.validate_pd_data(temp):
                    self.data = temp
                    return self.data
                raise TypeError("Data does not meet sample or column requirements to train a model")
        except():
            raise FileNotFoundError("Path to SCSV file was not found")


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
                raise TypeError("Data does not meet sample or column requirements to train a model")
        except():
            raise FileNotFoundError("Path to TSV file was not found")


class LoaderCreator:
    __instance = None
    _types: dict = {"CSV": CSVDataLoader(""), "TSV": TSVDataLoader(""), "JSON": JSONDataLoader(""),
                    "SCSV": SCSVDataLoader("")}

    @staticmethod
    def get_instance() -> "LoaderCreator":
        """Static access method."""
        if LoaderCreator.__instance is None:
            LoaderCreator()
        return LoaderCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if LoaderCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            LoaderCreator.__instance = self

    def create_loader(self, file_path: str, loader_type: str) -> DataLoader:
        # transform param to capital letters and then replace white spaces
        key = loader_type.upper().replace(" ", "")
        if key in self._types.keys():
            loader = self._types[key]
            loader.file_path = file_path
            return loader
        raise ValueError("Loader type value is wrong. It should be: CSV, TSV, SCSV or JSON")

    def get_available_types(self) -> tuple:
        available_types = [k for k in self._types.keys()]
        types = tuple(available_types)
        return types
