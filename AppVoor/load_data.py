import json
from abc import ABC, abstractmethod

import pandas as pd

from is_data import is_dataframe


class ABCDataLoader(ABC):

    def __init__(self, full_path: str):
        # initialization when obj is created. By default _data is None
        self._full_path = full_path
        self._data = None

    # method to treat class as str
    def __str__(self):
        return self._full_path

    # method to treat class with len
    def __len__(self):
        return len(self._data)

    @abstractmethod
    def get_file_transformed(self):
        pass


class JSONDataTypeLoader(ABCDataLoader):

    # abstract class method implementation
    def get_file_transformed(self):
        # try to load file and set data, if error raise FileNotFoundError
        try:
            with open(self._full_path, 'r', encoding="utf-8") as f:
                temp = json.load(f)
                self._data = temp
                return self._data
        except():
            raise FileNotFoundError


class CSVDataTypeLoader(ABCDataLoader):

    # abstract class method implementation
    def get_file_transformed(self):
        # initialize separator  as ","
        separator = ","
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self._full_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if is_dataframe(temp):
                    self._data = temp
                    return self._data
                raise TypeError
        except():
            raise FileNotFoundError


class TSVDataTypeLoader(ABCDataLoader):

    # abstract class method implementation
    def get_file_transformed(self):
        # initialize separator  as "\t"
        separator = "\t"
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        try:
            with open(self._full_path, 'r', encoding="utf-8") as f:
                temp = pd.read_csv(f, sep=separator)
                if is_dataframe(temp):
                    self._data = temp
                    return self._data
                raise TypeError
        except():
            raise FileNotFoundError
