import json
import pathlib
from abc import ABC, abstractmethod

import pandas as pd

from is_data import is_dataframe


class ABCDataLoader(ABC):

    def __init__(self, full_path: str):
        # initialization
        self._full_path = full_path
        self._data = None

    # method to treat class as str
    def __str__(self):
        return self._full_path

    @abstractmethod
    def get_file_transformed(self):
        pass


class JSONDataTypeLoader(ABCDataLoader):

    # abstract class method implementation
    def get_file_transformed(self):
        # try to load file and set data, if error raise FileNotFoundError
        try:
            with open(self._full_path, 'r', encoding="utf-8") as f:
                self._data = json.load(f)
                return self._data
        except():
            raise FileNotFoundError


class CSVDataTypeLoader(ABCDataLoader):

    # abstract class method implementation
    def get_file_transformed(self):
        # initialize separator  as ","
        separator = ","
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        file = pathlib.Path(self._full_path)
        if file.exists():
            self._data = pd.read_csv(self._full_path, sep=separator)
            if is_dataframe(self._data):
                return self._data
            raise TypeError
        raise FileNotFoundError


class TSVDataTypeLoader(ABCDataLoader):

    # abstract class method implementation
    def get_file_transformed(self):
        # initialize separator  as "\t"
        separator = "\t"
        # try to load file. Raise TypeError if it does not meet requirements, else raise FileNotFoundError
        file = pathlib.Path(self._full_path)
        if file.exists():
            self._data = pd.read_csv(self._full_path, sep=separator)
            if is_dataframe(self._data):
                return self._data
            raise TypeError
        raise FileNotFoundError
