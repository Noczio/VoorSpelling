import pathlib
from abc import ABC

import pandas as pd

from is_data import is_dataframe


class ABCDataLoader(ABC):

    def __init__(self, full_path: str):
        self._full_path = full_path
        self._df = None

    def __str__(self):
        return self._full_path

    def get_file_as_dataframe(self, separator: str):
        pass


class CSVDataTypeLoader(ABCDataLoader):

    def get_file_as_dataframe(self, separator=","):
        file = pathlib.Path(self._full_path)
        if file.exists():
            self._df = pd.read_csv(self._full_path, sep=separator)
            if is_dataframe(self._df):
                return self._df
            raise TypeError
        raise FileNotFoundError


class TSVDataTypeLoader(ABCDataLoader):

    def get_file_as_dataframe(self, separator="\t"):
        file = pathlib.Path(self._full_path)
        if file.exists():
            self._df = pd.read_csv(self._full_path, sep=separator)
            if is_dataframe(self._df):
                return self._df
            raise TypeError
        raise FileNotFoundError
