import pathlib
from abc import ABC

import pandas as pd

from parse_file import DataParseEnsurer


class ABCDataLoader(ABC):

    def __init__(self, full_path: str):
        self._full_path = full_path

    def __str__(self):
        return self._full_path

    def get_file_as_dataframe(self, separator: str):
        pass


class CSVDataTypeLoader(ABCDataLoader):

    def get_file_as_dataframe(self, separator=","):
        file = pathlib.Path(self._full_path)
        if file.exists():
            df = pd.read_csv(self._full_path, sep=separator)
            if DataParseEnsurer.is_dataframe(df):
                return df
            raise TypeError
        raise FileNotFoundError


class TSVDataTypeLoader(ABCDataLoader):

    def get_file_as_dataframe(self, separator="\t"):
        file = pathlib.Path(self._full_path)
        if file.exists():
            df = pd.read_csv(self._full_path, sep=separator)
            if DataParseEnsurer.is_dataframe(df):
                return df
            raise TypeError
        raise FileNotFoundError
