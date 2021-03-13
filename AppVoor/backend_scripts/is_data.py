from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

DataFrame = pd.DataFrame


class ValidData(ABC):

    @abstractmethod
    def data_is_valid(self, data: Any, expected: Any) -> bool:
        pass


class PyData(ValidData):

    def data_is_valid(self, data: Any, expected: Any) -> bool:
        if isinstance(data, expected):
            return True
        return False


class PdData(ValidData):

    def data_is_valid(self, data: Any, expected: Any) -> bool:
        # this method is supposed to be used to valid if a dataframe has enough samples and features to train
        if isinstance(data, expected):
            # x_shape is number of rows (samples) and y_shape is number of columns (features)
            x_shape, y_shape = data.shape
            min_x_data, min_y_data = 100, 2
            if y_shape >= min_y_data and x_shape >= min_x_data:
                not_valid = ("", ",", ";", "\t")
                columns = [value for value in data]
                # if all columns seem ok then proceed to check for rows
                for col in columns:
                    for col_char in str(col):
                        if col_char in not_valid:
                            return False
                """for every row, for every column: check if current data has a not valid info. if true 
                return a false (not a valid data)"""
                rows = [0, 1, min_x_data - 1, min_x_data, min_x_data + 1, x_shape - 2, x_shape - 1]
                for i in rows:
                    for j in columns:
                        for row_char in str(data.at[i, j]):
                            if row_char in not_valid:
                                return False
                return True
            return False
        return False


class DataEnsurer:

    @staticmethod
    def validate_py_data(data: Any, type_expected: Any) -> bool:
        py_data_validator: ValidData = PyData()
        return py_data_validator.data_is_valid(data, type_expected)

    @staticmethod
    def validate_pd_data(data: Any) -> bool:
        pd_data_validator: ValidData = PdData()
        return pd_data_validator.data_is_valid(data, pd.DataFrame)
