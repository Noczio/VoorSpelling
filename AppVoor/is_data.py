from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

DataFrame = pd.DataFrame


class IData(ABC):

    @abstractmethod
    def data_is_valid(self, data: Any, expected: Any):
        pass


class PyData(IData):

    def data_is_valid(self, data: Any, expected: Any):
        if isinstance(data, expected):
            return True
        return False


class PdData(IData):

    def data_is_valid(self, data: Any, expected: Any) -> bool:
        # this method is supposed to be used to valid if a dataframe has enough samples and features to train
        min_data = 100
        if isinstance(data, expected):
            # x_shape is number of rows (samples) and y_shape is number of columns (features)
            x_shape, y_shape = data.shape
            if x_shape <= min_data or y_shape == 1:
                return False
            return True
        return False


class DataEnsurer:

    @staticmethod
    def validate_py_data(data: Any, type_expected: Any) -> bool:
        py_data_validator: IData = PyData()
        return py_data_validator.data_is_valid(data, type_expected)

    @staticmethod
    def validate_pd_data(data: Any) -> bool:
        pd_data_validator: IData = PdData()
        return pd_data_validator.data_is_valid(data, pd.DataFrame)
