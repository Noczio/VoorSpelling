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
        min_data = 100
        if isinstance(data, pd.DataFrame):
            x_shape, y_shape = data.shape
            if x_shape <= min_data or y_shape == 1:
                return False
            return True
        return False


class DataEnsurer:

    def __init__(self) -> None:
        self._py_data_validator: IData = PyData()
        self._pd_data_validator: IData = PdData()

    def validate_py_data(self, data: Any, type_expected: Any) -> bool:
        return self._py_data_validator.data_is_valid(data, type_expected)

    def validate_pd_data(self, data: Any) -> bool:
        return self._pd_data_validator.data_is_valid(data, pd.DataFrame)
