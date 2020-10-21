from typing import Any

import pandas as pd


class IsData:

    @staticmethod
    def is_dict(data: Any) -> bool:
        if isinstance(data, dict):
            return True
        return False

    @staticmethod
    def is_tuple(data: Any) -> bool:
        if isinstance(data, tuple):
            return True
        return False

    @staticmethod
    def is_list(data: Any) -> bool:
        if isinstance(data, list):
            return True
        return False

    @staticmethod
    def is_str(data: Any) -> bool:
        if isinstance(data, str):
            return True
        return False

    @staticmethod
    def is_float(data: Any) -> bool:
        if isinstance(data, float):
            return True
        return False

    @staticmethod
    def is_int(data: Any) -> bool:
        if isinstance(data, int):
            return True
        return False

    @staticmethod
    def is_dataframe(data: Any, min_data=100) -> bool:
        if isinstance(data, pd.DataFrame):
            x_shape, y_shape = data.shape
            if x_shape <= min_data or y_shape == 1:
                return False
            return True
        return False
